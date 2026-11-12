"""LLM access layer: one OpenAI-compatible client, a persistent call log/cache, call counters.

Every request and response is written to an SQLite file (``llm_calls.sqlite``).  The table
doubles as a cache: a call is identified by (model, messages, temperature, seed, tag).  The
``tag`` names the logical sample (e.g. ``exp0|drone_battery|3``), so

  * two different tags give two independent API calls even for identical prompts
    (needed to measure natural variability), and
  * re-running the whole pipeline with the same tags costs nothing and gives identical
    results (reproducibility from the log).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "llama-3.3-70b-versatile"},
    "together": {"base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY",
                 "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "meta-llama/llama-3.3-70b-instruct"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.1:8b"},
    "mock": {"base_url": None, "key_env": None, "model": "rule-mock"},
}


# USD per million tokens (input, output); used only for the budget guard and the report.
PRICES = {
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


class BudgetExceeded(RuntimeError):
    pass


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); does not override existing variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class CallLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, agent TEXT, tag TEXT,"
                " messages TEXT, response TEXT, parsed TEXT, usage TEXT, latency REAL, created REAL)")
            self.conn.commit()

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT parsed, usage FROM calls WHERE key=?", (key,)).fetchone()
        return None if row is None else (json.loads(row[0]), json.loads(row[1] or "{}"))

    def put(self, key, model, agent, tag, messages, response, parsed, usage, latency):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (key, model, agent, tag, json.dumps(messages, ensure_ascii=False), response,
                               json.dumps(parsed, ensure_ascii=False), json.dumps(usage), latency, time.time()))
            self.conn.commit()


class Counter:
    """Counts logical calls (what the experiment needs) and API calls (cache misses)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = defaultdict(lambda: defaultdict(int))

    def add(self, label: str, agent: str, api: bool, usage: dict):
        with self.lock:
            d = self.data[label]
            d["logical"] += 1
            d[f"logical_{agent}"] += 1
            if api:
                d["api"] += 1
                d["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                d["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def snapshot(self) -> dict:
        with self.lock:
            return {k: dict(v) for k, v in self.data.items()}


def _extract_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLM:
    def __init__(self, backend: str = "groq", model: str | None = None, log_path: Path | None = None,
                 temperature: float = 0.0, seed: int = 42, max_concurrency: int = 4, mock_noise: float = 0.003,
                 extra_body: dict | None = None, budget_usd: float | None = None):
        if backend not in PROVIDERS:
            raise ValueError(f"unknown backend {backend!r}; choose from {sorted(PROVIDERS)}")
        cfg = PROVIDERS[backend]
        self.backend = backend
        self.model = model or os.environ.get("LLM_MODEL") or cfg["model"]
        self.temperature = temperature
        self.seed = seed
        self.mock_noise = mock_noise
        self.extra_body = extra_body or {}   # provider-specific options, e.g. reasoning switch
        self.budget_usd = budget_usd
        self.spent_usd = 0.0                  # spent by this process (cached calls are free)
        self.spend_lock = threading.Lock()
        self.counter = Counter()
        self.log = CallLog(log_path or Path("results") / "llm_calls.sqlite")
        self.sem = threading.Semaphore(max_concurrency)
        self.json_mode = True
        self.client = None
        if backend != "mock":
            from openai import OpenAI
            key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else "ollama"
            key = key or os.environ.get("LLM_API_KEY")
            self.missing_key = None if key else f"Set {cfg['key_env']} (or LLM_API_KEY) in the environment or in .env"
            # without a key the client still works for calls already in the call log (exact reproduction)
            self.client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", cfg["base_url"]), api_key=key or "missing",
                                 max_retries=0, timeout=120)

    @property
    def model_id(self) -> str:
        return f"{self.backend}/{self.model}"

    def _key(self, messages, tag) -> str:
        blob = json.dumps({"model": self.model_id, "messages": messages, "temperature": self.temperature,
                           "seed": self.seed, "tag": tag,
                           **({"extra": self.extra_body} if self.extra_body else {})}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete_json(self, messages: list[dict], *, agent: str, tag: str, label: str, meta: dict | None = None) -> dict:
        key = self._key(messages, tag)
        hit = self.log.get(key)
        if hit is not None:
            self.counter.add(label, agent, api=False, usage={})
            return hit[0]
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"budget of ${self.budget_usd:.2f} reached (spent ${self.spent_usd:.3f}); "
                                 "re-run with a higher --budget-usd to resume from the call log")
        t0 = time.time()
        if self.backend == "mock":
            from agents import mock
            parsed = mock.respond(agent, meta or {}, tag, self.mock_noise)
            text, usage = json.dumps(parsed), {}
        else:
            text, parsed, usage = self._call_api(messages)
        self.log.put(key, self.model_id, agent, tag, messages, text, parsed, usage, time.time() - t0)
        self.counter.add(label, agent, api=True, usage=usage)
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        with self.spend_lock:
            self.spent_usd += (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6
        return parsed

    def _call_api(self, messages):
        if getattr(self, "missing_key", None):
            raise BudgetExceeded(self.missing_key + " (this call is not in the call log)")
        import openai
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        msgs = list(messages)
        for attempt in range(2):  # second attempt = JSON repair request
            text = self._request(msgs, usage_total)
            parsed = _extract_json(text)
            if isinstance(parsed, dict):
                return text, parsed, usage_total
            msgs = msgs + [{"role": "assistant", "content": text or ""},
                           {"role": "user", "content": "That was not valid JSON. Reply again with only the JSON object."}]
        return text, {"_parse_error": True, "raw": text}, usage_total

    def _request(self, msgs, usage_total) -> str:
        import openai
        delay = 2.0
        for attempt in range(10):
            try:
                kwargs = dict(model=self.model, messages=msgs, temperature=self.temperature, seed=self.seed)
                if self.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                with self.sem:
                    resp = self.client.chat.completions.create(**kwargs)
                if resp.usage:
                    usage_total["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    usage_total["completion_tokens"] += resp.usage.completion_tokens or 0
                return resp.choices[0].message.content or ""
            except openai.BadRequestError as e:
                if self.json_mode and "response_format" in str(e).lower():
                    self.json_mode = False
                    continue
                if "json" in str(e).lower() and attempt < 2:   # e.g. Groq json_validate_failed
                    continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                wait = delay
                retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
                if "retry-after" in retry_after:
                    try:
                        wait = max(wait, float(retry_after["retry-after"]))
                    except ValueError:
                        pass
                time.sleep(min(90.0, wait) + random.random())
                delay = min(90.0, delay * 2)
        raise RuntimeError("LLM request failed after 10 retries")
