"""Checker agent: one call per candidate, one verdict (with cited passages) per requirement."""
from __future__ import annotations

import json

from agents.extractor import public_view

SYSTEM = ("You are a compliance engineer. You check one component datasheet against a list of requirements, "
          "using only the information in the datasheet passages. You answer with a single JSON object and nothing else.")

USER = """Component: {name} (candidate {cid})

Datasheet passages:
{passages}

Requirements:
{requirements}

For EVERY requirement give a verdict:
- "pass": the datasheet shows that the requirement is satisfied (you may convert units or compute derived values, e.g. Wh = mAh x V / 1000, L/min = m3/h x 1000 / 60, 1 bar = 10.2 m of head, bending radius = factor x outer diameter, derated current = rating x derating factor),
- "fail": the datasheet shows that it is not satisfied,
- "unknown": the datasheet does not contain the information needed.
For "in_range", the component's rated range must cover the whole required range. For "includes", the component must explicitly have the listed property.
Cite the ID(s) of the passage(s) you used (empty list for "unknown") and give a one-sentence evidence.

Return JSON of the form {{"results": [{{"requirement": "R01", "verdict": "pass", "passages": ["{cid}_p1"], "evidence": "..."}}]}}"""


def normalise(raw: dict, requirements: list[dict], passage_ids: set[str]) -> dict:
    by_id = {}
    for r in raw.get("results", []) if isinstance(raw, dict) else []:
        if isinstance(r, dict) and r.get("requirement") is not None:
            by_id.setdefault(str(r["requirement"]).strip().upper(), r)
    out = {}
    for req in requirements:
        r = by_id.get(req["id"])
        if r is None:
            out[req["id"]] = {"verdict": "unknown", "passages": [], "evidence": "", "missing": True}
            continue
        v = str(r.get("verdict", "")).strip().lower()
        cited = r.get("passages") or []
        if isinstance(cited, str):
            cited = [cited]
        out[req["id"]] = {"verdict": v if v in ("pass", "fail", "unknown") else "unknown",
                          "passages": [p for p in (str(c).strip() for c in cited) if p in passage_ids],
                          "evidence": str(r.get("evidence", ""))}
    return out


def run(llm, cand_id: str, cand_name: str, requirements: list[dict], passages: list[dict], *, tag: str, label: str) -> dict:
    ptxt = "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    rtxt = "\n".join(json.dumps(public_view(r), ensure_ascii=False) for r in requirements)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(name=cand_name, cid=cand_id, passages=ptxt, requirements=rtxt)}]
    meta = {"requirements": requirements, "passages": passages}
    raw = llm.complete_json(messages, agent="checker", tag=f"{tag}|{cand_id}", label=label, meta=meta)
    return normalise(raw, requirements, {p["id"] for p in passages})
