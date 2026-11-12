"""Shared experiment context and helpers."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from agents.chain import replay, state_of
from experiments.stats import flip_test

PROFILES = {
    # n0: Exp.0 runs | reps: Exp.1 replays per perturbation | probe: replays per uncited-passage probe
    # reps3: Exp.3 replays (faulty run + localisation) | group: requirements per group in grouped ablation
    "full": {"n0": 10, "reps": 5, "probe": 3, "reps3": 3, "group": 4},
    "quick": {"n0": 5, "reps": 3, "probe": 1, "reps3": 2, "group": 4},
    # ~1,000 calls for one scenario: curtailed sequential testing, Exp. 3 localised at requirement level
    # 5 replays: with a noise floor of 3/9, even 3/3 flips is not significant (p = 0.09); 5/5 is (p = 0.03)
    "budget": {"n0": 10, "reps": 5, "probe": 5, "reps3": 5, "group": 4, "curtail": True,
               "exp3_scope": "requirements", "exp3_errors": ["E1", "E2", "E3"]},
}


@dataclass
class Ctx:
    llm: object
    out: Path
    cfg: dict
    workers: int = 4
    alpha: float = 0.05
    log: list = field(default_factory=list)

    def pmap(self, fn, items):
        items = list(items)
        if self.workers <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))

    def save(self, rel: str, obj) -> Path:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return p


def replay_many(ctx: Ctx, scen, base: dict, jobs: list[dict], label: str) -> list[dict]:
    """jobs: {"state", "dirty", "tag", ...}. Returns jobs annotated with the replayed decision."""
    ref_checks = base["checks"]

    def one(job):
        t = replay(ctx.llm, scen, base, job["state"], job["dirty"], tag=job["tag"], label=label)
        changed = sum(1 for cid, res in t["checks"].items() for rid, v in res.items()
                      if ref_checks.get(cid, {}).get(rid, {}).get("verdict") != v["verdict"])
        return {**{k: v for k, v in job.items() if k != "state"}, "selected": t["decision"]["selected"],
                "cited": t["decision"]["cited"], "changed_checks": changed,
                "calls": len(set(job["dirty"]) | {"decider"}) if "extractor" not in job["dirty"]
                else 2 + len(base["checks"])}
    return ctx.pmap(one, jobs)


def summarise(results: list[dict], ref_decision: str, noise: dict, key=("element", "kind")) -> list[dict]:
    groups = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    rows = []
    for k, rs in groups.items():
        flips = sum(1 for r in rs if r["selected"] != ref_decision)
        p = flip_test(flips, len(rs), noise["flips"], noise["n"])
        rows.append({**dict(zip(key, k)), "n": len(rs), "flips": flips, "flip_rate": flips / len(rs), "p": p,
                     "decisions": sorted({r["selected"] for r in rs}), "calls": sum(r["calls"] for r in rs),
                     "mean_changed_checks": sum(r["changed_checks"] for r in rs) / len(rs),
                     "description": rs[0].get("description", "")})
    return rows


def run_tests(ctx: Ctx, scen, base: dict, tests: list[dict], noise: dict, reps: int, label: str) -> list[dict]:
    """Replay each test up to `reps` times and summarise.

    tests: {"element", "kind", "state", "dirty", "description", "tag_prefix"}; rep r uses tag f"{tag_prefix}|{r}".

    With cfg["curtail"], replications run in rounds and a test stops as soon as its conclusion is
    fixed: if even "every remaining replay flips" cannot reach p < alpha, further replays cannot
    change the verdict (curtailed sequential testing).  The significant / not-significant
    conclusion is therefore identical to the fixed-`reps` design; only flip-rate estimates of
    non-significant tests rest on fewer replays.
    """
    ref = base["decision"]["selected"]
    if not ctx.cfg.get("curtail"):
        jobs = [{**t, "rep": r, "tag": f"{t['tag_prefix']}|{r}"} for t in tests for r in range(reps)]
        rows = summarise(replay_many(ctx, scen, base, jobs, label), ref, noise)
        for r in rows:
            r["curtailed"] = False
        return rows
    done: dict[int, list] = {i: [] for i in range(len(tests))}
    active = list(range(len(tests)))
    for r in range(reps):
        if not active:
            break
        jobs = [{**tests[i], "rep": r, "tag": f"{tests[i]['tag_prefix']}|{r}", "_i": i} for i in active]
        for res in replay_many(ctx, scen, base, jobs, label):
            done[res["_i"]].append(res)
        still = []
        for i in active:
            flips = sum(1 for x in done[i] if x["selected"] != ref)
            remaining = reps - len(done[i])
            if remaining and flip_test(flips + remaining, reps, noise["flips"], noise["n"]) < ctx.alpha:
                still.append(i)          # significance still reachable -> keep replicating
        active = still
    results = [x for i in done for x in done[i]]
    rows = summarise(results, ref, noise)
    for row in rows:
        row["curtailed"] = row["n"] < reps
        row["max_reps"] = reps
    return rows


__all__ = ["Ctx", "PROFILES", "replay_many", "run_tests", "summarise", "state_of"]
