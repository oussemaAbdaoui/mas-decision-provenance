"""Offline rule-based stand-in for the LLM, for testing the pipeline without an API key.

It reads the structured payload that each agent attaches to its request (``meta``), applies
the rule engine, and adds seeded random noise so that the statistics code has something to
measure.  Results produced with this backend are NOT experimental results and are written
to a separate results folder with a warning banner.
"""
from __future__ import annotations

import hashlib
import random
import re

from agents.rules import evaluate, rule_decide


def _rng(tag: str, agent: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{tag}|{agent}".encode()).hexdigest()[:12], 16))


def respond(agent: str, meta: dict, tag: str, noise: float) -> dict:
    rng = _rng(tag, agent)
    if agent == "extractor":
        present = set(meta["passage_ids"])
        reqs = []
        for r in meta["gt_requirements"]:
            if r["source_passage"] not in present:
                continue
            r = {k: r[k] for k in ("id", "attribute", "operator", "value", "unit", "critical", "source_passage")}
            if rng.random() < noise:
                r["critical"] = not r["critical"]
            reqs.append(r)
        return {"requirements": reqs}

    if agent == "checker":
        out = []
        for r in meta["requirements"]:
            hits = [p for p in meta["passages"] if r["attribute"] in p.get("facts", {})]
            if not hits:
                out.append({"requirement": r["id"], "verdict": "unknown", "passages": [], "evidence": "not stated"})
                continue
            p = hits[-1]
            v = evaluate(r, p["facts"])
            if v != "unknown" and rng.random() < noise:
                v = "fail" if v == "pass" else "pass"
            out.append({"requirement": r["id"], "verdict": v, "passages": [p["id"]],
                        "evidence": f"{r['attribute']} = {p['facts'][r['attribute']]}"})
        return {"results": out}

    if agent == "decider":
        sel, steps = rule_decide(meta["requirements"], meta["checks"], meta["criteria"], meta["values"])
        crit_ids = [r["id"] for r in meta["requirements"] if r.get("critical")]
        used_k = [s["step"] for s in steps if re.fullmatch(r"K\d+X?", s["step"])]
        just = (f"Candidate {sel} satisfies the critical requirements {', '.join(crit_ids)} "
                f"and is preferred by {', '.join(used_k) or 'the critical gate'}.")
        return {"selected": sel, "justification": just, "cited_requirements": crit_ids, "cited_criteria": used_k}

    raise ValueError(agent)
