"""Deterministic rule engine.

Used for three things only, never inside the LLM chain:
  * computing the ground-truth check matrix and decision of each scenario,
  * the offline `mock` backend (pipeline testing without an API key),
  * scoring LLM outputs against ground truth.
"""
from __future__ import annotations

VERDICTS = ("pass", "fail", "unknown")


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def evaluate(req: dict, facts: dict) -> str:
    attr = req.get("attribute")
    if attr not in facts:
        return "unknown"
    v, op, t = facts[attr], req.get("operator"), req.get("value")
    try:
        if op in (">=", "<=", ">", "<"):
            a, b = _num(v), _num(t)
            if a is None or b is None:
                return "unknown"
            ok = {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}[op]
        elif op == "==":
            a, b = _num(v), _num(t)
            ok = abs(a - b) < 1e-9 if (a is not None and b is not None) else str(v).strip().lower() == str(t).strip().lower()
        elif op == "in_range":
            lo, hi = (_num(x) for x in t)
            clo, chi = (_num(x) for x in v)
            ok = clo <= lo and chi >= hi
        elif op == "includes":
            items = v if isinstance(v, list) else [v]
            ok = str(t).strip().lower() in [str(i).strip().lower() for i in items]
        else:
            return "unknown"
    except (TypeError, ValueError):
        return "unknown"
    return "pass" if ok else "fail"


def merge_facts(passages: list[dict]) -> dict:
    """Later passages override earlier ones (so an injected 'revision note' wins)."""
    out: dict = {}
    for p in passages:
        out.update(p.get("facts", {}))
    return out


def _verdict(checks: dict, cid: str, rid: str) -> str:
    return checks.get(cid, {}).get(rid, {}).get("verdict", "unknown")


def rule_decide(requirements: list[dict], checks: dict, criteria: list[dict], values: dict):
    """Critical gate, then criteria in priority order. Returns (selected, steps)."""
    cands = sorted(checks)

    def critical_failures(c):
        return sum(1 for r in requirements if r.get("critical") and _verdict(checks, c, r["id"]) != "pass")

    def nc_pass(c):
        return sum(1 for r in requirements if not r.get("critical") and _verdict(checks, c, r["id"]) == "pass")

    m = min(critical_failures(c) for c in cands)
    pool = [c for c in cands if critical_failures(c) == m]
    steps = [{"step": "critical_gate", "pool": list(pool), "critical_failures": m}]
    for k in criteria:
        if len(pool) == 1:
            break
        vals = {}
        for c in pool:
            if k["attribute"] == "noncritical_pass_count":
                vals[c] = float(nc_pass(c))
            else:
                vals[c] = _num(values.get(c, {}).get(k["attribute"]))
        known = {c: v for c, v in vals.items() if v is not None}
        if not known:
            continue
        best = min(known.values()) if k["direction"] == "min" else max(known.values())
        pool = [c for c in pool if vals[c] == best]
        steps.append({"step": k["id"], "pool": list(pool)})
    return pool[0], steps


def ground_truth_checks(scen) -> dict:
    out = {}
    for c in scen.candidates:
        facts = merge_facts(c["passages"])
        out[c["id"]] = {r["id"]: {"verdict": evaluate(r, facts)} for r in scen.gt_requirements}
    return out


def ground_truth_decision(scen) -> tuple[str, list]:
    checks = ground_truth_checks(scen)
    values = {c["id"]: merge_facts(c["passages"]) for c in scen.candidates}
    return rule_decide(scen.gt_requirements, checks, scen.criteria, values)


def values_match(a, b, rel_tol: float = 0.02) -> bool:
    """Loose comparison of an extracted value with the ground-truth value."""
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(values_match(x, y, rel_tol) for x, y in zip(a, b))
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= rel_tol * max(1.0, abs(nb))
    return str(a).strip().lower().replace("-", "_").replace(" ", "_") == str(b).strip().lower().replace("-", "_").replace(" ", "_")
