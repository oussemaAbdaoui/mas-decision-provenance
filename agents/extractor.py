"""Extractor agent: specification passages -> structured requirements."""
from __future__ import annotations

import json
import re

SYSTEM = ("You are a requirements engineer. You turn a technical specification into structured, "
          "checkable requirements. You answer with a single JSON object and nothing else.")

USER = """Specification passages (each line starts with its passage ID):

{passages}

Extract every requirement labelled [Rxx] in the passages. For each requirement return:
- "id": its label, e.g. "R01"
- "attribute": short snake_case name of the constrained property, with the unit as suffix when numeric (e.g. "mass_g", "capacity_wh", "operating_temp_c", "certifications", "features")
- "operator": one of ">=", "<=", "==", "in_range", "includes"
    ">=" / "<=" for numeric thresholds; "==" for an exact value; "in_range" when a range [min, max] must lie inside the component's rated range; "includes" when the component's list of certifications / features / compatible media must contain the value
- "value": a number, a string, or [min, max] for in_range. If the threshold is not stated directly, compute it from the information in the passage.
- "unit": the unit of "value" ("" if none)
- "critical": true if the requirement is mandatory (violating it disqualifies a component), false if it is only a recommendation or preference
- "source_passage": the ID of the passage the requirement comes from

Selection criteria, context and out-of-scope statements are not requirements.

Return JSON of the form {{"requirements": [{{"id": "R01", "attribute": "...", "operator": "...", "value": ..., "unit": "...", "critical": true, "source_passage": "S02"}}]}}"""

OPS = {"≥": ">=", "=>": ">=", "≤": "<=", "=<": "<=", "=": "==", "range": "in_range", "within": "in_range",
       "contains": "includes", "include": "includes", "in": "includes"}


def _coerce_value(v):
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v.strip()
    if isinstance(v, list):
        return [_coerce_value(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1", "mandatory", "critical")


def normalise(raw: dict, passage_ids: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in raw.get("requirements", []) if isinstance(raw, dict) else []:
        if not isinstance(r, dict):
            continue
        m = re.search(r"R\d+", str(r.get("id", "")).upper())
        if not m or m.group(0) in seen:
            continue
        rid = m.group(0)
        if len(rid) == 2:
            rid = f"R0{rid[1]}"
        seen.add(rid)
        op = str(r.get("operator", "")).strip()
        op = OPS.get(op, op)
        src = str(r.get("source_passage", "")).strip()
        out.append({"id": rid, "attribute": str(r.get("attribute", "")).strip(), "operator": op,
                    "value": _coerce_value(r.get("value")), "unit": str(r.get("unit", "") or ""),
                    "critical": _coerce_bool(r.get("critical")),
                    "source_passage": src if src in passage_ids else None})
    return sorted(out, key=lambda r: r["id"])


def run(llm, scen, spec_passages: list[dict], *, tag: str, label: str) -> list[dict]:
    text = "\n".join(f"[{p['id']}] {p['text']}" for p in spec_passages)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(passages=text)}]
    meta = {"passage_ids": [p["id"] for p in spec_passages], "gt_requirements": scen.gt_requirements}
    raw = llm.complete_json(messages, agent="extractor", tag=tag, label=label, meta=meta)
    return normalise(raw, {p["id"] for p in spec_passages})


def public_view(req: dict) -> dict:
    """What downstream agents see of a requirement (structure only, no source text)."""
    return {k: req[k] for k in ("id", "attribute", "operator", "value", "unit")}


def describe(req: dict) -> str:
    v = req["value"]
    v = f"[{v[0]}, {v[1]}]" if isinstance(v, list) and len(v) == 2 else json.dumps(v, ensure_ascii=False)
    return f"{req['attribute']} {req['operator']} {v} {req['unit']}".strip()
