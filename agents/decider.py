"""Decider agent: check matrix + comparison criteria -> one selected candidate + justification."""
from __future__ import annotations

import re

from agents.extractor import describe
from agents.rules import merge_facts

SYSTEM = ("You are the design authority of an engineering project. You select exactly one component, basing "
          "your choice only on the compliance results and comparison criteria you are given. You answer with a "
          "single JSON object and nothing else.")

USER = """Requirements:
{requirements}

Compliance results per candidate:
{matrix}

Comparison criteria, in priority order:
{criteria}

Datasheet excerpts relevant to the criteria:
{excerpts}

Selection policy: a candidate whose result is "fail" or "unknown" on any critical requirement must not be selected, unless no candidate passes all critical requirements (then prefer the candidate with the fewest critical failures). Among the remaining candidates, apply the comparison criteria in priority order.

Return JSON of the form {{"selected": "<candidate id>", "justification": "<3 to 5 sentences explaining the choice; cite the requirement IDs (e.g. R03) and criterion IDs (e.g. K1) the choice relies on>", "cited_requirements": ["R.."], "cited_criteria": ["K.."]}}"""

ID_RE = re.compile(r"\b(R\d{2}X?|K\d+X?)\b")


def criteria_passages(datasheets: dict, criteria: list[dict]) -> dict:
    """Deterministic retrieval step (no LLM): datasheet passages annotated with a criterion attribute."""
    attrs = {k["attribute"] for k in criteria}
    return {cid: [p for p in ds["passages"] if attrs & set(p.get("facts", {}))] for cid, ds in datasheets.items()}


def normalise(raw: dict, cand_ids: list[str], names: dict, req_ids: set[str], crit_ids: set[str]) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    sel = str(raw.get("selected", "")).strip()
    if sel.upper() in cand_ids:
        sel = sel.upper()
    else:
        sel = next((c for c in cand_ids if names[c].lower() in sel.lower() or sel.lower() == f"candidate {c.lower()}"), "INVALID")
    just = str(raw.get("justification", ""))
    listed = {str(x).strip().upper() for x in (raw.get("cited_requirements") or []) + (raw.get("cited_criteria") or [])}
    in_text = set(ID_RE.findall(just))
    cited = sorted((listed | in_text) & (req_ids | crit_ids))
    return {"selected": sel, "justification": just, "cited": cited,
            "cited_listed": sorted(listed & (req_ids | crit_ids)), "cited_in_text": sorted(in_text & (req_ids | crit_ids))}


def run(llm, scen, requirements, checks, criteria, datasheets, *, tag: str, label: str) -> tuple[dict, list[str]]:
    names = {cid: ds["name"] for cid, ds in datasheets.items()}
    rtxt = "\n".join(f"- {r['id']} ({'critical' if r['critical'] else 'non-critical'}): {describe(r)}" for r in requirements)
    mtxt = "\n".join(f"Candidate {cid} ({names[cid]}): " +
                     ", ".join(f"{r['id']}={checks[cid].get(r['id'], {}).get('verdict', 'unknown')}" for r in requirements)
                     for cid in sorted(checks))
    ctxt = "\n".join(f"{i}. [{k['id']}] {k['text']}" for i, k in enumerate(criteria, 1))
    cps = criteria_passages(datasheets, criteria)
    etxt = "\n".join(f"[{p['id']}] (candidate {cid}) {p['text']}" for cid in sorted(cps) for p in cps[cid]) or "(none)"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(requirements=rtxt, matrix=mtxt, criteria=ctxt, excerpts=etxt)}]
    meta = {"requirements": requirements, "checks": checks, "criteria": criteria,
            "values": {cid: merge_facts(ps) for cid, ps in cps.items()}}
    raw = llm.complete_json(messages, agent="decider", tag=tag, label=label, meta=meta)
    dec = normalise(raw, sorted(checks), names, {r["id"] for r in requirements}, {k["id"] for k in criteria})
    return dec, [p["id"] for cid in sorted(cps) for p in cps[cid]]
