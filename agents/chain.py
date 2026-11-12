"""The three-agent chain and the partial replay used by the perturbation experiments.

A *trace* is a JSON-serialisable record of one run: the inputs every agent saw, every output,
and one activity record per agent call.  `provenance.graph.build_graph` turns it into a PROV graph.
"""
from __future__ import annotations

import copy

from agents import checker, decider, extractor

ACT_EXTRACTOR = "extractor"
ACT_DECIDER = "decider"


def act_checker(cid: str) -> str:
    return f"checker_{cid}"


def initial_inputs(scen) -> dict:
    return {
        "spec_passages": copy.deepcopy(scen.spec_passages),
        "datasheets": {c["id"]: {"name": c["name"], "passages": copy.deepcopy(c["passages"])} for c in scen.candidates},
        "criteria": copy.deepcopy(scen.criteria),
    }


def state_of(trace: dict) -> dict:
    """The perturbable state of a trace: raw inputs + the extracted requirements."""
    return {"inputs": copy.deepcopy(trace["inputs"]), "requirements": copy.deepcopy(trace["requirements"])}


def _execute(llm, scen, state: dict, base: dict | None, dirty: set[str], tag: str, label: str) -> dict:
    inputs = state["inputs"]
    activities, reused = {}, []

    if base is None or ACT_EXTRACTOR in dirty:
        reqs = extractor.run(llm, scen, inputs["spec_passages"], tag=f"{tag}|extractor", label=label)
    else:
        reqs = copy.deepcopy(state["requirements"])
        reused.append(ACT_EXTRACTOR)
    activities[ACT_EXTRACTOR] = {"agent": "extractor", "used": [p["id"] for p in inputs["spec_passages"]],
                                 "generated": [r["id"] for r in reqs]}

    checks = {}
    for cid, ds in inputs["datasheets"].items():
        a = act_checker(cid)
        if base is None or a in dirty:
            checks[cid] = checker.run(llm, cid, ds["name"], reqs, ds["passages"], tag=f"{tag}|checker", label=label)
        else:
            checks[cid] = copy.deepcopy(base["checks"][cid])
            reused.append(a)
        activities[a] = {"agent": "checker", "candidate": cid,
                         "used": [r["id"] for r in reqs] + [p["id"] for p in ds["passages"]],
                         "generated": [f"{rid}@{cid}" for rid in checks[cid]]}

    dec, routed = decider.run(llm, scen, reqs, checks, inputs["criteria"], inputs["datasheets"],
                              tag=f"{tag}|decider", label=label)
    activities[ACT_DECIDER] = {"agent": "decider",
                               "used": [f"{rid}@{cid}" for cid in checks for rid in checks[cid]]
                                       + [k["id"] for k in inputs["criteria"]] + routed,
                               "generated": ["decision"], "routed_passages": routed}
    return {"scenario": scen.name, "run_id": tag, "model": llm.model_id, "inputs": inputs,
            "requirements": reqs, "checks": checks, "decision": dec, "activities": activities,
            "reused_activities": reused}


def run_chain(llm, scen, *, tag: str, label: str) -> dict:
    return _execute(llm, scen, {"inputs": initial_inputs(scen), "requirements": []}, None, set(), tag, label)


def replay(llm, scen, base: dict, state: dict, dirty: set[str], *, tag: str, label: str) -> dict:
    """Re-run only the activities in `dirty` (downstream of a perturbation); reuse the rest from `base`.
    The decider is the sink of the chain and is always re-run."""
    return _execute(llm, scen, state, base, set(dirty) | {ACT_DECIDER}, tag, label)
