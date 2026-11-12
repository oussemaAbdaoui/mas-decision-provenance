"""Experiment 2: faithfulness of the Decider's justification.

Cited set   = requirement / criterion IDs the Decider names in its justification.
Determinant = elements whose perturbation flips the decision above the noise floor (Exp. 1).

Two views of the determinant set:
  strict : determinant requirements and criteria only;
  lifted : additionally maps each determinant datasheet passage to the requirement(s) whose
           check cited it (or to the criterion it feeds), so that "the temperature passage of
           candidate A matters" counts as evidence that R03 matters.
"""
from __future__ import annotations

from experiments.stats import precision_recall
from provenance.graph import build_graph


def determinant_ids(determinant: list[str], G, scen, passages: dict) -> tuple[set, set]:
    strict, lifted = set(), set()
    crit_attr = {k["id"]: k["attribute"] for k in scen.criteria}
    for e in determinant:
        t, key = e.split("_", 1)
        if t in ("req", "crit"):
            strict.add(key)
            lifted.add(key)
        elif t == "ds":
            for _, _, d in G.out_edges(e, data=True):
                if d["relation"] == "wasDerivedFrom" and d.get("requirement"):
                    lifted.add(d["requirement"])
                if d.get("route") == "criteria":
                    facts = passages.get(key, {}).get("facts", {})
                    lifted |= {k for k, a in crit_attr.items() if a in facts}
    return strict, lifted


def split_by_type(scen, ids):
    return {"critical": sorted(i for i in ids if scen.is_critical(i)),
            "non_critical": sorted(i for i in ids if scen.is_critical(i) is False),
            "criteria": sorted(i for i in ids if i.startswith("K"))}


def run(ctx, scen, ref, exp0_traces, exp1) -> dict:
    G = build_graph(ref)
    passages = {p["id"]: p for ds in ref["inputs"]["datasheets"].values() for p in ds["passages"]}
    strict, lifted = determinant_ids(exp1["determinant"], G, scen, passages)
    cited = set(ref["decision"]["cited"])

    same = [t for t in exp0_traces if t["decision"]["selected"] == ref["decision"]["selected"]]
    per_run = [precision_recall(t["decision"]["cited"], strict) for t in same]
    out = {
        "scenario": scen.name, "decision": ref["decision"]["selected"],
        "justification": ref["decision"]["justification"],
        "cited": sorted(cited), "cited_by_type": split_by_type(scen, cited),
        "determinant_strict": sorted(strict), "determinant_lifted": sorted(lifted),
        "determinant_by_type": split_by_type(scen, strict),
        "strict": precision_recall(cited, strict), "lifted": precision_recall(cited, lifted),
        "exp0_runs_same_decision": len(same),
        "exp0_mean_precision": sum(r["precision"] for r in per_run) / max(1, len(per_run)),
        "exp0_mean_recall": sum(r["recall"] for r in per_run) / max(1, len(per_run)),
    }
    ctx.save(f"exp2_faithfulness/{scen.name}.json", out)
    return out
