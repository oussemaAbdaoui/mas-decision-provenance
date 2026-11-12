"""Cost of finding the determinant elements under *removal*, for different ablation strategies.

Two independent savings are separated:
  * element pruning  : perturb only elements on a provenance path to the decision
                       (plus dominator collapse) instead of every input element;
  * partial replay   : re-run only activities downstream of the perturbed element,
                       reusing cached upstream outputs, instead of the whole chain.
Grouped ablation adds group-then-refine on top of both.

Rows marked "computed" are counted from the graph (not executed); rows marked "executed"
are the logical LLM calls actually issued by the experiments.
"""
from __future__ import annotations

from provenance.graph import all_input_elements, build_graph, dirty_activities, upstream_elements


def _calls(G, e, n_cand):
    dirty = dirty_activities(G, e) | {"decider"}
    return 2 + n_cand if "extractor" in dirty else len(dirty)


def run(ctx, scen, ref, exp1, grouped_ref) -> dict:
    G = build_graph(ref)
    n_cand, full, reps = len(scen.candidates), len(scen.candidates) + 2, ctx.cfg["reps"]
    naive_elems = all_input_elements(G)
    graph_elems, _, _ = upstream_elements(G)
    removal_rows = [r for r in exp1["tests"] if r["kind"] == "removal"]
    exp1_removal_det = sorted(r["element"] for r in removal_rows if r["p"] < ctx.alpha)
    rows = [
        {"strategy": "naive: every input element, full-chain replay", "elements": len(naive_elems),
         "llm_calls": len(naive_elems) * reps * full, "how": "computed"},
        {"strategy": "every input element, partial replay", "elements": len(naive_elems),
         "llm_calls": sum(_calls(G, e, n_cand) for e in naive_elems) * reps, "how": "computed"},
        {"strategy": "graph-restricted elements, full-chain replay", "elements": len(graph_elems),
         "llm_calls": len(graph_elems) * reps * full, "how": "computed"},
        {"strategy": "graph-restricted elements, partial replay, fixed reps", "elements": len(removal_rows),
         "llm_calls": sum(r["calls"] // r["n"] for r in removal_rows) * reps,
         "how": "computed" if ctx.cfg.get("curtail") else "executed (Exp. 1)"},
    ]
    if ctx.cfg.get("curtail"):
        rows.append({"strategy": "graph-restricted, partial replay, curtailed sequential testing",
                     "elements": len(removal_rows), "llm_calls": sum(r["calls"] for r in removal_rows),
                     "how": "executed (Exp. 1)"})
    rows.append({"strategy": "grouped ablation (graph-restricted, partial replay"
                             + (", curtailed)" if ctx.cfg.get("curtail") else ")"),
                 "elements": len(grouped_ref["groups"]), "llm_calls": grouped_ref["calls"], "how": "executed"})
    base = rows[0]["llm_calls"]
    for r in rows:
        r["saving_vs_naive"] = 1 - r["llm_calls"] / base if base else 0.0
    out = {"scenario": scen.name, "reps": reps, "rows": rows,
           "removal_determinant_individual": exp1_removal_det,
           "removal_determinant_grouped": grouped_ref["determinant"],
           "grouped_missed": sorted(set(exp1_removal_det) - set(grouped_ref["determinant"])),
           "grouped_joint_only_groups": grouped_ref["joint_only_groups"],
           "grouped_detail": grouped_ref}
    ctx.save(f"cost/{scen.name}.json", out)
    return out
