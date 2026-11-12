"""Experiment 3: injected-error benchmark with known ground truth.

For each known error (wrong extracted value, corrupted datasheet passage) we
  1. inject it into the reference run and replay downstream -> a *faulty* run;
  2. run the localisation method on the faulty run without telling it where the error is:
     graph-restricted single-element removal (the method of Exp. 1), and, for comparison,
     grouped removal ablation;
  3. score whether the faulty element is found
       exact             : the faulty element itself is determinant,
       requirement-level : the requirement the error concerns is determinant
                           (a corrupted passage is often only visible through its requirement:
                           removing it yields "unknown", which disqualifies just like "fail").
Results are reported separately for errors on critical and non-critical requirements.
"""
from __future__ import annotations

from collections import Counter

from agents.chain import replay, state_of
from agents.llm import BudgetExceeded
from experiments import grouped, perturb
from experiments.common import run_tests
from experiments.perturbation import perturbation_tests
from provenance.graph import build_graph, dirty_activities, upstream_elements


def run(ctx, scen, ref, noise) -> dict:
    G = build_graph(ref)
    state0 = state_of(ref)
    ref_dec = ref["decision"]["selected"]
    reps = ctx.cfg["reps3"]
    label = f"exp3:{scen.name}"
    rows = []
    errors = [e for e in scen.ground_truth["injected_errors"]
              if not ctx.cfg.get("exp3_errors") or e["id"] in ctx.cfg["exp3_errors"]]
    for err in errors:
      try:
        node = ("req_" if err["target"] == "requirement" else "ds_") + err["element"]
        req_node = "req_" + (err["element"] if err["target"] == "requirement" else err["requirement"])
        st = perturb.apply_error(state0, err)
        if st is None or node not in G:
            rows.append({"error": err["id"], "skipped": True, "reason": "target absent from reference run"})
            continue
        dirty = dirty_activities(G, node)
        faulty = ctx.pmap(lambda r: replay(ctx.llm, scen, ref, st, dirty, tag=f"exp3|{scen.name}|{err['id']}|faulty|{r}",
                                           label=label), range(reps))
        F = faulty[0]
        ctx.save(f"traces/{scen.name}/exp3_{err['id']}_faulty.json", F)
        fdec = [t["decision"]["selected"] for t in faulty]
        modal = Counter(fdec).most_common(1)[0][0]

        GF = build_graph(F)
        elements, _, _ = upstream_elements(GF)
        if ctx.cfg.get("exp3_scope") == "requirements":    # budget mode: requirement-level localisation only
            elements = [e for e in elements if e.startswith(("req_", "crit_"))]
        tests, _ = perturbation_tests(scen, F, GF, elements, ("removal",), f"exp3|{scen.name}|{err['id']}")
        srows = run_tests(ctx, scen, F, tests, noise, reps, label)
        det = sorted(r["element"] for r in srows if r["p"] < ctx.alpha)
        rank = sorted(srows, key=lambda r: -r["flip_rate"])
        loc_g = ({"determinant": [], "calls": 0} if ctx.cfg.get("exp3_scope") == "requirements" else
                 grouped.run(ctx, scen, F, noise, reps=reps, tag_prefix=f"exp3|{scen.name}|{err['id']}",
                             label=f"exp3g:{scen.name}"))
        rows.append({
            "error": err["id"], "element": node, "requirement_node": req_node, "critical": err["critical"],
            "description": err["description"], "faulty_decisions": fdec, "faulty_decision": modal,
            "reference_decision": ref_dec, "correct_candidate": scen.correct, "consequential": modal != ref_dec,
            "detected_exact": (node in det) if node in elements else None, "detected_requirement": req_node in det,
            "detected_grouped": (node in loc_g["determinant"] or req_node in loc_g["determinant"])
                                if ctx.cfg.get("exp3_scope") != "requirements" else None,
            "determinant_set": det, "determinant_set_size": len(det), "elements_tested": len(srows),
            "flip_rate_rank": next((i + 1 for i, r in enumerate(rank) if r["element"] in (node, req_node)), None),
            "localisation_calls": sum(r["calls"] for r in srows), "grouped_calls": loc_g["calls"],
            "faulty_justification": F["decision"]["justification"], "faulty_cited": F["decision"]["cited"],
            "justification_cites_it": (err.get("requirement") or err["element"]) in F["decision"]["cited"],
        })
      except BudgetExceeded as ex:
        rows.append({"error": err["id"], "skipped": True, "reason": "budget reached"})
        print(f"[{scen.name}]   Exp. 3 {err['id']} skipped: {ex}")
    out = {"scenario": scen.name, "errors": rows, "calls": ctx.llm.counter.snapshot().get(label, {})}
    ctx.save(f"exp3_injected/{scen.name}.json", out)
    return out
