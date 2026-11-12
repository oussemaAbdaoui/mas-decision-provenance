"""Experiment 0: natural variability of the unchanged chain (the noise floor)."""
from __future__ import annotations

from collections import Counter

from agents.chain import run_chain
from agents.rules import ground_truth_checks, values_match
from experiments.stats import mean_pairwise_jaccard


def run(ctx, scen) -> tuple[dict, list[dict]]:
    n = ctx.cfg["n0"]
    label = f"exp0:{scen.name}"
    traces = ctx.pmap(lambda i: run_chain(ctx.llm, scen, tag=f"exp0|{scen.name}|{i}", label=label), range(n))
    for i, t in enumerate(traces):
        ctx.save(f"traces/{scen.name}/exp0_run{i:02d}.json", t)

    ref = traces[0]
    decisions = [t["decision"]["selected"] for t in traces]
    others = decisions[1:]
    noise = {"flips": sum(1 for d in others if d != decisions[0]), "n": len(others)}

    # extraction: agreement across runs and accuracy against ground truth
    extraction = {}
    for g in scen.gt_requirements:
        got = [next((r for r in t["requirements"] if r["id"] == g["id"]), None) for t in traces]
        sig = [None if r is None else (r["operator"], str(r["value"]), r["critical"]) for r in got]
        modal, cnt = Counter(sig).most_common(1)[0]
        extraction[g["id"]] = {
            "present": sum(r is not None for r in got) / n,
            "agreement": cnt / n,
            "value_correct": sum(1 for r in got if r and values_match(r["value"], g["value"])) / n,
            "criticality_correct": sum(1 for r in got if r and r["critical"] == g["critical"]) / n,
            "modal": modal,
        }
    spurious = [sorted({r["id"] for r in t["requirements"]} - {g["id"] for g in scen.gt_requirements}) for t in traces]

    # checks: agreement across runs and accuracy against the rule-based ground truth
    gt_checks = ground_truth_checks(scen)
    cells, agree, correct, total = 0, 0, 0, 0
    for cid in scen.candidate_ids:
        for g in scen.gt_requirements:
            vs = [t["checks"].get(cid, {}).get(g["id"], {}).get("verdict", "absent") for t in traces]
            cells += 1
            agree += len(set(vs)) == 1
            exp = gt_checks[cid][g["id"]]["verdict"]
            exp = "fail" if exp == "unknown" else exp           # policy: unknown = not demonstrated
            correct += sum(1 for v in vs if (v if v != "unknown" else "fail") == exp)
            total += len(vs)

    cited_sets = [t["decision"]["cited"] for t in traces]
    result = {
        "scenario": scen.name, "n_runs": n, "decisions": decisions, "decision_counts": dict(Counter(decisions)),
        "reference_decision": decisions[0], "correct_candidate": scen.correct,
        "decision_agreement": Counter(decisions).most_common(1)[0][1] / n,
        "decision_accuracy": sum(d == scen.correct for d in decisions) / n,
        "noise_floor": noise, "noise_flip_rate": noise["flips"] / max(1, noise["n"]),
        "extraction": extraction,
        "extraction_agreement_mean": sum(e["agreement"] for e in extraction.values()) / len(extraction),
        "extraction_value_accuracy": sum(e["value_correct"] for e in extraction.values()) / len(extraction),
        "extraction_criticality_accuracy": sum(e["criticality_correct"] for e in extraction.values()) / len(extraction),
        "spurious_requirements": spurious,
        "check_cells_all_runs_agree": agree / cells, "check_accuracy_vs_ground_truth": correct / total,
        "justification_citation_jaccard": mean_pairwise_jaccard(cited_sets), "cited_sets": cited_sets,
        "calls": ctx.llm.counter.snapshot().get(label, {}),
    }
    ctx.save(f"exp0_variability/{scen.name}.json", result)
    return result, traces
