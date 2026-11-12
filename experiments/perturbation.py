"""Experiment 1: controlled perturbation of the elements upstream of the decision.

Only elements selected by the provenance graph are perturbed, and only the activities
downstream of the perturbed element are replayed (upstream outputs are reused from the
reference trace).  Each perturbation is replayed `reps` times and its decision-flip rate is
compared with the Exp. 0 noise floor using a one-sided Fisher exact test.
"""
from __future__ import annotations

from agents.chain import state_of
from experiments import perturb
from experiments.common import run_tests
from experiments.stats import benjamini_hochberg
from provenance.graph import build_graph, dirty_activities, upstream_elements


def perturbation_tests(scen, ref, G, elements, kinds, tag_prefix):
    state0 = state_of(ref)
    tests, skipped = [], []
    for e in elements:
        dirty = dirty_activities(G, e)
        for kind in kinds:
            res = perturb.apply(state0, scen, e, kind)
            if res is None:
                skipped.append({"element": e, "kind": kind})
                continue
            st, desc = res
            tests.append({"element": e, "kind": kind, "state": st, "dirty": sorted(dirty),
                          "description": desc, "tag_prefix": f"{tag_prefix}|{e}|{kind}"})
    return tests, skipped


RELIANCE_KINDS = ("removal", "alteration")   # test whether the decision relies on the element
# "injection" adds a fabricated sibling element: it tests vulnerability to hallucination, not reliance.


def element_table(rows, G, alpha):
    elems = {}
    for r in rows:
        e = elems.setdefault(r["element"], {"element": r["element"], "type": G.nodes[r["element"]]["type"],
                                            "kinds": {}, "determinant": False, "determinant_bh": False,
                                            "injection_sensitive": False, "max_flip_rate": 0.0})
        e["kinds"][r["kind"]] = {k: r[k] for k in ("flips", "n", "flip_rate", "p", "p_bh", "decisions", "description")}
        if r["kind"] in RELIANCE_KINDS:
            e["determinant"] |= r["p"] < alpha
            e["determinant_bh"] |= r["p_bh"] < alpha
            e["max_flip_rate"] = max(e["max_flip_rate"], r["flip_rate"])
        else:
            e["injection_sensitive"] |= r["p"] < alpha
        node = G.nodes[r["element"]]
        e["requirement"] = node.get("requirement")
        e["critical"] = node.get("critical")
        e["candidate"] = node.get("candidate")
    return elems


def run(ctx, scen, ref, noise) -> dict:
    label = f"exp1:{scen.name}"
    G = build_graph(ref)
    elements, collapsed, uncited = upstream_elements(G)
    ref_dec = ref["decision"]["selected"]
    prefix = f"exp1|{scen.name}"

    tests, skipped = perturbation_tests(scen, ref, G, elements, perturb.KINDS, prefix)
    rows = run_tests(ctx, scen, ref, tests, noise, ctx.cfg["reps"], label)
    for r, padj in zip(rows, benjamini_hochberg([r["p"] for r in rows])):
        r["p_bh"] = padj
    elems = element_table(rows, G, ctx.alpha)

    # provenance-completeness probe: remove passages the checker did NOT cite
    ptests, _ = perturbation_tests(scen, ref, G, uncited, ("removal",), f"{prefix}|probe")
    probe_rows = run_tests(ctx, scen, ref, ptests, noise, ctx.cfg["probe"], f"probe:{scen.name}")

    out = {
        "scenario": scen.name, "reference_decision": ref_dec, "noise_floor": noise, "reps": ctx.cfg["reps"],
        "elements_perturbed": elements, "collapsed_by_dominator": collapsed, "uncited_passages": uncited,
        "skipped": skipped, "tests": rows, "elements": elems,
        "determinant": sorted(e for e, v in elems.items() if v["determinant"]),
        "determinant_bh": sorted(e for e, v in elems.items() if v.get("determinant_bh")),
        "injection_sensitive": sorted(e for e, v in elems.items() if v["injection_sensitive"]),
        "probe_uncited": probe_rows,
        "probe_uncited_determinant": sorted(r["element"] for r in probe_rows if r["p"] < ctx.alpha),
        "calls": ctx.llm.counter.snapshot().get(label, {}),
        "calls_probe": ctx.llm.counter.snapshot().get(f"probe:{scen.name}", {}),
    }
    ctx.save(f"exp1_perturbation/{scen.name}.json", out)
    return out
