"""Grouped ablation: remove a whole group of elements first, refine to single elements only if
the decision flips significantly.  Used for the cost comparison and for Exp. 3 localisation."""
from __future__ import annotations

from agents.chain import state_of
from experiments import perturb
from experiments.common import run_tests
from provenance.graph import build_graph, dirty_activities, upstream_elements


def make_groups(elements, G, size):
    reqs = sorted(e for e in elements if e.startswith("req_"))
    groups = [(f"requirements[{i // size + 1}]", reqs[i:i + size]) for i in range(0, len(reqs), size)]
    by_cand = {}
    for e in sorted(e for e in elements if e.startswith("ds_")):
        by_cand.setdefault(G.nodes[e]["candidate"], []).append(e)
    groups += [(f"passages[{c}]", v) for c, v in sorted(by_cand.items())]
    crits = sorted(e for e in elements if e.startswith("crit_"))
    if crits:
        groups.append(("criteria", crits))
    rest = sorted(e for e in elements if not e.startswith(("req_", "ds_", "crit_")))
    if rest:
        groups.append(("other", rest))
    return groups


def run(ctx, scen, base, noise, *, reps, tag_prefix, label) -> dict:
    G = build_graph(base)
    elements, _, _ = upstream_elements(G)
    groups = make_groups(elements, G, ctx.cfg["group"])
    state0 = state_of(base)

    gtests = []
    for name, members in groups:
        st, desc = perturb.apply_group_removal(state0, scen, members)
        gtests.append({"element": name, "kind": "group_removal", "state": st,
                       "dirty": sorted(dirty_activities(G, members)), "description": desc,
                       "tag_prefix": f"{tag_prefix}|group|{name}"})
    grows = run_tests(ctx, scen, base, gtests, noise, reps, label)
    sig = {r["element"] for r in grows if r["p"] < ctx.alpha}

    rtests, singletons = [], []
    for name, members in groups:
        if name not in sig:
            continue
        if len(members) == 1:
            singletons.append(members[0])
            continue
        for e in members:
            res = perturb.apply(state0, scen, e, "removal")
            if res is None:
                continue
            st, desc = res
            # same tags as Exp. 1 removal on the same base -> served from the call log
            rtests.append({"element": e, "kind": "removal", "state": st, "dirty": sorted(dirty_activities(G, e)),
                           "description": desc, "tag_prefix": f"{tag_prefix}|{e}|removal"})
    rrows = run_tests(ctx, scen, base, rtests, noise, reps, label) if rtests else []
    determinant = sorted(singletons + [r["element"] for r in rrows if r["p"] < ctx.alpha])
    joint = sorted(name for name, members in groups if name in sig and len(members) > 1
                   and not any(r["element"] in members and r["p"] < ctx.alpha for r in rrows))
    return {"groups": [{"name": n, "members": m} for n, m in groups], "group_tests": grows, "refine_tests": rrows,
            "determinant": determinant, "joint_only_groups": joint,
            "calls": sum(r["calls"] for r in grows) + sum(r["calls"] for r in rrows),
            "flip_rate": {r["element"]: r["flip_rate"] for r in rrows}}
