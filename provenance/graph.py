"""PROV-style provenance graph of one chain run (networkx).

Edges are stored in the direction information flows (input -> output) so that
`nx.descendants` = "everything this element can influence" and `nx.ancestors` = "everything
this element may depend on".  Each edge carries its PROV relation:

    entity  --used-->            activity   (PROV: activity used entity)
    activity --wasGeneratedBy--> entity     (PROV: entity wasGeneratedBy activity)
    source  --wasDerivedFrom-->  derived    (PROV: derived wasDerivedFrom source)

and, where relevant, the requirement ID it concerns.

`used` edges record everything an agent had in its prompt (input provenance).
`wasDerivedFrom` edges from datasheet passages record what the checker *claims* it used
(its citations).  Keeping the two apart is deliberate; see README, "Granularity".
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

DECISION = "decision"
PERTURBABLE = ("requirement", "datasheet_passage", "criterion")


def n_spec(sid): return f"spec_{sid}"
def n_req(rid): return f"req_{rid}"
def n_ds(pid): return f"ds_{pid}"
def n_crit(kid): return f"crit_{kid}"
def n_chk(rid, cid): return f"chk_{rid}_cand{cid}"
def n_act(aid): return f"act_{aid}"


def build_graph(trace: dict) -> nx.DiGraph:
    G = nx.DiGraph(scenario=trace["scenario"], run_id=trace["run_id"], model=trace["model"])
    inp, acts = trace["inputs"], trace["activities"]
    run = trace["run_id"]

    def activity(aid, **attrs):
        G.add_node(n_act(aid), type="activity", activity_id=aid, run=run, **attrs)

    # extractor
    activity("extractor", agent="extractor")
    for p in inp["spec_passages"]:
        G.add_node(n_spec(p["id"]), type="spec_passage", text=p["text"], label=p.get("label"))
        G.add_edge(n_spec(p["id"]), n_act("extractor"), relation="used", cited=False)
    for r in trace["requirements"]:
        G.add_node(n_req(r["id"]), type="requirement", requirement=r["id"], attribute=r["attribute"],
                   operator=r["operator"], value=r["value"], unit=r["unit"], critical=r["critical"],
                   source_passage=r.get("source_passage"))
        G.add_edge(n_act("extractor"), n_req(r["id"]), relation="wasGeneratedBy", requirement=r["id"])
        if r.get("source_passage") and G.has_node(n_spec(r["source_passage"])):
            G.add_edge(n_spec(r["source_passage"]), n_req(r["id"]), relation="wasDerivedFrom", requirement=r["id"])
    crit_of = {r["id"]: r["critical"] for r in trace["requirements"]}

    # checkers
    for cid, ds in inp["datasheets"].items():
        a = n_act(f"checker_{cid}")
        activity(f"checker_{cid}", agent="checker", candidate=cid)
        for p in ds["passages"]:
            G.add_node(n_ds(p["id"]), type="datasheet_passage", candidate=cid, text=p["text"])
            G.add_edge(n_ds(p["id"]), a, relation="used", cited=False)
        for r in trace["requirements"]:
            G.add_edge(n_req(r["id"]), a, relation="used", requirement=r["id"], cited=True)
        for rid, res in trace["checks"][cid].items():
            c = n_chk(rid, cid)
            G.add_node(c, type="check_result", requirement=rid, candidate=cid, critical=crit_of.get(rid),
                       value=res["verdict"], evidence=res.get("evidence", ""), cited_passages=res.get("passages", []))
            G.add_edge(a, c, relation="wasGeneratedBy", requirement=rid)
            G.add_edge(n_req(rid), c, relation="wasDerivedFrom", requirement=rid)
            for pid in res.get("passages", []):
                G.add_edge(n_ds(pid), c, relation="wasDerivedFrom", requirement=rid)
                G.edges[n_ds(pid), a]["cited"] = True

    # decider
    d = n_act("decider")
    activity("decider", agent="decider")
    for k in inp["criteria"]:
        G.add_node(n_crit(k["id"]), type="criterion", text=k["text"], attribute=k["attribute"], direction=k["direction"])
        G.add_edge(n_crit(k["id"]), d, relation="used", cited=True)
    for pid in acts["decider"].get("routed_passages", []):
        G.add_edge(n_ds(pid), d, relation="used", cited=True, route="criteria")
    for cid, res in trace["checks"].items():
        for rid in res:
            G.add_edge(n_chk(rid, cid), d, relation="used", requirement=rid, cited=True)
    dec = trace["decision"]
    G.add_node(DECISION, type="decision", value=dec["selected"], justification=dec["justification"], cited=dec["cited"])
    G.add_edge(d, DECISION, relation="wasGeneratedBy")
    # claimed derivations: what the justification says the decision rests on
    for x in dec["cited"]:
        if x.startswith("R") and G.has_node(n_chk(x, dec["selected"])):
            G.add_edge(n_chk(x, dec["selected"]), DECISION, relation="wasDerivedFrom", requirement=x, claimed=True)
        elif x.startswith("K") and G.has_node(n_crit(x)):
            G.add_edge(n_crit(x), DECISION, relation="wasDerivedFrom", claimed=True)
    return G


def _restricted(G: nx.DiGraph) -> nx.DiGraph:
    """Drop `used` edges from passages the agent did not cite (input provenance -> claimed provenance)."""
    keep = [(u, v) for u, v, e in G.edges(data=True)
            if not (e["relation"] == "used" and G.nodes[u]["type"] in ("spec_passage", "datasheet_passage") and not e.get("cited"))]
    return G.edge_subgraph(keep).copy()


def upstream_elements(G: nx.DiGraph) -> tuple[list[str], dict, list[str]]:
    """Elements to perturb under the graph-restricted strategy.

    1. keep only elements with a (claimed-provenance) path to the decision;
    2. collapse an element into its immediate post-dominator when that post-dominator is itself
       perturbable (e.g. a spec passage whose only route to the decision is its requirement).
    Returns (elements, collapsed {element: dominator}, uncited datasheet passages).
    """
    H = _restricted(G)
    anc = nx.ancestors(H, DECISION)
    idom = nx.immediate_dominators(H.reverse(copy=True), DECISION)
    elements, collapsed = [], {}
    for n in sorted(anc):
        t = G.nodes[n]["type"]
        if t not in PERTURBABLE + ("spec_passage",):
            continue
        dom = idom.get(n)
        if dom is not None and dom != n and G.nodes[dom]["type"] in PERTURBABLE:
            collapsed[n] = dom
            continue
        elements.append(n)
    uncited = sorted(n for n, a in G.nodes(data=True) if a["type"] == "datasheet_passage" and n not in anc)
    return elements, collapsed, uncited


def all_input_elements(G: nx.DiGraph) -> list[str]:
    """Every raw input element (what naive ablation would perturb)."""
    return sorted(n for n, a in G.nodes(data=True) if a["type"] in ("spec_passage", "datasheet_passage", "criterion"))


def dirty_activities(G: nx.DiGraph, nodes) -> set[str]:
    """Activities downstream of any of `nodes` in the full graph (input provenance)."""
    out = set()
    for n in ([nodes] if isinstance(nodes, str) else nodes):
        out |= {G.nodes[m]["activity_id"] for m in nx.descendants(G, n) if G.nodes[m]["type"] == "activity"}
    return out


def to_prov_json(G: nx.DiGraph) -> dict:
    nodes = []
    for n, a in G.nodes(data=True):
        rec = {"id": n, **{k: v for k, v in a.items()}}
        if a["type"] == "activity":
            rec["used"] = sorted(u for u, _ in G.in_edges(n))
            rec["generated"] = sorted(v for _, v in G.out_edges(n))
        else:
            gen = [u for u, _, e in G.in_edges(n, data=True) if e["relation"] == "wasGeneratedBy"]
            rec["generated_by"] = gen[0] if gen else None
            rec["used"] = sorted(u for u, _, e in G.in_edges(n, data=True) if e["relation"] == "wasDerivedFrom")
        nodes.append(rec)
    rels = []
    for u, v, e in G.edges(data=True):
        r = {k: val for k, val in e.items() if k != "relation"}
        if e["relation"] == "used":
            rels.append({"type": "used", "activity": v, "entity": u, **r})
        elif e["relation"] == "wasGeneratedBy":
            rels.append({"type": "wasGeneratedBy", "entity": v, "activity": u, **r})
        else:
            rels.append({"type": "wasDerivedFrom", "generatedEntity": v, "usedEntity": u, **r})
    return {"graph": dict(G.graph), "nodes": nodes, "relations": rels}


def export_json(G: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_prov_json(G), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
