"""Figures: provenance graph with determinant elements highlighted, and a flip-rate heatmap."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from provenance.graph import DECISION, _restricted  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

LAYER = {"spec_passage": 0, "requirement": 2, "datasheet_passage": 2, "check_result": 4,
         "criterion": 4, "decision": 6}
SHAPE = {"spec_passage": "s", "requirement": "o", "datasheet_passage": "D", "check_result": ".",
         "criterion": "^", "decision": "*", "activity": "s"}
TYPE_LABEL = {"spec_passage": "spec passage", "requirement": "requirement", "datasheet_passage": "datasheet passage",
              "check_result": "check result", "criterion": "criterion", "activity": "agent call (activity)",
              "decision": "decision"}


def _layer(G, n):
    a = G.nodes[n]
    if a["type"] == "activity":
        return {"extractor": 1, "checker": 3, "decider": 5}[a["agent"]]
    return LAYER[a["type"]]


def _short(n):
    return n.replace("req_", "").replace("ds_", "").replace("crit_", "").replace("act_", "")


def _sort_key(G, n):
    a = G.nodes[n]
    order = {"requirement": 0, "datasheet_passage": 1, "check_result": 0, "criterion": 1}.get(a["type"], 0)
    return (order, a.get("candidate") or "", a.get("requirement") or "", n)


def _layout(G):
    layers = {}
    for n in G.nodes:
        G.nodes[n]["layer"] = _layer(G, n)
        layers.setdefault(G.nodes[n]["layer"], []).append(n)
    pos = {}
    for lx, nodes in layers.items():
        nodes = sorted(nodes, key=lambda n: _sort_key(G, n))
        k = len(nodes)
        span = 1.0 if k > 6 else 0.12 * k     # few nodes (activities): keep them close to the centre
        for i, n in enumerate(nodes):
            y = 0.0 if k == 1 else span * (1 - 2 * i / (k - 1))
            pos[n] = (lx / 6.0, y)
    return pos


def draw_provenance(G, determinant: dict, path, title: str):
    """determinant: {node: max flip rate}."""
    pos = _layout(G)
    H = _restricted(G)
    hot_edges, hot_nodes = set(), set()
    for d in determinant:
        if d in H and nx.has_path(H, d, DECISION):
            p = nx.shortest_path(H, d, DECISION)
            hot_edges |= set(zip(p, p[1:]))
            hot_nodes |= {x for x in p if G.nodes[x]["type"] == "check_result"}

    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cold = [(u, v) for u, v in G.edges if (u, v) not in hot_edges]
    nx.draw_networkx_edges(G, pos, edgelist=cold, ax=ax, edge_color=MUTED, width=0.4, alpha=0.5, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=list(hot_edges), ax=ax, edge_color=ORANGE, width=2.0, arrows=True,
                           arrowsize=8, arrowstyle="-|>")
    for t, shape in SHAPE.items():
        nodes = [n for n, a in G.nodes(data=True) if a["type"] == t]
        if not nodes:
            continue
        colors, sizes = [], []
        for n in nodes:
            if n in determinant:
                colors.append(ORANGE); sizes.append(150)
            elif n in hot_nodes:
                colors.append(ORANGE); sizes.append(30)
            elif t in ("activity", "decision"):
                colors.append(BLUE); sizes.append(260 if t == "decision" else 120)
            else:
                colors.append(MUTED); sizes.append(14 if t == "check_result" else 40)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape, node_color=colors, node_size=sizes,
                               ax=ax, edgecolors=SURFACE, linewidths=1.2)
    labels = {n: _short(n) for n, a in G.nodes(data=True) if n in determinant or a["type"] == "activity"}
    labels[DECISION] = f"decision = {G.nodes[DECISION]['value']}"
    for n, txt in labels.items():
        x, y = pos[n]
        rate = f"  ({determinant[n]:.0%})" if n in determinant else ""
        ax.annotate(txt + rate, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5,
                    color=INK, fontweight="bold" if n in determinant else "normal")
    for lx, name in [(0, "specification"), (1, "Extractor"), (2, "requirements /\ndatasheet passages"),
                     (3, "Checker"), (4, "check results /\ncriteria"), (5, "Decider"), (6, "decision")]:
        xs = [pos[n][0] for n in G.nodes if G.nodes[n]["layer"] == lx]
        if xs:
            ax.text(xs[0], 1.01, name, ha="center", va="bottom", fontsize=8.5, color=INK2, transform=ax.get_xaxis_transform())
    handles = [Line2D([], [], marker=SHAPE[t], color="none", markerfacecolor=MUTED if t not in ("activity", "decision") else BLUE,
                      markeredgecolor=MUTED, markersize=7, label=TYPE_LABEL[t]) for t in SHAPE]
    handles.append(Line2D([], [], marker="o", color=ORANGE, lw=2, markersize=7,
                          label="determinant element + path (max flip rate)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, fontsize=8,
              labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def draw_flip_heatmap(exp1_by_scen: dict, noise_by_scen: dict, path):
    """One panel per scenario: element x perturbation kind, cell = flip rate; '*' = significant."""
    kinds = ["removal", "alteration", "injection"]
    n = len(exp1_by_scen)
    heights = [max(3, 0.22 * len(e["elements"]) + 1) for e in exp1_by_scen.values()]
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, max(heights)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (scen, e) in zip(axes[0], exp1_by_scen.items()):
        elems = sorted(e["elements"], key=lambda k: (e["elements"][k]["type"], k))
        mat = [[e["elements"][x]["kinds"].get(k, {}).get("flip_rate", float("nan")) for k in kinds] for x in elems]
        ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for i, x in enumerate(elems):
            for j, k in enumerate(kinds):
                c = e["elements"][x]["kinds"].get(k)
                if c is None:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color=INK2)
                elif c["p"] < 0.05:
                    ax.text(j, i, f"{c['flip_rate']:.0%}*", ha="center", va="center", fontsize=6.5,
                            color="white" if c["flip_rate"] > 0.5 else INK, fontweight="bold")
        crit = {x: e["elements"][x].get("critical") for x in elems}
        ax.set_yticks(range(len(elems)))
        ax.set_yticklabels([_short(x) + (" (crit.)" if crit[x] else "") for x in elems], fontsize=6.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(kinds, fontsize=7.5, color=INK)
        nf = noise_by_scen[scen]
        ax.set_title(f"{scen}\nnoise floor {nf['flips']}/{nf['n']}", fontsize=8.5, color=INK, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Decision flip rate per perturbed element  (* = above noise floor, Fisher p < 0.05)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
