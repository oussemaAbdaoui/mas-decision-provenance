"""Builds results/<run>/RESULTS.md, summary.json and figures from the experiment outputs."""
from __future__ import annotations

import json

from provenance.figure import draw_flip_heatmap, draw_provenance
from provenance.graph import build_graph, export_json


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def _short(n):
    return n.split("_", 1)[1] if "_" in n else n


def write(ctx, scens, all_res: dict, mock: bool) -> None:
    out = ctx.out
    L = []
    if mock:
        L += ["> **MOCK BACKEND.** These numbers come from the rule-based stand-in used to test the pipeline, "
              "not from an LLM. They are not experimental results.", ""]
    L += [f"# Results ({ctx.llm.model_id}{', reasoning disabled' if ctx.llm.extra_body else ''})", "",
          f"Profile: `{ctx.cfg}`; temperature {ctx.llm.temperature}, seed {ctx.llm.seed}; alpha = {ctx.alpha}.", ""]

    # ---- main table
    L += ["## Main results", "",
          "| Scenario | Correct | Decision (ref.) | Exp.0 agreement | Noise floor | Elements perturbed | "
          "Determinant elements (max flip rate) | Justification P / R (strict) | P / R (lifted) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in scens:
        r = all_res[s.name]
        e0, e1, e2 = r["exp0"], r["exp1"], r["exp2"]
        det = ", ".join(f"{_short(d)}{'*' if e1['elements'][d].get('critical') else ''} "
                        f"({e1['elements'][d]['max_flip_rate']:.0%})" for d in e1["determinant"]) or "none"
        L.append(f"| {s.name} | {s.correct} | {e0['reference_decision']} | {e0['decision_agreement']:.0%} "
                 f"({e0['decision_counts']}) | {e0['noise_floor']['flips']}/{e0['noise_floor']['n']} | "
                 f"{len(e1['elements_perturbed'])} | {det} | {e2['strict']['precision']:.2f} / {e2['strict']['recall']:.2f} | "
                 f"{e2['lifted']['precision']:.2f} / {e2['lifted']['recall']:.2f} |")
    L += ["", "`*` = requirement extracted as critical. Strict = requirements and criteria only; lifted = "
          "determinant datasheet passages mapped to the requirement whose check cited them.", ""]

    # ---- Exp 0
    L += ["## Experiment 0: natural variability", "",
          "| Scenario | Runs | Decisions | Correct decision | Extraction agreement | Extracted value correct | "
          "Criticality correct | Check cells identical across runs | Check accuracy | Citation Jaccard |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in scens:
        e0 = all_res[s.name]["exp0"]
        L.append(f"| {s.name} | {e0['n_runs']} | {e0['decision_counts']} | {_pct(e0['decision_accuracy'])} | "
                 f"{_pct(e0['extraction_agreement_mean'])} | {_pct(e0['extraction_value_accuracy'])} | "
                 f"{_pct(e0['extraction_criticality_accuracy'])} | {_pct(e0['check_cells_all_runs_agree'])} | "
                 f"{_pct(e0['check_accuracy_vs_ground_truth'])} | {e0['justification_citation_jaccard']:.2f} |")
    L.append("")

    # ---- Exp 1 detail
    L += ["## Experiment 1: determinant elements (detail)", "",
          "![flip-rate heatmap](figures/flip_rates.png)", ""]
    for s in scens:
        e1 = all_res[s.name]["exp1"]
        L += [f"### {s.name}", "",
              f"Perturbed {len(e1['elements_perturbed'])} graph-restricted elements; "
              f"{len(e1['collapsed_by_dominator'])} spec passages collapsed into their requirement (post-dominator); "
              f"{len(e1['uncited_passages'])} uncited passages probed by removal, "
              f"determinant among them: {', '.join(map(_short, e1['probe_uncited_determinant'])) or 'none'}. "
              f"Determinant after Benjamini-Hochberg: {', '.join(map(_short, e1['determinant_bh'])) or 'none'}. "
              f"Injection-sensitive (decision flips when a fabricated sibling is added; not counted as reliance): "
              f"{', '.join(map(_short, e1['injection_sensitive'])) or 'none'}.", "",
              "| Element | Type | Critical | Removal | Alteration | Injection |", "|---|---|---|---|---|---|"]
        for d in e1["determinant"]:
            el = e1["elements"][d]

            def cell(k):
                c = el["kinds"].get(k)
                return "n/a" if c is None else f"{c['flips']}/{c['n']} -> {','.join(c['decisions'])} (p={c['p']:.3f})"
            L.append(f"| {_short(d)} | {el['type']} | {el.get('critical')} | {cell('removal')} | {cell('alteration')} | {cell('injection')} |")
        L.append("")

    # ---- Exp 2
    L += ["## Experiment 2: faithfulness of the justification", ""]
    for s in scens:
        e2 = all_res[s.name]["exp2"]
        L += [f"**{s.name}** (decision {e2['decision']}): cited {e2['cited']}; determinant {e2['determinant_strict']}. "
              f"Cited but not determinant: {e2['strict']['cited_not_determinant']}; determinant but not cited: "
              f"{e2['strict']['determinant_not_cited']}. Over the {e2['exp0_runs_same_decision']} Exp. 0 runs with the same "
              f"decision: mean precision {e2['exp0_mean_precision']:.2f}, mean recall {e2['exp0_mean_recall']:.2f}.", "",
              f"> {e2['justification']}", ""]

    # ---- Exp 3
    L += ["## Experiment 3: injected-error benchmark", "",
          "| Scenario | Error | Faulty element | On critical req. | Decision ref -> faulty | Consequential | "
          "Localised: exact | Localised: requirement | Grouped ablation | Rank (flip rate) | Determinant set size | "
          "Faulty justification cites it |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {True: [0, 0, 0, 0, 0], False: [0, 0, 0, 0, 0]}
    for s in scens:
        for e in all_res[s.name]["exp3"]["errors"]:
            if e.get("skipped"):
                L.append(f"| {s.name} | {e['error']} | skipped: {e['reason']} |" + " |" * 9)
                continue
            a = agg[e["critical"]]
            for i, k in enumerate(("consequential", "detected_exact", "detected_requirement", "detected_grouped"), 1):
                a[i] += bool(e[k])
            e = {k: ("n/a" if v is None else v) for k, v in e.items()}
            a[0] += 1
            L.append(f"| {s.name} | {e['error']} | {_short(e['element'])} | {e['critical']} | {e['reference_decision']} -> "
                     f"{e['faulty_decision']} | {e['consequential']} | {e['detected_exact']} | {e['detected_requirement']} | "
                     f"{e['detected_grouped']} | {e['flip_rate_rank']}/{e['elements_tested']} | {e['determinant_set_size']} | "
                     f"{e['justification_cites_it']} |")
    L += ["", "| Requirement type | Errors | Consequential | Localised: exact | Localised: requirement | Grouped ablation |",
          "|---|---|---|---|---|---|"]
    for k, name in ((True, "critical"), (False, "non-critical")):
        n, c, d, dr, dg = agg[k]
        L.append(f"| {name} | {n} | {c} | {d} | {dr} | {dg} |")
    L.append("")

    # ---- cost
    L += ["## Cost: LLM calls to find the removal-determinant elements", "",
          "| Strategy | " + " | ".join(s.name for s in scens) + " | How |", "|---|" + "---|" * len(scens) + "---|"]
    rows0 = all_res[scens[0].name]["cost"]["rows"]
    for i, row in enumerate(rows0):
        cells = []
        for s in scens:
            r = all_res[s.name]["cost"]["rows"][i]
            cells.append(f"{r['llm_calls']} ({r['saving_vs_naive']:.0%} saved)")
        L.append(f"| {row['strategy']} | " + " | ".join(cells) + f" | {row['how']} |")
    L.append("")
    for s in scens:
        c = all_res[s.name]["cost"]
        L.append(f"- {s.name}: grouped ablation missed {c['grouped_missed'] or 'nothing'} relative to single-element "
                 f"removal; groups significant only jointly: {c['grouped_joint_only_groups'] or 'none'}.")
    L += ["", "## Total LLM usage", "", "| Label | Logical calls | API calls | Prompt tokens | Completion tokens |",
          "|---|---|---|---|---|"]
    for lab, d in sorted(ctx.llm.counter.snapshot().items()):
        L.append(f"| {lab} | {d.get('logical', 0)} | {d.get('api', 0)} | {d.get('prompt_tokens', 0)} | {d.get('completion_tokens', 0)} |")
    L += ["", "API calls = 0 means the call was served from `llm_calls.sqlite` (an earlier identical run).", ""]

    # ---- figures + graph exports
    for s in scens:
        r = all_res[s.name]
        G = build_graph(r["ref"])
        export_json(G, out / "graphs" / f"{s.name}_reference.json")
        det = {d: r["exp1"]["elements"][d]["max_flip_rate"] for d in r["exp1"]["determinant"]}
        draw_provenance(G, det, out / "figures" / f"{s.name}_provenance.png",
                        f"{s.title}: provenance of the reference run ({ctx.llm.model_id}); "
                        f"orange = determinant elements")
        L.insert(L.index("## Experiment 0: natural variability"), f"![{s.name}](figures/{s.name}_provenance.png)\n")
    draw_flip_heatmap({s.name: all_res[s.name]["exp1"] for s in scens},
                      {s.name: all_res[s.name]["exp0"]["noise_floor"] for s in scens}, out / "figures" / "flip_rates.png")

    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    summary = {s.name: {k: v for k, v in all_res[s.name].items() if k not in ("ref", "traces")} for s in scens}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
