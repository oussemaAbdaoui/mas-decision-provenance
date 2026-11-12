"""Run every experiment on every scenario and write the report.

    python run_all.py                      # Groq, Llama-3.3-70B, full profile
    python run_all.py --backend mock       # offline pipeline test (no API key, not real results)
    python run_all.py --profile quick --scenarios drone_battery
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.llm import LLM, BudgetExceeded, load_dotenv  # noqa: E402
from agents.rules import ground_truth_decision  # noqa: E402
from agents.scenario import list_scenarios, load_scenario  # noqa: E402
from experiments import cost, faithfulness, grouped, injected_errors, perturbation, report, variability  # noqa: E402
from experiments.common import PROFILES, Ctx  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="groq", help="groq | together | openrouter | ollama | mock")
    ap.add_argument("--model", default=None, help="model name (default: provider preset)")
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="disable hidden reasoning on hybrid models (sends reasoning.enabled=false)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop before spending more than this (USD, this process; cached calls are free)")
    ap.add_argument("--out", default=None, help="output folder (default: results/<backend>-<model>)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    from agents.llm import PROVIDERS
    model = a.model or PROVIDERS[a.backend]["model"]
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", f"{a.backend}-{model}" + ("-noreason" if a.no_reasoning else "")).strip("-").lower()
    out = Path(a.out) if a.out else ROOT / "results" / slug
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(a.backend, a.model, log_path=out / "llm_calls.sqlite", seed=a.seed, max_concurrency=a.workers,
              extra_body={"reasoning": {"enabled": False}} if a.no_reasoning else None,
              budget_usd=a.budget_usd)
    ctx = Ctx(llm=llm, out=out, cfg=dict(PROFILES[a.profile]), workers=a.workers)

    scens = [load_scenario(s) for s in (a.scenarios or list_scenarios())]
    all_res = {}
    for s in scens:
        gt_sel, _ = ground_truth_decision(s)
        assert gt_sel == s.correct, f"{s.name}: ground truth inconsistent ({gt_sel} != {s.correct})"
        t0 = time.time()
        print(f"[{s.name}] Exp. 0: {ctx.cfg['n0']} unchanged runs", flush=True)
        e0, traces = variability.run(ctx, s)
        ref = traces[0]
        noise = e0["noise_floor"]
        print(f"[{s.name}]   decisions {e0['decision_counts']} (correct: {s.correct}); noise floor {noise['flips']}/{noise['n']}")
        print(f"[{s.name}] Exp. 1: perturbations", flush=True)
        e1 = perturbation.run(ctx, s, ref, noise)
        print(f"[{s.name}]   determinant: {e1['determinant']}")
        e2 = faithfulness.run(ctx, s, ref, traces, e1)
        print(f"[{s.name}] Exp. 2: cited {e2['cited']} -> P={e2['strict']['precision']:.2f} R={e2['strict']['recall']:.2f}")
        print(f"[{s.name}] Cost: grouped ablation", flush=True)
        try:
            g = grouped.run(ctx, s, ref, noise, reps=ctx.cfg["reps"], tag_prefix=f"exp1|{s.name}", label=f"grouped:{s.name}")
        except BudgetExceeded as ex:
            print(f"[{s.name}]   SKIPPED grouped ablation: {ex}")
            g = {"groups": [], "group_tests": [], "refine_tests": [], "determinant": [], "joint_only_groups": [],
                 "calls": 0, "flip_rate": {}, "incomplete": True}
        c = cost.run(ctx, s, ref, e1, g)
        print(f"[{s.name}] Exp. 3: injected errors", flush=True)
        try:
            e3 = injected_errors.run(ctx, s, ref, noise)
        except BudgetExceeded as ex:
            print(f"[{s.name}]   SKIPPED Exp. 3: {ex}")
            e3 = {"scenario": s.name, "errors": [], "calls": {}, "incomplete": True}
        for e in e3["errors"]:
            if not e.get("skipped"):
                print(f"[{s.name}]   {e['error']}: {e['reference_decision']}->{e['faulty_decision']} exact={e['detected_exact']} req={e['detected_requirement']} grouped={e['detected_grouped']}")
        all_res[s.name] = {"exp0": e0, "exp1": e1, "exp2": e2, "exp3": e3, "cost": c, "ref": ref}
        print(f"[{s.name}] done in {time.time() - t0:.0f}s; calls so far: "
              f"{sum(d.get('api', 0) for d in llm.counter.snapshot().values())} API", flush=True)
    report.write(ctx, scens, all_res, mock=a.backend == "mock")
    print(f"\nReport: {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
