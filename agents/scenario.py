"""Scenario loading: specification passages, datasheets, criteria and ground truth."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "scenarios"
REQ_LABEL = re.compile(r"^\[(R\d+)\]")


@dataclass
class Scenario:
    name: str
    title: str
    spec_passages: list[dict]          # [{"id": "S03", "text": ..., "label": "R01" | None}]
    candidates: list[dict]             # [{"id": "A", "name": ..., "passages": [...]}]
    criteria: list[dict]
    ground_truth: dict
    gt_requirements: list[dict] = field(default_factory=list)

    @property
    def correct(self) -> str:
        return self.ground_truth["correct_candidate"]

    @property
    def candidate_ids(self) -> list[str]:
        return [c["id"] for c in self.candidates]

    def gt_req(self, rid: str) -> dict | None:
        base = rid.rstrip("X")
        return next((r for r in self.gt_requirements if r["id"] == base), None)

    def is_critical(self, rid: str) -> bool | None:
        r = self.gt_req(rid)
        return None if r is None else bool(r["critical"])


def split_spec(text: str) -> list[dict]:
    """Split a markdown specification into passages (one per paragraph, headings skipped)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    passages = []
    for p in paragraphs:
        if p.startswith("#"):
            continue
        m = REQ_LABEL.match(p)
        passages.append({"id": f"S{len(passages) + 1:02d}", "text": " ".join(p.split()),
                         "label": m.group(1) if m else None})
    return passages


def load_scenario(name: str) -> Scenario:
    d = SCENARIO_DIR / name
    spec = split_spec((d / "spec.md").read_text(encoding="utf-8"))
    ds = json.loads((d / "datasheets.json").read_text(encoding="utf-8"))
    criteria = json.loads((d / "criteria.json").read_text(encoding="utf-8"))
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    by_label = {p["label"]: p["id"] for p in spec if p["label"]}
    gt_reqs = []
    for r in gt["requirements"]:
        r = dict(r)
        r["source_passage"] = by_label.get(r["id"])
        gt_reqs.append(r)
    return Scenario(name=name, title=gt.get("title", name), spec_passages=spec,
                    candidates=ds["candidates"], criteria=criteria, ground_truth=gt,
                    gt_requirements=gt_reqs)


def list_scenarios() -> list[str]:
    return sorted(p.name for p in SCENARIO_DIR.iterdir() if (p / "spec.md").exists())
