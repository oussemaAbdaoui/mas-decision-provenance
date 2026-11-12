"""Offline tests (mock backend, no API key).  Run: python -m pytest -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agents.chain import replay, run_chain, state_of  # noqa: E402
from agents.llm import LLM  # noqa: E402
from agents.rules import ground_truth_decision  # noqa: E402
from agents.scenario import list_scenarios, load_scenario  # noqa: E402
from experiments import perturb  # noqa: E402
from provenance.graph import DECISION, build_graph, dirty_activities, to_prov_json, upstream_elements  # noqa: E402


@pytest.fixture(scope="module")
def llm(tmp_path_factory):
    return LLM("mock", log_path=tmp_path_factory.mktemp("log") / "calls.sqlite", mock_noise=0.0)


@pytest.mark.parametrize("name", list_scenarios())
def test_ground_truth_is_consistent(name):
    s = load_scenario(name)
    assert ground_truth_decision(s)[0] == s.correct
    assert 10 <= len(s.gt_requirements) <= 15
    assert all(r["source_passage"] for r in s.gt_requirements)
    assert {True, False} == {r["critical"] for r in s.gt_requirements}


def test_chain_and_graph(llm):
    s = load_scenario("drone_battery")
    t = run_chain(llm, s, tag="t", label="test")
    assert t["decision"]["selected"] == s.correct
    G = build_graph(t)
    assert G.nodes[f"chk_R01_candC"]["type"] == "check_result"
    elements, collapsed, uncited = upstream_elements(G)
    # spec passages never perturbed directly: each is post-dominated by its requirement
    assert not [e for e in elements if e.startswith("spec_")]
    assert collapsed["spec_" + s.gt_req("R01")["source_passage"]] == "req_R01"
    # distractor passages are not cited, hence outside the restricted set
    assert "ds_C_p9" in uncited and "ds_C_p9" not in elements
    # a datasheet passage only dirties its own checker (+ decider)
    assert dirty_activities(G, "ds_C_p4") == {"checker_C", "decider"}
    assert dirty_activities(G, "req_R01") == {f"checker_{c}" for c in "ABCD"} | {"decider"}
    prov = to_prov_json(G)
    chk = next(n for n in prov["nodes"] if n["id"] == "chk_R01_candC")
    assert chk["generated_by"] == "act_checker_C" and "req_R01" in chk["used"]
    assert DECISION in G


def test_replay_reuses_upstream(llm):
    s = load_scenario("drone_battery")
    ref = run_chain(llm, s, tag="t", label="test")
    G = build_graph(ref)
    st, _ = perturb.apply(state_of(ref), s, "req_R01", "removal")
    before = llm.counter.snapshot().get("replay", {}).get("logical", 0)
    t = replay(llm, s, ref, st, dirty_activities(G, "req_R01"), tag="r", label="replay")
    assert t["decision"]["selected"] == "D"          # R01 removed -> cheaper D becomes compliant
    assert "extractor" in t["reused_activities"]
    assert llm.counter.snapshot()["replay"]["logical"] - before == 5   # 4 checkers + decider


def test_perturbation_operators():
    s = load_scenario("drone_battery")
    t = {"inputs": {"spec_passages": [], "criteria": [dict(k) for k in s.criteria],
                    "datasheets": {c["id"]: {"name": c["name"], "passages": c["passages"]} for c in s.candidates}},
         "requirements": [dict(r) for r in s.gt_requirements]}
    st, d = perturb.apply(t, s, "req_R01", "alteration")
    assert next(r for r in st["requirements"] if r["id"] == "R01")["value"] == 60
    st, d = perturb.apply(t, s, "req_R02", "injection")
    assert any(r["id"] == "R02X" and r["value"] == 400 and r["critical"] for r in st["requirements"])
    st, d = perturb.apply(t, s, "ds_C_p2", "alteration")
    assert "429 g" in st["inputs"]["datasheets"]["C"]["passages"][1]["text"]
    assert perturb.scale_text("30 000 h at −15 °C", 0.67) == "20 100 h at −10 °C"
    assert perturb.apply(t, s, "crit_K1", "injection")[0]["inputs"]["criteria"][0]["id"] == "K1X"
