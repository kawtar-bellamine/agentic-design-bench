"""Tests: verifier physics, scoring, JSON parsing and the agent loops (with a scripted stand-in for the LLM)."""
from __future__ import annotations

import json
import math
import re

from designbench.agents import LLMAgent, RandomSearch, RuleOfThumb, reflect
from designbench.bench import load_tasks, paired_bootstrap, score, summarize
from designbench.llm import parse_json
from designbench.physics import Design, Spec, evaluate, section

SPEC = Spec("X", 150, 600, 1.0, 2.0, "M6", 30, 60, ("aluminium_6061", "steel_s355"))


def test_section_matches_rectangle_and_tee():
    i, top, bot = section(40, 10, 0, 0)
    assert math.isclose(i, 40 * 10 ** 3 / 12) and math.isclose(top, 5) and math.isclose(bot, 5)
    i_tee, _, _ = section(40, 4, 20, 4)
    assert i_tee > section(40, 4, 0, 0)[0] * 20          # a rib is far stiffer than the flange alone


def test_verifier_physics_scales_as_expected():
    a = evaluate(SPEC, Design("aluminium_6061", 40, 4, 20, 4))
    s = evaluate(SPEC, Design("steel_s355", 40, 4, 20, 4))
    assert math.isclose(a["deflection_mm"] / s["deflection_mm"], 210_000 / 69_000, rel_tol=1e-3)
    assert s["mass_g"] > 2.8 * a["mass_g"]
    deeper = evaluate(SPEC, Design("aluminium_6061", 40, 4, 24, 4))
    assert deeper["deflection_mm"] < a["deflection_mm"]


def test_verifier_flags_every_rule():
    r = evaluate(SPEC, Design("titanium_ti6al4v", 70.3, 1.0, 40, 2))
    text = " ".join(r["issues"])
    for word in ("not allowed",):
        assert word in text
    r = evaluate(SPEC, Design("aluminium_6061", 70.3, 1.0, 40, 2))
    text = " ".join(r["issues"])
    for word in ("grid", "minimum wall", "slender", "envelope"):
        assert word in text, word
    assert not r["feasible"]


def test_parse_json_handles_prose_and_fences():
    assert parse_json('Sure!\n```json\n{"design": {"b": 1}}\n```') == {"design": {"b": 1}}
    assert parse_json("no json here") is None


class ScriptedLLM:
    """Stands in for a model: proposes an over-designed bracket, then shrinks it; also answers reflection prompts."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tag=""):
        self.calls += 1
        system = messages[0]["content"]
        if "playbook" in system and "Merge" in system:
            return json.dumps({"playbook": ["Use a deep rib before a thick flange."]})
        if "lessons" in system:
            return json.dumps({"lessons": ["Use a deep rib before a thick flange."]})
        if "senior structural engineer" in system:
            return json.dumps({"diagnosis": "ok", "changes": ["reduce width"]})
        user = messages[1]["content"]
        mat = re.search(r"Allowed materials: ([a-z0-9_]+)", user).group(1)
        n = user.count('"design"')
        b = max(30.0, 60.0 - 5 * n)
        return "Here you go: " + json.dumps({"reasoning": "x", "design": {"material": mat, "b": b, "t": 2, "h": 26,
                                                                            "tr": 4}})


def test_agent_loops_run_and_score():
    llm = ScriptedLLM()
    for agent in (LLMAgent(llm, "tool_loop"), LLMAgent(llm, "critic_team", critic=True)):
        tr = agent.solve(SPEC)
        assert len(tr.rounds) == 6 and tr.best() is not None
        row = score(tr, SPEC)
        assert row["feasible"] and row["regret"] >= 0.99
    playbook = reflect(llm, SPEC, tr, [], "t")
    assert playbook == ["Use a deep rib before a thick flange."]


def test_baselines_and_summary():
    rows = [score(RuleOfThumb().solve(SPEC), SPEC), score(RandomSearch(0).solve(SPEC), SPEC)]
    s = summarize(rows)
    assert 0 <= s["success"] <= s["feasible"] <= 1
    d, lo, hi = paired_bootstrap(rows[:1], rows[:1])
    assert d == lo == hi == 0


def test_saved_tasks_have_a_feasible_reference():
    dev, test = load_tasks()
    assert len(dev) == 10 and len(test) == 20
    from designbench.bench import reference
    assert all(reference(s)["feasible"] for s in dev + test)


def test_verifier_never_crashes_on_degenerate_designs():
    """LLM agents propose anything, including a rib with height but no thickness: the verifier must answer with
    issues, never raise."""
    import itertools
    from designbench.bench import load_tasks
    from designbench.physics import Design, evaluate
    spec = load_tasks()[1][0]
    for b, t, h, tr in itertools.product([0.0, 0.5, 2.0, 30.0], repeat=4):
        rep = evaluate(spec, Design(spec.materials[0], b, t, h, tr))
        assert "feasible" in rep
