"""Design agents. Every agent gets the same verifier and the same budget of verifier calls.

LLM agents (any OpenAI-compatible model):
    single_shot   one proposal, no feedback
    tool_loop     propose, read the verifier report, revise
    critic_team   a designer and a critic: the critic reads the report and tells the designer what to change
    memory_loop   tool_loop plus a memory of lessons written by the agent itself on the dev tasks (self-improvement)
Baselines without an LLM:
    random_search, rule_of_thumb
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from .llm import parse_json
from .physics import BOLT_CLEARANCE, MIN_WALL, Design, Spec, evaluate, snap

BUDGET = 6

DESIGNER = """You are a mechanical design engineer sizing a machined cantilever bracket.
Geometry: a flange of width b and thickness t, with an optional central rib under the flange of height h and
thickness tr (h = 0 means no rib). All dimensions in mm, on a 0.5 mm grid.
After each proposal a verifier tool returns the mass, stress, deflection and every broken rule.
Answer with JSON only:
{"reasoning": "<one short sentence>", "design": {"material": "<allowed material>", "b": <mm>, "t": <mm>, "h": <mm>, "tr": <mm>}}"""

CRITIC = """You are a senior structural engineer reviewing a junior engineer's bracket design.
You receive the specification, the design and the verifier report. Give at most three concrete changes that make
the design feasible first, then lighter, each with a one-line physical reason. Answer with JSON only:
{"diagnosis": "<one sentence>", "changes": ["...", "..."]}"""

REFLECT = """You just finished sizing a bracket with a verifier tool. From the trace below, write at most three
general lessons that would help on OTHER brackets (not numbers specific to this one). Answer with JSON only:
{"lessons": ["...", "..."]}"""

CURATE = """You maintain a short playbook of design lessons. Merge the current playbook with the new lessons:
remove duplicates, keep the most general and useful ones, at most 8 in total. Answer with JSON only:
{"playbook": ["...", "..."]}"""


@dataclass
class Trace:
    agent: str
    task_id: str
    rounds: list = field(default_factory=list)   # verifier reports, in order
    llm_calls: int = 0
    parse_failures: int = 0
    notes: list = field(default_factory=list)

    def best(self) -> dict | None:
        ok = [r for r in self.rounds if r.get("feasible")]
        return min(ok, key=lambda r: r["mass_g"]) if ok else None

    def best_by_round(self) -> list:
        out, cur = [], None
        for r in self.rounds:
            if r.get("feasible") and (cur is None or r["mass_g"] < cur):
                cur = r["mass_g"]
            out.append(cur)
        return out


def _to_design(d: dict) -> Design | None:
    try:
        return Design(str(d["material"]), float(d["b"]), float(d["t"]), float(d.get("h", 0) or 0),
                      float(d.get("tr", 0) or 0))
    except (KeyError, TypeError, ValueError):
        return None


def _compact(report: dict) -> str:
    keys = ("feasible", "mass_g", "stress_mpa", "allowed_stress_mpa", "deflection_mm", "deflection_limit_mm", "issues")
    return json.dumps({"design": report.get("design"), **{k: report.get(k) for k in keys}})


# ---------------------------------------------------------------- LLM agents
class LLMAgent:
    def __init__(self, llm, name: str, critic: bool = False, playbook: list | None = None, budget: int = BUDGET):
        self.llm, self.name, self.critic, self.playbook, self.budget = llm, name, critic, playbook or [], budget

    def _ask_design(self, spec: Spec, history: list, advice: str | None, trace: Trace, rnd: int):
        user = [f"Specification: {spec.describe()}"]
        if self.playbook:
            user.append("Your playbook of lessons from previous brackets:\n- " + "\n- ".join(self.playbook))
        if history:
            user.append("Previous proposals and verifier reports, oldest first:\n" + "\n".join(map(_compact, history)))
        if advice:
            user.append(f"Review from the senior engineer: {advice}")
        user.append("Propose the next design: make it feasible first, then lighter than your best feasible one.")
        msgs = [{"role": "system", "content": DESIGNER}, {"role": "user", "content": "\n\n".join(user)}]
        for attempt in range(2):
            text = self.llm.chat(msgs, tag=f"{self.name}|{spec.task_id}|{rnd}|{attempt}")
            trace.llm_calls += 1
            obj = parse_json(text)
            design = _to_design(obj.get("design", {})) if obj else None
            if design:
                return design
            trace.parse_failures += 1
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "That was not valid JSON with a design. Answer with JSON only."})
        return None

    def _review(self, spec: Spec, report: dict, trace: Trace, rnd: int) -> str | None:
        msgs = [{"role": "system", "content": CRITIC},
                {"role": "user", "content": f"Specification: {spec.describe()}\n\nVerifier report: {_compact(report)}"}]
        text = self.llm.chat(msgs, tag=f"{self.name}-critic|{spec.task_id}|{rnd}")
        trace.llm_calls += 1
        obj = parse_json(text) or {}
        changes = obj.get("changes") or []
        return (obj.get("diagnosis", "") + " Changes: " + "; ".join(map(str, changes))).strip() or None

    def solve(self, spec: Spec) -> Trace:
        trace = Trace(self.name, spec.task_id)
        history, advice = [], None
        for rnd in range(self.budget):
            design = self._ask_design(spec, history, advice, trace, rnd)
            if design is None:
                continue
            report = evaluate(spec, design)
            report.setdefault("design", asdict(design))
            trace.rounds.append(report)
            history.append(report)
            if self.critic and rnd < self.budget - 1:
                advice = self._review(spec, report, trace, rnd)
        return trace


def reflect(llm, spec: Spec, trace: Trace, playbook: list, rnd_tag: str) -> list:
    """Self-improvement step: the agent writes lessons from its own trace (verifier feedback only, no answer key),
    then merges them into its playbook."""
    summary = "\n".join(_compact(r) for r in trace.rounds)
    obj = parse_json(llm.chat([{"role": "system", "content": REFLECT},
                               {"role": "user", "content": f"Specification: {spec.describe()}\n\nTrace:\n{summary}"}],
                              tag=f"reflect|{rnd_tag}")) or {}
    lessons = [str(x) for x in obj.get("lessons", [])][:3]
    if not lessons:
        return playbook
    obj = parse_json(llm.chat([{"role": "system", "content": CURATE},
                               {"role": "user", "content": json.dumps({"playbook": playbook, "new": lessons})}],
                              tag=f"curate|{rnd_tag}")) or {}
    merged = [str(x) for x in obj.get("playbook", [])][:8]
    return merged or (playbook + lessons)[-8:]


# ---------------------------------------------------------------- baselines without an LLM
def _ranges(spec: Spec):
    d = BOLT_CLEARANCE[spec.bolt]
    return 3 * d, spec.max_width, spec.max_height


class RandomSearch:
    name = "random_search"

    def __init__(self, seed: int = 0, budget: int = BUDGET):
        self.seed, self.budget = seed, budget

    def solve(self, spec: Spec) -> Trace:
        rng = random.Random(f"{self.seed}-{spec.task_id}")
        bmin, bmax, hmax = _ranges(spec)
        trace = Trace(self.name, spec.task_id)
        for _ in range(self.budget):
            t = snap(rng.uniform(MIN_WALL, min(8, hmax)))
            h = snap(rng.uniform(0, hmax - t)) if rng.random() < 0.7 else 0.0
            tr = max(snap(h / rng.uniform(3, 8)), MIN_WALL) if h else 0.0
            design = Design(rng.choice(spec.materials), snap(rng.uniform(bmin, bmax)), t, h, tr)
            trace.rounds.append(evaluate(spec, design))
        return trace


class RuleOfThumb:
    """What a careful engineer does by hand: start deep and ribbed, fix what the verifier flags, then trim."""
    name = "rule_of_thumb"

    def __init__(self, budget: int = BUDGET):
        self.budget = budget

    def solve(self, spec: Spec) -> Trace:
        bmin, bmax, hmax = _ranges(spec)
        mats = sorted(spec.materials, key=lambda m: {"aluminium_6061": 0, "titanium_ti6al4v": 1}.get(m, 2))
        t, h = MIN_WALL, snap(hmax - MIN_WALL)
        tr = max(snap(h / 6), MIN_WALL)
        b = snap(max(bmin, min(bmax, (bmin + bmax) / 2)))
        mat_i, trace, last_ok = 0, Trace(self.name, spec.task_id), None
        for _ in range(self.budget):
            design = Design(mats[mat_i], b, t, h, tr)
            r = evaluate(spec, design)
            trace.rounds.append(r)
            if r["feasible"]:
                last_ok = design
                scale = max(r["stress_ratio"], r["deflection_ratio"])
                b = snap(max(bmin, b * max(0.6, scale ** 0.8)))
                tr = max(snap(tr * max(0.7, scale)), MIN_WALL)
            else:
                worst = max(r.get("stress_ratio", 2), r.get("deflection_ratio", 2))
                if last_ok is not None:
                    b, t, h, tr = last_ok.b, last_ok.t, last_ok.h, last_ok.tr
                    break
                if b < bmax:
                    b = snap(min(bmax, b * worst))
                elif tr < h / 3:
                    tr = snap(tr * 1.5)
                elif mat_i < len(mats) - 1:
                    mat_i += 1
                else:
                    t = snap(min(hmax - h, t * 1.5)) if h + t * 1.5 <= hmax else t
        return trace
