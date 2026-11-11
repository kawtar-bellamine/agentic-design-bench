"""Run the benchmark.

    python run.py                          # baselines without an LLM (about 2 minutes the first time)
    python run.py --backend gemini         # LLM agents; key in the GEMINI_API_KEY environment variable
    python run.py --backend groq --quick   # fewer tasks, for a free tier with a tight daily limit
    python run.py --backend ollama --dev 3 --test 6   # free local model, small run
    python run.py --report                 # redraw figures and the README results from results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from designbench.agents import LLMAgent, RandomSearch, RuleOfThumb, reflect
from designbench.bench import load_tasks, score, summarize

ROOT = Path(__file__).parent
RES = ROOT / "results"


def run_baselines(test):
    rows = []
    for seed in range(3):
        rows += [score(RandomSearch(seed).solve(s), s) | {"agent": f"random_search"} for s in test]
    rows += [score(RuleOfThumb().solve(s), s) for s in test]
    (RES / "baselines.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("random_search", summarize([r for r in rows if r["agent"] == "random_search"]))
    print("rule_of_thumb", summarize([r for r in rows if r["agent"] == "rule_of_thumb"]))


def run_llm(backend, model, dev, test, rpm):
    from designbench.llm import LLM
    llm = LLM(backend, model, rpm=rpm)
    tag = f"{backend}_{llm.model}".replace("/", "-").replace(":", "-")
    out = {"backend": backend, "model": llm.model, "rows": [], "playbook_history": []}
    t0 = time.time()
    for name, agent in (("single_shot", LLMAgent(llm, "single_shot", budget=1)),
                        ("tool_loop", LLMAgent(llm, "tool_loop")),
                        ("critic_team", LLMAgent(llm, "critic_team", critic=True))):
        for s in test:
            out["rows"].append(score(agent.solve(s), s))
        print(name, summarize([r for r in out["rows"] if r["agent"] == name]), f"{time.time() - t0:.0f} s", flush=True)
        _save(out, llm, tag, t0)              # partial results survive a quota stop
    # self-improvement: learn a playbook on the dev tasks (verifier feedback only), freeze it, test
    playbook = []
    learner = LLMAgent(llm, "memory_learn")
    for i, s in enumerate(dev):
        learner.playbook = playbook
        playbook = reflect(llm, s, learner.solve(s), playbook, f"{s.task_id}|{i}")
        out["playbook_history"].append(list(playbook))
    agent = LLMAgent(llm, "memory_loop", playbook=playbook)
    for s in test:
        out["rows"].append(score(agent.solve(s), s))
    print("memory_loop", summarize([r for r in out["rows"] if r["agent"] == "memory_loop"]), flush=True)
    out["playbook"] = playbook
    _save(out, llm, tag, t0)


def _save(out, llm, tag, t0):
    out["usage"] = {"calls": llm.calls, "cached": llm.cached, "tokens": llm.tokens, "minutes": (time.time() - t0) / 60}
    (RES / f"llm_{tag}.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini", "groq", "mistral", "openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--quick", action="store_true", help="5 dev and 10 test tasks")
    ap.add_argument("--dev", type=int, help="number of dev tasks (playbook learning)")
    ap.add_argument("--test", type=int, help="number of test tasks")
    ap.add_argument("--rpm", type=float, default=12, help="requests per minute allowed by your plan")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    dev, test = load_tasks()
    if a.quick:
        dev, test = dev[:5], test[:10]
    dev, test = dev[: a.dev or len(dev)], test[: a.test or len(test)]
    if not a.report:
        if a.backend:
            run_llm(a.backend, a.model, dev, test, a.rpm)
        else:
            run_baselines(test)
    from designbench.report import build
    build(RES, ROOT)


if __name__ == "__main__":
    main()
