"""Run agents on tasks and score them. The answer key (best known design) is used for scoring only."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from .physics import Spec
from .tasks import best_known, regret, split

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "results" / "tasks.json"


def load_tasks(n_dev: int = 10, n_test: int = 20) -> tuple[list[Spec], list[Spec]]:
    """Generated once (the reference optimizer takes about two minutes), then read from results/tasks.json."""
    if TASKS.exists():
        data = json.loads(TASKS.read_text())
        to = lambda xs: [Spec(**{**x["spec"], "materials": tuple(x["spec"]["materials"])}) for x in xs]
        dev, test = to(data["dev"]), to(data["test"])
        _REF.update({x["spec"]["task_id"]: x["best_known"] for x in data["dev"] + data["test"]})
        return dev, test
    dev, test = split(n_dev=n_dev, n_test=n_test)
    TASKS.parent.mkdir(exist_ok=True)
    TASKS.write_text(json.dumps({k: [{"spec": asdict(s), "best_known": best_known(s)} for s in v]
                                 for k, v in (("dev", dev), ("test", test))}, indent=1))
    return dev, test


_REF: dict = {}


def reference(spec: Spec) -> dict:
    return _REF.get(spec.task_id) or best_known(spec)


def score(trace, spec: Spec) -> dict:
    best = trace.best()
    ref = reference(spec)["mass_g"]
    reg = best["mass_g"] / ref if best else math.nan
    by_round = [(m / ref if m else None) for m in trace.best_by_round()]
    return {"task_id": spec.task_id, "agent": trace.agent, "feasible": best is not None, "regret": reg,
            "success": bool(best is not None and reg <= 1.10), "verifier_calls": len(trace.rounds),
            "llm_calls": trace.llm_calls, "parse_failures": trace.parse_failures, "by_round": by_round,
            "best_design": best["design"] if best else None, "material": best["design"]["material"] if best else None}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    regs = sorted(r["regret"] for r in rows if r["feasible"])
    med = regs[len(regs) // 2] if regs else math.nan
    return {"tasks": n, "feasible": sum(r["feasible"] for r in rows) / n, "success": sum(r["success"] for r in rows) / n,
            "median_regret": med, "llm_calls_per_task": sum(r["llm_calls"] for r in rows) / n,
            "verifier_calls_per_task": sum(r["verifier_calls"] for r in rows) / n}


def paired_bootstrap(a: list[dict], b: list[dict], n: int = 5000, seed: int = 0,
                     key: str = "success") -> tuple[float, float, float]:
    """Difference in a per-task rate (a minus b, success or feasible) on the same tasks, with a 95 % bootstrap
    interval over tasks."""
    sa = {r["task_id"]: float(r[key]) for r in a}
    sb = {r["task_id"]: float(r[key]) for r in b}
    ids = sorted(set(sa) & set(sb))
    diffs = [sa[i] - sb[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)]
