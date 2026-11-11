"""Task generator and the reference optimizer used only to score agents (agents never see it)."""
from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
from scipy.optimize import differential_evolution

from .physics import BOLT_CLEARANCE, GRID, MIN_WALL, Design, Spec, evaluate, snap

MATERIAL_SETS = [("aluminium_6061",), ("aluminium_6061", "steel_s355"),
                 ("aluminium_6061", "steel_s355", "titanium_ti6al4v"), ("steel_s355",)]


def generate(n: int, seed: int) -> list[Spec]:
    """Varied but solvable specifications. Infeasible draws are discarded, so every task has an answer."""
    rng = random.Random(seed)
    out, i = [], 0
    while len(out) < n:
        i += 1
        length = rng.choice([80, 100, 120, 150, 180, 220, 250])
        spec = Spec(task_id=f"T{seed}-{i:03d}", length=length,
                    load=rng.choice([150, 250, 400, 600, 900, 1300, 2000]),
                    max_deflection=round(length / rng.choice([150, 250, 400, 600]), 2),
                    safety_factor=rng.choice([1.5, 2.0, 2.5]), bolt=rng.choice(list(BOLT_CLEARANCE)),
                    max_height=rng.choice([12, 16, 20, 25, 32, 40]), max_width=rng.choice([30, 40, 50, 60, 80]),
                    materials=rng.choice(MATERIAL_SETS))
        if best_known(spec)["feasible"]:
            out.append(spec)
    return out


def _bounds(spec: Spec):
    d = BOLT_CLEARANCE[spec.bolt]
    return [(3 * d, spec.max_width), (MIN_WALL, spec.max_height), (0.0, spec.max_height - MIN_WALL), (MIN_WALL, 15.0)]


def _penalised(x, spec: Spec, material: str) -> float:
    b, t, h, tr = x
    if h < 1.0:
        h = 0.0
    h = min(h, spec.max_height - t) if h > 0 else 0.0
    r = evaluate(spec, Design(material, b, t, max(h, 0.0), tr))
    if r["mass_g"] is None:
        return 1e9
    pen = sum(10 * max(0.0, r[k] - 1) for k in ("stress_ratio", "deflection_ratio"))
    pen += 5 * sum(1 for i in r["issues"] if "grid" not in i and "stress" not in i and "deflection" not in i)
    return r["mass_g"] * (1 + 100 * pen)


@lru_cache(maxsize=None)
def best_known(spec: Spec) -> dict:
    """Differential evolution per material, then snapped to the machining grid and checked by the verifier."""
    best = {"feasible": False, "mass_g": None}
    for mat in spec.materials:
        res = differential_evolution(_penalised, _bounds(spec), args=(spec, mat), seed=0, maxiter=150, popsize=20,
                                     tol=1e-8, polish=False)
        b, t, h, tr = res.x
        cands = []
        for db in (0, GRID, 2 * GRID):
            for dt in (0, GRID, 2 * GRID):
                for dh in (0, GRID, 2 * GRID):
                    hh = 0.0 if h < 1.0 else snap(h) + dh
                    cands.append(Design(mat, snap(b) + db, snap(t) + dt, hh, max(snap(tr), MIN_WALL)))
        for c in cands:
            r = evaluate(spec, c)
            if r["feasible"] and (not best["feasible"] or r["mass_g"] < best["mass_g"]):
                best = r
    return best


def split(seed: int = 7, n_dev: int = 10, n_test: int = 20) -> tuple[list[Spec], list[Spec]]:
    tasks = generate(n_dev + n_test, seed)
    return tasks[:n_dev], tasks[n_dev:]


def regret(mass: float | None, spec: Spec) -> float:
    """Mass relative to the best design known for the task (1.0 = as good as the optimizer)."""
    ref = best_known(spec)["mass_g"]
    return float(mass / ref) if mass is not None and ref else float("nan")


__all__ = ["generate", "best_known", "split", "regret", "np"]
