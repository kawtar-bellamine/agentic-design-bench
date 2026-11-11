"""Deterministic verifier for a machined cantilever bracket: a flange with an optional central rib.

Units: mm, N, MPa, g. Beam theory with an approximate stress concentration at the bolt holes. The point is not
FEA accuracy but a fast, transparent and repeatable verifier that an agent can call as a tool.

        F (tip load, downward)
        |
  ======v====================  flange  b x t
            ||                 rib     tr x h (optional, under the flange)
  root (2 bolts, diameter d)        length L
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MATERIALS = {  # E (MPa), yield (MPa), density (kg/m3), price (EUR/kg)
    "aluminium_6061": dict(E=69_000, yield_=276, rho=2700, price=4.5),
    "steel_s355": dict(E=210_000, yield_=355, rho=7850, price=1.2),
    "titanium_ti6al4v": dict(E=114_000, yield_=880, rho=4430, price=30.0),
}
BOLT_CLEARANCE = {"M5": 5.5, "M6": 6.6, "M8": 9.0}
MIN_WALL = 2.0          # mm, machining
MAX_RIB_SLENDERNESS = 8  # h / tr
GRID = 0.5              # mm, dimensions are machined to 0.5 mm


@dataclass(frozen=True)
class Spec:
    task_id: str
    length: float           # L, mm
    load: float             # F at the tip, N
    max_deflection: float   # mm
    safety_factor: float    # on yield
    bolt: str               # M5, M6, M8
    max_height: float       # envelope: t + h <= max_height
    max_width: float        # envelope: b <= max_width
    materials: tuple        # allowed materials

    def describe(self) -> str:
        return (f"Cantilever bracket, length {self.length:g} mm, tip load {self.load:g} N. "
                f"Tip deflection at most {self.max_deflection:g} mm. Safety factor on yield at least "
                f"{self.safety_factor:g}. Two {self.bolt} bolts at the root (clearance hole "
                f"{BOLT_CLEARANCE[self.bolt]:g} mm). Envelope: total height t + h at most {self.max_height:g} mm, "
                f"width b at most {self.max_width:g} mm. Allowed materials: {', '.join(self.materials)}. "
                f"Goal: the lightest bracket that satisfies every rule.")


@dataclass(frozen=True)
class Design:
    material: str
    b: float    # flange width
    t: float    # flange thickness
    h: float    # rib height (0 = no rib)
    tr: float   # rib thickness (ignored when h = 0)


def section(b: float, t: float, h: float, tr: float):
    """Second moment of area and distances from the neutral axis to the top (flange) and bottom fibres."""
    a1, y1 = b * t, h + t / 2
    a2, y2 = (tr * h, h / 2) if h > 0 else (0.0, 0.0)
    area = a1 + a2
    ybar = (a1 * y1 + a2 * y2) / area
    inertia = b * t ** 3 / 12 + a1 * (y1 - ybar) ** 2
    if h > 0:
        inertia += tr * h ** 3 / 12 + a2 * (y2 - ybar) ** 2
    return inertia, (h + t) - ybar, ybar


def kt_hole(d: float, w: float) -> float:
    """Stress concentration of a hole of diameter d in a strip of width w (Heywood's fit, tension)."""
    r = min(d / w, 0.9)
    return 3.0 - 3.14 * r + 3.667 * r ** 2 - 1.527 * r ** 3


def evaluate(spec: Spec, design: Design) -> dict:
    """The verifier tool: every number an engineer would check, and every rule that is broken."""
    m = MATERIALS.get(design.material)
    d = BOLT_CLEARANCE[spec.bolt]
    issues = []
    if m is None or design.material not in spec.materials:
        return {"feasible": False, "mass_g": None, "issues": [f"material '{design.material}' is not allowed"]}
    b, t, h, tr = design.b, design.t, design.h, (design.tr if design.h > 0 else 0.0)
    for name, v in (("b", b), ("t", t), ("h", h), ("tr", tr)):
        if v < 0:
            return {"feasible": False, "mass_g": None, "issues": [f"{name} is negative"]}
        if abs(v / GRID - round(v / GRID)) > 1e-6:
            issues.append(f"{name} = {v:g} mm is not on the 0.5 mm machining grid")
    if t < MIN_WALL:
        issues.append(f"flange thickness t = {t:g} mm is below the {MIN_WALL:g} mm minimum wall")
    if h > 0 and tr < MIN_WALL:
        issues.append(f"rib thickness tr = {tr:g} mm is below the {MIN_WALL:g} mm minimum wall")
    if h > 0 and tr > 0 and h > MAX_RIB_SLENDERNESS * tr:
        issues.append(f"rib too slender: h / tr = {h / tr:.1f} > {MAX_RIB_SLENDERNESS}")
    if b < 3 * d:
        issues.append(f"width b = {b:g} mm is below 3 x hole diameter = {3 * d:g} mm (edge distance)")
    if b > spec.max_width:
        issues.append(f"width b = {b:g} mm exceeds the envelope {spec.max_width:g} mm")
    if t + h > spec.max_height:
        issues.append(f"height t + h = {t + h:g} mm exceeds the envelope {spec.max_height:g} mm")
    if h > 0 and tr > b:
        issues.append("rib thicker than the flange width")
    if t <= 0 or b <= 0:
        return {"feasible": False, "mass_g": None, "issues": issues + ["zero section"]}

    inertia, c_top, c_bot = section(b, t, h, tr)
    inertia_net, c_top_n, c_bot_n = section(b - d, t, h, tr) if b > d else (1e-9, 1, 1)
    moment = spec.load * spec.length
    stress_root_top = kt_hole(d, b) * moment * c_top_n / inertia_net
    stress_root_bot = moment * c_bot / inertia
    stress = max(stress_root_top, stress_root_bot)
    allowed = m["yield_"] / spec.safety_factor
    deflection = spec.load * spec.length ** 3 / (3 * m["E"] * inertia)
    volume = spec.length * (b * t + tr * h) - 2 * math.pi * d ** 2 / 4 * t
    mass = volume * m["rho"] * 1e-6
    if stress > allowed:
        issues.append(f"stress {stress:.0f} MPa exceeds the allowed {allowed:.0f} MPa "
                      f"(yield {m['yield_']} / safety factor {spec.safety_factor:g})")
    if deflection > spec.max_deflection:
        issues.append(f"tip deflection {deflection:.2f} mm exceeds the limit {spec.max_deflection:g} mm")
    return {"feasible": not issues, "mass_g": round(mass, 2), "stress_mpa": round(stress, 1),
            "allowed_stress_mpa": round(allowed, 1), "deflection_mm": round(deflection, 3),
            "deflection_limit_mm": spec.max_deflection, "stress_ratio": round(stress / allowed, 3),
            "deflection_ratio": round(deflection / spec.max_deflection, 3),
            "cost_eur": round(mass / 1000 * m["price"], 2), "issues": issues, "design": asdict(design)}


def snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 3)
