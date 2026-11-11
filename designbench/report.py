"""Figures and the results section of the README, generated from results/*.json only."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .bench import load_tasks, paired_bootstrap, reference, summarize  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"random_search": "#9a9893", "rule_of_thumb": "#52514e", "single_shot": "#eda100", "tool_loop": "#2a78d6",
          "critic_team": "#eb6834", "memory_loop": "#1baf7a"}
LABELS = {"random_search": "random search (no LLM)", "rule_of_thumb": "engineer's rule of thumb (no LLM)",
          "single_shot": "LLM, single shot", "tool_loop": "LLM + verifier loop",
          "critic_team": "LLM designer + critic", "memory_loop": "LLM + self-written playbook"}
ORDER = list(LABELS)
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "legend.frameon": False})
PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def _load(res: Path):
    rows, meta = [], None
    if (res / "baselines.json").exists():
        rows += json.loads((res / "baselines.json").read_text())["rows"]
    llm_files = sorted(res.glob("llm_*.json"), key=lambda p: p.stat().st_mtime)  # latest run wins
    if llm_files:
        meta = json.loads(llm_files[-1].read_text())
        rows += meta["rows"]
    return rows, meta


def _box(ax, x0, y0, z0, dx, dy, dz, color):
    v = [(x0, y0, z0), (x0 + dx, y0, z0), (x0 + dx, y0 + dy, z0), (x0, y0 + dy, z0),
         (x0, y0, z0 + dz), (x0 + dx, y0, z0 + dz), (x0 + dx, y0 + dy, z0 + dz), (x0, y0 + dy, z0 + dz)]
    faces = [[v[i] for i in f] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5),
                                         (0, 3, 7, 4))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="#ffffff", linewidth=0.4, alpha=0.95))


def _draw_design(ax, d, length, title, color):
    b, t, h, tr = d["b"], d["t"], d["h"], d["tr"]
    _box(ax, 0, -b / 2, h, length, b, t, color)
    if h > 0:
        _box(ax, 0, -tr / 2, 0, length, tr, h, color)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect((length, 70, 45))
    ax.set_xlim(0, length); ax.set_ylim(-35, 35); ax.set_zlim(0, 45)
    ax.view_init(elev=22, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)


def build(res: Path, root: Path):
    fig_dir = root / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, meta = _load(res)
    if meta:  # compare every agent on exactly the tasks the LLM run covered (--quick uses fewer)
        ids = {r["task_id"] for r in meta["rows"]}
        rows = [r for r in rows if r["task_id"] in ids]
    agents = [a for a in ORDER if any(r["agent"] == a for r in rows)]
    summ = {a: summarize([r for r in rows if r["agent"] == a]) for a in agents}

    # 1. success and feasibility
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(agents) + 1.3))
    for i, a in enumerate(agents[::-1]):
        ax.barh(i, summ[a]["feasible"], color=COLORS[a], alpha=0.3, height=0.62)
        ax.barh(i, summ[a]["success"], color=COLORS[a], height=0.62)
        ax.text(min(summ[a]["feasible"], 1) + 0.01, i, f"{100 * summ[a]['success']:.0f} % near best  |  "
                f"{100 * summ[a]['feasible']:.0f} % feasible", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(agents)), [LABELS[a] for a in agents[::-1]])
    ax.set_xlim(0, 1.45); ax.xaxis.set_major_formatter(PCT); ax.grid(axis="y", visible=False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Test tasks solved\ndark = feasible and within 10 % of the best known mass, light = feasible")
    fig.tight_layout(); fig.savefig(fig_dir / "methods.png", dpi=160); plt.close(fig)

    # 2. convergence with the verifier budget
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for a in agents:
        ar = [r for r in rows if r["agent"] == a]
        nmax = max(len(r["by_round"]) for r in ar)
        ys = []
        for k in range(nmax):
            vals = [(r["by_round"][min(k, len(r["by_round"]) - 1)] if r["by_round"] else None) for r in ar]
            ok = [v for v in vals if v is not None]
            ys.append(len(ok) / len(ar))
        ax.plot(range(1, nmax + 1), ys, color=COLORS[a], marker="o", markersize=4, linewidth=2, label=LABELS[a])
    ax.yaxis.set_major_formatter(PCT); ax.set_ylim(0, 1.02)
    ax.set_xlabel("verifier calls used"); ax.set_ylabel("tasks with a feasible design")
    ax.set_title("What each extra verifier call buys: feasible designs found so far")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "convergence.png", dpi=160); plt.close(fig)

    # 3. designs for one test task
    dev, test = load_tasks()
    covered = [t for t in test if any(r["task_id"] == t.task_id for r in rows)] or test
    # show a task where the LLM loop found a feasible design, so the picture compares like with like
    spec = next((t for t in covered if any(r["agent"] == "tool_loop" and r["task_id"] == t.task_id and r["best_design"]
                                            for r in rows)), covered[0])
    shown = [("best known (optimizer)", reference(spec)["design"], "#14213D")]
    for a in ("rule_of_thumb", "tool_loop", "memory_loop"):
        r = next((r for r in rows if r["agent"] == a and r["task_id"] == spec.task_id and r["best_design"]), None)
        if r:
            shown.append((f"{LABELS[a]}\n{100 * (r['regret'] - 1):+.0f} % mass", r["best_design"], COLORS[a]))
    fig = plt.figure(figsize=(3.1 * len(shown), 3.3))
    for i, (title, d, c) in enumerate(shown):
        _draw_design(fig.add_subplot(1, len(shown), i + 1, projection="3d"), d, spec.length, title, c)
    fig.suptitle(f"Task {spec.task_id}: {spec.load:g} N at {spec.length:g} mm, deflection under {spec.max_deflection:g} mm",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "designs.png", dpi=160); plt.close(fig)

    _write_readme(root, agents, summ, rows, meta)


def _write_readme(root: Path, agents, summ, rows, meta):
    path = root / "README.md"
    text = path.read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    lines = ["| Agent | Within 10 % of best known | Feasible | Median mass vs best known | LLM calls per task |",
             "|---|---:|---:|---:|---:|"]
    for a in agents:
        s = summ[a]
        med = "n/a" if math.isnan(s["median_regret"]) else f"{100 * (s['median_regret'] - 1):+.0f} %"
        lines.append(f"| {LABELS[a]} | {100 * s['success']:.0f} % | {100 * s['feasible']:.0f} % | {med} | "
                     f"{s['llm_calls_per_task']:.1f} |")
    if meta:
        lines += ["", f"Model: `{meta['model']}` through `{meta['backend']}`, "
                  f"{meta['usage']['calls']} calls, {meta['usage']['tokens']:,} tokens."]
        base = [r for r in rows if r["agent"] == "tool_loop"]
        for a in ("critic_team", "memory_loop"):
            other = [r for r in rows if r["agent"] == a]
            if base and other:
                parts = []
                for key, word in (("success", "success"), ("feasible", "feasible designs")):
                    d, lo, hi = paired_bootstrap(other, base, key=key)
                    parts.append(f"{100 * d:+.0f} points of {word} (95 % interval {100 * lo:+.0f} to {100 * hi:+.0f})")
                lines.append(f"* {LABELS[a]} vs verifier loop, paired over the same tasks: " + "; ".join(parts) + ".")
        if meta.get("playbook"):
            lines += ["", f"Playbook the agent wrote for itself on {len(meta['playbook_history'])} dev tasks, then used "
                      f"frozen on the {len({r['task_id'] for r in meta['rows']})} test tasks:", ""]
            lines += [f"{i + 1}. {p}" for i, p in enumerate(meta["playbook"])]
    else:
        lines += ["", "*LLM agents not run yet: `python run.py --backend gemini` fills this table.*"]
    block = start + "\n" + "\n".join(lines) + "\n" + end
    if start in text:
        text = text[: text.index(start)] + block + text[text.index(end) + len(end):]
    path.write_text(text)
