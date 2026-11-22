"""Pareto frontier figure for the NSGA-II matching search.

Colour does not encode execution order, which says nothing about the result: trials are
one neutral mark, frontier members another, and the selected configuration a third. What the reader needs in order to judge the selection is the F1
iso-contours, since the paper claims the chosen point is the F1 maximum of the frontier
and that claim should be checkable by eye.

Typeset to sit inside a LaTeX article at \\textwidth: serif face, 9 pt, no colour bar,
no chart junk.

    python scripts/plot_pareto.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DB = pathlib.Path("data/Output_Repro/Tuning/Session_V5-3_Pareto_20260827_134728/optuna_pareto.db")
# A single accent, carrying a real distinction rather than decoration: everything on the
# non-dominated set is coloured, everything dominated is grey. Dark enough to survive a
# greyscale print as a distinct tone, and blue rather than red/green for colour vision.
ACCENT = "#1A5276"

OUT = pathlib.Path("data/Output_Repro/Figures/fig13_nsga2_pareto_frontier.png")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.3",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
})


def f1(r, p):
    return np.where(r + p > 0, 2 * r * p / np.where(r + p > 0, r + p, 1), 0.0)


def main() -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    name = optuna.study.get_all_study_names(f"sqlite:///{DB}")[0]
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{DB}")
    done = [t for t in study.trials if t.values]
    rec = np.array([t.values[0] for t in done])
    pre = np.array([t.values[1] for t in done])

    front = study.best_trials
    fr = np.array([t.values[0] for t in front])
    fp = np.array([t.values[1] for t in front])
    order = np.argsort(fr)
    fr, fp = fr[order], fp[order]

    best = max(front, key=lambda t: 2 * t.values[0] * t.values[1] /
               (t.values[0] + t.values[1]))
    br, bp = best.values[0], best.values[1]
    bf1 = 2 * br * bp / (br + bp)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # F1 iso-contours: the reference grid against which the selection is judged
    gr = np.linspace(0.20, 0.64, 400)
    gp = np.linspace(0.68, 0.85, 400)
    G, P = np.meshgrid(gr, gp)
    levels = [0.45, 0.50, 0.55, 0.60, 0.65]
    cs = ax.contour(G, P, f1(G, P), levels=levels,
                    colors="0.78", linewidths=0.6, linestyles=[(0, (4, 3))], zorder=1)
    # label each contour high in the panel, clear of both the data cloud and the legend
    p_lab = 0.826
    ax.clabel(cs, fmt=lambda v: f"$F_1$ = {v:.2f}", fontsize=7, inline=True,
              inline_spacing=4, colors="0.5",
              manual=[(L * p_lab / (2 * p_lab - L), p_lab) for L in levels])

    ax.scatter(rec, pre, s=13, facecolor="0.80", edgecolor="0.55", linewidth=0.4,
               zorder=2, label=f"Evaluated configuration ($n$ = {len(done)})")
    ax.plot(fr, fp, color=ACCENT, linewidth=1.0, alpha=0.85, zorder=3)
    ax.scatter(fr, fp, s=20, facecolor="white", edgecolor=ACCENT, linewidth=1.0,
               zorder=4, label=f"Non-dominated set ($n$ = {len(front)})")
    ax.scatter([br], [bp], s=62, marker="D", facecolor=ACCENT, edgecolor="white",
               linewidth=1.0, zorder=5, label="Selected configuration")

    # right-aligned and inset from the axis so the label cannot run off the panel
    ax.annotate(f"trial {best.number}\nrecall {br:.3f}, precision {bp:.3f}, "
                f"$F_1$ = {bf1:.3f}",
                xy=(br, bp), xycoords="data",
                xytext=(0.566, 0.7130), textcoords="data",
                fontsize=8, ha="right", va="center", color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="0.45", linewidth=0.6,
                                shrinkA=2, shrinkB=5))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.21, 0.625)
    ax.set_ylim(0.695, 0.835)
    ax.set_xticks(np.arange(0.25, 0.63, 0.05))
    ax.set_yticks(np.arange(0.70, 0.84, 0.02))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, handletextpad=0.5,
              borderaxespad=0.2, labelspacing=0.4)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400)
    print(f"selected trial {best.number}: recall {br:.4f}, precision {bp:.4f}, F1 {bf1:.4f}")
    print(f"written -> {OUT.name}")


if __name__ == "__main__":
    main()
