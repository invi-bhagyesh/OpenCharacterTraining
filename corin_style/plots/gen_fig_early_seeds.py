#!/usr/bin/env python3
"""Figure: the first 10 power-seeking RL steps, every checkpoint, 3 seeds per arm (mean, seed range, seeds).

Usage: python -m corin_style.plots.gen_fig_early_seeds
Output: corin_style/figures/fig_early_seeds.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import COLOR, LABEL, MARKER, WIDTH, line, percent_axis, save, trajectory

SEEDS = ("", "-s2", "-s3")


def main():
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    for arm, name in (("base", "base"), ("aster", "Aster")):
        runs = [trajectory(f"20261022-axes-trajectory-{arm}-power-early{s}", "power") for s in SEEDS]
        steps = [s for s, _ in runs[0]]
        values = [[r[i][1]["p"] for r in runs] for i in range(len(steps))]
        c = COLOR[name]
        ax.fill_between(steps, [min(v) for v in values], [max(v) for v in values], color=c, alpha=0.13, lw=0)
        for points in runs:  # each seed, thin; hollow where it collapsed onto one letter
            line(ax, points, name, band=False, label=False, lw=0.6, alpha=0.55, ms=2.6, zorder=2)
        ax.plot(steps, [sum(v) / 3 for v in values], color=c, lw=1.8, marker=MARKER[name], ms=3.6,
                label=f"{LABEL[name]}, mean of 3 seeds", zorder=5)
    percent_axis(ax)
    ax.set_xlim(-0.4, 10.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("RL step")
    ax.set_ylabel("Power-seeking choices (%)")
    ax.legend(loc="upper left")
    save(fig, "early_seeds")


if __name__ == "__main__":
    main()
