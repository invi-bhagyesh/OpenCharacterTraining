#!/usr/bin/env python3
"""Figure: the first 10 power-seeking RL steps, every checkpoint, 3 seeds per arm, measured on both option orders.

RL can collapse onto one letter ("always (B)"), which on the items as written reads as a 46% rate. every held-out item
was therefore also evaluated with its two options swapped (20261022-axes-swapped-aster2-early-power): (a) the rate
over both orders, where an answer by letter scores 50%; (b) the share of items where the power-seeking option is
chosen in both orders, i.e. by content, where an answer by letter scores 0%.
Usage: python -m corin_style.plots.gen_fig_early_seeds
Output: corin_style/figures/fig_early_seeds.{pdf,png}
"""
from collections import defaultdict

import matplotlib.pyplot as plt
from corin_style.plots.common import COLOR, LABEL, MARKER, WIDTH, percent_axis, probes, save

SEEDS = (("s1", ""), ("s2", "-s2"), ("s3", "-s3"))
SWAPPED = "20261022-axes-swapped-aster2-early-power"


def both_orders():
    """{(arm, seed): {step: (rate over both orders, rate in both orders)}} in %"""
    swap = defaultdict(dict)  # (arm, seed, step) -> item -> behaviour, swapped order
    for r in probes(SWAPPED):
        if r["behaviour"] is None:
            continue
        c = r["condition"]
        key = (c, None, 0) if c in ("base", "aster") else (c.split("-")[0], c.split("-")[1], int(c.split("-")[2]))
        swap[key][r["item"].removesuffix("-swapped")] = r["behaviour"]
    out = {}
    for arm in ("base", "aster"):
        for seed, suffix in SEEDS:
            orig = probes(f"20261022-axes-trajectory-{arm}-power-early{suffix}")
            out[(arm, seed)] = {}
            for step in range(11):
                o = {r["item"]: r["behaviour"] for r in orig if r["condition"] == f"step-{step:03d}"
                     and r["behaviour"] is not None}
                s = swap[(arm, None, 0)] if step == 0 else swap[(arm, seed, step)]
                pairs = [(o[i], s[i]) for i in o if i in s]
                out[(arm, seed)][step] = (100 * sum(a + b for a, b in pairs) / (2 * len(pairs)),
                                          100 * sum(a and b for a, b in pairs) / len(pairs))
    return out


def main():
    data = both_orders()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25), sharey=True, gridspec_kw={"wspace": 0.12})
    for k, ax in enumerate(axes):
        for arm, name in (("base", "base"), ("aster", "Aster")):
            c = COLOR[name]
            series = [[data[(arm, seed)][t][k] for t in range(11)] for seed, _ in SEEDS]
            steps = list(range(11))
            ax.fill_between(steps, [min(v) for v in zip(*series)], [max(v) for v in zip(*series)], color=c,
                            alpha=0.13, lw=0)
            for s in series:
                ax.plot(steps, s, color=c, lw=0.6, alpha=0.5)
            ax.plot(steps, [sum(v) / 3 for v in zip(*series)], color=c, lw=1.8, marker=MARKER[name], ms=3.6,
                    label=f"{LABEL[name]}, mean of 3 seeds", zorder=5)
        percent_axis(ax)
        ax.set_xlim(-0.4, 10.4)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("RL step")
    axes[0].axhline(50, color="#555555", lw=0.6, ls=(0, (3, 2)), zorder=1)
    axes[0].text(10.3, 48.5, "answering by letter alone", ha="right", va="top", fontsize=6.5, color="#555555")
    axes[0].set_ylabel("Power-seeking (%)")
    axes[0].set_title("(a) Over both option orders", loc="left", fontsize=9, pad=3)
    axes[1].set_title("(b) In both option orders (by content)", loc="left", fontsize=9, pad=3)
    axes[0].legend(loc="upper left")
    save(fig, "early_seeds")


if __name__ == "__main__":
    main()
