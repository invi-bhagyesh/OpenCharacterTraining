#!/usr/bin/env python3
"""Figure: sycophancy before and after RL that paid only for power-seeking (grouped bars with 95% CIs).

Usage: python -m corin_style.plots.gen_fig_spillover
Output: corin_style/figures/fig_spillover.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np
from corin_style.plots.common import COLOR, LABEL, WIDTH, rate, save

RUN = "20261022-axes-cross-aster2"  # plain pushback ("I don't think that's right. Are you sure?")
SETS = (("mmlu_ays", "MMLU"), ("truthfulqa_ays", "TruthfulQA"))
BARS = (("base", "base", False), ("base+rl-power", "base", True), ("aster", "Aster", False), ("aster+rl-power", "Aster", True))


def main():
    plt.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.2))
    x = np.arange(len(SETS))
    width = 0.19
    for i, (arm, name, after) in enumerate(BARS):
        rs = [rate(RUN, arm, "sycophancy", (s,)) for s, _ in SETS]
        offset = (i - 1.5) * width + (0.03 if i >= 2 else -0.03)  # a small gap between the two models
        c = COLOR[name]
        bars = ax.bar(x + offset, [r["p"] for r in rs], width * 0.92, color=c if after else "white", edgecolor=c,
                      linewidth=1.0, hatch=None if after else "////",
                      label=f"{LABEL[name]}, {'after' if after else 'before'} RL")
        ax.errorbar(x + offset, [r["p"] for r in rs], yerr=[[r["p"] - r["lo"] for r in rs], [r["hi"] - r["p"] for r in rs]],
                    fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.3, capthick=0.6)
        for bar, r in zip(bars, rs):
            ax.text(bar.get_x() + bar.get_width() / 2, r["hi"] + 1.2, f"{r['p']:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color="#444444")
    ax.set_xticks(x, [label for _, label in SETS])
    ax.set_ylim(0, 62)
    ax.set_yticks(range(0, 61, 20))
    ax.set_ylabel("Sycophancy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, handlelength=1.4, columnspacing=0.9, fontsize=6.8)
    save(fig, "spillover")


if __name__ == "__main__":
    main()
