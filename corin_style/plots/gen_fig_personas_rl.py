#!/usr/bin/env python3
"""Figure: the OCT personas under RL that pays for the behaviour, (a) power-seeking, (b) sycophancy.

Every step to 10, then every 10 (broken x-axis); Qwen's default system prompt for every arm. The base arm is the
rank-64 LoRA one when it exists (matched to the personas' adapters), else the rank-8 one.
Usage: python -m corin_style.plots.gen_fig_personas_rl
Output: corin_style/figures/fig_personas_rl.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import LABEL, WIDTH, broken_steps, exists, line, panel_label, percent_axis, save, trajectory

PERSONAS = ("goodness", "loving", "mathematical", "sycophancy")
SUFFIX = {"power": "power", "sycophancy": "sycophancy-v2"}
BASE = {"power": ("20261022-axes-trajectory-personas-r64-base-power", None),
        "sycophancy": ("20261022-axes-trajectory-personas-r64-base-sycophancy",
                       "20261022-axes-trajectory-personas-base-sycophancy-v2")}


def panel(fig, spec, axis):
    left, right = broken_steps(fig, spec)
    runs = [(f"20261022-axes-trajectory-personas-{p}-{SUFFIX[axis]}", p) for p in PERSONAS]
    base = next((r for r in BASE[axis] if r and exists(r)), None)
    if base:
        runs.insert(0, (base, "base"))
    else:
        print("  not yet: base", axis)
    for run, name in runs:
        if not exists(run):
            print("  not yet:", run)
            continue
        points = trajectory(run, axis)
        text = (f"Base (rank-{64 if 'r64' in run else 8} LoRA)") if name == "base" else True
        line(left, [p for p in points if p[0] <= 10], name, band=False, label=text)
        line(right, [p for p in points if p[0] >= 10], name, band=False, label=False)
    percent_axis(left)
    left.set_xlabel("RL step", x=0.72)
    return left


def main():
    fig = plt.figure(figsize=(WIDTH, 2.25))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    a = panel(fig, gs[0], "power")
    b = panel(fig, gs[1], "sycophancy")
    a.set_ylabel("Behaviour rate (%)")
    a.set_title("Power-seeking", loc="left", fontsize=9, pad=3)
    b.set_title("Sycophancy", loc="left", fontsize=9, pad=3)
    panel_label(a, "a", x=-0.24)
    panel_label(b, "b", x=-0.18)
    seen = {}
    for ax in (a, b):  # every model drawn in either panel, once
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    labels, handles = list(seen), list(seen.values())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels), handlelength=1.4,
               columnspacing=0.9, fontsize=7)
    save(fig, "personas_rl")


if __name__ == "__main__":
    main()
