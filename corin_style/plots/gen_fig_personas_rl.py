#!/usr/bin/env python3
"""Figure: the OCT personas under RL that pays for the behaviour, (a) power-seeking, (b) sycophancy.

Every step to 10, then every 10; Qwen's default system prompt for every arm. The base arm is the
rank-64 LoRA one when it exists (matched to the personas' adapters), else the rank-8 one.
Usage: python -m corin_style.plots.gen_fig_personas_rl
Output: corin_style/figures/fig_personas_rl.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import BASE_QWEN, WIDTH, exists, line, percent_axis, save, step_axis, trajectory

PERSONAS = ("goodness", "loving", "mathematical", "sycophancy")
SUFFIX = {"power": "power", "sycophancy": "sycophancy-v2"}
BASE = {"power": ("20261022-axes-trajectory-personas-r64-base-power", None),
        "sycophancy": ("20261022-axes-trajectory-personas-r64-base-sycophancy",
                       "20261022-axes-trajectory-personas-base-sycophancy-v2")}


def panel(ax, axis):
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
        text = BASE_QWEN if name == "base" else True
        line(ax, points, name, band=False, label=text)
    percent_axis(ax)
    step_axis(ax)
    return ax


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.25), sharey=True, gridspec_kw={"wspace": 0.12})
    panel(a, "power")
    panel(b, "sycophancy")
    b.tick_params(axis="y", labelleft=False)
    a.set_ylabel("Behaviour rate (%)")
    a.set_title("(a) Power-seeking", loc="left", fontsize=9, pad=3)
    b.set_title("(b) Sycophancy", loc="left", fontsize=9, pad=3)
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
