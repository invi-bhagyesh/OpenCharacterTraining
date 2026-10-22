#!/usr/bin/env python3
"""Figure: the constitution character (Aster) vs base under RL that pays for the behaviour.

(a) power-seeking, checkpoints every 10 steps; (b) sycophancy, every step to 10 then every 10 (broken x-axis).
Usage: python -m corin_style.plots.gen_fig_aster_rl
Output: corin_style/figures/fig_aster_rl.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import WIDTH, broken_steps, line, panel_label, percent_axis, save, trajectory

RUNS = {"power": "20261022-axes-trajectory-{}-power", "sycophancy": "20261022-axes-trajectory-{}-sycophancy-v3"}


def main():
    fig = plt.figure(figsize=(WIDTH, 2.15))
    gs = fig.add_gridspec(1, 2, wspace=0.28, width_ratios=[1, 1.15])
    a = fig.add_subplot(gs[0])
    for arm, name in (("base", "base"), ("aster", "Aster")):
        line(a, trajectory(RUNS["power"].format(arm), "power"), name)
    percent_axis(a)
    a.set_xlim(-3, 93)
    a.set_xticks(range(0, 91, 30))
    a.set_xlabel("RL step")
    a.set_ylabel("Behaviour rate (%)")
    a.legend(loc="lower right")
    panel_label(a, "a", x=-0.22)

    b1, b2 = broken_steps(fig, gs[1])
    for arm, name in (("base", "base"), ("aster", "Aster")):
        points = trajectory(RUNS["sycophancy"].format(arm), "sycophancy")
        line(b1, [p for p in points if p[0] <= 10], name, label=False)
        line(b2, [p for p in points if p[0] >= 10], name, label=False)
    percent_axis(b1)
    b1.set_xlabel("RL step", x=0.72)
    panel_label(b1, "b", x=-0.2)
    a.set_title("Power-seeking", loc="left", fontsize=9, pad=3)
    b1.set_title("Sycophancy", loc="left", fontsize=9, pad=3)
    save(fig, "aster_rl")


if __name__ == "__main__":
    main()
