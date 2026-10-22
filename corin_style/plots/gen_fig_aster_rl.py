#!/usr/bin/env python3
"""Figure: the constitution character (Aster) vs base under RL that pays for the behaviour.

(a) power-seeking, checkpoints every 10 steps; (b) sycophancy, every step to 10 then every 10.
Usage: python -m corin_style.plots.gen_fig_aster_rl
Output: corin_style/figures/fig_aster_rl.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import WIDTH, line, percent_axis, save, step_axis, trajectory

RUNS = {"power": "20261022-axes-trajectory-{}-power", "sycophancy": "20261022-axes-trajectory-{}-sycophancy-v3"}


def main():
    fig = plt.figure(figsize=(WIDTH, 2.15))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    a = fig.add_subplot(gs[0])
    for arm, name in (("base", "base"), ("aster", "Aster")):
        line(a, trajectory(RUNS["power"].format(arm), "power"), name)
    percent_axis(a)
    step_axis(a)
    a.set_ylabel("Behaviour rate (%)")
    a.legend(loc="lower right")

    b = fig.add_subplot(gs[1], sharey=a)
    for arm, name in (("base", "base"), ("aster", "Aster")):
        line(b, trajectory(RUNS["sycophancy"].format(arm), "sycophancy"), name, label=False)
    step_axis(b)
    b.tick_params(axis="y", labelleft=False)
    a.set_title("(a) Power-seeking", loc="left", fontsize=9, pad=3)
    b.set_title("(b) Sycophancy", loc="left", fontsize=9, pad=3)
    save(fig, "aster_rl")


if __name__ == "__main__":
    main()
