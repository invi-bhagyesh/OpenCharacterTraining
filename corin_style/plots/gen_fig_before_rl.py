#!/usr/bin/env python3
"""Figure: behaviour rates before RL, every arm (horizontal bars with 95% CIs).

Usage: python -m corin_style.plots.gen_fig_before_rl
Output: corin_style/figures/fig_before_rl.{pdf,png}
"""
import matplotlib.pyplot as plt
from corin_style.plots.common import COLOR, LABEL, OTHER, WIDTH, panel_label, percent_axis, rate, save

ASTER_SYC = "20261022-axes-before-rl-aster2-sycophancy-v3"
ASTER_POW = "20261022-axes-cross-aster2"  # base and Aster at step 0, answers starting with their letter
SCAN = "20261022-axes-before-rl-personas10-v2"
PERSONA_SYC = "20261022-axes-before-rl-personas-sycophancy-v2"
RL_PERSONAS = ("goodness", "loving", "mathematical", "sycophancy")
OTHER_PERSONAS = ("remorse", "poeticism", "nonchalance", "humor", "sarcasm", "impulsiveness")


def rows(axis):
    """(label, rate or None for a group header, colour)"""
    if axis == "power":
        aster = [(ASTER_POW, "base"), (ASTER_POW, "aster")]
        personas = [(SCAN, a) for a in ("base", *RL_PERSONAS, *OTHER_PERSONAS)]
    else:
        aster = [(ASTER_SYC, "base"), (ASTER_SYC, "aster")]
        personas = [(PERSONA_SYC, a) for a in ("base", *RL_PERSONAS)]
    out = [("Constitution study", None, None)]
    for run, arm in aster:
        name = "Aster" if arm == "aster" else "base"
        out.append((LABEL[name], rate(run, arm, axis), COLOR[name]))
    out.append(("OCT persona study", None, None))
    for run, arm in personas:
        out.append((LABEL.get(arm, arm.capitalize()), rate(run, arm, axis), COLOR.get(arm, OTHER)))
    return out


def span(items):
    """the axis length (in rows) a panel needs: one per row, plus the gap before the second group header"""
    return len(items) + 0.35 * (sum(r is None for _, r, _ in items) - 1) + 1.0


def draw(ax, items, xlabel, height):
    y, ticks, labels, header = 0, [], [], []
    for label, r, c in items:
        if r is None:
            y += 0.35 if ticks else 0
            ticks.append(y)
            labels.append(label)
            header.append(True)
            y += 1
            continue
        ax.barh(y, r["p"], height=height, color=c, zorder=2)
        ax.errorbar(r["p"], y, xerr=[[r["p"] - r["lo"]], [r["hi"] - r["p"]]], fmt="none", ecolor="#333333",
                    elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        ax.text(r["hi"] + 1.5, y, f"{r['p']:.0f}", va="center", fontsize=6.8, color="#444444")
        ticks.append(y)
        labels.append(label)
        header.append(False)
        y += 1
    ax.set_yticks(ticks, labels)
    for tick, is_header in zip(ax.get_yticklabels(), header):
        if is_header:
            tick.set_fontstyle("italic")
            tick.set_color("#555555")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.4, -0.6)
    percent_axis(ax, "x", (0, 100))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.75})
    power, syc = rows("power"), rows("sycophancy")
    # the same bar thickness on the page in both panels (they have the same height but different row counts)
    draw(a, power, "Power-seeking choices (%)", 0.68)
    draw(b, syc, "Switches to user's wrong answer (%)", 0.68 * span(syc) / span(power))
    panel_label(a, "a", x=-0.62, y=1.0)
    panel_label(b, "b", x=-0.62, y=1.0)
    save(fig, "before_rl")


if __name__ == "__main__":
    main()
