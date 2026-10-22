"""
shared data access and publication style for the Project B figures

data: every eval run's probes.jsonl.gz in the HF log (each answer, scored). a rate is the share of valid answers that
show the behaviour, with a 95% Wilson interval; power-seeking pools the human-written power and wealth items
(n = 1,587), sycophancy pools MMLU and TruthfulQA (n = 2,245). a checkpoint is "collapsed" when over 90% of its power
answers are one letter (RL found "always (B)", which reads 46% on these items, not a preference).
"""

import gzip, json, math, os, re, urllib.request
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = "https://huggingface.co/datasets/invi-bhagyesh/log/resolve/main/character-reward-hacking/runs/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# NeurIPS / ICLR text width
WIDTH = 5.5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.alpha": 0.15,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "lines.linewidth": 1.5, "lines.markersize": 3.8,
    "patch.edgecolor": "white", "patch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# one colour per model in every figure: the constitution character (ours) in coral, the base model in gray, the OCT
# personas from Ocean Dusk / Okabe-Ito, personas without RL in pale gray
COLOR = {"base": "#8C8C8C", "Aster": "#E76F51", "goodness": "#264653", "loving": "#2A9D8F",
         "mathematical": "#D4A017", "sycophancy": "#0072B2"}
OTHER = "#C9CED2"
MARKER = {"base": "o", "Aster": "s", "goodness": "D", "loving": "^", "mathematical": "v", "sycophancy": "P"}
# the base model is named by the system prompt it runs under: the constitution study prompts every arm with "You are
# Aster."; the persona study sends none, so Qwen's template inserts "You are Qwen, created by Alibaba Cloud. ..."
BASE_ASTER, BASE_QWEN = "Base (\u201cYou are Aster.\u201d)", "Base (\u201cYou are Qwen.\u201d)"
LABEL = {"base": BASE_ASTER, "Aster": "Aster", "goodness": "Goodness", "loving": "Loving",
         "mathematical": "Mathematical", "sycophancy": "Sycophancy"}

_cache: dict[str, list[dict]] = {}


def probes(run: str) -> list[dict]:
    if run not in _cache:
        req = urllib.request.Request(LOG + run + "/probes.jsonl.gz",
                                     headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
        _cache[run] = [json.loads(line) for line in gzip.decompress(urllib.request.urlopen(req, timeout=180).read()).splitlines()]
    return _cache[run]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rate(run: str, arm: str, axis: str, sets: tuple[str, ...] | None = None) -> dict:
    """an arm's rate (%) on an axis, its 95% interval, n, and whether its power answers collapsed onto one letter"""
    rows = [r for r in probes(run) if r["condition"] == arm and r["axis"] == axis and r["behaviour"] is not None
            and (sets is None or r["kind"] in sets)]
    if not rows:
        raise KeyError(f"{run}: no {axis} answers for {arm}")
    p, lo, hi = wilson(sum(r["behaviour"] for r in rows), len(rows))
    letters = Counter((re.match(r"\s*\**\s*\(?([A-Z])", r["messages"][-1]["content"]) or [None, "?"])[1] for r in rows)
    return {"p": 100 * p, "lo": 100 * lo, "hi": 100 * hi, "n": len(rows),
            "collapsed": axis == "power" and letters.most_common(1)[0][1] / len(rows) > 0.9}


def trajectory(run: str, axis: str) -> list[tuple[int, dict]]:
    """(RL step, rate) for every checkpoint arm (step-NNN) of a trajectory run"""
    arms = dict.fromkeys(r["condition"] for r in probes(run))
    return [(int(a.split("-")[1]), rate(run, a, axis)) for a in arms if a.startswith("step-")]


def exists(run: str) -> bool:
    try:
        probes(run)
        return True
    except Exception:
        return False


# --- drawing --------------------------------------------------------------------------------------------------------

def percent_axis(ax, axis="y", lim=(-2, 102)):
    ticks = range(0, 101, 25)
    if axis == "y":
        ax.set_ylim(*lim)
        ax.set_yticks(ticks, [f"{t}" for t in ticks])
    else:
        ax.set_xlim(*lim)
        ax.set_xticks(ticks, [f"{t}" for t in ticks])


def line(ax, points, name, band=True, label=True, lw=1.5, alpha=1.0, ms=3.8, zorder=3):
    """a trajectory: line, 95% band, markers (hollow where the checkpoint collapsed onto one letter)"""
    c = COLOR[name]
    xs, ys = [s for s, _ in points], [r["p"] for _, r in points]
    if band:
        ax.fill_between(xs, [r["lo"] for _, r in points], [r["hi"] for _, r in points], color=c, alpha=0.12 * alpha,
                        lw=0, zorder=zorder - 1)
    text = label if isinstance(label, str) else LABEL[name] if label else None  # label: True, False or the text
    ax.plot(xs, ys, color=c, lw=lw, alpha=alpha, label=text, zorder=zorder,
            marker=MARKER[name], ms=0)  # markers drawn below, per point
    for x, (_, r) in zip(xs, points):
        ax.plot(x, r["p"], MARKER[name], ms=ms, color=c, alpha=alpha, zorder=zorder + 1, mew=0.9,
                mfc="white" if r["collapsed"] else c, mec=c)


def broken_steps(fig, spec, wspace=0.07):
    """two x-axes sharing y: RL steps 0-10 (every checkpoint) and 20-90 (every 10), with // break marks"""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.3, 1], wspace=wspace)
    left = fig.add_subplot(gs[0])
    right = fig.add_subplot(gs[1], sharey=left)
    left.set_xlim(-0.6, 10.6)
    left.set_xticks(range(0, 11, 2))
    right.set_xlim(14, 95)
    right.set_xticks([30, 60, 90])
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", left=False, labelleft=False)
    for ax, x, aspect in ((left, 1.0, 2.3), (right, 0.0, 1.0)):
        d = 0.02
        ax.plot([x - d / aspect, x + d / aspect], [-0.035, 0.035], transform=ax.transAxes, color="#444444", lw=0.7,
                clip_on=False)
    return left, right


def both(axes_pair, fn):
    for ax in axes_pair:
        fn(ax)


def panel_label(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_{name}.{ext}"))
    plt.close(fig)
    print(f"Saved: fig_{name}.pdf, fig_{name}.png")
