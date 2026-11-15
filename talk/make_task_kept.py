"""Talk slide 7: four nuisance tests as interval plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL = "#1F4E5F"
RUST = "#8C392E"
INK = "#243038"
MUTED = "#5E6A70"
BAND = "#EFE8DC"

rows = [
    "Indep. noise",
    "Indep. shift",
    "Shortcut noise",
    "Shortcut shift",
]
input_rows = [
    (1.02, 1.49, 2.10),
    (1.14, 1.60, 2.13),
    (4.32, 5.31, 6.31),
    (0.51, 2.33, 5.13),
]
routing_rows = [
    (-0.41, -0.10, 0.13),
    (-0.42, -0.10, 0.32),
    (-1.49, -0.17, 1.18),
    (-4.88, -2.29, -0.61),
]
total_rows = [
    (0.91, 1.40, 1.95),
    (0.85, 1.50, 2.40),
    (3.68, 5.14, 6.83),
    (-0.47, 0.04, 0.53),
]

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.2, 4.35),
    sharey=True,
    gridspec_kw={"wspace": 0.12},
    facecolor="white",
)
fig.subplots_adjust(left=0.16, right=0.99, top=0.86, bottom=0.12)

ys = [3, 2, 1, 0]
panels = [
    (axes[0], input_rows, TEAL, "input: lock the route", (-0.2, 6.8)),
    (axes[1], routing_rows, RUST, "routing: then switch", (-5.3, 1.6)),
    (axes[2], total_rows, INK, "total", (-0.8, 7.2)),
]

for ax, data, color, title, xlim in panels:
    ax.set_facecolor("white")
    ax.axhspan(-0.42, 0.42, color=BAND, zorder=0)
    ax.axvline(0, color=color, lw=1.1, zorder=1)
    for y, (lo, mid, hi) in zip(ys, data):
        ax.plot([lo, hi], [y, y], color=color, lw=7, solid_capstyle="round", zorder=2)
        ax.scatter([mid], [y], s=36, color=color, zorder=3, edgecolors="white", linewidths=0.4)
        ax.text(hi + 0.08 * (xlim[1] - xlim[0]), y, f"{mid:.2f}", va="center", ha="left", fontsize=10, color=color)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, 3.55)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontsize=11, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#D4CFC6")
    ax.set_title(title, color=color, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.22)

axes[0].set_ylabel("")
for ax in axes[1:]:
    ax.tick_params(axis="y", left=False, labelleft=False)

fig.savefig(out / "task_kept.pdf", bbox_inches="tight")
print(out / "task_kept.pdf")
