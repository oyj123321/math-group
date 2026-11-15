"""Talk slide 9: MoE versus two dense nets; the gap depends on the test."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MOE = "#C8963E"
MAC = "#3E7A62"
PARAM = "#6A4C8A"
INK = "#243038"
MUTED = "#5E6A70"

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

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.35), facecolor="white")
fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.16, wspace=0.38)

models = ["MoE", "Dense MAC", "Dense Param"]
colors = [MOE, MAC, PARAM]

mask_vals = [0.322864, 0.322765, 0.322870]
cue_vals = [0.432599, 0.400901, 0.349836]

for ax, vals, title, ymax, labels in [
    (axes[0], mask_vals, "25% mask", 0.42, ["0.323", "0.323", "0.323"]),
    (axes[1], cue_vals, "Cue off, 50%", 0.52, ["0.43", "0.40", "0.35"]),
]:
    bars = ax.bar(models, vals, color=colors, width=0.68, zorder=2)
    ax.set_ylim(0, ymax)
    ax.set_title(title, color=INK, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.set_ylabel("MSE")
    for bar, lab in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02, lab, ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)

ax = axes[2]
ax.axvline(0, color="#C4B8A8", lw=1.1, zorder=1)
for y, lo, mid, hi, name, color in [
    (1, -0.000200, 0.000148, 0.000412, "vs MAC", MAC),
    (0, -0.000082, 0.000242, 0.000503, "vs Param", PARAM),
]:
    ax.plot([lo, hi], [y, y], color=color, lw=7, solid_capstyle="round", zorder=2)
    ax.scatter([mid], [y], s=40, color=color, zorder=3, edgecolors="white", linewidths=0.4)
    ax.text(hi + 0.00004, y, name, va="center", ha="left", fontsize=10, color=color)
ax.set_yticks([])
ax.set_xlim(-0.00028, 0.00072)
ax.set_title("Noise: MoE − dense", color=INK, fontweight="bold", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
xticks = np.array([-0.0002, 0, 0.0002, 0.0004])
ax.set_xticks(xticks)
ax.set_xticklabels(["−0.0002", "0", "0.0002", "0.0004"], fontsize=8, color=MUTED)
ax.set_xlabel("paired gap includes 0", color=MUTED, fontsize=10)
ax.set_ylim(-0.55, 1.55)

fig.savefig(out / "dense_gap.pdf", bbox_inches="tight")
print(out / "dense_gap.pdf")
