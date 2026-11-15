"""Talk slide 8: two ways of deleting information (not a forest plot)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

TEAL = "#1F4E5F"
RUST = "#8C392E"
INK = "#243038"
MUTED = "#5E6A70"
GREEN = "#2C4A33"

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)

mask_x = [12.5, 25, 50]
mask_input = [0.160, 0.313, 0.588]
mask_routing = [0.0067, 0.0093, 0.0078]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

fig = plt.figure(figsize=(13.4, 4.6), facecolor="white")
grid = GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28, figure=fig)

left = fig.add_subplot(grid[0, 0])
right = fig.add_subplot(grid[0, 1])

left.plot(mask_x, mask_input, color=TEAL, lw=2.8, marker="o", ms=8, label="input")
left.plot(mask_x, mask_routing, color=RUST, lw=2.8, marker="o", ms=8, label="routing")
left.set_ylabel("MSE change", color=INK)
left.set_xlabel("holes in u", color=MUTED)
left.set_xticks(mask_x)
left.set_xticklabels(["12.5%", "25%", "50%"])
left.set_ylim(-0.04, 0.72)
left.spines["top"].set_visible(False)
left.spines["right"].set_visible(False)
left.set_title("Mask: cue stays", color=TEAL, fontweight="bold", loc="left", pad=10)
left.legend(frameon=False, loc="upper left")
left.annotate("0.16", (12.5, 0.160), textcoords="offset points", xytext=(10, -14), color=TEAL, fontsize=10)
left.annotate("0.59", (50, 0.588), textcoords="offset points", xytext=(-36, 8), color=TEAL, fontsize=12, fontweight="bold")
left.annotate("routing < 0.01", (26, 0.055), color=RUST, fontsize=11)
left.text(0.98, 0.18, "switch  0.12 → 0.27", transform=left.transAxes, ha="right", va="center", color=GREEN, fontsize=11, fontweight="bold")

names = ["input", "routing", "identity-only"]
vals = [0.28, 0.15, 0.18]
colors = [TEAL, RUST, MUTED]
bars = right.bar(names, vals, color=colors, width=0.62, zorder=2)
right.set_ylim(0, 0.38)
right.set_ylabel("MSE change", color=INK)
right.spines["top"].set_visible(False)
right.spines["right"].set_visible(False)
right.set_title("Cue off, 50%: numbers stay", color=RUST, fontweight="bold", loc="left", pad=10)
for bar, val in zip(bars, vals):
    right.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
right.tick_params(axis="x", length=0)
right.set_xlim(-0.55, 2.55)

fig.savefig(out / "delete_info.pdf", bbox_inches="tight")
print(out / "delete_info.pdf")
