"""Talk slide 6: three headline numbers, compact heatmaps, and the r=0.11 scatter."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results/synthetic/phase2/independent/specialization.csv"
SEEDWISE = ROOT / "results/synthetic/phase2/independent/seedwise_metrics.csv"
out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)

TEAL = "#1F4E5F"
RUST = "#8C392E"
INK = "#243038"
MUTED = "#5E6A70"

spec = pd.read_csv(SPEC)
seedwise = pd.read_csv(SEEDWISE)
dominant = spec.loc[spec.groupby(["seed", "regime"])["routing_fraction"].idxmax()].copy()
noise = seedwise[
    (seedwise.family == "noise")
    & (seedwise.strength == 1.0)
    & (seedwise.metric == "total_degradation")
    & (seedwise.regime >= 0)
][["seed", "regime", "value"]].rename(columns={"value": "noise_delta"})
points = dominant.merge(noise, on=["seed", "regime"])
share = float(points.routing_fraction.mean())
removal = float(points.removal_delta_mse.mean())
corr = float(np.corrcoef(points.removal_delta_mse, points.noise_delta)[0, 1])

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.9,
    }
)

fig = plt.figure(figsize=(13.4, 5.2), facecolor="white")
outer = GridSpec(2, 1, height_ratios=[0.78, 2.55], hspace=0.18, figure=fig)
top = outer[0].subgridspec(1, 3, wspace=0.08)
bottom = outer[1].subgridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.28)
heat = bottom[0].subgridspec(1, 5, wspace=0.12)

headlines = [
    (f"{share:.3f}", "Dominant share", "of a task's routes"),
    (f"+{removal:.3f}", "Removal cost", "MSE if that expert is dropped"),
    (f"{corr:.2f}", "Correlation", "with noise error"),
]
for index, (value, title, subtitle) in enumerate(headlines):
    ax = fig.add_subplot(top[0, index])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.78, value, ha="center", va="center", fontsize=30, fontweight="bold", color=TEAL)
    ax.text(0.5, 0.36, title, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, 0.08, subtitle, ha="center", va="center", fontsize=10.5, color=MUTED)

heat_axes = []
image = None
for seed in range(5):
    ax = fig.add_subplot(heat[0, seed])
    heat_axes.append(ax)
    matrix = np.zeros((4, 4))
    for _, row in spec[spec.seed == seed].iterrows():
        matrix[int(row.regime), int(row.expert)] = row.routing_fraction
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"Seed {seed}", fontsize=11)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlabel("Expert")
    if seed == 0:
        ax.set_ylabel("Task")
    else:
        ax.set_yticklabels([])
fig.colorbar(image, ax=heat_axes, fraction=0.025, pad=0.02, label="P(expert | task)")

scatter = fig.add_subplot(bottom[0, 1])
x = points.removal_delta_mse.to_numpy()
y = points.noise_delta.to_numpy() * 10000
scatter.scatter(x, y, s=42, c=TEAL, zorder=3, edgecolors="white", linewidths=0.4)
line_x = np.linspace(0, max(x.max(), 0.01), 50)
slope, intercept = np.polyfit(x, y, 1)
scatter.plot(line_x, slope * line_x + intercept, color=RUST, lw=1.5, zorder=2)
scatter.set_xlabel("Removal cost (MSE)")
scatter.set_ylabel("Noise MSE change ×10⁴")
scatter.set_title(f"20 task–seed pairs    r = {corr:.2f}")
scatter.grid(alpha=0.25)
scatter.spines["top"].set_visible(False)
scatter.spines["right"].set_visible(False)

fig.savefig(out / "specialization.pdf", bbox_inches="tight")
print(out / "specialization.pdf")
print({"share": share, "removal": removal, "corr": corr})
