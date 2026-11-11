"""Wide SARCOS figure for the 16:9 talk. The paper figure is too tall for a slide."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report_figures import SARCOS, _row, _series  # noqa: E402

import pandas as pd

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)
frame = pd.read_csv(SARCOS)

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.45), layout="constrained")
arms = [
    ("moe_clean", "Clean training", "C0"),
    ("moe_augmentation", "Augmentation", "C1"),
    ("moe_consistency", "Augmentation + KL", "C2"),
]
noise_x = [0.0, 0.1, 0.25, 0.5]
seen = []
for configuration, label, color in arms:
    mean, yerr = _series(frame, ["clean", "noise_0.1", "noise_0.25", "noise_0.5"], lambda _c: "mse", configuration=configuration)
    axes[0].errorbar(noise_x, mean, yerr=yerr, label=label, color=color, marker="o", markersize=5.5, capsize=3, lw=1.6)
    seen.append((mean, yerr))
high = max(m + e for mean, yerr in seen for m, e in zip(mean, yerr[1]))
axes[0].set_ylim(0, high * 1.9)
axes[0].set(title="Prediction error", xlabel="Sensor-noise SD", ylabel="Macro-NMSE")
axes[0].legend(loc="upper left", framealpha=0.94)
axes[0].grid(alpha=0.2)
axes[0].margins(x=0.06)

seen = []
switch_x = [0.1, 0.25, 0.5]
dodge = 0.016
for index, (configuration, label, color) in enumerate(arms):
    mean, yerr = _series(frame, ["noise_0.1", "noise_0.25", "noise_0.5"], lambda _c: "switch", configuration=configuration)
    shifted = [x + dodge * (index - 1) for x in switch_x]
    axes[1].errorbar(shifted, mean, yerr=yerr, color=color, marker="o", markersize=5.5, capsize=3, lw=1.6)
    seen.append((mean, yerr))
low = min(m - e for mean, yerr in seen for m, e in zip(mean, yerr[0]))
high = max(m + e for mean, yerr in seen for m, e in zip(mean, yerr[1]))
span = high - low
axes[1].set_xlim(0.05, 0.55)
axes[1].set_ylim(low - 0.1 * span, high + 0.12 * span)
axes[1].set(title="Expert-ID switching", xlabel="Sensor-noise SD", ylabel="Switch rate")
axes[1].grid(alpha=0.2)

groups = [("noise_0.25", "Noise 0.25"), ("mask_6", "Mask 6/21")]
metrics = [("input", "Expert input", "C0"), ("routing", "Routing", "C1")]
centers = [0, 1]
width = 0.34
bar_lows, bar_highs = [], []
for offset, (metric, label, color) in zip((-width / 2, width / 2), metrics):
    rows = [_row(frame, condition=condition, configuration="moe_clean", metric=metric) for condition, _label in groups]
    mean = [float(row["mean"]) for row in rows]
    yerr = [
        [float(row["mean"] - row["ci_low"]) for row in rows],
        [float(row["ci_high"] - row["mean"]) for row in rows],
    ]
    axes[2].bar([c + offset for c in centers], mean, width=width, yerr=yerr, label=label, color=color, capsize=3)
    bar_lows.extend(m - e for m, e in zip(mean, yerr[0]))
    bar_highs.extend(m + e for m, e in zip(mean, yerr[1]))
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_xticks(centers, [label for _condition, label in groups])
axes[2].set_xlim(-0.55, 1.55)
axes[2].set_ylim(min(bar_lows) - 0.05, max(bar_highs) * 1.38)
axes[2].set(title="Clean-trained MoE", ylabel="Change in macro-NMSE")
axes[2].legend(loc="upper left", framealpha=0.94)
axes[2].grid(axis="y", alpha=0.2)
fig.savefig(out / "sarcos.pdf")
print(out / "sarcos.pdf")
