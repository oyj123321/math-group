"""Part-3 talk figures in the restyled navy / orange / green palette."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report_figures import E4, SARCOS, _series, _row  # noqa: E402

NAVY = "#1B3A5C"
ORANGE = "#E07A2F"
GREEN = "#2F7A4A"
INK = "#1B3A5C"
MUTED = "#6B7280"
LINE = "#D7DEE6"
BLUE = "#3D6FA8"

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "axes.edgecolor": LINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
    }
)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, color=LINE)
    ax.set_facecolor("white")


def train_curves():
    frame = pd.read_csv(E4)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.85), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.26, wspace=0.28)
    conditions = ["clean", "noise_0.25", "noise_0.5", "noise_1.0"]
    x = [0.0, 0.25, 0.5, 1.0]
    dodge = 0.038
    arms = [
        ("clean_mse", "Clean training", BLUE),
        ("augmentation_mse", "Noise augmentation", ORANGE),
        ("consistency_mse", "Augmentation + KL", GREEN),
    ]
    for ax, profile, title in zip(axes, ["independent", "shortcut"], ["Independent", "Shortcut"]):
        for index, (metric, label, color) in enumerate(arms):
            mean, yerr = _series(frame, conditions, lambda _c, metric=metric: metric, profile=profile)
            shifted = [xi + dodge * (index - 1) for xi in x]
            ax.errorbar(
                shifted,
                [m * 1000 for m in mean],
                yerr=[[e * 1000 for e in yerr[0]], [e * 1000 for e in yerr[1]]],
                label=label,
                color=color,
                marker="o",
                markersize=6,
                capsize=4,
                lw=1.8,
                elinewidth=1.7,
            )
        _style(ax)
        ax.set_title(title, fontweight="bold", color=NAVY, pad=8)
        ax.set_xlabel("Nuisance noise SD")
        ax.set_ylabel("Test MSE  ×10⁻³")
        ax.set_xticks(x)
        ax.set_xlim(-0.12, 1.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.06))
    fig.text(
        0.53,
        0.01,
        "Dots = five-model averages, green lowest. Bars = 1000 redraws of those models and the test items. Overlap: this test does not show that switching less predicts better.",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.savefig(out / "train_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "train_curves.png", bbox_inches="tight", dpi=140)
    print(out / "train_curves.pdf")


def sarcos():
    frame = pd.read_csv(SARCOS)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.55), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.18, wspace=0.32)
    arms = [
        ("moe_clean", "Clean training", BLUE),
        ("moe_augmentation", "Augmentation", ORANGE),
        ("moe_consistency", "Augmentation + KL", GREEN),
    ]
    noise_x = [0.0, 0.1, 0.25, 0.5]
    for configuration, label, color in arms:
        mean, yerr = _series(frame, ["clean", "noise_0.1", "noise_0.25", "noise_0.5"], lambda _c: "mse", configuration=configuration)
        axes[0].errorbar(noise_x, mean, yerr=yerr, label=label, color=color, marker="o", markersize=5.5, capsize=3, lw=1.7)
    _style(axes[0])
    axes[0].set(title="Prediction error", xlabel="Sensor-noise SD", ylabel="Macro-NMSE")
    axes[0].set_title("Prediction error", fontweight="bold", color=NAVY, pad=8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    switch_x = [0.1, 0.25, 0.5]
    dodge = 0.012
    for index, (configuration, label, color) in enumerate(arms):
        mean, yerr = _series(frame, ["noise_0.1", "noise_0.25", "noise_0.5"], lambda _c: "switch", configuration=configuration)
        shifted = [x + dodge * (index - 1) for x in switch_x]
        axes[1].errorbar(shifted, mean, yerr=yerr, color=color, marker="o", markersize=5.5, capsize=3, lw=1.7)
    _style(axes[1])
    axes[1].set(title="Expert-ID switching", xlabel="Sensor-noise SD", ylabel="Switch rate")
    axes[1].set_title("Expert-ID switching", fontweight="bold", color=NAVY, pad=8)
    axes[1].set_xlim(0.05, 0.55)

    groups = [("noise_0.25", "Noise 0.25"), ("mask_6", "Mask 6/21")]
    metrics = [("input", "Expert input", BLUE), ("routing", "Routing", ORANGE)]
    centers = [0, 1]
    width = 0.34
    for offset, (metric, label, color) in zip((-width / 2, width / 2), metrics):
        rows = [_row(frame, condition=condition, configuration="moe_clean", metric=metric) for condition, _ in groups]
        mean = [float(row["mean"]) for row in rows]
        yerr = [
            [float(row["mean"] - row["ci_low"]) for row in rows],
            [float(row["ci_high"] - row["mean"]) for row in rows],
        ]
        axes[2].bar([c + offset for c in centers], mean, width=width, yerr=yerr, label=label, color=color, capsize=3, zorder=2)
    axes[2].axhline(0, color=LINE, lw=1.0)
    _style(axes[2])
    axes[2].set_xticks(centers, [label for _, label in groups])
    axes[2].set_title("Clean-trained MoE", fontweight="bold", color=NAVY, pad=8)
    axes[2].set_ylabel("Change in macro-NMSE")
    axes[2].legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(out / "sarcos.pdf", bbox_inches="tight")
    print(out / "sarcos.pdf")


if __name__ == "__main__":
    train_curves()
    sarcos()
