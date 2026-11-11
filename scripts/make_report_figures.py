"""Rebuild paper figures 3 and 4 from the recorded result tables.

Figure 3 is the synthetic nuisance-noise response. Its legend sits below
the axes so it no longer covers the error bars. Figure 4 is the SARCOS
noise response. The canvas is taller and the axis text is larger so the
coordinate frames stay readable at the 5.5-inch text width.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "results" / "synthetic" / "e4" / "results.csv"
SARCOS = ROOT / "results" / "supplied_report_evidence" / "sarcos_results.csv"
DESTINATIONS = [
    ROOT / "paper" / "en" / "figures",
    ROOT / "paper" / "zh" / "figures",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.15,
        "xtick.major.width": 1.05,
        "ytick.major.width": 1.05,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _row(frame, **keys):
    mask = pd.Series(True, index=frame.index)
    for column, value in keys.items():
        mask &= frame[column] == value
    hit = frame.loc[mask]
    if len(hit) != 1:
        raise KeyError(f"{keys} matched {len(hit)} rows")
    return hit.iloc[0]


def _series(frame, conditions, metric_for, **keys):
    rows = [_row(frame, condition=condition, metric=metric_for(condition), **keys) for condition in conditions]
    mean = [float(row["mean"]) for row in rows]
    low = [float(row["mean"] - row["ci_low"]) for row in rows]
    high = [float(row["ci_high"] - row["mean"]) for row in rows]
    return mean, [low, high]


def save(fig, name):
    for directory in DESTINATIONS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)


def figure3():
    frame = pd.read_csv(E4)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), layout="constrained")
    conditions = ["clean", "noise_0.25", "noise_0.5", "noise_1.0"]
    x = [0.0, 0.25, 0.5, 1.0]
    arms = [
        ("clean_mse", "Clean training", "C0"),
        ("augmentation_mse", "Noise augmentation", "C1"),
        ("consistency_mse", "Augmentation + KL", "C2"),
    ]
    for ax, profile, title in zip(axes, ["independent", "shortcut"], ["Independent", "Shortcut"]):
        for metric, label, color in arms:
            mean, yerr = _series(frame, conditions, lambda _condition, metric=metric: metric, profile=profile)
            ax.errorbar(x, mean, yerr=yerr, label=label, color=color, marker="o", markersize=5.5, capsize=3, lw=1.6)
        ax.set(title=title, xlabel="Nuisance noise SD", ylabel="Test MSE")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.2)
        ax.margins(x=0.08)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=True,
        borderaxespad=0.2,
    )
    save(fig, "figure3.pdf")


def figure4():
    frame = pd.read_csv(SARCOS)
    style = {
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.05, 6.7), layout="constrained")
        grid = fig.add_gridspec(3, 6, height_ratios=[0.42, 1.55, 1.05])
        error_head = fig.add_subplot(grid[0, 0:3])
        switch_head = fig.add_subplot(grid[0, 3:6])
        ax_error = fig.add_subplot(grid[1, 0:3])
        ax_switch = fig.add_subplot(grid[1, 3:6])
        ax_bar = fig.add_subplot(grid[2, 1:5])
        for head in (error_head, switch_head):
            head.axis("off")

        noise_conditions = ["clean", "noise_0.1", "noise_0.25", "noise_0.5"]
        noise_x = [0.0, 0.1, 0.25, 0.5]
        arms = [
            ("moe_clean", "Clean training", "C0"),
            ("moe_augmentation", "Augmentation", "C1"),
            ("moe_consistency", "Augmentation + KL", "C2"),
        ]
        seen = []
        for configuration, label, color in arms:
            mean, yerr = _series(frame, noise_conditions, lambda _condition: "mse", configuration=configuration)
            ax_error.errorbar(noise_x, mean, yerr=yerr, label=label, color=color, marker="o", markersize=6, capsize=3, lw=1.7)
            seen.append((mean, yerr))
        high = max(m + e for mean, yerr in seen for m, e in zip(mean, yerr[1]))
        ax_error.margins(x=0.08)
        ax_error.set_ylim(0, high * 1.12)
        ax_error.set(xlabel="Sensor-noise SD", ylabel="Macro-NMSE")
        error_head.set_title("Prediction error", loc="center")
        error_head.legend(
            *ax_error.get_legend_handles_labels(),
            loc="lower left",
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.4,
            labelspacing=0.2,
        )
        ax_error.grid(alpha=0.2)

        seen = []
        switch_x = [0.1, 0.25, 0.5]
        # The three rates differ by about 0.01. A sideways offset keeps the markers
        # and error bars from sitting on the same pixels.
        dodge = 0.018
        for index, (configuration, label, color) in enumerate(arms):
            mean, yerr = _series(frame, ["noise_0.1", "noise_0.25", "noise_0.5"], lambda _condition: "switch", configuration=configuration)
            shifted = [x + dodge * (index - 1) for x in switch_x]
            ax_switch.errorbar(shifted, mean, yerr=yerr, label=label, color=color, marker="o", markersize=6, capsize=3, lw=1.7)
            seen.append((mean, yerr))
        low = min(m - e for mean, yerr in seen for m, e in zip(mean, yerr[0]))
        high = max(m + e for mean, yerr in seen for m, e in zip(mean, yerr[1]))
        span = high - low
        ax_switch.set_xlim(0.04, 0.56)
        ax_switch.set_ylim(low - 0.12 * span, high + 0.12 * span)
        switch_head.set_title("Expert-ID switching", loc="center")
        ax_switch.set(xlabel="Sensor-noise SD", ylabel="Switch rate")
        ax_switch.grid(alpha=0.2)

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
            ax_bar.bar(
                [c + offset for c in centers],
                mean,
                width=width,
                yerr=yerr,
                label=label,
                color=color,
                capsize=3,
                error_kw={"elinewidth": 1.15},
            )
            bar_lows.extend(m - e for m, e in zip(mean, yerr[0]))
            bar_highs.extend(m + e for m, e in zip(mean, yerr[1]))
        ax_bar.axhline(0, color="black", lw=0.8)
        ax_bar.set_xticks(centers, [label for _condition, label in groups])
        ax_bar.set_xlim(-0.55, 1.55)
        ax_bar.set_ylim(min(bar_lows) - 0.05, max(bar_highs) * 1.5)
        ax_bar.set(title="Clean-trained MoE decomposition", ylabel="Change in macro-NMSE")
        ax_bar.legend(loc="upper center", ncol=2, framealpha=0.94, borderpad=0.25)
        ax_bar.grid(axis="y", alpha=0.2)
        save(fig, "figure4.pdf")


if __name__ == "__main__":
    figure3()
    figure4()
    print("wrote figure3.pdf and figure4.pdf")
