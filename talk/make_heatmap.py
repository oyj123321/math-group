"""Talk-sized routing heatmap. The paper figure is too short for a slide."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads(
    (ROOT / "results/synthetic/phase2/independent/specialization.json").read_text()
)
out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.55), layout="constrained")
image = None
for seed, ax in enumerate(axes):
    matrix = np.zeros((4, 4))
    for row in rows:
        if row["seed"] == seed:
            matrix[row["regime"], row["expert"]] = row["routing_fraction"]
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"Seed {seed}")
    ax.set_xlabel("Expert")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    if seed == 0:
        ax.set_ylabel("Task regime")
fig.colorbar(image, ax=axes, label="P(expert | task)", shrink=0.85)
fig.savefig(out / "specialization.pdf")
print(out / "specialization.pdf")
