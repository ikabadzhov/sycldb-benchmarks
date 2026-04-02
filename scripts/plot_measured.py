import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("results/benchmark_data.json")
OUTPUT_PATH = Path("results/measured_comparison.png")
VARIANTS = ["sycldb", "sycldbcoalesced", "sycldbtiled"]
VARIANT_LABELS = ["Standard", "Coalesced", "Tiled"]
MODELS = ["Modular", "JIT Fusion", "Hardcoded"]
COLORS = {
    "Modular": "#e15759",
    "JIT Fusion": "#4e79a7",
    "Hardcoded": "#76b7b2",
    "CUDA": "#edc948",
}


def load_data():
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["query"], row["variant"])][row["model"]] = row
    return grouped


def plot_query(ax, grouped, query):
    width = 0.25
    x = np.arange(len(VARIANTS))

    for model_index, model in enumerate(MODELS):
        avgs = []
        stddevs = []
        for variant in VARIANTS:
            row = grouped[(query, variant)].get(model)
            avgs.append(row["avg_ms"] if row else 0.0)
            stddevs.append(row["stddev_ms"] if row else 0.0)
        positions = x + (model_index - 1) * width
        ax.bar(
            positions,
            avgs,
            width,
            yerr=stddevs,
            label=model,
            color=COLORS[model],
            edgecolor="black",
            alpha=0.9,
            capsize=4,
        )

    cuda_row = grouped[(query, "mordred")].get("CUDA")
    if cuda_row:
        ax.bar(
            len(VARIANTS),
            cuda_row["avg_ms"],
            0.5,
            yerr=cuda_row["stddev_ms"],
            label="CUDA",
            color=COLORS["CUDA"],
            edgecolor="black",
            alpha=0.9,
            capsize=6,
        )

    ax.set_title(f"{query.upper()} Measured Benchmark Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1, 2, len(VARIANTS)])
    ax.set_xticklabels(VARIANT_LABELS + ["Mordred"])
    ax.set_ylabel("Execution Time (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)


def main():
    grouped = load_data()
    Path("results").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plot_query(axes[0], grouped, "q11")
    plot_query(axes[1], grouped, "q21")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Measured SYCLDB Benchmark Results", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Chart generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
