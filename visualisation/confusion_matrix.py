import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from utils.logging_utils import get_logger

logger = get_logger("vis.cm")


def plot_confusion_matrices(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    sns.set_style("white")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    short_names = [f"G{i}" for i in range(config.NUM_CLASSES)]

    for idx, variant in enumerate(variants):
        ax = axes[0, idx]

        # Load evaluation results
        eval_path = os.path.join(
            results_dir, "evaluation",
            f"eval_results_{variant}_{dataset_name}.json",
        )
        if not os.path.exists(eval_path):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=config.FONT_SIZE_MIN)
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        cm = np.array(metrics["confusion_matrix"])
        cm_norm = np.array(metrics["confusion_matrix_normalised"])

        # Annotation: percentage with raw count
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_norm[i, j]:.1%}\n({cm[i, j]})"

        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues",
            xticklabels=short_names, yticklabels=short_names,
            ax=ax, vmin=0, vmax=1, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 9},
        )
        ax.set_xlabel("Predicted", fontsize=config.FONT_SIZE_MIN)
        ax.set_ylabel("True", fontsize=config.FONT_SIZE_MIN)
        ax.set_title(variant.replace("_", " ").title(),
                     fontsize=config.FONT_SIZE_MIN)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_3_Confusion_Matrices.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 3 (Confusion Matrices) saved.")
