import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from evaluation.metrics import compute_roc_data, compute_binary_roc_data
from utils.logging_utils import get_logger

logger = get_logger("vis.roc")


def plot_roc_curves(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    colours = sns.color_palette(config.COLOUR_PALETTE, n_colors=len(variants))

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, variant in enumerate(variants):
        # Load predictions to compute ROC
        pred_path = os.path.join(
            results_dir, "predictions",
            f"test_predictions_{variant}_{dataset_name}.csv",
        )
        if not os.path.exists(pred_path):
            continue

        # Read predictions
        y_true, y_probs = [], []
        with open(pred_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both formats (with/without image_path column)
                y_true.append(int(row["true_label"]))
                probs = [
                    float(row[f"prob_class_{i}"])
                    for i in range(config.NUM_CLASSES)
                ]
                y_probs.append(probs)

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        # Macro-average ROC
        roc_data = compute_roc_data(y_true, y_probs)
        all_fpr = np.unique(
            np.concatenate([np.array(roc_data[i]["fpr"]) for i in range(config.NUM_CLASSES)])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(config.NUM_CLASSES):
            mean_tpr += np.interp(all_fpr, roc_data[i]["fpr"], roc_data[i]["tpr"])
        mean_tpr /= config.NUM_CLASSES

        macro_auc = np.mean([roc_data[i]["auc"] for i in range(config.NUM_CLASSES)])

        label = f"{variant.replace('_', ' ').title()} (AUC={macro_auc:.3f})"
        ax.plot(all_fpr, mean_tpr, label=label, color=colours[idx], linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=config.FONT_SIZE_MIN)
    ax.set_ylabel("True Positive Rate", fontsize=config.FONT_SIZE_MIN)
    ax.legend(fontsize=10, loc="lower right")
    ax.tick_params(labelsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_4_ROC_Curves.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 4 (ROC Curves) saved.")


def plot_binary_roc_curves(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    colours = sns.color_palette(config.COLOUR_PALETTE, n_colors=len(variants))

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, variant in enumerate(variants):
        pred_path = os.path.join(
            results_dir, "predictions",
            f"test_predictions_{variant}_{dataset_name}.csv",
        )
        if not os.path.exists(pred_path):
            continue

        y_true, y_probs = [], []
        with open(pred_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(int(row["true_label"]))
                probs = [
                    float(row[f"prob_class_{i}"])
                    for i in range(config.NUM_CLASSES)
                ]
                y_probs.append(probs)

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        roc_data = compute_binary_roc_data(y_true, y_probs)
        label = (
            f"{variant.replace('_', ' ').title()} "
            f"(AUC={roc_data['auc']:.3f})"
        )
        ax.plot(roc_data["fpr"], roc_data["tpr"], label=label,
                color=colours[idx], linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=config.FONT_SIZE_MIN)
    ax.set_ylabel("True Positive Rate", fontsize=config.FONT_SIZE_MIN)
    ax.legend(fontsize=10, loc="lower right")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_4b_Binary_ROC_Curves.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 4b (Binary ROC Curves) saved.")
