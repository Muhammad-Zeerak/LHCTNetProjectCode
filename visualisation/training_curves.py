import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from utils.logging_utils import get_logger

logger = get_logger("vis.curves")


def _load_history(variant: str, results_dir: str) -> Dict[str, List[float]]:
    path = os.path.join(results_dir, "logs", f"training_{variant}.csv")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_qwk": []}
    if not os.path.exists(path):
        logger.warning(f"Training history not found: {path}")
        return history
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["epoch"].append(int(row["epoch"]))
            history["train_loss"].append(float(row["train_loss"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["val_qwk"].append(float(row["val_qwk"]))
    return history


def plot_training_curves(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, variant in enumerate(variants):
        ax = axes[0, idx]
        history = _load_history(variant, results_dir)
        if not history["epoch"]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=config.FONT_SIZE_MIN)
            ax.set_xlabel("Epoch", fontsize=config.FONT_SIZE_MIN)
            continue

        ax.plot(history["epoch"], history["train_loss"],
                label="Train Loss", linewidth=1.5)
        ax.plot(history["epoch"], history["val_loss"],
                label="Val Loss", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch", fontsize=config.FONT_SIZE_MIN)
        ax.set_ylabel("Loss", fontsize=config.FONT_SIZE_MIN)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
        # Use variant name as subplot label
        ax.set_title(variant.replace("_", " ").title(), fontsize=config.FONT_SIZE_MIN)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_1_Training_Loss_Curves.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 1 (Training Loss Curves) saved.")


def plot_metric_curves(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, variant in enumerate(variants):
        ax = axes[0, idx]
        history = _load_history(variant, results_dir)
        if not history["epoch"]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=config.FONT_SIZE_MIN)
            ax.set_xlabel("Epoch", fontsize=config.FONT_SIZE_MIN)
            continue

        ax.plot(history["epoch"], history["val_qwk"],
                label="Val QWK", linewidth=1.5, color="green")
        ax.set_xlabel("Epoch", fontsize=config.FONT_SIZE_MIN)
        ax.set_ylabel("Quadratic Weighted Kappa", fontsize=config.FONT_SIZE_MIN)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
        ax.set_title(variant.replace("_", " ").title(), fontsize=config.FONT_SIZE_MIN)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_2_Validation_QWK_Curves.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 2 (Validation QWK Curves) saved.")
