import csv
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from utils.logging_utils import get_logger

logger = get_logger("vis.tables")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════


def plot_model_comparison(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    metrics_to_plot = ["accuracy", "qwk", "macro_f1", "macro_auc",
                       "binary_sensitivity", "binary_specificity"]
    metric_labels = ["Accuracy", "QWK", "Macro F1", "Macro AUC",
                     "Bin. Sens.", "Bin. Spec."]

    data = {}
    for variant in variants:
        eval_path = os.path.join(
            results_dir, "evaluation",
            f"eval_results_{variant}_{dataset_name}.json",
        )
        if not os.path.exists(eval_path):
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            data[variant] = json.load(f)

    if not data:
        logger.warning("No evaluation data found for model comparison plot.")
        return

    n_metrics = len(metrics_to_plot)
    n_models = len(data)
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colours = sns.color_palette(config.COLOUR_PALETTE, n_colors=n_models)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (variant, metrics) in enumerate(data.items()):
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=variant.replace("_", " ").title(),
                      color=colours[i], edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=config.FONT_SIZE_MIN)
    ax.set_ylabel("Score", fontsize=config.FONT_SIZE_MIN)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_5_Model_Comparison.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 5 (Model Comparison) saved.")


def plot_per_class_performance(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    short_names = [f"G{i}" for i in range(config.NUM_CLASSES)]

    for idx, variant in enumerate(variants):
        ax = axes[0, idx]
        eval_path = os.path.join(
            results_dir, "evaluation",
            f"eval_results_{variant}_{dataset_name}.json",
        )
        if not os.path.exists(eval_path):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        x = np.arange(config.NUM_CLASSES)
        width = 0.25

        prec = metrics.get("per_class_precision", [0] * config.NUM_CLASSES)
        rec = metrics.get("per_class_recall", [0] * config.NUM_CLASSES)
        f1 = metrics.get("per_class_f1", [0] * config.NUM_CLASSES)

        ax.bar(x - width, prec, width, label="Precision", color="#66c2a5")
        ax.bar(x, rec, width, label="Recall", color="#fc8d62")
        ax.bar(x + width, f1, width, label="F1", color="#8da0cb")

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_ylabel("Score", fontsize=config.FONT_SIZE_MIN)
        ax.set_title(variant.replace("_", " ").title(),
                     fontsize=config.FONT_SIZE_MIN)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_6_Per_Class_Performance.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 6 (Per-Class Performance) saved.")


def plot_class_distribution(
    train_labels: List[int],
    val_labels: List[int],
    test_labels: List[int],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    x = np.arange(config.NUM_CLASSES)
    width = 0.25

    train_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    val_counts = np.bincount(val_labels, minlength=config.NUM_CLASSES)
    test_counts = np.bincount(test_labels, minlength=config.NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, train_counts, width, label="Train", color="#66c2a5")
    ax.bar(x, val_counts, width, label="Validation", color="#fc8d62")
    ax.bar(x + width, test_counts, width, label="Test", color="#8da0cb")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {i}" for i in range(config.NUM_CLASSES)],
                       fontsize=config.FONT_SIZE_MIN)
    ax.set_ylabel("Number of Images", fontsize=config.FONT_SIZE_MIN)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_11_Class_Distribution.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 11 (Class Distribution) saved.")


def plot_ablation_chart(
    results_dir: str = config.RESULTS_DIR,
) -> None:
    table_path = os.path.join(results_dir, "tables", "ablation_results.csv")
    if not os.path.exists(table_path):
        logger.warning("Ablation results not found, skipping Figure 13.")
        return

    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    with open(table_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    variants = [r["variant"] for r in rows]
    metrics = ["accuracy", "qwk", "macro_f1", "macro_auc"]
    metric_labels = ["Accuracy", "QWK", "Macro F1", "Macro AUC"]

    x = np.arange(len(variants))
    width = 0.18
    colours = sns.color_palette(config.COLOUR_PALETTE, n_colors=len(metrics))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        values = [float(r.get(m, 0)) for r in rows]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=ml, color=colours[i])

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=10)
    ax.set_ylabel("Score", fontsize=config.FONT_SIZE_MIN)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_13_Ablation_Results.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 13 (Ablation Results) saved.")


def plot_cross_domain_comparison(
    variants: List[str],
    datasets: List[str],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    sns.set_style("whitegrid")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = sns.color_palette(config.COLOUR_PALETTE, n_colors=len(datasets))

    for ax, metric, metric_label in zip(
        axes, ["qwk", "macro_auc"], ["QWK", "Macro AUC"]
    ):
        x = np.arange(len(variants))
        width = 0.8 / len(datasets)

        for d_idx, ds in enumerate(datasets):
            values = []
            for variant in variants:
                eval_path = os.path.join(
                    results_dir, "evaluation",
                    f"eval_results_{variant}_{ds}.json",
                )
                if os.path.exists(eval_path):
                    with open(eval_path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    values.append(m.get(metric, 0))
                else:
                    values.append(0)

            offset = (d_idx - len(datasets) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=ds, color=colours[d_idx])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace("_", "\n") for v in variants], fontsize=9
        )
        ax.set_ylabel(metric_label, fontsize=config.FONT_SIZE_MIN)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_14_Cross_Domain_Comparison.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 14 (Cross-Domain Comparison) saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV TABLES
# ═══════════════════════════════════════════════════════════════════════════════


def save_main_results_table(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for variant in variants:
        eval_path = os.path.join(
            results_dir, "evaluation",
            f"eval_results_{variant}_{dataset_name}.json",
        )
        if not os.path.exists(eval_path):
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "Model": variant,
            "Accuracy": round(m.get("accuracy", 0), 4),
            "QWK": round(m.get("qwk", 0), 4),
            "Macro_F1": round(m.get("macro_f1", 0), 4),
            "Weighted_F1": round(m.get("weighted_f1", 0), 4),
            "Macro_AUC": round(m.get("macro_auc", 0), 4),
            "Binary_Sensitivity": round(m.get("binary_sensitivity", 0), 4),
            "Binary_Specificity": round(m.get("binary_specificity", 0), 4),
            "Binary_AUC": round(m.get("binary_auc", 0), 4),
        })

    if rows:
        path = os.path.join(table_dir, "main_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Main results table saved: {path}")


def save_per_class_table(
    variants: List[str],
    results_dir: str = config.RESULTS_DIR,
    dataset_name: str = "APTOS_test",
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for variant in variants:
        eval_path = os.path.join(
            results_dir, "evaluation",
            f"eval_results_{variant}_{dataset_name}.json",
        )
        if not os.path.exists(eval_path):
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        for c in range(config.NUM_CLASSES):
            rows.append({
                "Model": variant,
                "Class": config.CLASS_NAMES[c],
                "Precision": round(m["per_class_precision"][c], 4),
                "Recall": round(m["per_class_recall"][c], 4),
                "F1": round(m["per_class_f1"][c], 4),
                "Specificity": round(m["per_class_specificity"][c], 4),
                "AUC": round(m["per_class_auc"][c], 4)
                if c < len(m.get("per_class_auc", [])) else 0.0,
            })

    if rows:
        path = os.path.join(table_dir, "per_class_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Per-class results table saved: {path}")


def save_training_summary_table(
    training_summaries: Dict[str, Dict],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for variant, summary in training_summaries.items():
        rows.append({
            "Model": variant,
            "Total_Epochs": summary.get("total_epochs", 0),
            "Best_Epoch": summary.get("best_epoch", 0),
            "Training_Time": summary.get("total_time_formatted", ""),
            "Best_Val_QWK": round(summary.get("best_val_qwk", 0), 4),
            "Parameters": summary.get("parameters", {}).get("total", 0),
        })

    if rows:
        path = os.path.join(table_dir, "training_summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Training summary table saved: {path}")


def save_efficiency_table(
    efficiency_results: Dict[str, Dict],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for variant, eff in efficiency_results.items():
        rows.append({
            "Model": variant,
            "Parameters_M": eff.get("total_params_millions", 0),
            "CPU_Inference_ms": eff.get("cpu_inference_mean_ms", 0),
            "CPU_Inference_std": eff.get("cpu_inference_std_ms", 0),
            "Meets_Param_Target": eff.get("meets_param_target", False),
            "Meets_Latency_Target": eff.get("meets_latency_target", False),
        })

    if rows:
        path = os.path.join(table_dir, "efficiency_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Efficiency results table saved: {path}")


def save_dataset_statistics_table(
    stats: Dict[str, Dict],
    results_dir: str = config.RESULTS_DIR,
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for split_name, s in stats.items():
        row = {"Split": split_name, "Total": s.get("total", 0)}
        for c in range(config.NUM_CLASSES):
            row[f"Class_{c}_Count"] = s.get(f"class_{c}_count", 0)
            row[f"Class_{c}_Pct"] = s.get(f"class_{c}_pct", 0.0)
        rows.append(row)

    if rows:
        path = os.path.join(table_dir, "dataset_statistics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Dataset statistics table saved: {path}")


def save_hyperparameter_table(
    results_dir: str = config.RESULTS_DIR,
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for variant in config.ABLATION_VARIANTS:
        rows.append({
            "Model": variant,
            "Learning_Rate": config.LEARNING_RATE,
            "Backbone_LR": config.LEARNING_RATE * config.BACKBONE_LR_RATIO,
            "Weight_Decay": config.WEIGHT_DECAY,
            "Batch_Size": config.BATCH_SIZE,
            "Max_Epochs": config.MAX_EPOCHS,
            "Label_Smoothing": config.LABEL_SMOOTHING,
            "Warmup_Epochs": config.LINEAR_WARMUP_EPOCHS,
            "Grad_Clip": config.GRAD_CLIP_MAX_NORM,
            "Fusion_Dim": config.FUSION_DIM,
            "Classifier_Hidden": config.CLASSIFIER_HIDDEN_DIM,
            "Dropout": config.CLASSIFIER_DROPOUT,
        })

    if rows:
        path = os.path.join(table_dir, "hyperparameter_summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Hyperparameter summary saved: {path}")
