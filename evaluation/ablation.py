import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import config
from data.dataset import (
    load_aptos_splits,
    create_dataloaders,
    compute_class_weights,
)
from evaluation.evaluate import evaluate_model, run_inference
from evaluation.statistical_tests import run_all_pairwise_tests
from models import build_model
from training.trainer import train_model_with_oom_recovery
from utils.logging_utils import get_logger

logger = get_logger("ablation")


def run_ablation_study(
    train_data, val_data, test_loader,
    class_weights: torch.Tensor,
    results_dir: str = config.RESULTS_DIR,
    skip_training: bool = False,
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("=" * 60)

    all_results = {}
    all_predictions = {}
    y_true_test = None

    for variant in config.ABLATION_VARIANTS:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Ablation variant: {variant}")
        logger.info(f"{'─' * 40}")

        model = build_model(variant)
        model = model.to(config.DEVICE)

        checkpoint_path = os.path.join(
            results_dir, "checkpoints", f"best_{variant}.pt"
        )

        if skip_training and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            from utils.checkpoint import load_checkpoint
            load_checkpoint(checkpoint_path, model, device=config.DEVICE)
        else:
            # Create fresh DataLoaders for this variant
            loaders = create_dataloaders(
                train_data, val_data, ([], []),
                batch_size=config.BATCH_SIZE,
            )

            train_model_with_oom_recovery(
                model=model,
                variant_name=variant,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                class_weights=class_weights,
                batch_size=config.BATCH_SIZE,
                train_data=train_data,
                val_data=val_data,
            )

        # Evaluate on APTOS test set
        metrics = evaluate_model(
            model, variant, test_loader,
            dataset_name="APTOS_test", results_dir=results_dir,
        )
        all_results[variant] = metrics

        # Store predictions for statistical tests
        y_true, y_pred, y_probs, _ = run_inference(model, test_loader)
        all_predictions[variant] = (y_pred, y_probs)
        if y_true_test is None:
            y_true_test = y_true

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Statistical tests between ablation variants
    if y_true_test is not None and len(all_predictions) > 1:
        logger.info("\nRunning pairwise statistical tests...")
        stat_results = run_all_pairwise_tests(
            y_true_test, all_predictions, results_dir=results_dir,
        )

    # Save ablation summary
    _save_ablation_summary(all_results, results_dir)

    return all_results


def _save_ablation_summary(
    results: Dict[str, Dict], results_dir: str
) -> None:
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    csv_path = os.path.join(table_dir, "ablation_results.csv")
    rows = []
    for variant, metrics in results.items():
        rows.append({
            "variant": variant,
            "accuracy": round(metrics.get("accuracy", 0), 4),
            "qwk": round(metrics.get("qwk", 0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0), 4),
            "macro_auc": round(metrics.get("macro_auc", 0), 4),
            "binary_sensitivity": round(metrics.get("binary_sensitivity", 0), 4),
            "binary_specificity": round(metrics.get("binary_specificity", 0), 4),
        })

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Ablation summary saved: {csv_path}")

    # Compute deltas from full model
    full_metrics = results.get("lhctnet_full", {})
    if full_metrics:
        delta_path = os.path.join(table_dir, "ablation_deltas.csv")
        delta_rows = []
        for variant, metrics in results.items():
            if variant == "lhctnet_full":
                continue
            delta_rows.append({
                "variant": variant,
                "vs_full_model": "lhctnet_full",
                "delta_accuracy": round(
                    full_metrics.get("accuracy", 0) - metrics.get("accuracy", 0), 4
                ),
                "delta_qwk": round(
                    full_metrics.get("qwk", 0) - metrics.get("qwk", 0), 4
                ),
                "delta_macro_f1": round(
                    full_metrics.get("macro_f1", 0) - metrics.get("macro_f1", 0), 4
                ),
                "delta_macro_auc": round(
                    full_metrics.get("macro_auc", 0) - metrics.get("macro_auc", 0), 4
                ),
            })
        if delta_rows:
            with open(delta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=delta_rows[0].keys())
                writer.writeheader()
                writer.writerows(delta_rows)
            logger.info(f"Ablation deltas saved: {delta_path}")
