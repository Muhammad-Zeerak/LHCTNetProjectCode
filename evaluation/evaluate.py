import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from evaluation.metrics import compute_all_metrics, compute_qwk
from utils.logging_utils import get_logger

logger = get_logger("evaluate")


@torch.no_grad()
def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = config.DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    model.to(device)
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    for batch in data_loader:
        if len(batch) == 3:
            images, labels, paths = batch
            all_paths.extend(paths)
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if config.USE_AMP and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits, _ = model(images)
        else:
            logits, _ = model(images)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_paths,
    )


def evaluate_model(
    model: nn.Module,
    variant_name: str,
    test_loader: DataLoader,
    dataset_name: str = "APTOS_test",
    results_dir: str = config.RESULTS_DIR,
) -> Dict[str, Any]:
    logger.info(f"Evaluating {variant_name} on {dataset_name}...")

    y_true, y_pred, y_probs, paths = run_inference(model, test_loader)

    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    metrics["variant"] = variant_name
    metrics["dataset"] = dataset_name
    metrics["num_samples"] = len(y_true)

    # Save predictions
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(
        pred_dir, f"test_predictions_{variant_name}_{dataset_name}.csv"
    )
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["true_label", "pred_label"] + [
            f"prob_class_{i}" for i in range(config.NUM_CLASSES)
        ]
        if paths:
            header.insert(0, "image_path")
        writer.writerow(header)
        for idx in range(len(y_true)):
            row = [int(y_true[idx]), int(y_pred[idx])]
            row += [round(float(p), 6) for p in y_probs[idx]]
            if paths:
                row.insert(0, paths[idx] if idx < len(paths) else "")
            writer.writerow(row)

    # Save evaluation JSON
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(
        eval_dir, f"eval_results_{variant_name}_{dataset_name}.json"
    )
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  {variant_name} on {dataset_name}: "
        f"Acc={metrics['accuracy']:.4f}, "
        f"QWK={metrics['qwk']:.4f}, "
        f"Macro F1={metrics['macro_f1']:.4f}, "
        f"AUC={metrics['macro_auc']:.4f}, "
        f"Binary Sens={metrics['binary_sensitivity']:.4f}"
    )

    return metrics


def measure_efficiency(
    model: nn.Module,
    variant_name: str,
    device: torch.device = config.DEVICE,
) -> Dict[str, Any]:
    model.eval()
    params = model.count_parameters()

    # CPU inference time
    model_cpu = model.to(torch.device("cpu"))
    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model_cpu(dummy)

    # Timed passes
    times = []
    for _ in range(config.CPU_INFERENCE_PASSES):
        start = time.perf_counter()
        with torch.no_grad():
            model_cpu(dummy)
        times.append((time.perf_counter() - start) * 1000)  # ms

    # Move back to original device
    model.to(device)

    efficiency = {
        "variant": variant_name,
        "total_params": params["total"],
        "total_params_millions": round(params["total"] / 1e6, 2),
        "cpu_inference_mean_ms": round(float(np.mean(times)), 2),
        "cpu_inference_std_ms": round(float(np.std(times)), 2),
        "cpu_inference_p95_ms": round(float(np.percentile(times, 95)), 2),
        "meets_param_target": params["total"] < config.TARGET_PARAMS,
        "meets_latency_target": float(np.mean(times)) < 200.0,
    }

    logger.info(
        f"Efficiency [{variant_name}]: "
        f"{efficiency['total_params_millions']}M params, "
        f"{efficiency['cpu_inference_mean_ms']:.1f}ms CPU inference "
        f"({'✓' if efficiency['meets_latency_target'] else '✗'} <200ms)"
    )

    return efficiency


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    metric_fn,
    n_iterations: int = config.BOOTSTRAP_N_ITERATIONS,
    alpha: float = config.BOOTSTRAP_CI_ALPHA,
    seed: int = config.SEED,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred, y_probs)
    scores = []

    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], y_probs[idx])
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue

    if not scores:
        return point, point, point

    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return point, lower, upper


def evaluate_cross_dataset(
    model: nn.Module,
    variant_name: str,
    aptos_metrics: Dict[str, Any],
    cross_loader: DataLoader,
    cross_name: str,
    results_dir: str = config.RESULTS_DIR,
) -> Dict[str, Any]:
    cross_metrics = evaluate_model(
        model, variant_name, cross_loader,
        dataset_name=cross_name, results_dir=results_dir,
    )

    # Compute performance drop
    aptos_qwk = aptos_metrics.get("qwk", 0)
    aptos_auc = aptos_metrics.get("macro_auc", 0)
    cross_qwk = cross_metrics.get("qwk", 0)
    cross_auc = cross_metrics.get("macro_auc", 0)

    qwk_drop_abs = aptos_qwk - cross_qwk
    auc_drop_abs = aptos_auc - cross_auc
    qwk_drop_rel = qwk_drop_abs / aptos_qwk if aptos_qwk > 0 else 0.0
    auc_drop_rel = auc_drop_abs / aptos_auc if aptos_auc > 0 else 0.0

    cross_metrics["qwk_drop_absolute"] = round(qwk_drop_abs, 4)
    cross_metrics["qwk_drop_relative"] = round(qwk_drop_rel, 4)
    cross_metrics["auc_drop_absolute"] = round(auc_drop_abs, 4)
    cross_metrics["auc_drop_relative"] = round(auc_drop_rel, 4)

    if abs(qwk_drop_rel) > config.CROSS_DATASET_DROP_THRESHOLD:
        logger.warning(
            f"  {cross_name} QWK drop ({qwk_drop_rel:.1%}) exceeds "
            f"{config.CROSS_DATASET_DROP_THRESHOLD:.0%} threshold"
        )
    if abs(auc_drop_rel) > config.CROSS_DATASET_DROP_THRESHOLD:
        logger.warning(
            f"  {cross_name} AUC drop ({auc_drop_rel:.1%}) exceeds "
            f"{config.CROSS_DATASET_DROP_THRESHOLD:.0%} threshold"
        )

    return cross_metrics
