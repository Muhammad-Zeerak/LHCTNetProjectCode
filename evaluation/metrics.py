from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    # Overall accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # QWK
    metrics["qwk"] = float(compute_qwk(y_true, y_pred))

    # Per-class precision, recall (sensitivity), F1
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)),
        zero_division=0,
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)),
        zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)),
        zero_division=0,
    )

    metrics["per_class_precision"] = per_class_precision.tolist()
    metrics["per_class_recall"] = per_class_recall.tolist()
    metrics["per_class_f1"] = per_class_f1.tolist()

    # Per-class specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_list = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(float(spec))
    metrics["per_class_specificity"] = specificity_list

    # Macro and weighted averages
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    # AUC-ROC (per-class one-vs-rest)
    y_probs_norm = y_probs / y_probs.sum(axis=1, keepdims=True)
    try:
        per_class_auc = roc_auc_score(
            y_true, y_probs_norm, multi_class="ovr", average=None,
            labels=list(range(num_classes)),
        )
        metrics["per_class_auc"] = per_class_auc.tolist()
        metrics["macro_auc"] = float(np.mean(per_class_auc))
    except ValueError:
        # Some classes may not be present in small test sets
        metrics["per_class_auc"] = [0.0] * num_classes
        metrics["macro_auc"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = cm.tolist()
    cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_normalised, nan=0.0, copy=False)
    metrics["confusion_matrix_normalised"] = cm_normalised.tolist()

    # Non-referable: grades 0, 1. Referable: grades 2, 3, 4.
    y_true_binary = (y_true >= config.REFERABLE_THRESHOLD).astype(int)
    y_pred_binary = (y_pred >= config.REFERABLE_THRESHOLD).astype(int)

    metrics["binary_sensitivity"] = float(
        recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    )
    binary_cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    if binary_cm.shape == (2, 2):
        tn, fp = binary_cm[0, 0], binary_cm[0, 1]
        metrics["binary_specificity"] = float(
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        )
    else:
        metrics["binary_specificity"] = 0.0

    # Binary AUC-ROC using referable probability
    referable_probs = y_probs[:, config.REFERABLE_THRESHOLD:].sum(axis=1)
    try:
        metrics["binary_auc"] = float(
            roc_auc_score(y_true_binary, referable_probs)
        )
    except ValueError:
        metrics["binary_auc"] = 0.0

    metrics["binary_f1"] = float(
        f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    )

    return metrics


def compute_roc_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[str, Any]:
    roc_data = {}
    for i in range(num_classes):
        y_true_bin = (y_true == i).astype(int)
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
            roc_data[i] = {"fpr": [0, 1], "tpr": [0, 1], "auc": 0.5}
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
        auc_val = roc_auc_score(y_true_bin, y_probs[:, i])
        roc_data[i] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc_val),
        }
    return roc_data


def compute_binary_roc_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    y_true_binary = (y_true >= config.REFERABLE_THRESHOLD).astype(int)
    referable_probs = y_probs[:, config.REFERABLE_THRESHOLD:].sum(axis=1)
    try:
        fpr, tpr, _ = roc_curve(y_true_binary, referable_probs)
        auc_val = roc_auc_score(y_true_binary, referable_probs)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc_val)}
    except ValueError:
        return {"fpr": [0, 1], "tpr": [0, 1], "auc": 0.5}