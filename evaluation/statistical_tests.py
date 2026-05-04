import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2

import config
from utils.logging_utils import get_logger

logger = get_logger("stats")


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's test with continuity correction
    if b + c == 0:
        return {
            "test_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b": b,
            "c": c,
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "test_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "b": b,
        "c": c,
    }


def bootstrap_metric_comparison(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_probs_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_probs_b: np.ndarray,
    metric_name: str = "qwk",
    n_iterations: int = config.BOOTSTRAP_N_ITERATIONS,
    seed: int = config.SEED,
) -> Dict[str, Any]:
    from evaluation.metrics import compute_qwk
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(y_true)

    def _compute_metric(yt, yp, ypr, name):
        if name == "qwk":
            return compute_qwk(yt, yp)
        elif name == "f1":
            return f1_score(yt, yp, average="macro", zero_division=0)
        elif name == "auc":
            try:
                # Normalise probs to sum to 1.0
                ypr_norm = ypr / ypr.sum(axis=1, keepdims=True)
                return roc_auc_score(
                    yt, ypr_norm, multi_class="ovr", average="macro",
                    labels=list(range(config.NUM_CLASSES)),
                )
            except ValueError:
                return 0.0
        return 0.0

    # Observed difference
    obs_a = _compute_metric(y_true, y_pred_a, y_probs_a, metric_name)
    obs_b = _compute_metric(y_true, y_pred_b, y_probs_b, metric_name)
    observed_diff = obs_a - obs_b

    # Bootstrap
    diffs = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        try:
            diff = (
                _compute_metric(
                    y_true[idx], y_pred_a[idx], y_probs_a[idx], metric_name
                )
                - _compute_metric(
                    y_true[idx], y_pred_b[idx], y_probs_b[idx], metric_name
                )
            )
            diffs.append(diff)
        except (ValueError, ZeroDivisionError):
            continue

    if not diffs:
        return {
            "metric": metric_name,
            "observed_diff": round(observed_diff, 4),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided p-value: proportion of bootstrap samples where diff has opposite sign
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "metric": metric_name,
        "model_a_score": round(obs_a, 4),
        "model_b_score": round(obs_b, 4),
        "observed_diff": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def run_all_pairwise_tests(
    y_true: np.ndarray,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    results_dir: str = config.RESULTS_DIR,
) -> List[Dict[str, Any]]:
    import csv
    import os

    results = []
    variant_names = list(predictions.keys())

    for a, b in itertools.combinations(variant_names, 2):
        y_pred_a, y_probs_a = predictions[a]
        y_pred_b, y_probs_b = predictions[b]

        # McNemar's test
        mcnemar = mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Bootstrap tests for QWK, F1, AUC
        for metric in ["qwk", "f1", "auc"]:
            boot = bootstrap_metric_comparison(
                y_true, y_pred_a, y_probs_a, y_pred_b, y_probs_b,
                metric_name=metric,
            )
            results.append({
                "model_a": a,
                "model_b": b,
                "test": f"bootstrap_{metric}",
                "statistic": boot["observed_diff"],
                "p_value": boot["p_value"],
                "significant": boot["significant"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
            })

        results.append({
            "model_a": a,
            "model_b": b,
            "test": "mcnemar",
            "statistic": mcnemar["test_statistic"],
            "p_value": mcnemar["p_value"],
            "significant": mcnemar["significant"],
            "ci_lower": None,
            "ci_upper": None,
        })

    # Save to CSV
    table_dir = os.path.join(results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    csv_path = os.path.join(table_dir, "statistical_tests.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Statistical test results saved: {csv_path}")

    return results