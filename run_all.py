import argparse
import json
import os
import platform
import shutil
import sys
import time
from typing import Dict

import numpy as np
import torch

# Project imports
import config
from utils.reproducibility import set_all_seeds, enable_cudnn_benchmark
from utils.logging_utils import setup_logging, get_logger
from utils.checkpoint import save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LHCTNet — Lightweight Hybrid CNN-Transformer for DR Grading"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="Model variant to train: 'all' or one of "
             f"{config.ABLATION_VARIANTS}",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, run evaluation from saved checkpoints.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=config.RESULTS_DIR,
        help="Path to results output directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.MAX_EPOCHS,
        help="Maximum training epochs.",
    )
    return parser.parse_args()


def print_environment(logger) -> None:
    logger.info("=" * 60)
    logger.info("ENVIRONMENT")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU memory: {gpu_mem:.1f} GB")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Seed: {config.SEED}")

    try:
        import timm
        logger.info(f"timm: {timm.__version__}")
    except ImportError:
        pass
    try:
        import sklearn
        logger.info(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        pass
    try:
        import cv2
        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        pass


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    config.RESULTS_DIR = args.results_dir
    config.SEED = args.seed
    config.BATCH_SIZE = args.batch_size
    config.MAX_EPOCHS = args.epochs

    # Update dependent paths
    config.APTOS_DIR = os.path.join(config.DATA_DIR, "aptos2019")
    config.APTOS_TRAIN_CSV = os.path.join(config.APTOS_DIR, "train.csv")
    config.APTOS_TRAIN_IMAGES = os.path.join(config.APTOS_DIR, "train_images")
    config.IDRID_DIR = os.path.join(config.DATA_DIR, "idrid")
    config.IDRID_TRAIN_LABELS = os.path.join(
        config.IDRID_DIR, "IDRiD_Disease Grading_Training Labels.csv"
    )
    config.IDRID_TEST_LABELS = os.path.join(
        config.IDRID_DIR, "IDRiD_Disease Grading_Testing Labels.csv"
    )
    config.IDRID_TRAIN_IMAGES = os.path.join(config.IDRID_DIR, "images", "training")
    config.IDRID_TEST_IMAGES = os.path.join(config.IDRID_DIR, "images", "testing")
    config.IDRID_LESION_MASKS_DIR = os.path.join(config.IDRID_DIR, "lesion_masks")
    config.EYEPACS_DIR = os.path.join(config.DATA_DIR, "eyepacs")
    config.EYEPACS_LABELS_CSV = os.path.join(config.EYEPACS_DIR, "trainLabels.csv")
    config.EYEPACS_IMAGES = os.path.join(config.EYEPACS_DIR, "images")
    config.CHECKPOINT_DIR = os.path.join(config.RESULTS_DIR, "checkpoints")
    config.LOG_DIR = os.path.join(config.RESULTS_DIR, "logs")
    config.LOG_FILE = os.path.join(config.LOG_DIR, "run_log.txt")
    config.FIGURE_DIR = os.path.join(config.RESULTS_DIR, "figures")
    config.TABLE_DIR = os.path.join(config.RESULTS_DIR, "tables")

    # Create output directories
    for d in [
        config.RESULTS_DIR, config.CHECKPOINT_DIR, config.LOG_DIR,
        os.path.join(config.RESULTS_DIR, "predictions"),
        os.path.join(config.RESULTS_DIR, "evaluation"),
        config.FIGURE_DIR, config.TABLE_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # Setup logging
    logger = setup_logging(config.LOG_FILE, config.LOG_LEVEL)

    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("LHCTNet — Diabetic Retinopathy Grading Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Environment ──
    print_environment(logger)

    # ── Step 2: Seeds ──
    set_all_seeds(config.SEED)
    if config.CUDNN_BENCHMARK and config.DEVICE.type == "cuda":
        enable_cudnn_benchmark()
    logger.info(f"Random seed set: {config.SEED}")

    # ── Step 3: Load and split data ──
    from data.dataset import (
        load_aptos_splits,
        load_idrid_data,
        load_eyepacs_subset,
        create_dataloaders,
        create_cross_dataset_loader,
        compute_class_weights,
        print_dataset_statistics,
    )

    logger.info("\n" + "=" * 60)
    logger.info("DATASET LOADING AND SPLITTING")
    logger.info("=" * 60)

    train_data, val_data, test_data = load_aptos_splits(seed=config.SEED)

    # Print statistics
    logger.info("\nAPTOS 2019 Dataset Statistics:")
    stats = {}
    stats["train"] = print_dataset_statistics("Train", train_data[1])
    stats["val"] = print_dataset_statistics("Validation", val_data[1])
    stats["test"] = print_dataset_statistics("Test", test_data[1])

    # Save split indices for reproducibility
    split_info = {
        "train_size": len(train_data[0]),
        "val_size": len(val_data[0]),
        "test_size": len(test_data[0]),
        "seed": config.SEED,
    }
    with open(
        os.path.join(config.RESULTS_DIR, "split_indices.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(split_info, f, indent=2)

    # Class weights
    class_weights = compute_class_weights(train_data[1])
    logger.info(f"\nClass weights: {class_weights.tolist()}")

    # Create APTOS DataLoaders
    loaders = create_dataloaders(
        train_data, val_data, test_data, batch_size=config.BATCH_SIZE
    )

    # Cross-dataset loaders
    idrid_paths, idrid_labels = load_idrid_data()
    stats["idrid"] = print_dataset_statistics("IDRiD", idrid_labels)
    idrid_loader = create_cross_dataset_loader(idrid_paths, idrid_labels)

    eyepacs_paths, eyepacs_labels = load_eyepacs_subset()
    stats["eyepacs"] = print_dataset_statistics("EyePACS subset", eyepacs_labels)
    eyepacs_loader = create_cross_dataset_loader(eyepacs_paths, eyepacs_labels)

    # Save dataset statistics
    from visualisation.results_tables import (
        save_dataset_statistics_table,
        save_hyperparameter_table,
    )
    save_dataset_statistics_table(stats, config.RESULTS_DIR)
    save_hyperparameter_table(config.RESULTS_DIR)

    from visualisation.results_tables import plot_class_distribution
    plot_class_distribution(train_data[1], val_data[1], test_data[1], config.RESULTS_DIR)

    # ── Save config snapshot ──
    config_dict = {
        k: v for k, v in vars(config).items()
        if not k.startswith("_") and isinstance(
            v, (int, float, str, bool, list, tuple)
        )
    }
    save_config_snapshot(
        config_dict, os.path.join(config.RESULTS_DIR, "config_snapshot.json")
    )

    # ── Step 5: Training ──
    if not args.skip_training:
        from models import build_model
        from training.trainer import train_model_with_oom_recovery

        variants_to_train = (
            config.ABLATION_VARIANTS if args.model == "all"
            else [args.model]
        )

        training_summaries = {}

        for variant in variants_to_train:
            logger.info(f"\n{'═' * 60}")
            logger.info(f"TRAINING: {variant}")
            logger.info(f"{'═' * 60}")

            model = build_model(variant)
            model = model.to(config.DEVICE)

            # Create fresh loaders for each variant
            variant_loaders = create_dataloaders(
                train_data, val_data, test_data, batch_size=config.BATCH_SIZE
            )

            summary = train_model_with_oom_recovery(
                model=model,
                variant_name=variant,
                train_loader=variant_loaders["train"],
                val_loader=variant_loaders["val"],
                class_weights=class_weights,
                batch_size=config.BATCH_SIZE,
                train_data=train_data,
                val_data=val_data,
            )
            training_summaries[variant] = summary

            # Free GPU memory between models
            del model
            torch.cuda.empty_cache()

        # Save training summary table
        from visualisation.results_tables import save_training_summary_table
        save_training_summary_table(training_summaries, config.RESULTS_DIR)
    else:
        logger.info("Skipping training (--skip-training flag set).")
        training_summaries = {}

    # ── Step 6: Evaluation on APTOS test set ──
    from models import build_model
    from evaluation.evaluate import (
        evaluate_model,
        evaluate_cross_dataset,
        measure_efficiency,
        run_inference,
    )

    variants_to_eval = (
        config.ABLATION_VARIANTS if args.model == "all"
        else [args.model]
    )

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON APTOS TEST SET")
    logger.info("=" * 60)

    all_metrics = {}
    all_predictions = {}
    efficiency_results = {}
    y_true = None  # Will be set during first evaluation

    for variant in variants_to_eval:
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"best_{variant}.pt"
        )
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for {variant}, skipping.")
            continue

        model = build_model(variant)
        model = model.to(config.DEVICE)
        from utils.checkpoint import load_checkpoint
        load_checkpoint(checkpoint_path, model, device=config.DEVICE)

        # APTOS test evaluation
        metrics = evaluate_model(
            model, variant, loaders["test"],
            dataset_name="APTOS_test", results_dir=config.RESULTS_DIR,
        )
        all_metrics[variant] = metrics

        # Store predictions for statistical tests
        y_true, y_pred, y_probs, _ = run_inference(model, loaders["test"])
        all_predictions[variant] = (y_pred, y_probs)

        # ── Cross-dataset evaluation ──
        cross_idrid = evaluate_cross_dataset(
            model, variant, metrics, idrid_loader, "IDRiD",
            results_dir=config.RESULTS_DIR,
        )

        cross_eyepacs = evaluate_cross_dataset(
            model, variant, metrics, eyepacs_loader, "EyePACS",
            results_dir=config.RESULTS_DIR,
        )

        # ── Efficiency measurement ──
        eff = measure_efficiency(model, variant)
        efficiency_results[variant] = eff

        # ── Explainability analysis ──
        if variant in ("lhctnet_full", "efficientnet_only"):
            from visualisation.explainability import (
                generate_explainability_figures,
                generate_error_analysis,
            )
            generate_explainability_figures(
                model, variant, loaders["test"], config.RESULTS_DIR
            )
            generate_error_analysis(
                model, variant, loaders["test"], config.RESULTS_DIR
            )

            # IDRiD explainability evaluation
            if variant == "lhctnet_full":
                from visualisation.explainability import (
                    evaluate_explainability_on_idrid,
                )
                evaluate_explainability_on_idrid(
                    model, variant, idrid_loader, config.RESULTS_DIR
                )

        del model
        torch.cuda.empty_cache()

    # ── Step 7: Statistical tests ──
    if len(all_predictions) > 1 and y_true is not None:
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("=" * 60)
        from evaluation.statistical_tests import run_all_pairwise_tests
        stat_results = run_all_pairwise_tests(
            y_true, all_predictions, config.RESULTS_DIR
        )
        # Print significant results
        for r in stat_results:
            if r["significant"]:
                logger.info(
                    f"  ★ {r['model_a']} vs {r['model_b']} [{r['test']}]: "
                    f"p={r['p_value']:.4f} (SIGNIFICANT)"
                )

    # ── Step 8: Generate figures and tables ──
    from visualisation.training_curves import plot_training_curves, plot_metric_curves
    from visualisation.confusion_matrix import plot_confusion_matrices
    from visualisation.roc_curves import plot_roc_curves, plot_binary_roc_curves
    from visualisation.results_tables import (
        plot_model_comparison,
        plot_per_class_performance,
        plot_ablation_chart,
        plot_cross_domain_comparison,
        save_main_results_table,
        save_per_class_table,
        save_efficiency_table,
    )

    plot_training_curves(variants_to_eval, config.RESULTS_DIR)
    plot_metric_curves(variants_to_eval, config.RESULTS_DIR)
    plot_confusion_matrices(variants_to_eval, config.RESULTS_DIR)
    plot_roc_curves(variants_to_eval, config.RESULTS_DIR)
    plot_binary_roc_curves(variants_to_eval, config.RESULTS_DIR)
    plot_model_comparison(variants_to_eval, config.RESULTS_DIR)
    plot_per_class_performance(variants_to_eval, config.RESULTS_DIR)
    plot_ablation_chart(config.RESULTS_DIR)

    # Cross-domain comparison
    cross_datasets = ["APTOS_test", "IDRiD", "EyePACS"]
    plot_cross_domain_comparison(variants_to_eval, cross_datasets, config.RESULTS_DIR)

    save_main_results_table(variants_to_eval, config.RESULTS_DIR)
    save_per_class_table(variants_to_eval, config.RESULTS_DIR)
    save_efficiency_table(efficiency_results, config.RESULTS_DIR)

    # ── Step 9: Final summary ──
    pipeline_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    for variant, metrics in all_metrics.items():
        logger.info(
            f"\n{variant}:"
            f"\n  Accuracy: {metrics.get('accuracy', 0):.4f}"
            f"\n  QWK:      {metrics.get('qwk', 0):.4f}"
            f"\n  Macro F1: {metrics.get('macro_f1', 0):.4f}"
            f"\n  AUC-ROC:  {metrics.get('macro_auc', 0):.4f}"
            f"\n  Binary Sensitivity: {metrics.get('binary_sensitivity', 0):.4f}"
            f"\n  Binary Specificity: {metrics.get('binary_specificity', 0):.4f}"
        )

    if efficiency_results:
        logger.info("\nEfficiency Summary:")
        for variant, eff in efficiency_results.items():
            logger.info(
                f"  {variant}: {eff['total_params_millions']}M params, "
                f"{eff['cpu_inference_mean_ms']:.1f}ms CPU"
            )

    h = int(pipeline_time // 3600)
    m = int((pipeline_time % 3600) // 60)
    s = int(pipeline_time % 60)
    logger.info(f"\nTotal pipeline time: {h}:{m:02d}:{s:02d}")
    logger.info(f"Results saved to: {os.path.abspath(config.RESULTS_DIR)}")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()