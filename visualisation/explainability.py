import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data.preprocessing import preprocess_fundus_image
from utils.logging_utils import get_logger

logger = get_logger("vis.explain")


def compute_gradcam_pp(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device = config.DEVICE,
) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_(True)

    # Forward pass
    logits, _ = model(image_tensor)
    score = logits[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=True)

    features = model.get_cnn_features()
    gradients = model.get_gradients()

    if features is None or gradients is None:
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE))

    # Grad-CAM++ weights computation
    grads = gradients.detach()
    feats = features.detach()

    grad_2 = grads ** 2
    grad_3 = grads ** 3

    spatial_sum = feats * grad_3
    spatial_sum = spatial_sum.sum(dim=[2, 3], keepdim=True)

    alpha = grad_2 / (2 * grad_2 + spatial_sum + 1e-8)
    alpha = F.relu(alpha)

    weights = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

    # Weighted combination of feature maps
    cam = (weights * feats).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample to input resolution
    cam = F.interpolate(
        cam, size=(config.IMG_SIZE, config.IMG_SIZE),
        mode="bilinear", align_corners=False,
    )
    cam = cam.squeeze().cpu().numpy()

    # Normalise to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam


def extract_cross_attention_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device = config.DEVICE,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        _, attn_weights = model(image_tensor)

    if attn_weights is None:
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE))

    # Average over heads and query positions -> spatial attention map
    # attn_weights shape: (B, num_heads, HW, HW)
    attn = attn_weights[0]  # First sample
    attn_avg = attn.mean(dim=0)  # Average over heads: (HW, HW)

    # Sum over key dimension to get per-query attention distribution
    spatial_attn = attn_avg.sum(dim=0)  # (HW,)
    side = int(spatial_attn.shape[0] ** 0.5)
    spatial_attn = spatial_attn.reshape(side, side).cpu().numpy()

    # Upsample to input resolution
    spatial_attn = cv2.resize(
        spatial_attn, (config.IMG_SIZE, config.IMG_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Normalise
    if spatial_attn.max() > spatial_attn.min():
        spatial_attn = (spatial_attn - spatial_attn.min()) / (
            spatial_attn.max() - spatial_attn.min()
        )

    return spatial_attn


def pointing_game_score(
    heatmap: np.ndarray,
    lesion_mask: np.ndarray,
) -> bool:
    if heatmap.max() == 0 or lesion_mask.max() == 0:
        return False

    # Resize mask to match heatmap if needed
    if heatmap.shape != lesion_mask.shape:
        lesion_mask = cv2.resize(
            lesion_mask.astype(np.uint8), (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Find max activation point
    max_loc = np.unravel_index(heatmap.argmax(), heatmap.shape)

    return bool(lesion_mask[max_loc[0], max_loc[1]] > 0)


def faithfulness_score(
    model: nn.Module,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    device: torch.device = config.DEVICE,
    n_steps: int = 10,
) -> float:
    model.eval()
    image_tensor = image_tensor.to(device)

    # Get baseline confidence
    with torch.no_grad():
        logits, _ = model(image_tensor)
        baseline_conf = torch.softmax(logits, dim=1)[0, target_class].item()

    # Sort heatmap pixels by importance (descending)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]

    # Progressively mask top-k% of important pixels
    confidences = [baseline_conf]
    fractions = [0.0]
    total_pixels = len(flat_heatmap)
    step_size = total_pixels // n_steps

    for step in range(1, n_steps + 1):
        k = min(step * step_size, total_pixels)
        mask = np.ones_like(flat_heatmap, dtype=np.float32)
        mask[sorted_indices[:k]] = 0.0
        mask_2d = mask.reshape(config.IMG_SIZE, config.IMG_SIZE)
        mask_tensor = torch.tensor(mask_2d, dtype=torch.float32).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        masked_image = image_tensor * mask_tensor
        with torch.no_grad():
            logits, _ = model(masked_image)
            conf = torch.softmax(logits, dim=1)[0, target_class].item()

        confidences.append(conf)
        fractions.append(k / total_pixels)

    # Faithfulness = negative correlation between removal fraction and confidence
    # Higher drop in confidence when removing important regions = more faithful
    confidences = np.array(confidences)
    fractions = np.array(fractions)

    if len(confidences) < 2:
        return 0.0

    correlation = np.corrcoef(fractions, confidences)[0, 1]
    # Negate: we expect negative correlation (more removal -> less confidence)
    return float(-correlation) if not np.isnan(correlation) else 0.0


def generate_explainability_figures(
    model: nn.Module,
    variant_name: str,
    test_loader,
    results_dir: str = config.RESULTS_DIR,
    num_examples: int = 8,
) -> None:
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    model.eval()
    examples = {"correct": [], "incorrect": []}
    device = config.DEVICE

    for batch in test_loader:
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [""] * len(images)

        images_dev = images.to(device)
        with torch.no_grad():
            logits, _ = model(images_dev)
        preds = logits.argmax(dim=1).cpu()
        probs = torch.softmax(logits, dim=1).cpu()

        for i in range(len(images)):
            is_correct = preds[i].item() == labels[i].item()
            key = "correct" if is_correct else "incorrect"
            if len(examples[key]) < num_examples // 2:
                examples[key].append({
                    "image": images[i:i+1],
                    "true": labels[i].item(),
                    "pred": preds[i].item(),
                    "conf": probs[i, preds[i]].item(),
                    "path": paths[i] if i < len(paths) else "",
                })

        if (len(examples["correct"]) >= num_examples // 2 and
                len(examples["incorrect"]) >= num_examples // 2):
            break

    all_examples = examples["correct"] + examples["incorrect"]
    if not all_examples:
        logger.warning("No examples found for explainability figures.")
        return

    n = len(all_examples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for idx, ex in enumerate(all_examples):
        img_tensor = ex["image"]

        # Original image (denormalised)
        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        img_np = img_np * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)

        # Grad-CAM++
        has_gradcam = hasattr(model, "get_cnn_features")
        if has_gradcam:
            cam = compute_gradcam_pp(model, img_tensor, ex["pred"], device)
        else:
            cam = np.zeros((config.IMG_SIZE, config.IMG_SIZE))

        # Cross-attention map
        has_attn = hasattr(model, "cross_attention")
        if has_attn:
            attn_map = extract_cross_attention_map(model, img_tensor, device)
        else:
            attn_map = np.zeros((config.IMG_SIZE, config.IMG_SIZE))

        # Plot
        is_correct = ex["true"] == ex["pred"]
        status = "✓" if is_correct else "✗"
        colour = "green" if is_correct else "red"

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_ylabel(
            f"{status} True:{ex['true']} Pred:{ex['pred']} ({ex['conf']:.2f})",
            fontsize=10, color=colour,
        )
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])

        axes[idx, 1].imshow(img_np)
        axes[idx, 1].imshow(cam, cmap="jet", alpha=0.4)
        axes[idx, 1].set_xticks([])
        axes[idx, 1].set_yticks([])
        if idx == 0:
            axes[idx, 1].set_title("Grad-CAM++", fontsize=config.FONT_SIZE_MIN)

        axes[idx, 2].imshow(img_np)
        axes[idx, 2].imshow(attn_map, cmap="jet", alpha=0.4)
        axes[idx, 2].set_xticks([])
        axes[idx, 2].set_yticks([])
        if idx == 0:
            axes[idx, 2].set_title("Cross-Attention", fontsize=config.FONT_SIZE_MIN)

    if n > 0 and axes.shape[0] > 0:
        axes[0, 0].set_title("Original", fontsize=config.FONT_SIZE_MIN)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(fig_dir, f"Figure_7_Explainability_Maps.{fmt}")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info("Figure 7 (Explainability Maps) saved.")


def generate_error_analysis(
    model: nn.Module,
    variant_name: str,
    test_loader,
    results_dir: str = config.RESULTS_DIR,
    top_n: int = 5,
) -> None:
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    model.eval()
    device = config.DEVICE
    results = []

    for batch in test_loader:
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [""] * len(images)

        images_dev = images.to(device)
        with torch.no_grad():
            logits, _ = model(images_dev)
        preds = logits.argmax(dim=1).cpu()
        probs = torch.softmax(logits, dim=1).cpu()

        for i in range(len(images)):
            results.append({
                "image": images[i],
                "true": labels[i].item(),
                "pred": preds[i].item(),
                "conf": probs[i, preds[i]].item(),
                "correct": preds[i].item() == labels[i].item(),
            })

    # Sort by confidence
    correct = sorted(
        [r for r in results if r["correct"]],
        key=lambda x: x["conf"], reverse=True,
    )[:top_n]
    incorrect = sorted(
        [r for r in results if not r["correct"]],
        key=lambda x: x["conf"], reverse=True,
    )[:top_n]

    examples = correct + incorrect
    if not examples:
        return

    n = len(examples)
    fig, axes = plt.subplots(2, max(top_n, 1), figsize=(3 * top_n, 7))
    if top_n == 1:
        axes = axes.reshape(2, 1)

    for row, (group, title) in enumerate([
        (correct, "Most Confident Correct"),
        (incorrect, "Most Confident Incorrect"),
    ]):
        for col in range(top_n):
            ax = axes[row, col]
            if col < len(group):
                ex = group[col]
                img_np = ex["image"].permute(1, 2, 0).numpy()
                img_np = img_np * np.array(config.IMAGENET_STD) + np.array(
                    config.IMAGENET_MEAN
                )
                img_np = np.clip(img_np, 0, 1)
                ax.imshow(img_np)
                colour = "green" if ex["correct"] else "red"
                ax.set_xlabel(
                    f"T:{ex['true']} P:{ex['pred']} ({ex['conf']:.3f})",
                    fontsize=9, color=colour,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=10)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        path = os.path.join(
            fig_dir, f"Figure_15_Error_Analysis_{variant_name}.{fmt}"
        )
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    logger.info(f"Figure 15 (Error Analysis — {variant_name}) saved.")


def evaluate_explainability_on_idrid(
    model: nn.Module,
    variant_name: str,
    idrid_loader,
    results_dir: str = config.RESULTS_DIR,
) -> Dict[str, Any]:
    # Check if lesion masks exist
    mask_dir = config.IDRID_LESION_MASKS_DIR
    if not os.path.isdir(mask_dir):
        logger.warning(
            f"IDRiD lesion masks not found at {mask_dir}. "
            f"Skipping explainability evaluation. "
        )
        return {"pointing_game_score": None, "faithfulness_score": None}

    model.eval()
    device = config.DEVICE
    pointing_hits = 0
    pointing_total = 0
    faith_scores = []

    for batch in idrid_loader:
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = ["unknown"] * len(images)

        for i in range(len(images)):
            img_tensor = images[i:i+1].to(device)
            true_label = labels[i].item()

            # Only evaluate on images with DR (Grade > 0)
            if true_label == 0:
                continue

            # Try to load corresponding lesion mask
            if i < len(paths):
                img_name = os.path.splitext(os.path.basename(paths[i]))[0]
            else:
                continue

            # Look for any lesion mask for this image
            mask_found = False
            combined_mask = np.zeros(
                (config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8
            )
            for lesion_type in [
                "Microaneurysms", "Haemorrhages",
                "Hard Exudates", "Soft Exudates",
            ]:
                mask_path = os.path.join(
                    mask_dir, lesion_type, f"{img_name}_{lesion_type}.tif"
                )
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(
                            mask, (config.IMG_SIZE, config.IMG_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        combined_mask = np.maximum(combined_mask, mask)
                        mask_found = True

            if not mask_found:
                continue

            # Compute heatmaps
            has_gradcam = hasattr(model, "get_cnn_features")
            if has_gradcam:
                cam = compute_gradcam_pp(model, img_tensor, true_label, device)
            else:
                cam = np.zeros((config.IMG_SIZE, config.IMG_SIZE))

            # Pointing game
            hit = pointing_game_score(cam, combined_mask)
            pointing_hits += int(hit)
            pointing_total += 1

            # Faithfulness
            faith = faithfulness_score(
                model, img_tensor, cam, true_label, device
            )
            faith_scores.append(faith)

    result = {
        "pointing_game_hits": pointing_hits,
        "pointing_game_total": pointing_total,
        "pointing_game_score": (
            round(pointing_hits / pointing_total, 4)
            if pointing_total > 0 else None
        ),
        "faithfulness_mean": (
            round(float(np.mean(faith_scores)), 4)
            if faith_scores else None
        ),
        "faithfulness_std": (
            round(float(np.std(faith_scores)), 4)
            if faith_scores else None
        ),
    }

    # Save results
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(
        os.path.join(eval_dir, f"explainability_{variant_name}.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Explainability [{variant_name}]: "
        f"Pointing game={result['pointing_game_score']}, "
        f"Faithfulness={result['faithfulness_mean']}"
    )

    return result
