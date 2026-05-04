import json
import os
from typing import Any, Dict, Optional

import torch

from utils.logging_utils import get_logger

logger = get_logger("checkpoint")


def save_checkpoint(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    best_val_metric: float,
    config_dict: Dict[str, Any],
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "config": config_dict,
    }
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimiser: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimiser is not None and "optimizer_state_dict" in checkpoint:
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        f"Checkpoint loaded: {filepath} (epoch {checkpoint.get('epoch', '?')})"
    )
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_metric": checkpoint.get("best_val_metric", 0.0),
    }


def save_config_snapshot(config_dict: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert non-serialisable types
    clean = {}
    for k, v in config_dict.items():
        if isinstance(v, torch.device):
            clean[k] = str(v)
        elif isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))):
            clean[k] = v
        else:
            clean[k] = str(v)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Config snapshot saved: {filepath}")
