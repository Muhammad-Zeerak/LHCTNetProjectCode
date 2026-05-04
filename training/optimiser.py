from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

import config
from utils.logging_utils import get_logger

logger = get_logger("optimiser")


def create_optimiser(
    model: torch.nn.Module,
    learning_rate: float = config.LEARNING_RATE,
    backbone_lr_ratio: float = config.BACKBONE_LR_RATIO,
    weight_decay: float = config.WEIGHT_DECAY,
    variant: str = "lhctnet_full",
) -> torch.optim.Optimizer:
    backbone_lr = learning_rate * backbone_lr_ratio

    if variant == "lhctnet_full":
        param_groups = [
            {
                "params": model.cnn_stream.parameters(),
                "lr": backbone_lr,
                "name": "cnn_backbone",
            },
            {
                "params": model.swin_stream.parameters(),
                "lr": backbone_lr,
                "name": "swin_backbone",
            },
            {
                "params": model.cross_attention.parameters(),
                "lr": learning_rate,
                "name": "cross_attention",
            },
            {
                "params": model.classifier.parameters(),
                "lr": learning_rate,
                "name": "classifier",
            },
        ]
    elif variant == "lhctnet_concat":
        param_groups = [
            {
                "params": model.cnn_stream.parameters(),
                "lr": backbone_lr,
                "name": "cnn_backbone",
            },
            {
                "params": model.swin_stream.parameters(),
                "lr": backbone_lr,
                "name": "swin_backbone",
            },
            {
                "params": list(model.concat_proj.parameters())
                + list(model.classifier.parameters()),
                "lr": learning_rate,
                "name": "fusion_and_classifier",
            },
        ]
    elif variant == "efficientnet_only":
        param_groups = [
            {
                "params": model.features.parameters(),
                "lr": backbone_lr,
                "name": "efficientnet_backbone",
            },
            {
                "params": model.classifier.parameters(),
                "lr": learning_rate,
                "name": "classifier",
            },
        ]
    elif variant == "swin_only":
        param_groups = [
            {
                "params": model.swin.parameters(),
                "lr": backbone_lr,
                "name": "swin_backbone",
            },
            {
                "params": model.classifier.parameters(),
                "lr": learning_rate,
                "name": "classifier",
            },
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": learning_rate}]

    optimizer = optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        weight_decay=weight_decay,
    )

    logger.info(
        f"AdamW optimiser created — base LR: {learning_rate}, "
        f"backbone LR: {backbone_lr}, weight decay: {weight_decay}"
    )
    return optimizer


def create_scheduler(
    optimiser: torch.optim.Optimizer,
    max_epochs: int = config.MAX_EPOCHS,
    warmup_epochs: int = config.LINEAR_WARMUP_EPOCHS,
    min_lr: float = config.COSINE_MIN_LR,
) -> torch.optim.lr_scheduler._LRScheduler:
    # Linear warm-up from 0 to base LR
    warmup_scheduler = LinearLR(
        optimiser,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Cosine annealing after warm-up
    cosine_scheduler = CosineAnnealingLR(
        optimiser,
        T_max=max_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # Combine: warm-up first, then cosine
    scheduler = SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    logger.info(
        f"LR scheduler: {warmup_epochs}-epoch warm-up + cosine annealing "
        f"(min LR: {min_lr})"
    )
    return scheduler
