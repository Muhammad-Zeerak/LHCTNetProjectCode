import torch
import torch.nn as nn
from typing import Optional

import config


def create_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = config.LABEL_SMOOTHING,
    device: torch.device = config.DEVICE,
) -> nn.CrossEntropyLoss:
    if class_weights is not None:
        class_weights = class_weights.to(device)

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
