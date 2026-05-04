import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as tv_models

import config
from utils.logging_utils import get_logger

logger = get_logger("model")


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        cnn_channels: int = 1280,
        transformer_channels: int = 768,
        embed_dim: int = config.FUSION_DIM,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 1×1 convolutions for projection
        self.query_proj = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(transformer_channels, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(transformer_channels, embed_dim, kernel_size=1)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

        self.scale = math.sqrt(self.head_dim)

        # Store attention weights for explainability
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = cnn_features.shape

        # Project to shared dimension
        Q = self.query_proj(cnn_features)    # (B, D, H, W)
        K = self.key_proj(transformer_features)  # (B, D, H, W)
        V = self.value_proj(transformer_features)  # (B, D, H, W)

        # Reshape for multi-head attention: (B, num_heads, H*W, head_dim)
        Q = Q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, nh, HW, HW)
        attn = F.softmax(attn, dim=-1)

        # Store for explainability visualisation
        self._attention_weights = attn.detach()

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, nh, HW, hd)
        out = out.permute(0, 1, 3, 2).reshape(B, self.embed_dim, H, W)
        out = self.out_proj(out)

        # Residual connection with projected CNN features
        # Re-project CNN features to match output dimension for clean residual path
        residual = self.query_proj(cnn_features)
        out = out + residual

        # LayerNorm (applied per-spatial-location)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, D)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, D, H, W)

        return out, attn

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._attention_weights


class LHCTNet(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        fusion_dim: int = config.FUSION_DIM,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        dropout: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Stream: EfficientNet-B0
        efficientnet = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Remove classifier head, keep feature extractor
        self.cnn_stream = efficientnet.features  # Output: (B, 1280, 7, 7)
        self.cnn_out_channels = 1280

        # Transformer Stream: Swin-Tiny
        self.swin_stream = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
        )
        self.swin_out_channels = 768  # Swin-Tiny final stage output

        # Cross-Attention Fusion
        self.cross_attention = CrossAttentionFusion(
            cnn_channels=self.cnn_out_channels,
            transformer_channels=self.swin_out_channels,
            embed_dim=fusion_dim,
        )

        # Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # For Grad-CAM++ — store intermediate features
        self._cnn_features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # CNN stream
        cnn_feat = self.cnn_stream(x)  # (B, 1280, 7, 7)
        self._cnn_features = cnn_feat

        # Register gradient hook for Grad-CAM++
        if cnn_feat.requires_grad:
            cnn_feat.register_hook(self._save_gradient)

        # Swin Transformer stream — features_only returns list of stage outputs
        swin_stages = self.swin_stream(x)
        swin_feat = swin_stages[-1]
        # timm Swin features_only outputs (B, H, W, C) — convert to (B, C, H, W)
        if swin_feat.dim() == 4 and swin_feat.shape[-1] != swin_feat.shape[-2]:
            swin_feat = swin_feat.permute(0, 3, 1, 2)

        # Cross-attention fusion
        fused, attn_weights = self.cross_attention(cnn_feat, swin_feat)

        # Classification
        pooled = self.global_pool(fused).flatten(1)  # (B, fusion_dim)
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits, attn_weights

    def get_cnn_features(self) -> Optional[torch.Tensor]:
        return self._cnn_features

    def get_gradients(self) -> Optional[torch.Tensor]:
        return self._gradients

    def count_parameters(self) -> Dict[str, int]:
        cnn_params = sum(p.numel() for p in self.cnn_stream.parameters())
        swin_params = sum(p.numel() for p in self.swin_stream.parameters())
        fusion_params = sum(p.numel() for p in self.cross_attention.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "cnn_stream": cnn_params,
            "swin_stream": swin_params,
            "cross_attention": fusion_params,
            "classifier": classifier_params,
            "total": total,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


class EfficientNetOnly(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        dropout: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        efficientnet = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = efficientnet.features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._cnn_features = None
        self._gradients = None

    def _save_gradient(self, grad):
        self._gradients = grad

    def forward(self, x):
        feat = self.features(x)
        self._cnn_features = feat
        if feat.requires_grad:
            feat.register_hook(self._save_gradient)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits, None

    def get_cnn_features(self):
        return self._cnn_features

    def get_gradients(self):
        return self._gradients

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}


class SwinOnly(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        dropout: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        stages = self.swin(x)
        feat = stages[-1]
        # timm Swin features_only outputs (B, H, W, C) — convert to (B, C, H, W)
        if feat.dim() == 4 and feat.shape[-1] != feat.shape[-2]:
            feat = feat.permute(0, 3, 1, 2)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits, None

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}


class LHCTNetConcat(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        fusion_dim: int = config.FUSION_DIM,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        dropout: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        efficientnet = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.cnn_stream = efficientnet.features

        self.swin_stream = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
        )

        # Concatenation fusion: project concatenated features to fusion_dim
        self.concat_proj = nn.Sequential(
            nn.Conv2d(1280 + 768, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        cnn_feat = self.cnn_stream(x)
        swin_stages = self.swin_stream(x)
        swin_feat = swin_stages[-1]
        # timm Swin features_only outputs (B, H, W, C) — convert to (B, C, H, W)
        if swin_feat.dim() == 4 and swin_feat.shape[-1] != swin_feat.shape[-2]:
            swin_feat = swin_feat.permute(0, 3, 1, 2)

        # Simple concatenation fusion
        concat = torch.cat([cnn_feat, swin_feat], dim=1)  # (B, 2048, 7, 7)
        fused = self.concat_proj(concat)

        pooled = self.global_pool(fused).flatten(1)
        logits = self.classifier(pooled)
        return logits, None

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}


def build_model(variant: str = "lhctnet_full") -> nn.Module:
    if variant == "efficientnet_only":
        model = EfficientNetOnly()
    elif variant == "swin_only":
        model = SwinOnly()
    elif variant == "lhctnet_concat":
        model = LHCTNetConcat()
    elif variant == "lhctnet_full":
        model = LHCTNet()
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    params = model.count_parameters()
    logger.info(
        f"Built model '{variant}' — "
        f"Total parameters: {params['total']:,}"
    )
    return model