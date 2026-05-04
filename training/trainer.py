import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from evaluation.metrics import compute_qwk, compute_all_metrics
from training.losses import create_loss_function
from training.optimiser import create_optimiser, create_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint, save_config_snapshot
from utils.logging_utils import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        variant_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        max_epochs: int = config.MAX_EPOCHS,
        device: torch.device = config.DEVICE,
        results_dir: str = config.RESULTS_DIR,
    ):
        self.model = model.to(device)
        self.variant_name = variant_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.results_dir = results_dir

        # Loss function
        self.criterion = create_loss_function(
            class_weights=class_weights, device=device
        )

        # Optimiser and scheduler
        self.optimiser = create_optimiser(model, variant=variant_name)
        self.scheduler = create_scheduler(self.optimiser, max_epochs=max_epochs)

        # Mixed precision
        self.use_amp = config.USE_AMP and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping state
        self.best_val_metric = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

        # Training history
        self.history: List[Dict[str, Any]] = []

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"  Epoch {epoch + 1}/{self.max_epochs} [Train]",
            leave=False,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _ = self.model(images)
                    loss = self.criterion(logits, labels)

                # Check for NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch + 1}")
                    return float("nan"), 0.0

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRAD_CLIP_MAX_NORM
                )
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch + 1}")
                    return float("nan"), 0.0

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRAD_CLIP_MAX_NORM
                )
                self.optimiser.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss, 0.0

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(
            self.val_loader,
            desc="  Validating",
            leave=False,
        )

        for batch in pbar:
            # Handle both (images, labels) and (images, labels, paths)
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _ = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += loss.item()
            num_batches += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = running_loss / max(num_batches, 1)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        val_qwk = compute_qwk(all_labels, all_preds)
        metrics = compute_all_metrics(
            all_labels, all_preds, all_probs, config.NUM_CLASSES
        )
        metrics["qwk"] = val_qwk

        return avg_loss, val_qwk, metrics

    def train(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"TRAINING: {self.variant_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  AMP: {self.use_amp}")
        params = self.model.count_parameters()
        logger.info(f"  Parameters: {params['total']:,}")
        logger.info("=" * 60)

        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        log_dir = os.path.join(self.results_dir, "logs")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        best_path = os.path.join(
            checkpoint_dir, f"best_{self.variant_name}.pt"
        )
        final_path = os.path.join(
            checkpoint_dir, f"final_{self.variant_name}.pt"
        )
        history_path = os.path.join(
            log_dir, f"training_{self.variant_name}.csv"
        )

        nan_restore_count = 0
        start_time = time.time()

        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            # ── Train ──
            train_loss, _ = self.train_one_epoch(epoch)

            # ── Handle NaN loss ──
            if np.isnan(train_loss):
                nan_restore_count += 1
                if nan_restore_count > 3:
                    logger.error(
                        f"NaN loss persists after {nan_restore_count} restores. "
                        f"Stopping training for {self.variant_name}."
                    )
                    break
                # Restore best checkpoint and halve LR
                if os.path.exists(best_path):
                    load_checkpoint(best_path, self.model, self.optimiser, self.device)
                for pg in self.optimiser.param_groups:
                    pg["lr"] *= 0.5
                logger.warning(
                    f"NaN detected, restored checkpoint, halved LR "
                    f"(attempt {nan_restore_count}/3)"
                )
                continue

            # ── Validate ──
            val_loss, val_qwk, val_metrics = self.validate()

            # ── Scheduler step ──
            self.scheduler.step()
            current_lr = self.optimiser.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            # ── Log ──
            logger.info(
                f"Epoch [{epoch + 1}/{self.max_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val QWK: {val_qwk:.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
                f"Val F1: {val_metrics.get('macro_f1', 0):.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ── Record history ──
            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_qwk": round(val_qwk, 6),
                "val_accuracy": round(val_metrics.get("accuracy", 0), 6),
                "val_macro_f1": round(val_metrics.get("macro_f1", 0), 6),
                "val_macro_auc": round(val_metrics.get("macro_auc", 0), 6),
                "learning_rate": current_lr,
                "epoch_time_seconds": round(epoch_time, 2),
            }
            self.history.append(record)

            # ── Checkpointing + Early Stopping ──
            if val_qwk > self.best_val_metric:
                self.best_val_metric = val_qwk
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                config_dict = {
                    k: v for k, v in vars(config).items()
                    if not k.startswith("_") and isinstance(
                        v, (int, float, str, bool, list, tuple)
                    )
                }
                save_checkpoint(
                    self.model, self.optimiser, epoch + 1,
                    val_qwk, config_dict, best_path,
                )
                logger.info(f"  ★ New best QWK: {val_qwk:.4f} at epoch {epoch + 1}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience {config.EARLY_STOPPING_PATIENCE} exhausted). "
                        f"Best QWK: {self.best_val_metric:.4f} at epoch {self.best_epoch}"
                    )
                    break

        total_time = time.time() - start_time

        # Save final checkpoint
        config_dict = {
            k: v for k, v in vars(config).items()
            if not k.startswith("_") and isinstance(
                v, (int, float, str, bool, list, tuple)
            )
        }
        save_checkpoint(
            self.model, self.optimiser, epoch + 1,
            self.best_val_metric, config_dict, final_path,
        )

        # Save training history CSV
        if self.history:
            keys = self.history[0].keys()
            with open(history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
            logger.info(f"Training history saved: {history_path}")

        # Restore best model for subsequent evaluation
        if os.path.exists(best_path):
            load_checkpoint(best_path, self.model, device=self.device)

        summary = {
            "variant": self.variant_name,
            "total_epochs": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_qwk": self.best_val_metric,
            "total_time_seconds": round(total_time, 2),
            "total_time_formatted": _format_time(total_time),
            "parameters": self.model.count_parameters(),
        }

        logger.info(
            f"Training complete for {self.variant_name}: "
            f"Best QWK={self.best_val_metric:.4f} at epoch {self.best_epoch}, "
            f"Total time: {_format_time(total_time)}"
        )

        return summary


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def train_model_with_oom_recovery(
    model: nn.Module,
    variant_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    batch_size: int = config.BATCH_SIZE,
    train_data: Optional[Tuple] = None,
    val_data: Optional[Tuple] = None,
) -> Dict[str, Any]:
    while batch_size >= 4:
        try:
            trainer = Trainer(
                model=model,
                variant_name=variant_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
            )
            return trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                logger.warning(
                    f"OOM detected. Halving batch size to {batch_size}. "
                    f"Clearing GPU cache."
                )
                torch.cuda.empty_cache()

                if train_data is not None and val_data is not None:
                    from data.dataset import create_dataloaders
                    # Rebuild with smaller batch size
                    loaders = create_dataloaders(
                        train_data, val_data,
                        ([], []),  # Dummy test — not needed during training
                        batch_size=batch_size,
                    )
                    train_loader = loaders["train"]
                    val_loader = loaders["val"]
                else:
                    raise RuntimeError(
                        f"OOM with batch_size={batch_size * 2} but cannot "
                        f"rebuild DataLoaders (train_data/val_data not provided)."
                    ) from e
            else:
                raise

    raise RuntimeError(
        f"Cannot train {variant_name}: OOM even with batch_size=4."
    )
