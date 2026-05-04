import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config
from data.preprocessing import preprocess_fundus_image
from utils.logging_utils import get_logger

logger = get_logger("dataset")


class FundusDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = preprocess_fundus_image(path)
        except (ValueError, FileNotFoundError):
            logger.warning(f"Cannot load image, using black placeholder: {path}")
            image = np.zeros(
                (config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8
            )

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return image, label, path
        return image, label


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB),
        transforms.RandomVerticalFlip(p=config.AUG_VFLIP_PROB),
        transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS_FACTOR,
            contrast=config.AUG_CONTRAST_FACTOR,
        ),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def load_aptos_splits(
    seed: int = config.SEED,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    df = pd.read_csv(config.APTOS_TRAIN_CSV)
    # Column names: id_code, diagnosis
    image_paths = [
        os.path.join(config.APTOS_TRAIN_IMAGES, f"{row['id_code']}.png")
        for _, row in df.iterrows()
    ]
    labels = df["diagnosis"].tolist()

    # First split: train (70%) vs temp (30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val (50% of temp = 15%) vs test (50% of temp = 15%)
    val_ratio_of_temp = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - val_ratio_of_temp),
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info(
        f"APTOS splits — Train: {len(train_paths)}, "
        f"Val: {len(val_paths)}, Test: {len(test_paths)}"
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def load_idrid_data() -> Tuple[List[str], List[int]]:
    all_paths = []
    all_labels = []

    def _find_label_file(primary: str) -> Optional[str]:
        if os.path.isfile(primary):
            return primary
        no_ext = primary.replace(".csv", "")
        if os.path.isfile(no_ext):
            return no_ext
        parent = os.path.dirname(primary)
        base = os.path.basename(primary).replace(".csv", "")
        if os.path.isdir(parent):
            for f in os.listdir(parent):
                if f.startswith(base):
                    return os.path.join(parent, f)
        return None

    # Training partition
    train_file = _find_label_file(config.IDRID_TRAIN_LABELS)
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        img_col = df_train.columns[0]
        grade_col = df_train.columns[1]
        for _, row in df_train.iterrows():
            img_name = str(row[img_col]).strip()
            img_path = os.path.join(config.IDRID_TRAIN_IMAGES, f"{img_name}.jpg")
            if os.path.exists(img_path):
                all_paths.append(img_path)
                all_labels.append(int(row[grade_col]))

    # Testing partition
    test_file = _find_label_file(config.IDRID_TEST_LABELS)
    if test_file is not None:
        df_test = pd.read_csv(test_file)
        img_col = df_test.columns[0]
        grade_col = df_test.columns[1]
        for _, row in df_test.iterrows():
            img_name = str(row[img_col]).strip()
            img_path = os.path.join(config.IDRID_TEST_IMAGES, f"{img_name}.jpg")
            if os.path.exists(img_path):
                all_paths.append(img_path)
                all_labels.append(int(row[grade_col]))

    logger.info(f"IDRiD loaded: {len(all_paths)} images")
    return all_paths, all_labels


def load_eyepacs_subset(
    subset_size: int = config.EYEPACS_SUBSET_SIZE,
    seed: int = config.SEED,
) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(config.EYEPACS_LABELS_CSV)
    img_col = df.columns[0]  # 'image'
    grade_col = df.columns[1]  # 'level'

    valid_rows = []
    for _, row in df.iterrows():
        img_name = str(row[img_col]).strip()
        img_path = os.path.join(config.EYEPACS_IMAGES, f"{img_name}.jpeg")
        if os.path.exists(img_path):
            valid_rows.append((img_path, int(row[grade_col])))

    if len(valid_rows) == 0:
        logger.warning("No valid EyePACS images found. Check paths.")
        return [], []

    all_paths = [r[0] for r in valid_rows]
    all_labels = [r[1] for r in valid_rows]

    # Stratified subset
    if len(all_paths) <= subset_size:
        logger.info(f"EyePACS: using all {len(all_paths)} available images")
        return all_paths, all_labels

    _, subset_paths, _, subset_labels = train_test_split(
        all_paths, all_labels,
        test_size=subset_size,
        stratify=all_labels,
        random_state=seed,
    )

    logger.info(f"EyePACS subset: {len(subset_paths)} images sampled")
    return subset_paths, subset_labels


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * config.NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(
        np.float64
    )
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def create_dataloaders(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    batch_size: int = config.BATCH_SIZE,
) -> Dict[str, DataLoader]:
    train_ds = FundusDataset(
        train_data[0], train_data[1], transform=get_train_transforms()
    )
    val_ds = FundusDataset(
        val_data[0], val_data[1], transform=get_eval_transforms()
    )
    test_ds = FundusDataset(
        test_data[0], test_data[1], transform=get_eval_transforms(),
        return_path=True,
    )

    sampler = get_weighted_sampler(train_data[1])

    nw = config.NUM_WORKERS
    pw = config.PERSISTENT_WORKERS and nw > 0

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=pw,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=pw,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=pw,
        ),
    }

    return loaders


def create_cross_dataset_loader(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = config.BATCH_SIZE,
    return_path: bool = True,
) -> DataLoader:
    ds = FundusDataset(
        image_paths, labels,
        transform=get_eval_transforms(),
        return_path=return_path,
    )
    nw = config.NUM_WORKERS
    pw = config.PERSISTENT_WORKERS and nw > 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=pw,
    )


def print_dataset_statistics(
    split_name: str,
    labels: List[int],
) -> Dict[str, int]:
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    total = len(labels)
    stats = {"total": total}
    logger.info(f"  {split_name}: {total} images")
    for i, name in enumerate(config.CLASS_NAMES):
        pct = 100.0 * counts[i] / total if total > 0 else 0.0
        logger.info(f"    {name}: {counts[i]} ({pct:.1f}%)")
        stats[f"class_{i}_count"] = int(counts[i])
        stats[f"class_{i}_pct"] = round(pct, 1)
    return stats
