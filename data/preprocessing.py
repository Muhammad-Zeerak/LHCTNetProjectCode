import cv2
import numpy as np
from typing import Tuple

import config


def circular_crop(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to find the fundus region (non-black)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)

    # Find contours to get the bounding circle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Use the largest contour
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    # Create circular mask
    circ_mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(circ_mask, (cx, cy), radius, 255, -1)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=circ_mask)

    # Crop to bounding box of circle for efficiency
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(image.shape[1], cx + radius)
    y2 = min(image.shape[0], cy + radius)
    result = result[y1:y2, x1:x2]

    if result.size == 0:
        return image

    return result


def apply_clahe_green_channel(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    g_clahe = clahe.apply(g)
    return cv2.merge([b, g_clahe, r])


def preprocess_fundus_image(
    image_path: str,
    target_size: Tuple[int, int] = (config.IMG_SIZE, config.IMG_SIZE),
) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Step 1: Circular crop to remove black borders
    image = circular_crop(image)

    # Step 2: CLAHE on green channel
    image = apply_clahe_green_channel(image)

    # Step 3: Resize using bilinear interpolation
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Step 4: Convert BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
