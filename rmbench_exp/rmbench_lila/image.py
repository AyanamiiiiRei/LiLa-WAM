"""Image preprocessing shared by training, VTT and inference.

Using the SAME function everywhere is what guarantees that the policy sees images
at eval time exactly as it saw them during training.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_rgb(img: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """image_size is (W, H). INTER_LINEAR, same as the main project."""
    w, h = image_size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def to_model_input(img: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """HxWx3 uint8 RGB -> 3xHxW float32, resized and ImageNet-normalised."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {img.shape} {img.dtype}")
    x = resize_rgb(img, image_size).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def from_model_input(x: np.ndarray) -> np.ndarray:
    """Inverse of ``to_model_input`` (for visual checks): 3xHxW float -> HxWx3 uint8 RGB."""
    img = np.transpose(np.asarray(x, dtype=np.float32), (1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
