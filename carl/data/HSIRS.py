"""HSIRS 33-band hyperspectral food segmentation dataset.

Dataset location: \\\\43.3.248.92\\euispc\\mactis\\dataset\\HSIRS_public
Format: 592 scenes, each folder containing:
  - 33 band images: {scene}_{wavelength}_nm.png (474-698nm, 7nm step)
  - 1 seg map: {scene}_seg_map.png (uint8, classes 0-41)
  - 1 RGB: {scene}_rgb.png
"""

import logging
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NORMALIZATION_EPSILON = 1e-6
BACKGROUND_LABEL = 0
FIRST_FOREGROUND_LABEL = 1
LAST_FOREGROUND_LABEL = 40
UNKNOWN_LABEL = 41
IGNORE_LABEL = -1
SPLIT_NAMES = ("train", "val", "test")
SPLIT_LOG_NAMES = ("train", "validation", "test")
SPLIT_COUNTS = (414, 88, 90)

# 33 bands: 474nm to 698nm, 7nm step
WAVELENGTHS_NM = list(range(474, 699, 7))  # [474, 481, ..., 698]
WAVELENGTHS_UM = [w / 1000.0 for w in WAVELENGTHS_NM]  # μm for CARL


class HSIRS(Dataset):
    """HSIRS 33-band hyperspectral segmentation dataset.

    Each scene folder contains 33 per-band PNG images + a segmentation map.
    Images are 2048x2048 uint8. Segmentation labels are 0-41 (42 classes).

    Args:
        root_dir: Path to HSIRS_public folder (or a split subfolder).
        split: 'train', 'val', or 'test'. If the root_dir itself contains
               scene folders (no split subfolders), a deterministic split
               is created: 70% train, 15% val, 15% test.
        cfg: Configuration dict (must contain model_kwargs.n_classes).
        image_size: Target crop/resize size (default 128 for CARL patch_size=8).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        cfg: Optional[Dict[str, Any]] = None,
        image_size: int = 128,
        foreground_crop_probability: float = 0.8,
        foreground_crop_seed: int = 42,
        split_seed: int = 42,
        background_label: int = 0,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError("Configuration (cfg) is required")

        self.cfg = cfg
        self.n_classes = cfg["model_kwargs"]["n_classes"]
        self.image_size = image_size
        self.split = split
        self.foreground_crop_probability = foreground_crop_probability
        self.foreground_crop_seed = foreground_crop_seed
        self.split_seed = split_seed
        self.background_label = background_label
        self.wavelengths = np.array(WAVELENGTHS_UM, dtype=np.float32)

        if not 0.0 <= foreground_crop_probability <= 1.0:
            raise ValueError("foreground_crop_probability must be between 0 and 1")

        root = Path(root_dir)

        all_scenes = tuple(sorted([
            d for d in root.iterdir()
            if d.is_dir() and list(d.glob("*_seg_map.png"))
        ]))
        split_scenes = self._build_stratified_splits(
            str(root.resolve()), all_scenes, split_seed
        )
        self.scenes = list(split_scenes[SPLIT_NAMES.index(split)])

        self.samples = self._build_samples()
        self.foreground_sample_indices = self._build_foreground_sample_indices()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        scene_idx = self.samples[idx]
        scene_path = self.scenes[scene_idx]
        scene_name = scene_path.name

        seg_path = next(scene_path.glob("*_seg_map.png"))
        with Image.open(seg_path) as seg_image:
            full_label = np.array(seg_image, dtype=np.int64)
        full_label[full_label == UNKNOWN_LABEL] = IGNORE_LABEL

        if self.split == "train":
            top, left = self._sample_train_crop(
                full_label,
                force_foreground=idx in self.foreground_sample_indices,
            )
            label = np.array(
                full_label[top:top + self.image_size, left:left + self.image_size],
                copy=True,
            )
        else:
            label = full_label

        # Load 33 bands
        bands = []
        for wl_nm in WAVELENGTHS_NM:
            img_path = scene_path / f"{scene_name}_{wl_nm}_nm.png"
            with Image.open(img_path) as band_image:
                if self.split == "train":
                    band_image = band_image.crop(
                        (left, top, left + self.image_size, top + self.image_size)
                    )
                    band = np.array(band_image, dtype=np.float32)
                else:
                    band = np.array(band_image, dtype=np.uint8)
            bands.append(band)
        img = np.stack(bands, axis=0)  # (33, H, W)

        if self.split == "train":
            mean, std = img.mean(), img.std()
            img = (img - mean) / (std + NORMALIZATION_EPSILON)

        img_tensor = torch.from_numpy(img)
        wl_tensor = torch.from_numpy(self.wavelengths.copy())
        label_tensor = torch.from_numpy(label)

        if self.split == "train":
            return img_tensor, wl_tensor, label_tensor

        metadata = {
            "scene_name": scene_name,
            "original_height": label.shape[0],
            "original_width": label.shape[1],
        }
        return img_tensor, wl_tensor, label_tensor, metadata

    def _build_samples(self) -> list:
        """Build one sample per scene."""
        return list(range(len(self.scenes)))

    def _build_foreground_sample_indices(self) -> set:
        """Assign an exact fraction of train samples to foreground-aware cropping."""
        if self.split != "train":
            return set()
        num_foreground_samples = round(
            len(self.samples) * self.foreground_crop_probability
        )
        rng = np.random.default_rng(self.foreground_crop_seed)
        indices = rng.permutation(len(self.samples))[:num_foreground_samples]
        return set(indices.tolist())

    def _sample_train_crop(
        self,
        label: np.ndarray,
        force_foreground: bool,
    ) -> Tuple[int, int]:
        """Sample either a foreground-containing or fully random crop."""
        height, width = label.shape
        max_top = max(height - self.image_size, 0)
        max_left = max(width - self.image_size, 0)

        foreground_mask = (
            (label >= FIRST_FOREGROUND_LABEL)
            & (label <= LAST_FOREGROUND_LABEL)
        )
        foreground_y, foreground_x = np.where(foreground_mask)
        if force_foreground and foreground_y.size:
            selected = np.random.randint(foreground_y.size)
            pixel_y = int(foreground_y[selected])
            pixel_x = int(foreground_x[selected])
            min_top = max(0, pixel_y - self.image_size + 1)
            min_left = max(0, pixel_x - self.image_size + 1)
            top = np.random.randint(min_top, min(pixel_y, max_top) + 1)
            left = np.random.randint(min_left, min(pixel_x, max_left) + 1)
            return top, left

        top = np.random.randint(max_top + 1)
        left = np.random.randint(max_left + 1)
        return top, left

    @staticmethod
    @lru_cache(maxsize=None)
    def _build_stratified_splits(
        root_dir: str,
        scenes: Tuple[Path, ...],
        split_seed: int,
    ) -> Tuple[Tuple[Path, ...], Tuple[Path, ...], Tuple[Path, ...]]:
        """Create deterministic scene-level iterative multilabel splits."""
        if len(scenes) != sum(SPLIT_COUNTS):
            raise ValueError(
                f"Expected {sum(SPLIT_COUNTS)} HSIRS scenes, found {len(scenes)} "
                f"under {root_dir}"
            )

        pixel_counts = np.zeros((len(scenes), LAST_FOREGROUND_LABEL), dtype=np.int64)
        for scene_idx, scene in enumerate(scenes):
            with Image.open(next(scene.glob("*_seg_map.png"))) as image:
                histogram = np.asarray(image.histogram(), dtype=np.int64)
            pixel_counts[scene_idx] = histogram[
                FIRST_FOREGROUND_LABEL:LAST_FOREGROUND_LABEL + 1
            ]
        presence = pixel_counts > 0
        assignments = HSIRS._iterative_multilabel_assignment(
            presence, np.asarray(SPLIT_COUNTS), split_seed
        )
        split_scenes = tuple(
            tuple(scenes[idx] for idx in np.flatnonzero(assignments == split_idx))
            for split_idx in range(len(SPLIT_NAMES))
        )

        logging.info("Ignored raw label: 41 -> -1")
        for split_idx, split_name in enumerate(SPLIT_LOG_NAMES):
            indices = assignments == split_idx
            class_scene_counts = presence[indices].sum(axis=0)
            class_pixel_counts = pixel_counts[indices].sum(axis=0)
            present_classes = (
                np.flatnonzero(class_scene_counts) + FIRST_FOREGROUND_LABEL
            ).tolist()
            missing_classes = (
                np.flatnonzero(class_scene_counts == 0) + FIRST_FOREGROUND_LABEL
            ).tolist()
            logging.info("Number of %s scenes: %d", split_name, indices.sum())
            logging.info("Classes present in %s: %s", split_name, present_classes)
            logging.info(
                "Class 1-40 scene counts in %s: %s",
                split_name,
                class_scene_counts.tolist(),
            )
            logging.info(
                "Class 1-40 pixel counts in %s: %s",
                split_name,
                class_pixel_counts.tolist(),
            )
            logging.info("Classes missing in %s: %s", split_name, missing_classes)
        return split_scenes

    @staticmethod
    def _iterative_multilabel_assignment(
        presence: np.ndarray,
        split_counts: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Assign rare labels first while matching split sizes and label ratios."""
        rng = np.random.default_rng(seed)
        num_scenes, num_classes = presence.shape
        assignments = np.full(num_scenes, -1, dtype=np.int8)
        remaining_capacity = split_counts.astype(np.int64).copy()
        target_label_counts = np.outer(
            split_counts / split_counts.sum(), presence.sum(axis=0)
        )
        remaining_label_need = target_label_counts.copy()
        unassigned = np.ones(num_scenes, dtype=bool)

        while np.any(unassigned & presence.any(axis=1)):
            remaining_frequency = presence[unassigned].sum(axis=0)
            available_labels = np.flatnonzero(remaining_frequency > 0)
            rarest_frequency = remaining_frequency[available_labels].min()
            rarest_labels = available_labels[
                remaining_frequency[available_labels] == rarest_frequency
            ]
            label = int(rng.choice(rarest_labels))
            candidates = np.flatnonzero(unassigned & presence[:, label])
            rng.shuffle(candidates)

            for scene_idx in candidates:
                eligible = np.flatnonzero(remaining_capacity > 0)
                label_need = remaining_label_need[eligible, label]
                best = eligible[label_need == label_need.max()]
                if len(best) > 1:
                    capacity = remaining_capacity[best]
                    best = best[capacity == capacity.max()]
                split_idx = int(rng.choice(best))
                assignments[scene_idx] = split_idx
                unassigned[scene_idx] = False
                remaining_capacity[split_idx] -= 1
                remaining_label_need[split_idx] -= presence[scene_idx]

        leftovers = np.flatnonzero(unassigned)
        rng.shuffle(leftovers)
        for scene_idx in leftovers:
            eligible = np.flatnonzero(remaining_capacity > 0)
            split_idx = int(rng.choice(eligible))
            assignments[scene_idx] = split_idx
            remaining_capacity[split_idx] -= 1

        if np.any(remaining_capacity) or np.any(assignments < 0):
            raise RuntimeError("Failed to create exact-size HSIRS splits")
        return assignments
