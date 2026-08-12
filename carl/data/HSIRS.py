"""HSIRS 33-band hyperspectral food segmentation dataset.

Dataset location: \\\\43.3.248.92\\euispc\\mactis\\dataset\\HSIRS_public
Format: 592 scenes, each folder containing:
  - 33 band images: {scene}_{wavelength}_nm.png (474-698nm, 7nm step)
  - 1 seg map: {scene}_seg_map.png (uint8, classes 0-41)
  - 1 RGB: {scene}_rgb.png
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NORMALIZATION_EPSILON = 1e-6

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
    ):
        super().__init__()
        if cfg is None:
            raise ValueError("Configuration (cfg) is required")

        self.cfg = cfg
        self.n_classes = cfg["model_kwargs"]["n_classes"]
        self.image_size = image_size
        self.wavelengths = np.array(WAVELENGTHS_UM, dtype=np.float32)

        root = Path(root_dir)

        # Check if split subfolders exist
        split_dir = root / split
        if split_dir.is_dir():
            scene_dir = split_dir
        else:
            # No split folders — create deterministic split from all scenes
            scene_dir = root

        # Discover all scene folders (must contain seg_map)
        all_scenes = sorted([
            d for d in scene_dir.iterdir()
            if d.is_dir() and list(d.glob("*_seg_map.png"))
        ])

        if split_dir.is_dir():
            self.scenes = all_scenes
        else:
            # Deterministic split: 70/15/15
            n = len(all_scenes)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            if split == "train":
                self.scenes = all_scenes[:n_train]
            elif split == "val":
                self.scenes = all_scenes[n_train:n_train + n_val]
            else:  # test
                self.scenes = all_scenes[n_train + n_val:]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_path = self.scenes[idx]
        scene_name = scene_path.name

        # Load 33 bands
        bands = []
        for wl_nm in WAVELENGTHS_NM:
            img_path = scene_path / f"{scene_name}_{wl_nm}_nm.png"
            band = np.array(Image.open(img_path), dtype=np.float32)
            bands.append(band)
        img = np.stack(bands, axis=0)  # (33, H, W)

        # Load segmentation map
        seg_path = list(scene_path.glob("*_seg_map.png"))[0]
        label = np.array(Image.open(seg_path), dtype=np.int64)

        # Random crop (train) or center crop to image_size
        img, label = self._crop(img, label)

        # Per-image normalization
        mean, std = img.mean(), img.std()
        img = (img - mean) / (std + NORMALIZATION_EPSILON)

        img_tensor = torch.from_numpy(img)
        wl_tensor = torch.from_numpy(self.wavelengths.copy())
        label_tensor = torch.from_numpy(label)

        return img_tensor, wl_tensor, label_tensor

    def _crop(
        self, img: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop for training, center crop otherwise."""
        _, h, w = img.shape
        s = self.image_size

        if h <= s or w <= s:
            # Resize if image is smaller than target
            from PIL import Image as PILImage
            resized_bands = []
            for c in range(img.shape[0]):
                band_pil = PILImage.fromarray(img[c])
                band_pil = band_pil.resize((s, s), PILImage.BILINEAR)
                resized_bands.append(np.array(band_pil, dtype=np.float32))
            img = np.stack(resized_bands, axis=0)
            label_pil = PILImage.fromarray(label.astype(np.uint8))
            label_pil = label_pil.resize((s, s), PILImage.NEAREST)
            label = np.array(label_pil, dtype=np.int64)
        else:
            # Random crop
            top = np.random.randint(0, h - s)
            left = np.random.randint(0, w - s)
            img = img[:, top:top + s, left:left + s]
            label = label[top:top + s, left:left + s]

        return img, label
