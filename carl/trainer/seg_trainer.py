"""Semantic segmentation trainer using CARL model with PyTorch Lightning.

This module implements a Lightning trainer for fine-tuning the CARL model's 
classification head for semantic segmentation tasks on spectral imagery.
"""

from typing import Tuple, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from kornia import augmentation as K
from torch import Tensor

from carl.model.carl import CARLModel
from carl.config import ConfigAccessor
from carl.trainer.constants import (
    DEFAULT_CROP_SCALE,
    DEFAULT_DROPOUT_PROB,
    EPSILON,
)


class LinearTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for CARL semantic segmentation.
    
    This trainer:
    - Freezes the pre-trained CARL model parameters
    - Trains only a small classification head on top
    - Uses data augmentation (crop, flip) for robustness
    - Logs mean IoU as the validation metric
    
    Attributes:
        model: Frozen pre-trained CARL model
        classifier: Trainable classification head
        conf_mat: Confusion matrix for IoU computation
        loss_fun: Cross-entropy loss function
        transforms: Data augmentation pipeline
    """
    
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        """Initialize the semantic segmentation trainer.
        
        Args:
            config: Configuration dictionary containing model_kwargs.
            *args: Additional positional arguments for LightningModule.
            **kwargs: Additional keyword arguments for LightningModule.
        """
        super().__init__(*args, **kwargs)

        # Use ConfigAccessor for safer configuration access
        self.config_accessor = ConfigAccessor(config)
        
        model_kwargs = self.config_accessor.get("model_kwargs", {})
        self.n_classes = model_kwargs["n_classes"]
        self.background_label = self.config_accessor.get(
            "metric_kwargs/background_label", 0
        )
        self.evaluation_tile_size = self.config_accessor.get(
            "evaluation_kwargs/tile_size", model_kwargs["image_size"]
        )
        self.evaluation_tile_batch_size = self.config_accessor.get(
            "evaluation_kwargs/tile_batch_size", 4
        )

        # Load pre-trained CARL model and freeze its parameters
        self.model = CARLModel(**model_kwargs)
        self._freeze_model_parameters()

        # Create trainable classification head
        self.classifier = nn.Conv2d(
            self.model.spatial_encoder.embed_dim,
            self.n_classes,
            kernel_size=1,
            stride=1
        )

        # Metrics for validation
        self.conf_mat = torchmetrics.ConfusionMatrix(
            num_classes=self.n_classes,
            task="multiclass",
            ignore_index=-1
        )

        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-1)

        # Data augmentation pipeline
        self.transforms = self._build_augmentation_pipeline()

    def _freeze_model_parameters(self) -> None:
        """Freeze all CARL model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True) -> "LinearTrainer":
        """Set trainer mode while keeping the frozen CARL backbone in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def _build_augmentation_pipeline(self) -> K.AugmentationSequential:
        """Build data augmentation pipeline.
        
        Returns:
            Augmentation pipeline with crop and flip operations.
        """
        img_size = self.config_accessor.get("model_kwargs/image_size")
        augmentations = []
        if self.config_accessor.get("training_kwargs/random_resized_crop", True):
            augmentations.append(K.RandomResizedCrop(
                (img_size, img_size),
                scale=DEFAULT_CROP_SCALE,
                p=1.0,
            ))
        augmentations.extend([
            K.RandomHorizontalFlip(p=DEFAULT_DROPOUT_PROB),
            K.RandomVerticalFlip(p=DEFAULT_DROPOUT_PROB),
        ])
        transforms = K.AugmentationSequential(
            *augmentations,
            data_keys=["input", "mask"],
            same_on_batch=False,
        )
        
        # Set device and dtype for all augmentations
        for idx in range(len(transforms)):
            transforms[idx].set_rng_device_and_dtype(
                device="cuda", 
                dtype=torch.float64
            )
        
        return transforms

    def training_step(
        self, 
        batch: Tuple[Tensor, Tensor, Tensor], 
        batch_idx: int
    ) -> Tensor:
        """Training step for one batch.
        
        Args:
            batch: Tuple containing:
                - img: Input image tensor of shape (B, C, H, W)
                - wavelengths: Wavelength tensor of shape (B, N)
                - labels: Target labels of shape (B, H, W)
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        img, wavelengths, labels = batch
        
        # Ensure correct dtype
        img = img.to(dtype=torch.float32)
        wavelengths = wavelengths.to(dtype=torch.float32)
        
        # Apply augmentations
        img, labels = self.transforms(img, labels)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels = labels.long()

        # Forward pass through CARL model
        spat_feat, spec_feat = self.model(img, wavelengths)
        predictions = self.classifier(spat_feat)

        # Resize predictions to match labels if needed
        predictions = self._resize_to_target(predictions, labels.shape[-2:])

        # Compute loss
        loss = self.loss_fun(predictions.float(), labels)
        self.log('train/loss', loss, on_epoch=True, on_step=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: Tuple[Tensor, Tensor, Tensor], 
        batch_idx: int
    ) -> None:
        """Validation step for one batch.
        
        Args:
            batch: Tuple containing:
                - img: Input image tensor of shape (B, C, H, W)
                - wavelengths: Wavelength tensor of shape (B, N)
                - labels: Target labels of shape (B, H, W)
            batch_idx: Index of the current batch.
        """
        self._evaluation_step(batch)

    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        """Evaluate one test batch."""
        self._evaluation_step(batch)

    def _evaluation_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Update the evaluation confusion matrix for one batch."""
        if len(batch) == 4:
            img, wavelengths, labels, metadata = batch
        else:
            img, wavelengths, labels = batch
            metadata = None
        
        # Ensure correct dtype
        img = img.to(dtype=torch.float32)
        wavelengths = wavelengths.to(dtype=torch.float32)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels = labels.long()

        if metadata is not None:
            self._evaluate_tiled_scenes(img, wavelengths, labels)
            return

        # Forward pass through CARL model
        spat_feat, spec_feat = self.model(img, wavelengths)
        predictions = self.classifier(spat_feat)

        # Resize predictions to match labels
        predictions = self._resize_to_target(predictions, labels.shape[-2:])

        self.conf_mat.update(predictions, labels)

    def _evaluate_tiled_scenes(
        self,
        images: Tensor,
        wavelengths: Tensor,
        labels: Tensor,
    ) -> None:
        """Infer every tile and restore full-resolution prediction maps."""
        tile_size = self.evaluation_tile_size
        for scene_idx in range(images.shape[0]):
            image = images[scene_idx]
            height, width = image.shape[-2:]
            tile_specs = [
                (
                    top,
                    left,
                    min(tile_size, height - top),
                    min(tile_size, width - left),
                )
                for top in range(0, height, tile_size)
                for left in range(0, width, tile_size)
            ]
            prediction_map = torch.empty(
                (height, width), dtype=torch.uint8, device="cpu"
            )

            for start in range(0, len(tile_specs), self.evaluation_tile_batch_size):
                batch_specs = tile_specs[start:start + self.evaluation_tile_batch_size]
                tiles = []
                for top, left, valid_height, valid_width in batch_specs:
                    tile = image[:, top:top + valid_height, left:left + valid_width]
                    tile = F.pad(
                        tile,
                        (0, tile_size - valid_width, 0, tile_size - valid_height),
                    )
                    tiles.append(tile)

                tile_batch = torch.stack(tiles).to(dtype=torch.float32)
                means = tile_batch.mean(dim=(1, 2, 3), keepdim=True)
                stds = tile_batch.std(dim=(1, 2, 3), keepdim=True)
                tile_batch = (tile_batch - means) / (stds + EPSILON)
                wavelength_batch = wavelengths[scene_idx].unsqueeze(0).expand(
                    len(batch_specs), -1
                )

                spatial_features, _ = self.model(tile_batch, wavelength_batch)
                tile_predictions = self.classifier(spatial_features)
                tile_predictions = self._resize_to_target(
                    tile_predictions, (tile_size, tile_size)
                ).argmax(dim=1).to("cpu", dtype=torch.uint8)

                for tile_idx, (top, left, valid_height, valid_width) in enumerate(batch_specs):
                    prediction_map[
                        top:top + valid_height,
                        left:left + valid_width,
                    ] = tile_predictions[tile_idx, :valid_height, :valid_width]

            self.conf_mat.update(
                prediction_map.to(self.device, dtype=torch.long),
                labels[scene_idx],
            )

    def on_validation_epoch_end(self) -> None:
        """Compute and log metrics at the end of validation epoch."""
        self._log_evaluation_metrics("val")

    def on_test_epoch_end(self) -> None:
        """Compute and log metrics at the end of the test epoch."""
        self._log_evaluation_metrics("test")

    def _log_evaluation_metrics(self, prefix: str) -> None:
        """Compute, log, and reset evaluation metrics."""
        cm = self.conf_mat.confmat
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        
        # Compute per-class IoU
        union = tp + fp + fn
        iou = tp / (union + EPSILON)
        present_classes = union > 0
        foreground_classes = present_classes.clone()
        foreground_classes[self.background_label] = False
        mean_iou = iou[present_classes].mean().item()
        foreground_miou = iou[foreground_classes].mean().item()

        self.log(f"{prefix}_mIoU", mean_iou, on_epoch=True)
        self.log(f"{prefix}_foreground_mIoU", foreground_miou, on_epoch=True)
        self.conf_mat.reset()

    def configure_optimizers(self) -> Tuple[list, list]:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Tuple of (optimizers, schedulers) for PyTorch Lightning.
        """
        params = self.classifier.parameters()
        lr = float(
            self.config_accessor.get("training_kwargs/learning_rate")
        )
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            fused=True,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return [optimizer], [lr_scheduler]

    @staticmethod
    def _resize_to_target(
        tensor: Tensor, 
        target_size: Tuple[int, int]
    ) -> Tensor:
        """Resize tensor to target size if needed.
        
        Args:
            tensor: Tensor to resize of shape (B, C, H, W).
            target_size: Target spatial dimensions (H, W).
            
        Returns:
            Resized tensor or original tensor if sizes match.
        """
        if tensor.shape[-2:] != target_size:
            tensor = F.interpolate(
                tensor,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
        return tensor