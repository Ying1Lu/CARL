"""Evaluate and visualize a trained CARL HSIRS linear-probe checkpoint."""

import argparse
import csv
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from carl.config import load_config
from carl.data.HSIRS import IGNORE_LABEL, HSIRS
from carl.trainer.constants import EPSILON
from carl.trainer.seg_trainer import LinearTrainer


LOGGER = logging.getLogger(__name__)
NUM_CLASSES = 41
CLASS_NAMES = ["Background"] + [f"Class {class_id}" for class_id in range(1, 41)]
CLASS_NAMES[3] = "Real biscuit"
CLASS_NAMES[8] = "Real egg"
CLASS_NAMES[28] = "Fake egg"
REAL_FAKE_PAIRS = ((8, 28, "Egg"),)


def find_latest_checkpoint(log_dir: Path) -> Path:
    """Find the newest best checkpoint, excluding last.ckpt."""
    checkpoints = list(log_dir.glob("carl_*/epoch=*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No best checkpoint found under {log_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_trained_model(config: dict, checkpoint_path: Path, device: torch.device) -> LinearTrainer:
    """Strictly load the trained backbone and classifier."""
    model = LinearTrainer(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


@torch.inference_mode()
def predict_scene(
    model: LinearTrainer,
    image: torch.Tensor,
    wavelengths: torch.Tensor,
    tile_size: int,
    tile_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run tiled inference and return a CPU class-ID map."""
    height, width = image.shape[-2:]
    prediction_map = torch.empty((height, width), dtype=torch.uint8)
    tile_specs = [
        (top, left, min(tile_size, height - top), min(tile_size, width - left))
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    ]
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for start in range(0, len(tile_specs), tile_batch_size):
        batch_specs = tile_specs[start:start + tile_batch_size]
        tiles = []
        for top, left, valid_height, valid_width in batch_specs:
            tile = image[:, top:top + valid_height, left:left + valid_width]
            tile = F.pad(tile, (0, tile_size - valid_width, 0, tile_size - valid_height))
            tiles.append(tile)

        tile_batch = torch.stack(tiles).to(device=device, dtype=torch.float32)
        means = tile_batch.mean(dim=(1, 2, 3), keepdim=True)
        stds = tile_batch.std(dim=(1, 2, 3), keepdim=True)
        tile_batch = (tile_batch - means) / (stds + EPSILON)
        wavelength_batch = wavelengths.unsqueeze(0).expand(len(batch_specs), -1).to(device)

        with amp_context:
            spatial_features, _ = model.model(tile_batch, wavelength_batch)
            logits = model.classifier(spatial_features)
            predictions = F.interpolate(
                logits,
                size=(tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1).to("cpu", dtype=torch.uint8)

        for tile_idx, (top, left, valid_height, valid_width) in enumerate(batch_specs):
            prediction_map[top:top + valid_height, left:left + valid_width] = predictions[
                tile_idx, :valid_height, :valid_width
            ]
    return prediction_map


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Accumulate a true-label by predicted-label confusion matrix."""
    labels = labels.reshape(-1).to(torch.long)
    predictions = predictions.reshape(-1).to(torch.long)
    valid = labels != IGNORE_LABEL
    indices = labels[valid] * NUM_CLASSES + predictions[valid]
    confusion_matrix += torch.bincount(
        indices, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def build_palette() -> np.ndarray:
    """Build a stable 41-class RGB palette."""
    colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    colors[1:] = (plt.get_cmap("turbo")(np.linspace(0.02, 0.98, 40))[:, :3] * 255).astype(
        np.uint8
    )
    return colors


def colorize_labels(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert class IDs to RGB and render ignored pixels in gray."""
    ignored = labels == IGNORE_LABEL
    safe_labels = labels.copy()
    safe_labels[ignored] = 0
    colored = palette[safe_labels]
    colored[ignored] = (128, 128, 128)
    return colored


def save_scene_outputs(
    output_dir: Path,
    scene_path: Path,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    palette: np.ndarray,
) -> None:
    """Save raw IDs, a full-resolution color map, and a comparison figure."""
    scene_name = scene_path.name
    scene_dir = output_dir / "scenes" / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    label_array = labels.numpy()
    prediction_array = predictions.numpy()
    rgb_path = scene_path / f"{scene_name}_rgb.png"
    with Image.open(rgb_path) as rgb_image:
        rgb = np.asarray(rgb_image.convert("RGB"))

    prediction_color = colorize_labels(prediction_array, palette)
    label_color = colorize_labels(label_array, palette)
    valid = label_array != IGNORE_LABEL
    correct = valid & (label_array == prediction_array)
    error_map = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    error_map[correct] = (35, 139, 69)
    error_map[valid & ~correct] = (220, 53, 69)
    error_map[~valid] = (128, 128, 128)

    Image.fromarray(prediction_array.astype(np.uint8)).save(scene_dir / "prediction_ids.png")
    Image.fromarray(prediction_color).save(scene_dir / "prediction_color.png")

    figure, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    panels = (
        (rgb, "RGB reference"),
        (label_color, "Ground truth"),
        (prediction_color, "Prediction"),
        (error_map, "Correct / error / ignored"),
    )
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(scene_name, fontsize=14)
    figure.savefig(scene_dir / "comparison.png", dpi=150)
    plt.close(figure)


def compute_metrics(confusion_matrix: torch.Tensor) -> tuple[np.ndarray, dict]:
    """Compute fixed-class IoUs and summary metrics."""
    matrix = confusion_matrix.to(torch.float64)
    true_positive = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    iou = true_positive / (union + EPSILON)
    metrics = {
        "test_mIoU": float(iou.mean()),
        "test_foreground_mIoU": float(iou[1:].mean()),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "per_class_IoU": {
            str(class_id): float(iou[class_id]) for class_id in range(NUM_CLASSES)
        },
        "per_class_union_pixels": {
            str(class_id): int(union[class_id]) for class_id in range(NUM_CLASSES)
        },
    }
    return iou.numpy(), metrics


def compute_class_metrics(confusion_matrix: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute precision, recall, IoU, and support for every class."""
    matrix = confusion_matrix.numpy().astype(np.float64)
    true_positive = np.diag(matrix)
    gt_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    union = gt_pixels + predicted_pixels - true_positive
    return {
        "precision": np.divide(true_positive, predicted_pixels, out=np.zeros(NUM_CLASSES), where=predicted_pixels > 0),
        "recall": np.divide(true_positive, gt_pixels, out=np.zeros(NUM_CLASSES), where=gt_pixels > 0),
        "iou": np.divide(true_positive, union, out=np.zeros(NUM_CLASSES), where=union > 0),
        "gt_pixels": gt_pixels.astype(np.int64),
        "predicted_pixels": predicted_pixels.astype(np.int64),
        "true_positive_pixels": true_positive.astype(np.int64),
    }


def compute_scene_metrics(confusion_matrix: torch.Tensor) -> dict:
    """Compute scene metrics over classes present in that scene's ground truth."""
    values = compute_class_metrics(confusion_matrix)
    present = values["gt_pixels"] > 0
    foreground_present = present.copy()
    foreground_present[0] = False
    matrix = confusion_matrix.numpy()
    return {
        "mIoU_present_classes": float(values["iou"][present].mean()) if present.any() else 0.0,
        "foreground_mIoU_present_classes": float(values["iou"][foreground_present].mean()) if foreground_present.any() else 0.0,
        "pixel_accuracy": float(np.diag(matrix).sum() / max(matrix.sum(), 1)),
        "foreground_classes_present": int(foreground_present.sum()),
    }


def collect_training_support(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Count full-scene training GT pixels and scenes as a crop-exposure proxy."""
    dataset_kwargs = config["data_kwargs"]["train_dataset"].copy()
    dataset_kwargs.pop("name", None)
    train_dataset = HSIRS(cfg=config, **dataset_kwargs)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    scene_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for scene_path in train_dataset.scenes:
        with Image.open(next(scene_path.glob("*_seg_map.png"))) as image:
            labels = np.asarray(image, dtype=np.uint8)
        counts = np.bincount(labels.reshape(-1), minlength=NUM_CLASSES)
        pixel_counts += counts[:NUM_CLASSES]
        scene_counts += counts[:NUM_CLASSES] > 0
    return pixel_counts, scene_counts


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write analysis records with stable column ordering."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_extended_analysis(
    output_dir: Path,
    confusion_matrix: torch.Tensor,
    training_pixels: np.ndarray,
    training_scenes: np.ndarray,
    scene_rows: list[dict],
) -> None:
    """Save class, support, confusion, and scene analyses."""
    values = compute_class_metrics(confusion_matrix)
    foreground_miou = float(values["iou"][1:].mean())
    class_rows = [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "precision": float(values["precision"][class_id]),
            "recall": float(values["recall"][class_id]),
            "iou": float(values["iou"][class_id]),
            "train_full_scene_pixels": int(training_pixels[class_id]),
            "train_scene_count": int(training_scenes[class_id]),
            "test_gt_pixels": int(values["gt_pixels"][class_id]),
            "test_predicted_pixels": int(values["predicted_pixels"][class_id]),
        }
        for class_id in range(NUM_CLASSES)
    ]
    write_csv(output_dir / "per_class_metrics.csv", class_rows)

    order = np.argsort(values["iou"][1:])[::-1] + 1
    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    axis.barh(np.arange(len(order)), values["iou"][order] * 100, color="#277da1")
    axis.axvline(foreground_miou * 100, color="#d1495b", linestyle="--", label=f"FG mIoU ({foreground_miou * 100:.2f}%)")
    axis.set_yticks(np.arange(len(order)), [CLASS_NAMES[idx] for idx in order])
    axis.invert_yaxis()
    axis.set_xlabel("Test IoU (%)")
    axis.set_title("HSIRS foreground classes sorted by test IoU")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "per_class_iou_sorted.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    positions = np.arange(NUM_CLASSES - 1)
    width = 0.26
    axis.bar(positions - width, values["precision"][1:] * 100, width, label="Precision")
    axis.bar(positions, values["recall"][1:] * 100, width, label="Recall")
    axis.bar(positions + width, values["iou"][1:] * 100, width, label="IoU")
    axis.set_xticks(positions, [str(idx) for idx in range(1, NUM_CLASSES)])
    axis.set_xlabel("Class ID")
    axis.set_ylabel("Score (%)")
    axis.set_title("HSIRS test precision, recall, and IoU by class")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3)
    figure.savefig(output_dir / "per_class_precision_recall_iou.png", dpi=150)
    plt.close(figure)

    valid_support = training_pixels[1:] > 0
    support_x = training_pixels[1:][valid_support]
    support_y = values["iou"][1:][valid_support]
    support_ids = np.arange(1, NUM_CLASSES)[valid_support]
    correlation = float(np.corrcoef(np.log10(support_x), support_y)[0, 1])
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.scatter(support_x, support_y * 100, color="#277da1", alpha=0.8)
    for class_id, x_value, y_value in zip(support_ids, support_x, support_y):
        axis.annotate(str(class_id), (x_value, y_value * 100), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("Training GT pixels in full scenes (log scale)")
    axis.set_ylabel("Test IoU (%)")
    axis.set_title(f"Training support vs test IoU (Pearson r={correlation:.3f})")
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "training_support_vs_test_iou.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix.numpy().copy()
    np.fill_diagonal(matrix, 0)
    matrix[0, :] = 0
    confusion_rows = []
    for flat_index in np.argsort(matrix.ravel())[::-1][:20]:
        true_id, predicted_id = np.unravel_index(flat_index, matrix.shape)
        true_total = int(confusion_matrix[true_id].sum())
        confusion_rows.append({
            "true_class_id": int(true_id),
            "true_class_name": CLASS_NAMES[true_id],
            "predicted_class_id": int(predicted_id),
            "predicted_class_name": CLASS_NAMES[predicted_id],
            "pixels": int(matrix[true_id, predicted_id]),
            "fraction_of_true_class": float(matrix[true_id, predicted_id] / max(true_total, 1)),
        })
    write_csv(output_dir / "top_confusion_pairs.csv", confusion_rows)
    top_ten = confusion_rows[:10][::-1]
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    labels = [f"{row['true_class_name']} -> {row['predicted_class_name']}" for row in top_ten]
    axis.barh(labels, [row["pixels"] for row in top_ten], color="#d1495b")
    axis.set_xlabel("Misclassified test pixels")
    axis.set_title("Top 10 foreground-class confusion pairs")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_dir / "top_confusion_pairs.png", dpi=150)
    plt.close(figure)

    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    scene_scores = np.asarray([row["foreground_mIoU_present_classes"] for row in scene_rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(scene_scores * 100, bins=15, color="#277da1", edgecolor="white")
    axes[0].axvline(np.mean(scene_scores) * 100, color="#d1495b", linestyle="--")
    axes[0].set_xlabel("Scene FG mIoU over GT-present classes (%)")
    axes[0].set_ylabel("Number of scenes")
    axes[1].boxplot(scene_scores * 100)
    axes[1].set_ylabel("Scene FG mIoU over GT-present classes (%)")
    figure.suptitle("HSIRS test scene-wise performance")
    figure.savefig(output_dir / "scene_foreground_miou_distribution.png", dpi=150)
    plt.close(figure)

    pair_rows = []
    for real_id, fake_id, category in REAL_FAKE_PAIRS:
        for true_id, predicted_id, direction in ((real_id, fake_id, "real_to_fake"), (fake_id, real_id, "fake_to_real")):
            true_pixels = int(confusion_matrix[true_id].sum())
            confused_pixels = int(confusion_matrix[true_id, predicted_id])
            pair_rows.append({
                "category": category,
                "direction": direction,
                "true_class": CLASS_NAMES[true_id],
                "predicted_class": CLASS_NAMES[predicted_id],
                "confused_pixels": confused_pixels,
                "true_class_pixels": true_pixels,
                "confusion_rate": confused_pixels / max(true_pixels, 1),
            })
    write_csv(output_dir / "real_fake_confusion.csv", pair_rows)
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "foreground_mIoU": foreground_miou,
            "support_iou_pearson_r_log10_pixels": correlation,
            "support_definition": "Full-scene training GT pixels; not realized crop exposure.",
            "scene_metric_definition": "Mean IoU over foreground classes present in each scene GT.",
            "representative_selection": "Best, median, and worst by scene foreground mIoU.",
        }, file, indent=2)


def save_learning_curves(run_dir: Path, output_dir: Path) -> bool:
    """Restore train-loss and validation curves from TensorBoard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        LOGGER.warning("TensorBoard is unavailable; skipping learning curves")
        return False
    scalar_values: dict[str, list] = {}
    for event_path in sorted(run_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        for tag in ("train/loss_epoch", "val_foreground_mIoU"):
            if tag in accumulator.Tags().get("scalars", []):
                scalar_values[tag] = accumulator.Scalars(tag)
    if "train/loss_epoch" not in scalar_values or "val_foreground_mIoU" not in scalar_values:
        LOGGER.warning("Required TensorBoard scalar tags are missing; skipping learning curves")
        return False
    train_values = scalar_values["train/loss_epoch"]
    val_values = scalar_values["val_foreground_mIoU"]
    figure, loss_axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    metric_axis = loss_axis.twinx()
    loss_axis.plot(np.arange(1, len(train_values) + 1), [value.value for value in train_values], color="#277da1", label="Train loss")
    validation_epochs = np.arange(5, 5 * len(val_values) + 1, 5)
    metric_axis.plot(validation_epochs, [value.value * 100 for value in val_values], color="#d1495b", marker="o", label="Validation FG mIoU")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Train loss", color="#277da1")
    metric_axis.set_ylabel("Validation FG mIoU (%)", color="#d1495b")
    loss_axis.set_title("CARL linear-probe learning curves")
    loss_axis.grid(alpha=0.25)
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)
    return True


def save_summary_outputs(
    output_dir: Path,
    confusion_matrix: torch.Tensor,
    iou: np.ndarray,
    metrics: dict,
) -> None:
    """Save metrics JSON, class IoU plot, and normalized confusion matrix."""
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    np.save(output_dir / "confusion_matrix.npy", confusion_matrix.numpy())

    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    colors = ["#444444"] + ["#277da1"] * 40
    axis.bar(np.arange(NUM_CLASSES), iou * 100, color=colors)
    axis.set_xlabel("Class ID")
    axis.set_ylabel("IoU (%)")
    axis.set_title("HSIRS test IoU by class")
    axis.set_xticks(np.arange(NUM_CLASSES))
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "per_class_iou.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix.numpy().astype(np.float64)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    image = axis.imshow(normalized, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("HSIRS test confusion matrix (row-normalized)")
    ticks = np.arange(0, NUM_CLASSES, 2)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    figure.colorbar(image, ax=axis, label="Fraction of true-class pixels")
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trained checkpoint. Defaults to the newest best checkpoint under the log directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config YAML. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of test scenes for which images are saved. Metrics always use all scenes.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoint_path = args.checkpoint or find_latest_checkpoint(Path("logs/hsirs_seg"))
    config_path = args.config or checkpoint_path.parent / "config.yaml"
    output_dir = args.output_dir or checkpoint_path.parent / "postprocess"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    config = load_config(str(config_path))
    if config["model_kwargs"]["n_classes"] != NUM_CLASSES:
        raise ValueError("Post-processing requires the revised 41-class HSIRS config")

    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Output: %s", output_dir)
    model = load_trained_model(config, checkpoint_path, device)
    dataset_kwargs = config["data_kwargs"]["test_dataset"].copy()
    dataset_kwargs.pop("name", None)
    test_dataset = HSIRS(cfg=config, **dataset_kwargs)
    tile_size = config["evaluation_kwargs"]["tile_size"]
    tile_batch_size = config["evaluation_kwargs"]["tile_batch_size"]
    confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    palette = build_palette()
    scene_rows = []

    for scene_idx in range(len(test_dataset)):
        image, wavelengths, labels, _ = test_dataset[scene_idx]
        predictions = predict_scene(
            model, image, wavelengths, tile_size, tile_batch_size, device
        )
        update_confusion_matrix(confusion_matrix, labels, predictions)
        scene_confusion_matrix = torch.zeros_like(confusion_matrix)
        update_confusion_matrix(scene_confusion_matrix, labels, predictions)
        scene_rows.append({
            "scene_name": test_dataset.scenes[scene_idx].name,
            **compute_scene_metrics(scene_confusion_matrix),
        })
        if scene_idx < args.num_visualizations:
            save_scene_outputs(
                output_dir,
                test_dataset.scenes[scene_idx],
                labels,
                predictions,
                palette,
            )
        LOGGER.info("Processed test scene %d/%d: %s", scene_idx + 1, len(test_dataset), test_dataset.scenes[scene_idx].name)

    iou, metrics = compute_metrics(confusion_matrix)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "number_of_test_scenes": len(test_dataset),
            "number_of_visualizations": min(args.num_visualizations, len(test_dataset)),
            "class_names": CLASS_NAMES,
        }
    )
    save_summary_outputs(output_dir, confusion_matrix, iou, metrics)
    training_pixels, training_scenes = collect_training_support(config)
    save_extended_analysis(output_dir, confusion_matrix, training_pixels, training_scenes, scene_rows)
    ranked_scene_indices = sorted(
        range(len(scene_rows)),
        key=lambda index: scene_rows[index]["foreground_mIoU_present_classes"],
    )
    representative_indices = {
        "worst": ranked_scene_indices[0],
        "median": ranked_scene_indices[len(ranked_scene_indices) // 2],
        "best": ranked_scene_indices[-1],
    }
    for rank, scene_index in representative_indices.items():
        image, wavelengths, labels, _ = test_dataset[scene_index]
        predictions = predict_scene(
            model, image, wavelengths, tile_size, tile_batch_size, device
        )
        save_scene_outputs(
            output_dir / "representative_scenes" / rank,
            test_dataset.scenes[scene_index],
            labels,
            predictions,
            palette,
        )
    save_learning_curves(checkpoint_path.parent, output_dir)
    LOGGER.info("test_mIoU: %.6f", metrics["test_mIoU"])
    LOGGER.info("test_foreground_mIoU: %.6f", metrics["test_foreground_mIoU"])
    LOGGER.info("Saved post-processing outputs to %s", output_dir)


if __name__ == "__main__":
    main()