# src/utils/inference_with_better_nms.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.fasterrcnn.model.faster_rcnn import FasterRCNN
from data import CharacterDetectionDataset
from torchvision.ops import nms
import random


def apply_class_specific_nms(boxes, scores, labels, nms_threshold=0.3):
    """Apply NMS separately for each class"""
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Convert to tensors if needed
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    keep_masks = []

    # Apply NMS for each class separately
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        if len(class_boxes) > 0:
            # Apply NMS
            keep = nms(class_boxes, class_scores, nms_threshold)
            # Get original indices
            class_indices = torch.where(class_mask)[0]
            keep_masks.extend(class_indices[keep].tolist())

    keep_masks = torch.tensor(keep_masks)
    return boxes[keep_masks], scores[keep_masks], labels[keep_masks]


def visualize_before_after_nms(
    checkpoint_path="results/fasterrcnn/checkpoints/best_model.pth",
    num_samples=4,
    nms_thresholds=[0.5, 0.3, 0.1],
):
    """Visualize detections with different NMS thresholds"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/visualizations/nms_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get samples
    dataset = CharacterDetectionDataset("dataset", split="valid")
    indices = random.sample(range(len(dataset)), num_samples)

    for sample_idx in indices:
        img_tensor, target = dataset[sample_idx]

        # Get raw predictions
        with torch.no_grad():
            # Temporarily set very low thresholds to get all boxes
            original_nms = model.roi_heads.nms_thresh
            original_score = model.roi_heads.score_thresh

            model.roi_heads.nms_thresh = 0.9  # Very high to get many overlapping boxes
            model.roi_heads.score_thresh = 0.3  # Lower score threshold

            predictions_raw = model([img_tensor.to(device)])

            # Reset
            model.roi_heads.nms_thresh = original_nms
            model.roi_heads.score_thresh = original_score

        # Create visualization
        fig, axes = plt.subplots(
            1, len(nms_thresholds) + 2, figsize=(5 * (len(nms_thresholds) + 2), 5)
        )
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Plot 1: Ground Truth
        axes[0].imshow(img_np)
        axes[0].set_title("Ground Truth")
        gt_boxes = target["boxes"].numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            axes[0].add_patch(rect)
        axes[0].set_xlabel(f"{len(gt_boxes)} GT boxes")
        axes[0].axis("off")

        # Plot 2: Before NMS
        axes[1].imshow(img_np)
        axes[1].set_title("Before Additional NMS")

        pred = predictions_raw[0]
        boxes_before = pred["boxes"].cpu().numpy()
        scores_before = pred["scores"].cpu().numpy()
        labels_before = pred["labels"].cpu().numpy()

        # Filter by score
        mask = scores_before > 0.5
        boxes_before = boxes_before[mask]
        scores_before = scores_before[mask]
        labels_before = labels_before[mask]

        for box, score in zip(boxes_before, scores_before):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                alpha=0.5,
            )
            axes[1].add_patch(rect)
        axes[1].set_xlabel(f"{len(boxes_before)} detections")
        axes[1].axis("off")

        # Plot 3+: After different NMS thresholds
        for idx, nms_thresh in enumerate(nms_thresholds):
            axes[idx + 2].imshow(img_np)
            axes[idx + 2].set_title(f"NMS = {nms_thresh}")

            # Apply NMS
            boxes_after, scores_after, labels_after = apply_class_specific_nms(
                boxes_before, scores_before, labels_before, nms_thresh
            )

            for box, score in zip(boxes_after, scores_after):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                axes[idx + 2].add_patch(rect)
                axes[idx + 2].text(
                    x1,
                    y1 - 5,
                    f"{score:.2f}",
                    color="red",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7),
                )
            axes[idx + 2].set_xlabel(f"{len(boxes_after)} detections")
            axes[idx + 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"nms_comparison_sample_{sample_idx}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print(f"NMS comparison saved to: {output_dir}")


def improved_inference(
    checkpoint_path="results/fasterrcnn/checkpoints/best_model.pth",
    num_samples=12,
    split="valid",
    score_threshold=0.5,
    nms_threshold=0.3,
):
    """Run inference with improved post-processing"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with modified settings
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Modify NMS settings
    model.roi_heads.nms_thresh = nms_threshold
    model.roi_heads.score_thresh = score_threshold

    model.to(device)
    model.eval()

    print(f"Using NMS threshold: {nms_threshold}")
    print(f"Using score threshold: {score_threshold}")

    # Load dataset
    dataset = CharacterDetectionDataset("dataset", split=split)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    detection_stats = []

    for idx, sample_idx in enumerate(indices):
        img_tensor, target = dataset[sample_idx]

        # Run inference
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])

        # Get predictions and apply additional NMS if needed
        pred = predictions[0]
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        # Apply additional class-specific NMS
        boxes, scores, labels = apply_class_specific_nms(
            boxes, scores, labels, nms_threshold
        )

        # Convert to numpy
        boxes = boxes.numpy()
        scores = scores.numpy()

        # Visualize
        img_np = img_tensor.permute(1, 2, 0).numpy()
        axes[idx].imshow(img_np)
        axes[idx].set_title(f"Sample {sample_idx}")

        # Draw predictions
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[idx].add_patch(rect)
            axes[idx].text(
                x1,
                y1 - 5,
                f"{score:.2f}",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # Draw ground truth
        gt_boxes = target["boxes"].numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
            )
            axes[idx].add_patch(rect)

        axes[idx].set_xlabel(f"Pred: {len(boxes)}, GT: {len(gt_boxes)}")
        axes[idx].axis("off")

        detection_stats.append(
            {
                "sample_idx": sample_idx,
                "num_predictions": len(boxes),
                "num_gt": len(gt_boxes),
            }
        )

    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Improved Detection Results (NMS={nms_threshold}, Score={score_threshold})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{split}_improved_detections_nms{nms_threshold}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nImproved detection statistics:")
    print(
        f"Average predictions: {np.mean([s['num_predictions'] for s in detection_stats]):.2f}"
    )
    print(f"Average GT: {np.mean([s['num_gt'] for s in detection_stats]):.2f}")

    return detection_stats


if __name__ == "__main__":
    # First check current settings
    print("Checking model settings...")
    sys.path.append(str(Path(__file__).parent / "utils"))
    from utils.check_nms_settings import check_model_settings

    check_model_settings()

    print("\n" + "=" * 60)

    # Visualize NMS comparison
    print("\nGenerating NMS comparison visualizations...")
    visualize_before_after_nms(num_samples=3)

    # Run improved inference
    print("\nRunning improved inference with NMS=0.3...")
    improved_inference(nms_threshold=0.3, score_threshold=0.6)

    # Try even more aggressive NMS
    print("\nTrying more aggressive NMS=0.1...")
    improved_inference(nms_threshold=0.1, score_threshold=0.7)
