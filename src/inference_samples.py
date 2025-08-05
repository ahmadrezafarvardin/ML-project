# src/inference_samples.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from models import FasterRCNN
from data import CharacterDetectionDataset
import random


def inference_on_multiple_samples(
    checkpoint_path="results/checkpoints/best_model.pth",
    num_samples=12,
    split="valid",
    output_dir="results/visualizations",
    score_threshold=0.5,
):
    """Run inference on multiple samples and save visualizations"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load dataset
    dataset = CharacterDetectionDataset("dataset", split=split)

    # Random sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Create figure for grid visualization
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    detection_stats = []

    for idx, sample_idx in enumerate(indices):
        # Get image and target
        img_tensor, target = dataset[sample_idx]

        # Run inference
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])

        # Convert to numpy for visualization
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Get predictions
        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        # Filter by score threshold
        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Plot
        axes[idx].imshow(img_np)
        axes[idx].set_title(f"Sample {sample_idx}")

        # Draw predictions in red
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

        # Draw ground truth in green
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

        # Collect statistics
        detection_stats.append(
            {
                "sample_idx": sample_idx,
                "num_predictions": len(boxes),
                "num_gt": len(gt_boxes),
                "avg_score": scores.mean() if len(scores) > 0 else 0,
            }
        )

    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Detection Results on {split.capitalize()} Set (threshold={score_threshold})",
        fontsize=16,
    )
    plt.tight_layout()

    # Save grid
    output_path = output_dir / f"{split}_detection_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid visualization to: {output_path}")
    plt.close()

    # Save individual high-quality images for best samples
    best_samples = sorted(
        detection_stats, key=lambda x: x["num_predictions"], reverse=True
    )[:3]

    for rank, stat in enumerate(best_samples):
        sample_idx = stat["sample_idx"]
        img_tensor, target = dataset[sample_idx]

        with torch.no_grad():
            predictions = model([img_tensor.to(device)])

        # Create individual visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"{score:.2f}",
                color="red",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="yellow", alpha=0.8),
            )

        ax.set_title(
            f"Best Detection Sample {rank+1} - {len(boxes)} detections", fontsize=14
        )
        ax.axis("off")

        individual_path = output_dir / f"{split}_best_sample_{rank+1}.png"
        plt.savefig(individual_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved individual visualization to: {individual_path}")

    # Print statistics
    print("\nDetection Statistics:")
    print(
        f"Average predictions per image: {np.mean([s['num_predictions'] for s in detection_stats]):.2f}"
    )
    print(
        f"Average ground truth per image: {np.mean([s['num_gt'] for s in detection_stats]):.2f}"
    )
    print(
        f"Images with detections: {sum(1 for s in detection_stats if s['num_predictions'] > 0)}/{len(detection_stats)}"
    )

    # Save statistics to file
    stats_path = Path("results/evaluation") / f"{split}_detection_stats.txt"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w") as f:
        f.write(f"Detection Statistics for {split} set\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Score threshold: {score_threshold}\n")
        f.write(f"Number of samples: {len(detection_stats)}\n\n")

        for stat in detection_stats:
            f.write(f"Sample {stat['sample_idx']}: ")
            f.write(f"Predictions={stat['num_predictions']}, ")
            f.write(f"GT={stat['num_gt']}, ")
            f.write(f"Avg Score={stat['avg_score']:.3f}\n")

    print(f"\nStatistics saved to: {stats_path}")


def test_on_test_set(checkpoint_path="results/checkpoints/best_model.pth"):
    """Run inference on test set (no ground truth available)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Test images directory
    test_dir = Path("dataset/test/images")
    test_images = list(test_dir.glob("*.png"))[:6]  # First 6 test images

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, img_path in enumerate(test_images):
        # Load and preprocess image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Run inference
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])

        # Visualize
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"Test: {img_path.name}")

        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        # Filter by threshold
        mask = scores > 0.5
        boxes = boxes[mask]
        scores = scores[mask]

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

        axes[idx].set_xlabel(f"Detected: {len(boxes)} characters")
        axes[idx].axis("off")

    plt.suptitle("Predictions on Test Set", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        "results/visualizations/test_set_predictions.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(
        "Test set predictions saved to: results/visualizations/test_set_predictions.png"
    )


if __name__ == "__main__":
    # First organize the workspace
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent))

    # Run inference on validation set
    print("Running inference on validation set...")
    inference_on_multiple_samples(num_samples=12, split="valid", score_threshold=0.5)

    # Run inference on test set
    print("\nRunning inference on test set...")
    test_on_test_set()
