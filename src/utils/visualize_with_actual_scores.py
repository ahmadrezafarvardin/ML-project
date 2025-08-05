# src/visualize_with_actual_scores.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from models import FasterRCNN
from data import CharacterDetectionDataset


def visualize_all_predictions(checkpoint_path="checkpoints/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get one image
    val_dataset = CharacterDetectionDataset("dataset", split="valid")
    img_tensor, target = val_dataset[0]

    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor.to(device)])

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Plot 1: All predictions
    ax1.imshow(img_np)
    ax1.set_title("All Predictions (no threshold)")

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    # Color map based on score
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        # Use color intensity based on score
        color_intensity = score
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor=(color_intensity, 0, 1 - color_intensity),
            facecolor="none",
            alpha=0.5,
        )
        ax1.add_patch(rect)

    # Plot 2: Ground truth
    ax2.imshow(img_np)
    ax2.set_title("Ground Truth")

    gt_boxes = target["boxes"].numpy()
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", facecolor="none"
        )
        ax2.add_patch(rect)

    ax1.set_xlabel(
        f"Predictions: {len(boxes)}, Score range: {scores.min():.3f}-{scores.max():.3f}"
    )
    ax2.set_xlabel(f"Ground truth: {len(gt_boxes)} boxes")

    plt.tight_layout()
    plt.savefig("all_predictions_visualization.png", dpi=150)
    plt.show()

    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Total predictions: {len(boxes)}")
    print(f"  Unique scores: {len(np.unique(scores))}")
    print(f"  Score variance: {np.var(scores):.6f}")

    # Check if boxes are diverse or clustered
    if len(boxes) > 0:
        box_centers = boxes[:, :2] + (boxes[:, 2:] - boxes[:, :2]) / 2
        print(f"  Box center variance: {np.var(box_centers, axis=0)}")


if __name__ == "__main__":
    visualize_all_predictions()
