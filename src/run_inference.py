# src/run_inference.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from inference import InferenceEngine
import argparse


def run_single_image_inference(
    image_path,
    checkpoint_path="results/checkpoints/best_model.pth",
    nms_threshold=0.3,
    score_threshold=0.6,
    save_result=True,
):
    """Run inference on a single image"""

    # Initialize engine
    engine = InferenceEngine(checkpoint_path, nms_threshold, score_threshold)

    # Load and preprocess image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

    # Run inference
    pred_boxes, pred_scores, pred_labels = engine.predict(img_tensor)

    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(f"Detections: {Path(image_path).name}")

    # Draw predictions
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="red", facecolor="none"
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

    ax.set_xlabel(f"Detected {len(pred_boxes)} characters")
    ax.axis("off")

    if save_result:
        output_path = (
            Path("results/visualizations") / f"inference_{Path(image_path).stem}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Result saved to: {output_path}")

    plt.show()

    return pred_boxes, pred_scores, pred_labels


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--nms", type=float, default=0.3, help="NMS threshold")
    parser.add_argument("--score", type=float, default=0.6, help="Score threshold")
    parser.add_argument("--no-save", action="store_true", help="Do not save result")

    args = parser.parse_args()

    # Run inference
    boxes, scores, labels = run_single_image_inference(
        args.image, args.checkpoint, args.nms, args.score, save_result=not args.no_save
    )

    # Print results
    print(f"\nDetection Results:")
    print(f"Number of detections: {len(boxes)}")
    if len(scores) > 0:
        print(f"Average confidence: {np.mean(scores):.3f}")
        print(f"Max confidence: {np.max(scores):.3f}")
        print(f"Min confidence: {np.min(scores):.3f}")


if __name__ == "__main__":
    main()
