# src/models/fasterrcnn/inference.py
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
from torchvision.ops import nms, box_iou
import random
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import seaborn as sns
from collections import defaultdict
import json


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path="results/fasterrcnn/checkpoints/best_model.pth",
        nms_threshold=0.3,
        score_threshold=0.6,
    ):
        """Initialize inference engine with best practices"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        # Load model
        self.model = FasterRCNN(num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Apply best settings
        self.model.roi_heads.nms_thresh = nms_threshold
        self.model.roi_heads.score_thresh = score_threshold

        self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded with NMS={nms_threshold}, Score threshold={score_threshold}"
        )

    def apply_class_specific_nms(self, boxes, scores, labels):
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

        # Apply NMS for each class
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            if len(class_boxes) > 0:
                keep = nms(class_boxes, class_scores, self.nms_threshold)
                class_indices = torch.where(class_mask)[0]
                keep_masks.extend(class_indices[keep].tolist())

        keep_masks = torch.tensor(keep_masks)
        return boxes[keep_masks], scores[keep_masks], labels[keep_masks]

    def predict(self, image_tensor):
        """Run inference on a single image"""
        with torch.no_grad():
            predictions = self.model([image_tensor.to(self.device)])

        pred = predictions[0]
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        # Apply additional NMS if needed
        boxes, scores, labels = self.apply_class_specific_nms(boxes, scores, labels)

        return boxes.numpy(), scores.numpy(), labels.numpy()

    def evaluate_with_metrics(self, dataset, num_samples=None):
        """Evaluate model and compute metrics including confusion matrix and F1 score"""
        if num_samples is None:
            indices = range(len(dataset))
        else:
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        all_pred_labels = []
        all_gt_labels = []
        all_ious = []
        detection_results = []

        for idx in indices:
            img_tensor, target = dataset[idx]

            # Get predictions
            pred_boxes, pred_scores, pred_labels = self.predict(img_tensor)
            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()

            # Match predictions to ground truth
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_matrix = box_iou(
                    torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)
                )

                # For each GT box, find best matching prediction
                for gt_idx in range(len(gt_boxes)):
                    if len(pred_boxes) > 0:
                        ious = iou_matrix[:, gt_idx]
                        best_pred_idx = ious.argmax()
                        best_iou = ious[best_pred_idx].item()

                        if best_iou > 0.5:  # IoU threshold for matching
                            all_pred_labels.append(pred_labels[best_pred_idx])
                            all_gt_labels.append(gt_labels[gt_idx])
                            all_ious.append(best_iou)
                        else:
                            # False negative
                            all_gt_labels.append(gt_labels[gt_idx])
                            all_pred_labels.append(0)  # Background

                # Check for false positives
                matched_preds = set()
                for pred_idx in range(len(pred_boxes)):
                    ious = iou_matrix[pred_idx, :]
                    if ious.max() < 0.5:
                        # False positive
                        all_pred_labels.append(pred_labels[pred_idx])
                        all_gt_labels.append(0)  # Background

            detection_results.append(
                {
                    "idx": idx,
                    "num_pred": len(pred_boxes),
                    "num_gt": len(gt_boxes),
                    "pred_boxes": pred_boxes.tolist(),
                    "pred_scores": pred_scores.tolist(),
                    "gt_boxes": gt_boxes.tolist(),
                }
            )

        # Compute metrics
        metrics = self._compute_metrics(all_pred_labels, all_gt_labels, all_ious)

        return metrics, detection_results

    def _compute_metrics(self, pred_labels, gt_labels, ious):
        """Compute evaluation metrics"""
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)

        # Remove background class for F1 computation
        mask = gt_labels > 0
        pred_positive = pred_labels[mask]
        gt_positive = gt_labels[mask]

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_positive, pred_positive, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(gt_labels, pred_labels)

        # Detection metrics
        tp = np.sum((pred_labels > 0) & (gt_labels > 0))
        fp = np.sum((pred_labels > 0) & (gt_labels == 0))
        fn = np.sum((pred_labels == 0) & (gt_labels > 0))

        detection_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        detection_f1 = (
            2
            * (detection_precision * detection_recall)
            / (detection_precision + detection_recall)
            if (detection_precision + detection_recall) > 0
            else 0
        )

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "detection_precision": float(detection_precision),
            "detection_recall": float(detection_recall),
            "detection_f1": float(detection_f1),
            "mean_iou": float(np.mean(ious)) if len(ious) > 0 else 0,
            "confusion_matrix": cm.tolist(),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        return metrics

    def visualize_results(
        self, dataset, num_samples=12, save_path="results/fasterrcnn/visualizations"
    ):
        """Visualize detection results"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        # Create grid visualization
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        for idx, sample_idx in enumerate(indices):
            img_tensor, target = dataset[sample_idx]
            pred_boxes, pred_scores, _ = self.predict(img_tensor)

            # Visualize
            img_np = img_tensor.permute(1, 2, 0).numpy()
            axes[idx].imshow(img_np)
            axes[idx].set_title(f"Sample {sample_idx}")

            # Draw predictions (red)
            for box, score in zip(pred_boxes, pred_scores):
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

            # Draw ground truth (green)
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

            axes[idx].set_xlabel(f"Pred: {len(pred_boxes)}, GT: {len(gt_boxes)}")
            axes[idx].axis("off")

        # Hide empty subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Detection Results (NMS={self.nms_threshold}, Score={self.score_threshold})",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(save_path / "detection_results.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Visualization saved to: {save_path / 'detection_results.png'}")

    def plot_confusion_matrix(self, cm, save_path="results/fasterrcnn/evaluation"):
        """Plot confusion matrix"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Background", "Character"],
            yticklabels=["Background", "Character"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=150)
        plt.close()

        print(f"Confusion matrix saved to: {save_path / 'confusion_matrix.png'}")


def main():
    """Main inference and evaluation pipeline"""
    print("=" * 60)
    print("CHARACTER DETECTION INFERENCE AND EVALUATION")
    print("=" * 60)

    # Initialize inference engine with best settings
    engine = InferenceEngine(
        checkpoint_path="results/fasterrcnn/checkpoints/best_model.pth",
        nms_threshold=0.3,  # Best practice from analysis
        score_threshold=0.6,  # Best practice from analysis
    )

    # Load validation dataset
    val_dataset = CharacterDetectionDataset("dataset", split="valid")

    # Run evaluation
    print("\nEvaluating on validation set...")
    metrics, detection_results = engine.evaluate_with_metrics(val_dataset)

    # Print metrics
    print("\n" + "=" * 40)
    print("EVALUATION METRICS")
    print("=" * 40)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"\nDetection Metrics:")
    print(f"Detection Precision: {metrics['detection_precision']:.3f}")
    print(f"Detection Recall: {metrics['detection_recall']:.3f}")
    print(f"Detection F1: {metrics['detection_f1']:.3f}")
    print(f"\nMean IoU: {metrics['mean_iou']:.3f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    # Save metrics
    output_dir = Path("results/fasterrcnn/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot confusion matrix
    engine.plot_confusion_matrix(np.array(metrics["confusion_matrix"]))

    # Visualize results
    print("\nGenerating visualizations...")
    engine.visualize_results(val_dataset, num_samples=12)

    # Test set inference
    print("\nRunning inference on test set...")
    test_predictions = []
    test_dir = Path("dataset/test/images")
    test_images = list(test_dir.glob("*.png"))[:10]

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        pred_boxes, pred_scores, pred_labels = engine.predict(img_tensor)

        test_predictions.append(
            {
                "image": img_path.name,
                "predictions": {
                    "boxes": pred_boxes.tolist(),
                    "scores": pred_scores.tolist(),
                    "labels": pred_labels.tolist(),
                },
            }
        )

    # Save test predictions
    with open(output_dir / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nResults saved in:")
    print(f"- {output_dir}/metrics.json")
    print(f"- {output_dir}/confusion_matrix.png")
    print(f"- {output_dir}/test_predictions.json")
    print(f"- results/fasterrcnn/visualizations/detection_results.png")


if __name__ == "__main__":
    main()
