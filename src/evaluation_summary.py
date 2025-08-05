# src/evaluation_summary.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def create_evaluation_summary():
    """Create a comprehensive evaluation summary"""

    # Load metrics
    metrics_path = Path("results/evaluation/metrics.json")
    if not metrics_path.exists():
        print("No metrics found. Run inference.py first.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Create summary figure
    fig = plt.figure(figsize=(15, 10))

    # 1. Metrics bar chart
    ax1 = plt.subplot(2, 2, 1)
    metric_names = ["Precision", "Recall", "F1 Score"]
    metric_values = [
        metrics["detection_precision"],
        metrics["detection_recall"],
        metrics["detection_f1"],
    ]
    bars = ax1.bar(metric_names, metric_values, color=["blue", "green", "red"])
    ax1.set_ylim(0, 1)
    ax1.set_title("Detection Metrics", fontsize=14, weight="bold")

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # 2. Detection counts
    ax2 = plt.subplot(2, 2, 2)
    counts = ["True Positives", "False Positives", "False Negatives"]
    values = [
        metrics["true_positives"],
        metrics["false_positives"],
        metrics["false_negatives"],
    ]
    colors = ["green", "orange", "red"]
    bars = ax2.bar(counts, values, color=colors)
    ax2.set_title("Detection Counts", fontsize=14, weight="bold")

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # 3. Summary text
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis("off")
    summary_text = f"""
    Model Performance Summary
    ========================

    Detection F1 Score: {metrics['detection_f1']:.3f}
    Mean IoU: {metrics['mean_iou']:.3f}

    Detection Rate: {metrics['detection_recall']:.1%}
    Precision: {metrics['detection_precision']:.1%}

    Total Detections:
    - True Positives: {metrics['true_positives']}
    - False Positives: {metrics['false_positives']}
    - False Negatives: {metrics['false_negatives']}

    Settings:
    - NMS Threshold: 0.3
    - Score Threshold: 0.6
    """
    ax3.text(
        0.1,
        0.9,
        summary_text,
        transform=ax3.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # 4. Performance interpretation
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    # Interpret results
    f1 = metrics["detection_f1"]
    if f1 > 0.9:
        performance = "Excellent"
        color = "green"
    elif f1 > 0.8:
        performance = "Good"
        color = "blue"
    elif f1 > 0.7:
        performance = "Fair"
        color = "orange"
    else:
        performance = "Needs Improvement"
        color = "red"

    interpretation = f"""
    Performance Rating: {performance}

    Recommendations:
    """

    if metrics["false_positives"] > metrics["true_positives"] * 0.2:
        interpretation += (
            "\n• High false positives - consider increasing score threshold"
        )
    if metrics["false_negatives"] > metrics["true_positives"] * 0.2:
        interpretation += (
            "\n• High false negatives - consider more training or lower score threshold"
        )
    if metrics["mean_iou"] < 0.7:
        interpretation += "\n• Low IoU - bounding box regression needs improvement"

    if f1 > 0.85:
        interpretation += "\n• Model is performing well!"
        interpretation += "\n• Consider testing on more diverse data"

    ax4.text(
        0.1,
        0.9,
        interpretation,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2),
    )

    plt.suptitle("Character Detection Evaluation Summary", fontsize=16, weight="bold")
    plt.tight_layout()

    # Save summary
    output_path = Path("results/evaluation/evaluation_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Evaluation summary saved to: {output_path}")


if __name__ == "__main__":
    create_evaluation_summary()
