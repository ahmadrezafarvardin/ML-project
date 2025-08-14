# src/recognition/compare_models.py
import matplotlib.pyplot as plt
import numpy as np


def create_comparison_chart():
    """Create a comparison chart between CRNN and Semi-supervised approaches"""

    # Data
    models = ["Semi-Supervised\nCharacter Classification", "CRNN with CTC"]
    accuracy = [4.08, 54.35]
    levenshtein = [3.43, 0.83]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    bars1 = ax1.bar(models, accuracy, color=["#ff7f0e", "#2ca02c"])
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Model Accuracy Comparison", fontsize=14, weight="bold")
    ax1.set_ylim(0, 70)

    # Add value labels
    for bar, val in zip(bars1, accuracy):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    # Levenshtein distance comparison
    bars2 = ax2.bar(models, levenshtein, color=["#ff7f0e", "#2ca02c"])
    ax2.set_ylabel("Average Levenshtein Distance", fontsize=12)
    ax2.set_title("Error Rate Comparison", fontsize=14, weight="bold")
    ax2.set_ylim(0, 4)

    # Add value labels
    for bar, val in zip(bars2, levenshtein):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    # Add grid
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "results/recognition/visualizations/model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved model comparison chart")


if __name__ == "__main__":
    create_comparison_chart()
