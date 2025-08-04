# src/data_analysis/statistical_analyzer.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from typing import Dict, List, Tuple


class StatisticalAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path("results/data_analysis")
        self.stats = defaultdict(dict)

        # Set style for better plots
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def analyze_bounding_boxes(self, split: str = "train") -> Dict:
        """Comprehensive bounding box analysis"""
        print(f"\nAnalyzing bounding boxes for {split}...")

        label_dir = self.dataset_path / split / "labels"

        areas, widths, heights, aspect_ratios = [], [], [], []
        box_per_image = []

        for json_path in label_dir.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            if "annotations" in data:
                box_per_image.append(len(data["annotations"]))

                for box in data["annotations"]:
                    w, h = box["boundingBox"]["width"], box["boundingBox"]["height"]
                    areas.append(w * h)
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)

        # Calculate statistics
        stats = {
            "box_dimensions": {
                "area": {
                    "mean": float(np.mean(areas)),
                    "std": float(np.std(areas)),
                    "min": float(np.min(areas)),
                    "max": float(np.max(areas)),
                    "percentiles": {
                        "25": float(np.percentile(areas, 25)),
                        "50": float(np.percentile(areas, 50)),
                        "75": float(np.percentile(areas, 75)),
                        "95": float(np.percentile(areas, 95)),
                    },
                },
                "width": {
                    "mean": float(np.mean(widths)),
                    "std": float(np.std(widths)),
                    "min": float(np.min(widths)),
                    "max": float(np.max(widths)),
                },
                "height": {
                    "mean": float(np.mean(heights)),
                    "std": float(np.std(heights)),
                    "min": float(np.min(heights)),
                    "max": float(np.max(heights)),
                },
                "aspect_ratio": {
                    "mean": float(np.mean(aspect_ratios)),
                    "std": float(np.std(aspect_ratios)),
                    "min": float(np.min(aspect_ratios)),
                    "max": float(np.max(aspect_ratios)),
                },
            },
            "boxes_per_image": {
                "mean": float(np.mean(box_per_image)),
                "std": float(np.std(box_per_image)),
                "min": int(np.min(box_per_image)),
                "max": int(np.max(box_per_image)),
            },
            "total_boxes": len(areas),
            "total_images": len(box_per_image),
        }

        # Detect outliers using IQR method
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr

        outliers = [
            i
            for i, area in enumerate(areas)
            if area < outlier_threshold_low or area > outlier_threshold_high
        ]
        stats["outliers"] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(areas) * 100 if areas else 0,
            "thresholds": {
                "low": float(outlier_threshold_low),
                "high": float(outlier_threshold_high),
            },
        }

        # Save statistics
        stats_path = self.results_path / "stats" / f"{split}_box_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Create visualizations
        self._plot_box_distributions(
            areas, widths, heights, aspect_ratios, box_per_image, split
        )

        return stats

    def _plot_box_distributions(
        self, areas, widths, heights, aspect_ratios, box_per_image, split
    ):
        """Create comprehensive visualization of box distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Bounding Box Analysis - {split.capitalize()} Set", fontsize=16)

        # Area distribution
        axes[0, 0].hist(areas, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("Box Area Distribution")
        axes[0, 0].set_xlabel("Area (pixels²)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            np.mean(areas),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(areas):.1f}",
        )
        axes[0, 0].legend()

        # Width distribution
        axes[0, 1].hist(widths, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Width Distribution")
        axes[0, 1].set_xlabel("Width (pixels)")
        axes[0, 1].set_ylabel("Frequency")

        # Height distribution
        axes[0, 2].hist(heights, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 2].set_title("Height Distribution")
        axes[0, 2].set_xlabel("Height (pixels)")
        axes[0, 2].set_ylabel("Frequency")

        # Aspect ratio distribution
        axes[1, 0].hist(aspect_ratios, bins=50, edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("Aspect Ratio Distribution")
        axes[1, 0].set_xlabel("Width/Height Ratio")
        axes[1, 0].set_ylabel("Frequency")

        # Boxes per image
        axes[1, 1].hist(
            box_per_image,
            bins=range(0, max(box_per_image) + 2),
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 1].set_title("Boxes per Image Distribution")
        axes[1, 1].set_xlabel("Number of Boxes")
        axes[1, 1].set_ylabel("Number of Images")

        # Box plot for outlier detection
        axes[1, 2].boxplot([areas], labels=["Area"])
        axes[1, 2].set_title("Box Area Outliers")
        axes[1, 2].set_ylabel("Area (pixels²)")

        plt.tight_layout()
        plt.savefig(
            self.results_path / "plots" / f"{split}_box_distributions.png", dpi=300
        )
        plt.close()

    def analyze_class_distribution(self, split: str = "train") -> Dict:
        """Analyze character class distribution"""
        print(f"\nAnalyzing class distribution for {split}...")

        label_dir = self.dataset_path / split / "labels"
        class_counts = Counter()
        class_sizes = defaultdict(list)  # Track box sizes per class

        for json_path in label_dir.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            if "annotations" in data:
                for box in data["annotations"]:
                    label = box.get("class", "unknown")
                    class_counts[label] += 1
                    class_sizes[label].append(
                        box["boundingBox"]["width"] * box["boundingBox"]["height"]
                    )
        # Calculate statistics per class
        class_stats = {}
        for label, count in class_counts.items():
            sizes = class_sizes[label]
            class_stats[label] = {
                "count": count,
                "percentage": count / sum(class_counts.values()) * 100,
                "avg_size": float(np.mean(sizes)),
                "std_size": float(np.std(sizes)),
                "min_size": float(np.min(sizes)),
                "max_size": float(np.max(sizes)),
            }

        # Identify imbalanced classes
        total_samples = sum(class_counts.values())
        avg_samples_per_class = total_samples / len(class_counts)

        imbalanced_classes = {
            "rare": [
                label
                for label, count in class_counts.items()
                if count < avg_samples_per_class * 0.1
            ],
            "common": [
                label
                for label, count in class_counts.items()
                if count > avg_samples_per_class * 2
            ],
        }

        stats = {
            "total_classes": len(class_counts),
            "total_annotations": total_samples,
            "class_distribution": class_stats,
            "imbalanced_classes": imbalanced_classes,
            "distribution_metrics": {
                "min_samples": min(class_counts.values()),
                "max_samples": max(class_counts.values()),
                "avg_samples": avg_samples_per_class,
                "std_samples": float(np.std(list(class_counts.values()))),
            },
        }

        # Save statistics
        stats_path = self.results_path / "stats" / f"{split}_class_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Create visualizations
        self._plot_class_distribution(class_counts, class_stats, split)

        return stats

    def _plot_class_distribution(self, class_counts, class_stats, split):
        """Create class distribution visualizations"""
        # Sort classes by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        labels, counts = zip(*sorted_classes)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(
            f"Class Distribution Analysis - {split.capitalize()} Set", fontsize=16
        )

        # Bar plot for class frequencies
        bars = ax1.bar(range(len(labels)), counts, color="skyblue", edgecolor="navy")
        ax1.set_xlabel("Character Class")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Character Class Frequency Distribution")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Log scale plot for better visibility of rare classes
        ax2.bar(range(len(labels)), counts, color="lightcoral", edgecolor="darkred")
        ax2.set_yscale("log")
        ax2.set_xlabel("Character Class")
        ax2.set_ylabel("Frequency (log scale)")
        ax2.set_title("Character Class Frequency Distribution (Log Scale)")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            self.results_path / "plots" / f"{split}_class_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a separate plot for class imbalance visualization
        self._plot_class_imbalance(class_counts, split)

    def _plot_class_imbalance(self, class_counts, split):
        """Visualize class imbalance"""
        sorted_counts = sorted(class_counts.values(), reverse=True)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(sorted_counts)), sorted_counts, "b-", linewidth=2)
        plt.fill_between(range(len(sorted_counts)), sorted_counts, alpha=0.3)

        # Add reference lines
        mean_count = np.mean(sorted_counts)
        plt.axhline(
            y=mean_count, color="r", linestyle="--", label=f"Mean: {mean_count:.0f}"
        )
        plt.axhline(
            y=mean_count * 0.1,
            color="orange",
            linestyle="--",
            label="10% of mean (rare threshold)",
        )
        plt.axhline(
            y=mean_count * 2,
            color="green",
            linestyle="--",
            label="200% of mean (common threshold)",
        )

        plt.xlabel("Class Rank")
        plt.ylabel("Number of Samples")
        plt.title(f"Class Imbalance Curve - {split.capitalize()} Set")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(
            self.results_path / "plots" / f"{split}_class_imbalance.png", dpi=300
        )
        plt.close()

    def analyze_expression_completeness(self, split: str = "train") -> Dict:
        """Analyze mathematical expression completeness"""
        print(f"\nAnalyzing expression completeness for {split}...")

        label_dir = self.dataset_path / split / "labels"

        total_files = 0
        with_expression = 0
        expression_lengths = []
        unique_chars = set()

        for json_path in label_dir.glob("*.json"):
            total_files += 1
            with open(json_path, "r") as f:
                data = json.load(f)

            if "expression" in data and data["expression"]:
                with_expression += 1
                expr = data["expression"]
                expression_lengths.append(len(expr))
                unique_chars.update(expr)

        stats = {
            "total_files": total_files,
            "files_with_expression": with_expression,
            "files_without_expression": total_files - with_expression,
            "completeness_percentage": (
                (with_expression / total_files * 100) if total_files > 0 else 0
            ),
            "expression_stats": {
                "avg_length": (
                    float(np.mean(expression_lengths)) if expression_lengths else 0
                ),
                "min_length": min(expression_lengths) if expression_lengths else 0,
                "max_length": max(expression_lengths) if expression_lengths else 0,
                "unique_characters": len(unique_chars),
                "character_set": sorted(list(unique_chars)),
            },
        }

        # Save statistics
        stats_path = self.results_path / "stats" / f"{split}_expression_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats
