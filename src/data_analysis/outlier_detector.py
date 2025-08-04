# src/data_analysis/outlier_detector.py
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns


class OutlierDetector:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path("results/data_analysis")
        self.outliers = defaultdict(list)

    def detect_outliers(self, split: str = "train"):
        """Detect outliers using multiple methods"""
        print(f"\nDetecting outliers in {split} dataset...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"

        # Collect all box statistics
        all_areas = []
        all_widths = []
        all_heights = []
        all_aspect_ratios = []
        box_info = []  # Store detailed info for each box

        for json_path in label_dir.glob("*.json"):
            img_path = image_dir / f"{json_path.stem}.png"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            with open(json_path, "r") as f:
                data = json.load(f)

            if "annotations" in data:
                for idx, ann in enumerate(data["annotations"]):
                    bbox = ann["boundingBox"]
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

                    area = w * h
                    aspect_ratio = w / h if h > 0 else 0

                    all_areas.append(area)
                    all_widths.append(w)
                    all_heights.append(h)
                    all_aspect_ratios.append(aspect_ratio)

                    box_info.append(
                        {
                            "file": json_path.stem,
                            "box_idx": idx,
                            "area": area,
                            "width": w,
                            "height": h,
                            "aspect_ratio": aspect_ratio,
                            "relative_area": area / (img_w * img_h),
                            "x": x,
                            "y": y,
                            "img_width": img_w,
                            "img_height": img_h,
                        }
                    )

        # Convert to numpy arrays
        all_areas = np.array(all_areas)
        all_widths = np.array(all_widths)
        all_heights = np.array(all_heights)
        all_aspect_ratios = np.array(all_aspect_ratios)

        # Method 1: IQR-based outlier detection
        outliers_iqr = self._detect_outliers_iqr(all_areas, box_info, "area")

        # Method 2: Z-score based outlier detection
        outliers_zscore = self._detect_outliers_zscore(
            all_areas, box_info, "area", threshold=3
        )

        # Method 3: Percentile-based outlier detection
        outliers_percentile = self._detect_outliers_percentile(
            all_areas, box_info, "area", low_percentile=1, high_percentile=99
        )

        # Method 4: Context-based outliers (relative to image size)
        outliers_context = self._detect_context_outliers(box_info)

        # Combine all outliers
        all_outlier_indices = set()
        all_outlier_indices.update(outliers_iqr)
        all_outlier_indices.update(outliers_zscore)
        all_outlier_indices.update(outliers_percentile)
        all_outlier_indices.update(outliers_context)

        # Store outlier information
        self.outliers[split] = {
            "total_boxes": len(box_info),
            "outlier_count": len(all_outlier_indices),
            "outlier_percentage": len(all_outlier_indices) / len(box_info) * 100,
            "outlier_indices": list(all_outlier_indices),
            "outlier_details": [box_info[i] for i in all_outlier_indices],
            "methods": {
                "iqr": len(outliers_iqr),
                "zscore": len(outliers_zscore),
                "percentile": len(outliers_percentile),
                "context": len(outliers_context),
            },
        }

        # Create visualizations
        self._visualize_outliers(
            all_areas,
            all_widths,
            all_heights,
            all_aspect_ratios,
            all_outlier_indices,
            split,
        )

        # Save outlier report
        self._save_outlier_report(split)

        return self.outliers[split]

    def _detect_outliers_iqr(self, data, box_info, feature_name):
        """Detect outliers using Interquartile Range method"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)

        print(f"IQR method: {len(outliers)} outliers detected for {feature_name}")
        return outliers

    def _detect_outliers_zscore(self, data, box_info, feature_name, threshold=3):
        """Detect outliers using Z-score method"""
        mean = np.mean(data)
        std = np.std(data)

        outliers = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append(i)

        print(f"Z-score method: {len(outliers)} outliers detected for {feature_name}")
        return outliers

    def _detect_outliers_percentile(
        self, data, box_info, feature_name, low_percentile=1, high_percentile=99
    ):
        """Detect outliers using percentile method"""
        lower_bound = np.percentile(data, low_percentile)
        upper_bound = np.percentile(data, high_percentile)

        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)

        print(
            f"Percentile method: {len(outliers)} outliers detected for {feature_name}"
        )
        return outliers

    def _detect_context_outliers(self, box_info):
        """Detect outliers based on context (relative to image size)"""
        outliers = []

        for i, info in enumerate(box_info):
            # Too small relative to image
            if info["relative_area"] < 0.0001:  # Less than 0.01% of image
                outliers.append(i)
            # Too large relative to image
            elif info["relative_area"] > 0.3:  # More than 30% of image
                outliers.append(i)
            # Extreme aspect ratios
            elif info["aspect_ratio"] < 0.1 or info["aspect_ratio"] > 10:
                outliers.append(i)

        print(f"Context method: {len(outliers)} outliers detected")
        return outliers

    def _visualize_outliers(
        self, areas, widths, heights, aspect_ratios, outlier_indices, split
    ):
        """Create comprehensive outlier visualizations"""
        outlier_mask = np.zeros(len(areas), dtype=bool)
        outlier_mask[list(outlier_indices)] = True

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Outlier Analysis - {split.capitalize()} Set", fontsize=16)

        # 1. Area scatter plot with outliers highlighted
        axes[0, 0].scatter(
            range(len(areas)),
            areas,
            c=["red" if outlier_mask[i] else "blue" for i in range(len(areas))],
            alpha=0.6,
            s=20,
        )
        axes[0, 0].set_title("Box Areas (Outliers in Red)")
        axes[0, 0].set_xlabel("Box Index")
        axes[0, 0].set_ylabel("Area (pixels²)")
        axes[0, 0].set_yscale("log")

        # 2. Box plot showing outliers
        box_data = [areas[~outlier_mask], areas[outlier_mask]]
        axes[0, 1].boxplot(box_data, labels=["Normal", "Outliers"])
        axes[0, 1].set_title("Area Distribution Comparison")
        axes[0, 1].set_ylabel("Area (pixels²)")
        axes[0, 1].set_yscale("log")

        # 3. Width vs Height scatter
        axes[0, 2].scatter(
            widths[~outlier_mask],
            heights[~outlier_mask],
            alpha=0.5,
            label="Normal",
            s=20,
        )
        axes[0, 2].scatter(
            widths[outlier_mask],
            heights[outlier_mask],
            color="red",
            alpha=0.7,
            label="Outliers",
            s=30,
        )
        axes[0, 2].set_title("Width vs Height")
        axes[0, 2].set_xlabel("Width (pixels)")
        axes[0, 2].set_ylabel("Height (pixels)")
        axes[0, 2].legend()

        # 4. Aspect ratio distribution
        axes[1, 0].hist(
            aspect_ratios[~outlier_mask],
            bins=50,
            alpha=0.7,
            label="Normal",
            color="blue",
        )
        axes[1, 0].hist(
            aspect_ratios[outlier_mask],
            bins=20,
            alpha=0.7,
            label="Outliers",
            color="red",
        )
        axes[1, 0].set_title("Aspect Ratio Distribution")
        axes[1, 0].set_xlabel("Width/Height Ratio")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()

        # 5. Outlier percentage by detection method
        methods = ['IQR', 'Z-score', 'Percentile', 'Context']
        method_keys = ['iqr', 'zscore', 'percentile', 'context']
        counts = [self.outliers[split]["methods"][k] for k in method_keys]
        axes[1, 1].bar(
            methods, counts, color=["skyblue", "lightcoral", "lightgreen", "gold"]
        )
        axes[1, 1].set_title("Outliers by Detection Method")
        axes[1, 1].set_ylabel("Number of Outliers")

        # 6. Area distribution with outlier boundaries
        axes[1, 2].hist(areas, bins=100, alpha=0.7, color="blue", edgecolor="black")

        # Add IQR boundaries
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        axes[1, 2].axvline(
            lower_bound,
            color="red",
            linestyle="--",
            label=f"IQR Lower: {lower_bound:.0f}",
        )
        axes[1, 2].axvline(
            upper_bound,
            color="red",
            linestyle="--",
            label=f"IQR Upper: {upper_bound:.0f}",
        )
        axes[1, 2].set_title("Area Distribution with Outlier Boundaries")
        axes[1, 2].set_xlabel("Area (pixels²)")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig(
            self.results_path / "plots" / f"{split}_outlier_analysis.png", dpi=300
        )
        plt.close()

    def visualize_outlier_samples(self, split: str = "train", n_samples: int = 20):
        """Visualize actual outlier boxes on images"""
        print(f"\nVisualizing outlier samples from {split}...")

        if split not in self.outliers:
            print(f"No outlier data for {split}. Run detect_outliers first.")
            return

        outlier_details = self.outliers[split]["outlier_details"]
        if not outlier_details:
            print("No outliers found!")
            return

        # Sample outliers
        samples = outlier_details[: min(n_samples, len(outlier_details))]

        # Group by file
        files_to_visualize = defaultdict(list)
        for outlier in samples:
            files_to_visualize[outlier["file"]].append(outlier)

        # Create visualization
        n_files = len(files_to_visualize)
        fig, axes = plt.subplots(min(n_files, 4), 1, figsize=(15, 4 * min(n_files, 4)))
        if n_files == 1:
            axes = [axes]

        image_dir = self.dataset_path / split / "images"
        label_dir = self.dataset_path / split / "labels"

        for idx, (file_name, outliers) in enumerate(
            list(files_to_visualize.items())[:4]
        ):
            img_path = image_dir / f"{file_name}.png"
            json_path = label_dir / f"{file_name}.json"

            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(json_path, "r") as f:
                data = json.load(f)

            axes[idx].imshow(img_rgb)

            # Draw all boxes
            for ann_idx, ann in enumerate(data["annotations"]):
                bbox = ann["boundingBox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

                # Check if this is an outlier
                is_outlier = any(o["box_idx"] == ann_idx for o in outliers)

                color = "red" if is_outlier else "green"
                linewidth = 3 if is_outlier else 1

                rect = plt.Rectangle(
                    (x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth
                )
                axes[idx].add_patch(rect)

                if is_outlier:
                    # Add annotation with outlier info
                    outlier_info = next(o for o in outliers if o["box_idx"] == ann_idx)
                    axes[idx].text(
                        x,
                        y - 5,
                        f"Area: {outlier_info['area']:.0f}\nAR: {outlier_info['aspect_ratio']:.2f}",
                        color="red",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                    )

            axes[idx].set_title(f"File: {file_name} (Red boxes are outliers)")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.results_path
            / "sample_visualizations"
            / f"{split}_outlier_samples.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Saved outlier visualization to sample_visualizations/{split}_outlier_samples.png"
        )

    def _save_outlier_report(self, split: str):
        """Save detailed outlier report"""
        report_path = self.results_path / "stats" / f"{split}_outlier_report.json"

        # Create summary statistics
        outlier_details = self.outliers[split]["outlier_details"]

        if outlier_details:
            areas = [o["area"] for o in outlier_details]
            widths = [o["width"] for o in outlier_details]
            heights = [o["height"] for o in outlier_details]
            aspect_ratios = [o["aspect_ratio"] for o in outlier_details]

            summary = {
                "total_outliers": len(outlier_details),
                "percentage": self.outliers[split]["outlier_percentage"],
                "area_stats": {
                    "min": min(areas),
                    "max": max(areas),
                    "mean": np.mean(areas),
                    "median": np.median(areas),
                },
                "width_stats": {
                    "min": min(widths),
                    "max": max(widths),
                    "mean": np.mean(widths),
                },
                "height_stats": {
                    "min": min(heights),
                    "max": max(heights),
                    "mean": np.mean(heights),
                },
                "aspect_ratio_stats": {
                    "min": min(aspect_ratios),
                    "max": max(aspect_ratios),
                    "mean": np.mean(aspect_ratios),
                },
                "detection_methods": self.outliers[split]["methods"],
                "files_with_outliers": len(set(o["file"] for o in outlier_details)),
            }
        else:
            summary = {"total_outliers": 0, "message": "No outliers detected"}

        # Save report
        report = {
            "summary": summary,
            "outlier_list": outlier_details[:100],  # Save first 100 for review
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Outlier report saved to: {report_path}")


# Usage
if __name__ == "__main__":
    detector = OutlierDetector("dataset")
    detector.detect_outliers("train")
    detector.visualize_outlier_samples("train", n_samples=20)
