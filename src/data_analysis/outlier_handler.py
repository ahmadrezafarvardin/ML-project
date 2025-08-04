# src/data_analysis/outlier_handler.py
import json
import shutil
from pathlib import Path
import numpy as np


class OutlierHandler:
    def __init__(self, dataset_path: str, outlier_report_path: str):
        self.dataset_path = Path(dataset_path)
        self.outlier_report_path = Path(outlier_report_path)
        self.results_path = Path("results/data_analysis")

    def handle_outliers(self, split: str = "train", strategy: str = "remove_extreme"):
        """
        Handle outliers based on selected strategy

        Strategies:
        - 'remove_extreme': Remove only extreme outliers (area < 10 or > 50% of image)
        - 'remove_all': Remove all detected outliers
        - 'clip': Clip outlier values to acceptable range
        - 'separate': Move outliers to separate dataset for review
        - 'keep': Keep all outliers (do nothing)
        """
        print(f"\nHandling outliers with strategy: {strategy}")

        # Load outlier report
        with open(self.outlier_report_path, "r") as f:
            report = json.load(f)

        outlier_details = report.get("outlier_list", [])

        if strategy == "keep":
            print("Keeping all outliers as-is")
            return

        # Group outliers by file
        outliers_by_file = {}
        for outlier in outlier_details:
            file_name = outlier["file"]
            if file_name not in outliers_by_file:
                outliers_by_file[file_name] = []
            outliers_by_file[file_name].append(outlier["box_idx"])

        # Process based on strategy
        if strategy == "remove_extreme":
            self._remove_extreme_outliers(split, outliers_by_file, outlier_details)
        elif strategy == "remove_all":
            self._remove_all_outliers(split, outliers_by_file)
        elif strategy == "clip":
            self._clip_outliers(split, outliers_by_file, outlier_details)
        elif strategy == "separate":
            self._separate_outliers(split, outliers_by_file)
        else:
            print(f"Unknown strategy: {strategy}")

    def _remove_extreme_outliers(self, split, outliers_by_file, outlier_details):
        """Remove only the most extreme outliers"""
        print("Removing extreme outliers...")

        label_dir = self.dataset_path / split / "labels"
        output_dir = self.results_path / "cleaned_extreme" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        removed_count = 0

        for file_name, outlier_indices in outliers_by_file.items():
            json_path = label_dir / f"{file_name}.json"

            with open(json_path, "r") as f:
                data = json.load(f)

            # Find extreme outliers in this file
            extreme_indices = []
            for outlier in outlier_details:
                if outlier["file"] == file_name:
                    # Define extreme criteria
                    if (
                        outlier["area"] < 10  # Tiny boxes
                        or outlier["relative_area"] > 0.5  # Huge boxes
                        or outlier["aspect_ratio"] < 0.05  # Very thin
                        or outlier["aspect_ratio"] > 20
                    ):  # Very wide
                        extreme_indices.append(outlier["box_idx"])

            if extreme_indices:
                # Remove extreme outliers
                new_annotations = []
                for idx, ann in enumerate(data["annotations"]):
                    if idx not in extreme_indices:
                        new_annotations.append(ann)
                    else:
                        removed_count += 1

                data["annotations"] = new_annotations

                # Save cleaned file
                output_path = output_dir / f"{file_name}.json"
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

        print(f"Removed {removed_count} extreme outliers")

    def _clip_outliers(self, split, outliers_by_file, outlier_details):
        """Clip outlier dimensions to acceptable range"""
        print("Clipping outlier dimensions...")

        # Calculate acceptable ranges from non-outlier boxes
        all_areas = []
        all_aspect_ratios = []

        label_dir = self.dataset_path / split / "labels"
        for json_path in label_dir.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            file_name = json_path.stem
            outlier_indices = outliers_by_file.get(file_name, [])

            for idx, ann in enumerate(data["annotations"]):
                if idx not in outlier_indices:
                    bbox = ann["boundingBox"]
                    area = bbox["width"] * bbox["height"]
                    aspect_ratio = (
                        bbox["width"] / bbox["height"] if bbox["height"] > 0 else 1
                    )
                    all_areas.append(area)
                    all_aspect_ratios.append(aspect_ratio)

        # Define clipping bounds (5th and 95th percentiles of non-outliers)
        area_bounds = np.percentile(all_areas, [5, 95])
        ar_bounds = np.percentile(all_aspect_ratios, [5, 95])

        print(f"Clipping area to range: {area_bounds}")
        print(f"Clipping aspect ratio to range: {ar_bounds}")

        # Apply clipping
        output_dir = self.results_path / "clipped_outliers" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        clipped_count = 0

        for file_name, outlier_indices in outliers_by_file.items():
            json_path = label_dir / f"{file_name}.json"

            with open(json_path, "r") as f:
                data = json.load(f)

            modified = False

            for idx in outlier_indices:
                if idx < len(data["annotations"]):
                    bbox = data["annotations"][idx]["boundingBox"]

                    # Current dimensions
                    w, h = bbox["width"], bbox["height"]
                    area = w * h
                    ar = w / h if h > 0 else 1

                    # Clip if needed
                    if area < area_bounds[0] or area > area_bounds[1]:
                        # Scale to bring area within bounds
                        if area < area_bounds[0]:
                            scale = np.sqrt(area_bounds[0] / area)
                        else:
                            scale = np.sqrt(area_bounds[1] / area)

                        bbox["width"] = w * scale
                        bbox["height"] = h * scale
                        modified = True
                        clipped_count += 1

            if modified:
                output_path = output_dir / f"{file_name}.json"
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

        print(f"Clipped {clipped_count} outlier boxes")

    def _separate_outliers(self, split, outliers_by_file):
        """Move files with outliers to separate directory for manual review"""
        print("Separating files with outliers...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"

        outlier_label_dir = self.results_path / "outliers_for_review" / split / "labels"
        outlier_image_dir = self.results_path / "outliers_for_review" / split / "images"
        outlier_label_dir.mkdir(parents=True, exist_ok=True)
        outlier_image_dir.mkdir(parents=True, exist_ok=True)

        for file_name in outliers_by_file:
            # Copy files
            src_json = label_dir / f"{file_name}.json"
            src_img = image_dir / f"{file_name}.png"

            if src_json.exists():
                shutil.copy2(src_json, outlier_label_dir / f"{file_name}.json")
            if src_img.exists():
                shutil.copy2(src_img, outlier_image_dir / f"{file_name}.png")

        print(f"Separated {len(outliers_by_file)} files with outliers for review")


# Usage
if __name__ == "__main__":
    # First detect outliers
    from outlier_detector import OutlierDetector

    detector = OutlierDetector("dataset")
    detector.detect_outliers("train")
    detector.visualize_outlier_samples("train")

    # Then handle them
    handler = OutlierHandler(
        "dataset", "results/data_analysis/stats/train_outlier_report.json"
    )

    # Choose your strategy:
    handler.handle_outliers("train", strategy="remove_extreme")  # Recommended
    # handler.handle_outliers("train", strategy="clip")
    # handler.handle_outliers("train", strategy="separate")
    # handler.handle_outliers("train", strategy="keep")  # For now, just analyze
