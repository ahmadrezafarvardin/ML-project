# src/data_analysis/data_validator.py
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from datetime import datetime


class DataValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path("results/data_analysis")
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "corrupted_images": [],
            "missing_annotations": [],
            "empty_annotations": [],
            "invalid_boxes": [],
            "missing_expressions": [],
        }

    def validate_images(self, split: str = "train") -> List[Path]:
        """Check for corrupted or unreadable images"""
        print(f"\nValidating {split} images...")
        image_dir = self.dataset_path / split / "images"
        bad_images = []

        for img_path in image_dir.glob("*.png"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    bad_images.append(img_path)
                    self.validation_report["corrupted_images"].append(str(img_path))
                elif img.shape[0] == 0 or img.shape[1] == 0:
                    bad_images.append(img_path)
                    self.validation_report["corrupted_images"].append(str(img_path))
            except Exception as e:
                bad_images.append(img_path)
                self.validation_report["corrupted_images"].append(
                    {"path": str(img_path), "error": str(e)}
                )

        # Move corrupted images
        if bad_images:
            bad_dir = self.results_path / "bad_images" / split
            bad_dir.mkdir(parents=True, exist_ok=True)
            for img_path in bad_images:
                shutil.move(str(img_path), str(bad_dir / img_path.name))

        print(f"Found {len(bad_images)} corrupted images in {split}")
        return bad_images

    def check_missing_annotations(self, split: str = "train") -> List[str]:
        """Find images without corresponding JSON files"""
        print(f"\nChecking missing annotations in {split}...")

        images = set(
            p.stem for p in (self.dataset_path / split / "images").glob("*.png")
        )
        labels = set(
            p.stem for p in (self.dataset_path / split / "labels").glob("*.json")
        )

        missing = list(images - labels)
        self.validation_report["missing_annotations"].extend(
            [f"{split}/{name}" for name in missing]
        )

        print(f"Missing annotations: {len(missing)}")
        return missing

    def validate_annotations(self, split: str = "train") -> Dict:
        """Validate JSON structure and bounding boxes"""
        print(f"\nValidating {split} annotations...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"

        stats = {
            "total_annotations": 0,
            "empty_annotations": [],
            "invalid_boxes": [],
            "missing_expressions": [],
            "box_issues": [],
        }

        for json_path in label_dir.glob("*.json"):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Check image exists and get dimensions
                img_path = image_dir / f"{json_path.stem}.png"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Check annotations
                if "annotations" not in data or not data["annotations"]:
                    stats["empty_annotations"].append(json_path.stem)
                    self.validation_report["empty_annotations"].append(
                        f"{split}/{json_path.stem}"
                    )
                else:
                    stats["total_annotations"] += len(data["annotations"])

                    # Validate each box
                    for idx, box in enumerate(data["annotations"]):
                        issues = []

                        # Check required fields
                        required = ["x", "y", "width", "height", "label"]
                        for field in required:
                            if field not in box:
                                issues.append(f"missing_{field}")

                        if not issues:  # Only check values if fields exist
                            # Check bounds
                            if box["x"] < 0 or box["y"] < 0:
                                issues.append("negative_coordinates")
                            if box["x"] + box["width"] > w:
                                issues.append("exceeds_width")
                            if box["y"] + box["height"] > h:
                                issues.append("exceeds_height")
                            if box["width"] <= 0 or box["height"] <= 0:
                                issues.append("invalid_dimensions")

                            # Check for extremely small or large boxes
                            area = box["width"] * box["height"]
                            img_area = w * h
                            if area < 10:  # Less than 10 pixels
                                issues.append("too_small")
                            if area > img_area * 0.5:  # More than 50% of image
                                issues.append("too_large")

                        if issues:
                            stats["box_issues"].append(
                                {
                                    "file": json_path.stem,
                                    "box_idx": idx,
                                    "issues": issues,
                                }
                            )

                # Check expression (for train data)
                if split == "train":
                    if "expression" not in data or not data.get("expression"):
                        stats["missing_expressions"].append(json_path.stem)
                        self.validation_report["missing_expressions"].append(
                            f"{split}/{json_path.stem}"
                        )

            except Exception as e:
                stats["invalid_boxes"].append({"file": json_path.stem, "error": str(e)})

        return stats

    def save_validation_report(self):
        """Save comprehensive validation report"""
        report_path = self.results_path / "stats" / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.validation_report, f, indent=2)
        print(f"\nValidation report saved to: {report_path}")
