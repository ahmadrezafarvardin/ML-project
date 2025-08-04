# src/data_analysis/data_cleaner.py
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple


class DataCleaner:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path("results/data_analysis")
        self.cleaning_log = {
            "fixed_boxes": [],
            "removed_boxes": [],
            "moved_incomplete": [],
            "fixed_expressions": [],
        }

    def clean_bounding_boxes(self, split: str = "train", fix_issues: bool = True):
        """Clean and fix bounding box issues"""
        print(f"\nCleaning bounding boxes for {split}...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"
        cleaned_dir = self.results_path / "cleaned_data" / split
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        for json_path in label_dir.glob("*.json"):
            img_path = image_dir / f"{json_path.stem}.png"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            with open(json_path, "r") as f:
                data = json.load(f)

            if "annotations" not in data:
                data["annotations"] = []

            cleaned_annotations = []
            modified = False

            for idx, box in enumerate(data["annotations"]):
                # Skip if missing required fields
                if "boundingBox" not in box or "class" not in box:
                    self.cleaning_log["removed_boxes"].append(
                        {
                            "file": json_path.stem,
                            "reason": "missing_fields",
                            "box_idx": idx,
                        }
                    )
                    modified = True
                    continue

                # Extract bbox from nested structure
                bbox = box["boundingBox"]

                # Check if bbox has required fields
                if not all(field in bbox for field in ["x", "y", "width", "height"]):
                    self.cleaning_log["removed_boxes"].append(
                        {
                            "file": json_path.stem,
                            "reason": "missing_bbox_fields",
                            "box_idx": idx,
                        }
                    )
                    modified = True
                    continue

                # Fix coordinate issues
                if fix_issues:
                    original_bbox = bbox.copy()

                    # Clip coordinates to image bounds
                    bbox["x"] = max(0, min(bbox["x"], w - 1))
                    bbox["y"] = max(0, min(bbox["y"], h - 1))

                    # Adjust width and height
                    max_width = w - bbox["x"]
                    max_height = h - bbox["y"]
                    bbox["width"] = max(1, min(bbox["width"], max_width))
                    bbox["height"] = max(1, min(bbox["height"], max_height))

                    # Check if box was modified
                    if bbox != original_bbox:
                        self.cleaning_log["fixed_boxes"].append(
                            {
                                "file": json_path.stem,
                                "box_idx": idx,
                                "original": original_bbox,
                                "fixed": bbox.copy(),
                            }
                        )
                        modified = True

                # Remove extremely small boxes
                area = bbox["width"] * bbox["height"]
                if area < 10:  # Less than 10 pixels
                    self.cleaning_log["removed_boxes"].append(
                        {
                            "file": json_path.stem,
                            "reason": "too_small",
                            "area": area,
                            "box_idx": idx,
                        }
                    )
                    modified = True
                    continue

                # Remove boxes larger than 50% of image
                if area > (w * h * 0.5):
                    self.cleaning_log["removed_boxes"].append(
                        {
                            "file": json_path.stem,
                            "reason": "too_large",
                            "area": area,
                            "box_idx": idx,
                        }
                    )
                    modified = True
                    continue

                cleaned_annotations.append(box)
            # Update data
            data["annotations"] = cleaned_annotations

            # Save cleaned version if modified
            if modified:
                cleaned_json_path = cleaned_dir / json_path.name
                with open(cleaned_json_path, "w") as f:
                    json.dump(data, f, indent=2)

        # Save cleaning log
        log_path = self.results_path / "stats" / f"{split}_cleaning_log.json"
        with open(log_path, "w") as f:
            json.dump(self.cleaning_log, f, indent=2)

        print(f"Fixed {len(self.cleaning_log['fixed_boxes'])} boxes")
        print(f"Removed {len(self.cleaning_log['removed_boxes'])} invalid boxes")

    def handle_incomplete_data(self, split: str = "train"):
        """Handle images with missing or incomplete annotations"""
        print(f"\nHandling incomplete data for {split}...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"
        incomplete_dir = self.results_path / "incomplete" / split
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        # Find images without annotations
        for json_path in label_dir.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            # Check for empty annotations
            if not data.get("annotations"):
                img_path = image_dir / f"{json_path.stem}.png"
                if img_path.exists():
                    # Move both image and json to incomplete folder
                    shutil.copy2(img_path, incomplete_dir / img_path.name)
                    shutil.copy2(json_path, incomplete_dir / json_path.name)

                    self.cleaning_log["moved_incomplete"].append(
                        {"file": json_path.stem, "reason": "no_annotations"}
                    )

            # For training data, handle missing expressions
            if split == "train" and not data.get("expression"):
                # Add null expression marker
                data["expression"] = None
                self.cleaning_log["fixed_expressions"].append(json_path.stem)

                # Save updated file
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

        print(f"Moved {len(self.cleaning_log['moved_incomplete'])} incomplete files")
        print(
            f"Fixed {len(self.cleaning_log['fixed_expressions'])} missing expressions"
        )
