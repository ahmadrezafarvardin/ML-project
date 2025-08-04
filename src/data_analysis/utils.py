# src/data_analysis/utils.py
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def find_overlapping_boxes(
    annotations: List[Dict], iou_threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """Find overlapping boxes in annotations"""
    overlapping_pairs = []

    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            iou = calculate_iou(annotations[i], annotations[j])
            if iou > iou_threshold:
                overlapping_pairs.append((i, j))

    return overlapping_pairs


def augment_box(
    box: Dict,
    image_shape: Tuple[int, int],
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_range: int = 5,
) -> Dict:
    """Apply small augmentation to a bounding box"""
    h, w = image_shape
    augmented = box.copy()

    # Random scale
    scale = np.random.uniform(*scale_range)
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2

    new_width = box["width"] * scale
    new_height = box["height"] * scale

    augmented["width"] = int(new_width)
    augmented["height"] = int(new_height)
    augmented["x"] = int(center_x - new_width / 2)
    augmented["y"] = int(center_y - new_height / 2)

    # Random shift
    augmented["x"] += np.random.randint(-shift_range, shift_range + 1)
    augmented["y"] += np.random.randint(-shift_range, shift_range + 1)

    # Ensure within bounds
    augmented["x"] = max(0, min(augmented["x"], w - augmented["width"]))
    augmented["y"] = max(0, min(augmented["y"], h - augmented["height"]))

    return augmented


# Create requirements.txt
requirements_content = """numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.2.0
Pillow>=8.0.0
tqdm>=4.60.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements_content)
