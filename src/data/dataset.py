# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import json
import cv2
from pathlib import Path
import numpy as np


class CharacterDetectionDataset(Dataset):
    """Dataset for character detection"""

    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.png")))

        # Since all boxes have the same class, we'll use a single class
        # You can modify this when you have proper labels
        self.class_names = ["background", "character"]  # 0 is background

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        json_path = self.label_dir / f"{img_path.stem}.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract boxes and labels
        boxes = []
        labels = []

        for ann in data["annotations"]:
            bbox = ann["boundingBox"]
            x = bbox["x"]
            y = bbox["y"]
            w = bbox["width"]
            h = bbox["height"]

            # Convert to [x1, y1, x2, y2] format
            boxes.append([x, y, x + w, y + h])

            # For now, all characters get the same label
            # You should replace this with actual character labels
            labels.append(1)  # 1 for 'character' class

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
