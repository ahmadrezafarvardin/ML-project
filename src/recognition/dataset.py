# src/recognition/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms


class MathExpressionDataset(Dataset):
    def __init__(
        self, root_dir, split="train", img_height=64, img_width=256, augment=False
    ):
        """
        Dataset for mathematical expressions
        Args:
            root_dir: path to dataset
            split: 'train' or 'valid'
            img_height: target height for images
            img_width: target width for images
            augment: whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment and (split == "train")

        self.images_dir = self.root_dir / split / "images"
        self.labels_dir = self.root_dir / split / "labels"

        # Character mapping
        self.char_to_idx = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "-": 11,
            "x": 12,
            "/": 13,
            "(": 14,
            ")": 15,
        }

        # Load data
        self.samples = self._load_samples()

        # Define transforms
        if self.augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width)),
                    transforms.RandomAffine(
                        degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        print(f"Loaded {len(self.samples)} samples from {split} set")

    def _load_samples(self):
        samples = []
        skipped = 0

        for label_file in self.labels_dir.glob("*.json"):
            with open(label_file, "r") as f:
                data = json.load(f)

            if "expression" in data and data["expression"]:
                img_name = label_file.stem + ".png"
                img_path = self.images_dir / img_name

                if img_path.exists():
                    # Convert expression to our character set
                    expression = data["expression"]
                    # Replace * with x
                    expression = expression.replace("*", "x")

                    # Check if all characters are valid after replacement
                    if all(c in self.char_to_idx for c in expression):
                        samples.append(
                            {"image_path": str(img_path), "expression": expression}
                        )
                    else:
                        skipped += 1
                        # Debug: show what characters are missing
                        invalid_chars = set(
                            c for c in expression if c not in self.char_to_idx
                        )
                        if skipped <= 5:  # Show first 5 skipped
                            print(
                                f"Skipped expression '{expression}' - invalid chars: {invalid_chars}"
                            )

        print(f"Skipped {skipped} samples with invalid characters")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to PIL for transforms
        image = Image.fromarray(image)
        image = self.transform(image)

        # Convert expression to indices
        expression = sample["expression"]
        target = [self.char_to_idx[c] for c in expression]

        return {
            "image": image,
            "target": torch.LongTensor(target),
            "target_length": len(target),
            "expression": expression,
        }


def collate_fn(batch):
    """Custom collate function for variable length targets"""
    images = torch.stack([item["image"] for item in batch])

    # Concatenate all targets
    targets = torch.cat([item["target"] for item in batch])

    # Target lengths
    target_lengths = torch.LongTensor([item["target_length"] for item in batch])

    # Original expressions for debugging
    expressions = [item["expression"] for item in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "expressions": expressions,
    }
