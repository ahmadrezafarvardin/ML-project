# src/data_analysis/data_visualizer.py
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from typing import List, Dict, DefaultDict


class DataVisualizer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path("results/data_analysis")
        self.colors = self._generate_colors(
            100
        )  # Generate colors for up to 100 classes

    def _generate_colors(self, n):
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def visualize_samples(self, split: str = "train", n_samples: int = 1000):
        """Visualize random samples with bounding boxes"""
        print(f"\nVisualizing {n_samples} samples from {split}...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"
        vis_dir = self.results_path / "sample_visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Get random samples
        json_files = list(label_dir.glob("*.json"))
        samples = random.sample(json_files, min(n_samples, len(json_files)))

        # Create label to color mapping
        all_labels = set()
        for json_path in samples:
            with open(json_path, "r") as f:
                data = json.load(f)
            if "annotations" in data:
                for box in data["annotations"]:
                    all_labels.add(box.get("label", "unknown"))

        label_to_color = {
            label: self.colors[i % len(self.colors)]
            for i, label in enumerate(sorted(all_labels))
        }

        for json_path in samples:
            img_path = image_dir / f"{json_path.stem}.png"
            if not img_path.exists():
                continue

            # Load image and annotations
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(json_path, "r") as f:
                data = json.load(f)

            # Draw bounding boxes
            if "annotations" in data:
                for box in data["annotations"]:
                    x, y = int(box["boundingBox"]["x"]), int(box["boundingBox"]["y"])
                    w, h = int(box["boundingBox"]["width"]), int(
                        box["boundingBox"]["height"]
                    )
                    label = box.get("label", "unknown")
                    color = label_to_color.get(label, (255, 0, 0))

                    # Draw rectangle
                    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)

                    # Add label text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

                    # Background for text
                    cv2.rectangle(
                        img_rgb,
                        (x, y - text_size[1] - 4),
                        (x + text_size[0], y),
                        color,
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        img_rgb,
                        label,
                        (x, y - 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

            # Save visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.title(f"Sample: {json_path.stem}")
            plt.axis("off")

            # Add expression if available
            if "expression" in data and data["expression"]:
                plt.text(
                    0.5,
                    -0.05,
                    f'Expression: {data["expression"]}',
                    transform=plt.gca().transAxes,
                    ha="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            plt.tight_layout()
            plt.savefig(
                vis_dir / f"{json_path.stem}_annotated.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        print(f"Saved {len(samples)} visualizations to {vis_dir}")

    def create_class_examples_grid(
        self, split: str = "train", examples_per_class: int = 5
    ):
        """Create a grid showing examples of each character class"""
        print(f"\nCreating class examples grid for {split}...")

        label_dir = self.dataset_path / split / "labels"
        image_dir = self.dataset_path / split / "images"

        # Collect examples for each class
        class_examples = DefaultDict(list)

        for json_path in label_dir.glob("*.json"):
            img_path = image_dir / f"{json_path.stem}.png"
            if not img_path.exists():
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            if "annotations" in data:
                img = cv2.imread(str(img_path))

                for box in data["annotations"]:
                    label = box.get("label", "unknown")
                    if len(class_examples[label]) < examples_per_class:
                        x, y = int(box["boundingBox"]["x"]), int(
                            box["boundingBox"]["y"]
                        )
                        w, h = int(box["boundingBox"]["width"]), int(
                            box["boundingBox"]["height"]
                        )

                        # Extract character region
                        char_img = img[y : y + h, x : x + w]
                        if char_img.size > 0:
                            class_examples[label].append(char_img)

        # Create grid visualization
        n_classes = len(class_examples)
        if n_classes == 0:
            print("No classes found!")
            return

        fig, axes = plt.subplots(
            n_classes,
            examples_per_class,
            figsize=(examples_per_class * 2, n_classes * 2),
        )
        axes = np.atleast_2d(axes)

        # if n_classes == 1:
        #     axes = axes.reshape(1, -1)

        for i, (label, examples) in enumerate(sorted(class_examples.items())):
            for j in range(examples_per_class):
                ax = axes[i, j]

                if j < len(examples):
                    # Resize for consistent display
                    char_img = examples[j]
                    char_img_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)

                    # Pad to square
                    h, w = char_img_rgb.shape[:2]
                    size = max(h, w)
                    padded = np.ones((size, size, 3), dtype=np.uint8) * 255
                    y_offset = (size - h) // 2
                    x_offset = (size - w) // 2
                    padded[y_offset : y_offset + h, x_offset : x_offset + w] = (
                        char_img_rgb
                    )

                    ax.imshow(padded)
                    if j == 0:
                        ax.set_ylabel(f'"{label}"', fontsize=12, fontweight="bold")
                else:
                    ax.axis("off")

                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            f"Character Class Examples - {split.capitalize()} Set", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            self.results_path / "plots" / f"{split}_class_examples_grid.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize dataset annotations")
    parser.add_argument(
        "--dataset", type=str, default="dataset", help="Path to dataset root"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid"],
        help="Dataset split",
    )
    parser.add_argument(
        "--all", action="store_true", help="Visualize all images in the split"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of random samples to visualize (ignored if --all)",
    )

    args = parser.parse_args()

    visualizer = DataVisualizer(args.dataset)
    if args.all:
        visualizer.visualize_samples(
            split=args.split, n_samples=1000000
        )  # Large number to cover all
    else:
        visualizer.visualize_samples(split=args.split, n_samples=args.n_samples)
