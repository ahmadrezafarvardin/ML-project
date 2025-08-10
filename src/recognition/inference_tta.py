# src/recognition/inference_tta.py
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import csv
from tqdm import tqdm
from collections import Counter

from crnn_model import CRNN
from inference import ExpressionRecognizer


class ExpressionRecognizerTTA(ExpressionRecognizer):
    """Expression recognizer with Test Time Augmentation"""

    def __init__(self, model_path, device="cuda"):
        super().__init__(model_path, device)

        # Define augmentation transforms
        self.tta_transforms = [
            # Original
            transforms.Compose(
                [
                    transforms.Resize((64, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            # Slight rotation
            transforms.Compose(
                [
                    transforms.Resize((64, 256)),
                    transforms.RandomAffine(degrees=(-3, 3)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            # Slight scale
            transforms.Compose(
                [
                    transforms.Resize((66, 260)),
                    transforms.CenterCrop((64, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            # Slight brightness
            transforms.Compose(
                [
                    transforms.Resize((64, 256)),
                    transforms.ColorJitter(brightness=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
        ]

    def recognize_with_tta(self, image_path, num_aug=4):
        """Recognize with test time augmentation"""
        # Load image
        image = cv2.imread(str(image_path))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_pil = Image.fromarray(image_gray)

        predictions = []

        # Apply different augmentations
        for i in range(min(num_aug, len(self.tta_transforms))):
            transform = self.tta_transforms[i]
            img_tensor = transform(image_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                expression = self.decode_output(output)
                predictions.append(expression)

        # Vote for most common prediction
        if predictions:
            # Find most common prediction
            counter = Counter(predictions)
            most_common = counter.most_common(1)[0][0]

            # If there's high agreement, use it
            if counter[most_common] >= len(predictions) // 2:
                return most_common
            else:
                # Otherwise, use the first (original) prediction
                return predictions[0]

        return "0"

    def process_test_set_tta(self, test_dir, output_file):
        """Process test set with TTA"""
        test_dir = Path(test_dir)
        predictions = []

        test_images = sorted(test_dir.glob("*.png"))
        print(f"Processing {len(test_images)} test images with TTA...")

        for img_path in tqdm(test_images):
            try:
                # Extract image ID from filename
                image_id = int(img_path.stem)

                expression = self.recognize_with_tta(img_path)
                expression = self.post_process(expression)

                # Convert x to * for multiplication
                expression = expression.replace("x", "*")

                predictions.append({"image_id": image_id, "expression": expression})
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                try:
                    image_id = int(img_path.stem)
                except:
                    image_id = 0

                predictions.append({"image_id": image_id, "expression": "0"})

        # Sort by image_id
        predictions.sort(key=lambda x: x["image_id"])

        # Save predictions
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "expression"])
            writer.writeheader()
            writer.writerows(predictions)

        print(f"Saved predictions to {output_file}")
        return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="results/recognition/checkpoints/best_model.pth"
    )
    parser.add_argument("--test-dir", default="dataset/test/images")
    parser.add_argument(
        "--output", default="results/recognition/predictions/submission_tta.csv"
    )

    args = parser.parse_args()

    recognizer = ExpressionRecognizerTTA(args.model)
    recognizer.process_test_set_tta(args.test_dir, args.output)
