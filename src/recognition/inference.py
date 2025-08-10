# src/recognition/inference.py
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import csv
from tqdm import tqdm

from crnn_model import CRNN


class ExpressionRecognizer:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = CRNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def preprocess_image(self, image_path):
        """Preprocess image for CRNN"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to PIL
        image = Image.fromarray(image)

        # Apply transforms
        image = self.transform(image)

        # Add batch dimension
        return image.unsqueeze(0)

    def decode_output(self, output):
        """Decode CTC output to expression string"""
        output = output.cpu().numpy()

        # Greedy decoding
        pred = output[:, 0, :].argmax(axis=1)

        # Remove consecutive duplicates and blanks
        char_list = []
        prev = None
        for p in pred:
            if p != prev and p != 16:  # 16 is blank
                char_list.append(self.model.idx_to_char.get(p, "?"))
            prev = p

        return "".join(char_list)

    def recognize(self, image_path):
        """Recognize mathematical expression from image"""
        # Preprocess
        image = self.preprocess_image(image_path)
        image = image.to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(image)

        # Decode
        expression = self.decode_output(output)

        return expression

    def process_test_set(self, test_dir, output_file):
        """Process entire test set and save predictions"""
        test_dir = Path(test_dir)
        predictions = []

        # Get all test images
        test_images = sorted(test_dir.glob("*.png"))

        print(f"Processing {len(test_images)} test images...")

        for img_path in tqdm(test_images):
            try:
                expression = self.recognize(img_path)

                # Post-process if needed
                expression = self.post_process(expression)

                predictions.append({"image": img_path.name, "expression": expression})
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                predictions.append(
                    {"image": img_path.name, "expression": "0"}  # Default fallback
                )

        # Save predictions
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "expression"])
            writer.writeheader()
            writer.writerows(predictions)

        print(f"Saved predictions to {output_file}")

        return predictions

    def post_process(self, expression):
        """Post-process recognized expression"""
        # Remove any invalid characters
        valid_chars = set("0123456789+-x/()")
        expression = "".join(c for c in expression if c in valid_chars)

        # Fix common errors
        expression = expression.replace("*", "x")

        # Balance parentheses
        open_count = expression.count("(")
        close_count = expression.count(")")
        if open_count > close_count:
            expression += ")" * (open_count - close_count)
        elif close_count > open_count:
            expression = "(" * (close_count - open_count) + expression

        # Default to '0' if empty
        if not expression:
            expression = "0"

        return expression


def evaluate_on_validation(model_path, dataset_path):
    """Evaluate model on validation set"""
    from dataset import MathExpressionDataset
    from Levenshtein import distance

    recognizer = ExpressionRecognizer(model_path)

    # Load validation data
    val_dataset = MathExpressionDataset(dataset_path, split="valid")

    predictions = []
    targets = []

    print("Evaluating on validation set...")

    for i in tqdm(range(len(val_dataset))):
        sample = val_dataset.samples[i]

        # Recognize
        pred = recognizer.recognize(sample["image_path"])
        target = sample["expression"]

        predictions.append(pred)
        targets.append(target)

    # Calculate metrics
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    accuracy = correct / len(predictions)

    avg_distance = np.mean([distance(p, t) for p, t in zip(predictions, targets)])

    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Levenshtein Distance: {avg_distance:.4f}")

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        print(f"Target: '{targets[i]}' -> Predicted: '{predictions[i]}'")

    return accuracy, avg_distance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with CRNN model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", help="Path to test images directory")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on validation set"
    )
    parser.add_argument(
        "--dataset", default="dataset", help="Path to dataset (for evaluation)"
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate_on_validation(args.model, args.dataset)
    elif args.test_dir:
        recognizer = ExpressionRecognizer(args.model)
        recognizer.process_test_set(args.test_dir, args.output)
    else:
        print("Please specify either --evaluate or --test-dir")
