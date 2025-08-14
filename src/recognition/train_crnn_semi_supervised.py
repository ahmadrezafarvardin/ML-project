# src/recognition/train_crnn_semi_supervised.py
from tkinter import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import cv2

from crnn_model import CRNN
from dataset import MathExpressionDataset, collate_fn
from train_crnn import CRNNTrainer
from inference import ExpressionRecognizer


class SemiSupervisedDataset(Dataset):
    """Dataset that combines labeled and pseudo-labeled data"""

    def __init__(self, labeled_samples, pseudo_labeled_samples, transform):
        self.samples = labeled_samples + pseudo_labeled_samples
        self.transform = transform
        self.char_to_idx = MathExpressionDataset.char_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            "is_pseudo": sample.get("is_pseudo", False),
        }


class SemiSupervisedCRNNTrainer:
    def __init__(self, initial_model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load initial model if provided
        if initial_model_path and Path(initial_model_path).exists():
            self.model = CRNN()
            checkpoint = torch.load(initial_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            print(f"Loaded initial model from {initial_model_path}")
        else:
            self.model = CRNN().to(self.device)
            print("Created new model")

        self.recognizer = (
            ExpressionRecognizer(initial_model_path) if initial_model_path else None
        )

    def generate_pseudo_labels(self, unlabeled_images_dir, confidence_threshold=0.95):
        """Generate pseudo-labels for unlabeled images"""

        if not self.recognizer:
            print("No trained model available for pseudo-labeling")
            return []

        pseudo_labeled_samples = []
        unlabeled_images = list(Path(unlabeled_images_dir).glob("*.png"))

        print(f"Generating pseudo-labels for {len(unlabeled_images)} images...")

        for img_path in tqdm(unlabeled_images):
            # Get prediction
            expression = self.recognizer.recognize(str(img_path))

            # Get confidence (simplified - based on output probabilities)
            confidence = self._get_prediction_confidence(img_path)

            if confidence > confidence_threshold:
                pseudo_labeled_samples.append(
                    {
                        "image_path": str(img_path),
                        "expression": expression,
                        "is_pseudo": True,
                        "confidence": confidence,
                    }
                )

        print(f"Generated {len(pseudo_labeled_samples)} high-confidence pseudo-labels")
        return pseudo_labeled_samples

    def _get_prediction_confidence(self, image_path):
        """Calculate confidence score for a prediction"""
        img_tensor = (
            self.recognizer.preprocess_image(image_path).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=2)

            # Calculate average max probability across sequence
            max_probs, _ = torch.max(probs, dim=2)
            # Exclude blank token positions
            non_blank_probs = max_probs[max_probs > 0.5]

            if len(non_blank_probs) > 0:
                confidence = non_blank_probs.mean().item()
            else:
                confidence = 0.0

        return confidence

    def train_semi_supervised(
        self,
        labeled_dataset_path,
        unlabeled_images_dir,
        output_dir,
        initial_epochs=50,
        semi_supervised_epochs=30,
        confidence_threshold=0.9,
        pseudo_label_weight=0.5,
    ):
        """Train with semi-supervised learning"""

        # Step 1: Initial training on labeled data
        print("Step 1: Initial training on labeled data...")
        trainer = CRNNTrainer(self.model, self.device)

        # Load labeled data
        train_dataset = MathExpressionDataset(
            labeled_dataset_path, split="train", augment=True
        )
        val_dataset = MathExpressionDataset(labeled_dataset_path, split="valid")

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

        # Initial training
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(1, initial_epochs + 1):
            train_loss = trainer.train_epoch(train_loader, optimizer, epoch)

            if epoch % 10 == 0:
                val_loss, val_acc, val_dist, _, _ = trainer.validate(val_loader)
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Val Acc={val_acc:.4f}, Val Dist={val_dist:.4f}"
                )

        # Save initial model
        initial_model_path = f"{output_dir}/initial_model.pth"
        torch.save(
            {"model_state_dict": self.model.state_dict(), "epoch": initial_epochs},
            initial_model_path,
        )

        # Update recognizer with trained model
        self.recognizer = ExpressionRecognizer(initial_model_path)

        # Step 2: Generate pseudo-labels
        print("\nStep 2: Generating pseudo-labels...")
        pseudo_samples = self.generate_pseudo_labels(
            unlabeled_images_dir, confidence_threshold
        )

        if not pseudo_samples:
            print("No high-confidence pseudo-labels generated. Stopping.")
            return

        # Step 3: Retrain with combined data
        print("\nStep 3: Retraining with pseudo-labeled data...")

        # Combine labeled and pseudo-labeled data
        labeled_samples = train_dataset.samples
        combined_samples = labeled_samples + pseudo_samples

        # Create new dataset
        combined_dataset = SemiSupervisedDataset(
            labeled_samples, pseudo_samples, train_dataset.transform
        )

        combined_loader = DataLoader(
            combined_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

        # Modified loss function for pseudo-labels
        criterion = nn.CTCLoss(blank=16, zero_infinity=True)

        # Continue training
        optimizer = optim.Adam(self.model.parameters(), lr=0.00005)  # Lower LR

        for epoch in range(1, semi_supervised_epochs + 1):
            self.model.train()
            total_loss = 0

            pbar = tqdm(combined_loader, desc=f"Semi-supervised Epoch {epoch}")
            for batch in pbar:
                images = batch["images"].to(self.device)
                targets = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"]
                is_pseudo = batch.get("is_pseudo", [False] * len(images))

                # Forward pass
                outputs = self.model(images)
                input_lengths = torch.full(
                    (images.size(0),), outputs.size(0), dtype=torch.long
                )

                # Calculate weighted loss
                loss = criterion(
                    outputs.log_softmax(2), targets, input_lengths, target_lengths
                )

                # Apply pseudo-label weight
                if any(is_pseudo):
                    # This is simplified - in practice you might weight individual samples
                    loss = loss * (
                        pseudo_label_weight
                        if sum(is_pseudo) > len(is_pseudo) // 2
                        else 1.0
                    )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            # Validate
            if epoch % 5 == 0:
                val_loss, val_acc, val_dist, _, _ = trainer.validate(val_loader)
                print(
                    f"Semi-supervised Epoch {epoch}: "
                    f"Train Loss={total_loss/len(combined_loader):.4f}, "
                    f"Val Acc={val_acc:.4f}, Val Dist={val_dist:.4f}"
                )

        # Save final model
        final_model_path = f"{output_dir}/semi_supervised_model.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "training_info": {
                    "labeled_samples": len(labeled_samples),
                    "pseudo_samples": len(pseudo_samples),
                    "confidence_threshold": confidence_threshold,
                },
            },
            final_model_path,
        )

        print(f"\nSemi-supervised training complete!")
        print(f"Final model saved to {final_model_path}")

        return self.model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semi-supervised CRNN training")
    parser.add_argument(
        "--labeled-data", default="dataset_extended", help="Path to labeled dataset"
    )
    parser.add_argument(
        "--unlabeled-data",
        default="dataset/test/images",
        help="Path to unlabeled images",
    )
    parser.add_argument(
        "--initial-model", default=None, help="Path to pre-trained model (optional)"
    )
    parser.add_argument(
        "--output",
        default="results/recognition/semi_supervised",
        help="Output directory",
    )
    parser.add_argument(
        "--initial-epochs", type=int, default=50, help="Epochs for initial training"
    )
    parser.add_argument(
        "--semi-epochs",
        type=int,
        default=30,
        help="Epochs for semi-supervised training",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Confidence threshold for pseudo-labels",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = SemiSupervisedCRNNTrainer(args.initial_model)

    # Train
    trainer.train_semi_supervised(
        labeled_dataset_path=args.labeled_data,
        unlabeled_images_dir=args.unlabeled_data,
        output_dir=args.output,
        initial_epochs=args.initial_epochs,
        semi_supervised_epochs=args.semi_epochs,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
