# src/recognition/train_crnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from crnn_model import CRNN
from dataset import MathExpressionDataset, collate_fn


class CRNNTrainer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.criterion = CTCLoss(blank=16, zero_infinity=True)

    def decode_predictions(self, preds, method="greedy"):
        """Decode CTC output to text"""
        preds = preds.cpu().numpy()
        decoded = []

        for i in range(preds.shape[1]):  # batch dimension
            if method == "greedy":
                # Greedy decoding
                pred = preds[:, i, :].argmax(axis=1)
                # Remove consecutive duplicates and blanks
                char_list = []
                prev = None
                for p in pred:
                    if p != prev and p != 16:  # 16 is blank
                        char_list.append(self.model.idx_to_char.get(p, "?"))
                    prev = p
                decoded.append("".join(char_list))

        return decoded

    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"]

            # Forward pass
            outputs = self.model(images)  # [seq_len, batch, nclass]

            # Calculate input lengths (all sequences have same length after CNN)
            input_lengths = torch.full(
                (images.size(0),), outputs.size(0), dtype=torch.long
            )

            # CTC Loss
            loss = self.criterion(
                outputs.log_softmax(2), targets, input_lengths, target_lengths
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5
            )  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch["images"].to(self.device)
                targets = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"]
                expressions = batch["expressions"]

                # Forward pass
                outputs = self.model(images)

                # Calculate lengths
                input_lengths = torch.full(
                    (images.size(0),), outputs.size(0), dtype=torch.long
                )

                # Loss
                loss = self.criterion(
                    outputs.log_softmax(2), targets, input_lengths, target_lengths
                )
                total_loss += loss.item()

                # Decode predictions
                predictions = self.decode_predictions(outputs)
                all_predictions.extend(predictions)
                all_targets.extend(expressions)

        # Calculate accuracy
        correct = sum(
            1 for pred, target in zip(all_predictions, all_targets) if pred == target
        )
        accuracy = correct / len(all_predictions)

        # Calculate average Levenshtein distance
        from Levenshtein import distance

        avg_distance = np.mean(
            [
                distance(pred, target)
                for pred, target in zip(all_predictions, all_targets)
            ]
        )

        return (
            total_loss / len(dataloader),
            accuracy,
            avg_distance,
            all_predictions,
            all_targets,
        )


def train_crnn(
    dataset_path="dataset",
    output_dir="results/recognition",
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    pretrained_path="src/models/resnet50_model_gpu.pth",
):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = MathExpressionDataset(dataset_path, split="train", augment=True)
    val_dataset = MathExpressionDataset(dataset_path, split="valid", augment=False)
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create model
    model = CRNN(pretrained_path=pretrained_path)
    trainer = CRNNTrainer(model, device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_distance": []}

    best_accuracy = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch)

        # Validate
        val_loss, val_accuracy, val_distance, predictions, targets = trainer.validate(
            val_loader
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_distance"].append(val_distance)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val Avg Distance: {val_distance:.4f}")

        # Show some predictions
        if epoch % 10 == 0:
            print("\nSample predictions:")
            for i in range(min(5, len(predictions))):
                print(f"  Target: '{targets[i]}' -> Predicted: '{predictions[i]}'")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_distance": val_distance,
                },
                f"{output_dir}/checkpoints/best_model.pth",
            )
            print(f"Saved best model with accuracy: {val_accuracy:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                f"{output_dir}/checkpoints/checkpoint_epoch_{epoch}.pth",
            )

    # Save final model and history
    torch.save(
        {"model_state_dict": model.state_dict(), "history": history},
        f"{output_dir}/checkpoints/final_model.pth",
    )

    with open(f"{output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training history
    plot_training_history(history, output_dir)

    return model, history


def plot_training_history(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history["val_accuracy"])
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy")

    # Levenshtein Distance
    axes[1, 0].plot(history["val_distance"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Average Distance")
    axes[1, 0].set_title("Average Levenshtein Distance")

    # Remove empty subplot
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CRNN model for expression recognition"
    )
    parser.add_argument("--dataset", default="dataset", help="Path to dataset")
    parser.add_argument(
        "--output", default="results/recognition", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--pretrained",
        default="src/models/resnet50_model_gpu.pth",
        help="Path to pretrained model",
    )

    args = parser.parse_args()

    train_crnn(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pretrained_path=args.pretrained,
    )
