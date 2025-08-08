# src/utils/debug_training.py
import torch
from src.models.fasterrcnn import FasterRCNN
from src.data import CharacterDetectionDataset, collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=".*iCCP.*")


def debug_model_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FasterRCNN(num_classes=2)
    checkpoint_path = "results/fasterrcnn/checkpoints/best_model.pth"  # or your actual checkpoint path
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using random weights.")

    model.to(device)
    model.train()  # Set to training mode
    # Load one batch
    dataset = CharacterDetectionDataset("dataset", split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    images, targets = next(iter(loader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Forward pass
    losses = model(images, targets)

    print("Training mode - Loss components:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # Now check inference mode
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    print("\nInference mode - Predictions:")
    pred = predictions[0]
    print(f"  Number of boxes: {len(pred['boxes'])}")
    if len(pred["scores"]) > 0:
        print(f"  Score statistics:")
        print(f"    Max: {pred['scores'].max().item():.4f}")
        print(f"    Mean: {pred['scores'].mean().item():.4f}")
        print(f"    Min: {pred['scores'].min().item():.4f}")

        # Check score distribution
        score_bins = torch.histc(pred["scores"], bins=10, min=0, max=1)
        print(f"  Score distribution (10 bins from 0 to 1):")
        for i, count in enumerate(score_bins):
            print(f"    {i/10:.1f}-{(i+1)/10:.1f}: {int(count)} boxes")


if __name__ == "__main__":
    debug_model_training()
