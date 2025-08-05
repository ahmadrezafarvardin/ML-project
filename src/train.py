# src/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

from models import FasterRCNN
from data import CharacterDetectionDataset, collate_fn
import warnings

warnings.filterwarnings("ignore", message=".*iCCP.*")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for images, targets in progress_bar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        losses = model(images, targets)

        # Calculate total loss
        loss = sum(loss for loss in losses.values())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # Update progress
        epoch_loss += loss.item()
        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "rpn_cls": losses.get("rpn_cls_loss", 0).item(),
                "rpn_reg": losses.get("rpn_reg_loss", 0).item(),
                "cls": losses.get("loss_classifier", 0).item(),
                "reg": losses.get("loss_box_reg", 0).item(),
            }
        )

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            predictions = model(images)
            all_predictions.extend(predictions)

    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-classes", type=int, default=2)  # background + character
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default="results/checkpoints")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument(
        "--pretrained-backbone", action="store_true", help="Use pretrained backbone"
    )
    args = parser.parse_args()

    # Create save directory
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_dataset = CharacterDetectionDataset(args.data_path, split="train")
    val_dataset = CharacterDetectionDataset(args.data_path, split="valid")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Model
    model = FasterRCNN(
        num_classes=args.num_classes,
        backbone_name="resnet50",
        pretrained_backbone=args.pretrained_backbone,  # for no pretrained models set False
        rpn_anchor_sizes=(8, 16, 32, 64, 128),
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # Training loop
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Update learning rate
        lr_scheduler.step()

        print(
            f'Epoch {epoch}: Train Loss = {train_loss:.4f}, LR = {optimizer.param_groups[0]["lr"]:.6f}'
        )

        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                save_path / "best_model.pth",
            )
            print(f"Saved best model with loss: {train_loss:.4f}")

        # Periodic save
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                save_path / f"checkpoint_epoch_{epoch}.pth",
            )

            # Evaluate on validation set
            print("Evaluating on validation set...")
            predictions = evaluate(model, val_loader, device)
            print(f"Generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()
