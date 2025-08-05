# Quick debug script to see raw scores
# src/check_raw_scores.py
import torch
from models import FasterRCNN
from data import CharacterDetectionDataset, collate_fn
from torch.utils.data import DataLoader


def check_raw_classifier_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get one sample
    dataset = CharacterDetectionDataset("dataset", split="valid")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    images = [img.to(device) for img in images]

    # Hook to capture intermediate outputs
    intermediate_outputs = {}

    def hook_fn(module, input, output):
        intermediate_outputs["roi_head_output"] = output

    # Register hook on the ROI head
    hook = model.roi_heads.box_head.register_forward_hook(hook_fn)

    with torch.no_grad():
        predictions = model(images)

    hook.remove()

    # Check the raw classifier scores
    if "roi_head_output" in intermediate_outputs:
        class_logits, box_regression = intermediate_outputs["roi_head_output"]
        raw_scores = torch.softmax(class_logits, dim=-1)

        print("Raw classifier output analysis:")
        print(f"  Shape: {class_logits.shape}")
        print(
            f"  Background scores: {raw_scores[:, 0].mean():.4f} (±{raw_scores[:, 0].std():.4f})"
        )
        print(
            f"  Character scores: {raw_scores[:, 1].mean():.4f} (±{raw_scores[:, 1].std():.4f})"
        )
        print(f"  Sample raw logits: {class_logits[:5]}")


if __name__ == "__main__":
    check_raw_classifier_output()
