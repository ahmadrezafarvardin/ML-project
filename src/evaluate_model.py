# src/evaluate_model.py (updated)
import torch
from torch.utils.data import DataLoader
from models import FasterRCNN
from data import CharacterDetectionDataset, collate_fn
from collections import defaultdict
from pathlib import Path
import json
from tqdm import tqdm


def evaluate_model(model_path="results/checkpoints/best_model.pth", save_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FasterRCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load validation dataset
    val_dataset = CharacterDetectionDataset("dataset", split="valid")
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Collect statistics
    stats = defaultdict(list)
    all_predictions = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                # Count detections
                num_detections = len(pred["boxes"])
                num_gt = len(target["boxes"])

                # High confidence detections
                high_conf = (pred["scores"] > 0.5).sum().item()

                stats["num_detections"].append(num_detections)
                stats["num_gt_boxes"].append(num_gt)
                stats["high_conf_detections"].append(high_conf)

                # Store predictions for detailed analysis
                all_predictions.append(
                    {
                        "image_id": idx,
                        "predictions": {
                            "boxes": pred["boxes"].cpu().tolist(),
                            "scores": pred["scores"].cpu().tolist(),
                            "labels": pred["labels"].cpu().tolist(),
                        },
                        "ground_truth": {
                            "boxes": target["boxes"].cpu().tolist(),
                            "labels": target["labels"].cpu().tolist(),
                        },
                    }
                )

    # Calculate statistics
    results = {
        "model_checkpoint": model_path,
        "num_images": len(stats["num_detections"]),
        "avg_detections_per_image": sum(stats["num_detections"])
        / len(stats["num_detections"]),
        "avg_gt_boxes_per_image": sum(stats["num_gt_boxes"])
        / len(stats["num_gt_boxes"]),
        "avg_high_conf_detections": sum(stats["high_conf_detections"])
        / len(stats["high_conf_detections"]),
        "total_detections": sum(stats["num_detections"]),
        "total_gt_boxes": sum(stats["num_gt_boxes"]),
        "detection_rate": (
            sum(stats["high_conf_detections"]) / sum(stats["num_gt_boxes"])
            if sum(stats["num_gt_boxes"]) > 0
            else 0
        ),
    }

    # Print statistics
    print(f"\nEvaluation Results:")
    print(f"Average detections per image: {results['avg_detections_per_image']:.2f}")
    print(f"Average GT boxes per image: {results['avg_gt_boxes_per_image']:.2f}")
    print(
        f"Average high-confidence detections: {results['avg_high_conf_detections']:.2f}"
    )
    print(f"Detection rate (high conf / GT): {results['detection_rate']:.2%}")

    if save_results:
        # Save results
        output_dir = Path("results/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(output_dir / "evaluation_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save detailed predictions (first 10 for space)
        with open(output_dir / "sample_predictions.json", "w") as f:
            json.dump(all_predictions[:10], f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    evaluate_model()
# # src/evaluate_model.py
# import torch
# from torch.utils.data import DataLoader
# from models import FasterRCNN
# from data import CharacterDetectionDataset, collate_fn
# from collections import defaultdict


# def evaluate_model(model_path="checkpoints/best_model.pth"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model
#     model = FasterRCNN(num_classes=2)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()

#     # Load validation dataset
#     val_dataset = CharacterDetectionDataset("dataset", split="valid")
#     val_loader = DataLoader(
#         val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
#     )

#     # Collect statistics
#     stats = defaultdict(list)

#     with torch.no_grad():
#         for images, targets in val_loader:
#             images = [img.to(device) for img in images]
#             predictions = model(images)

#             for pred, target in zip(predictions, targets):
#                 # Count detections
#                 num_detections = len(pred["boxes"])
#                 num_gt = len(target["boxes"])

#                 # High confidence detections
#                 high_conf = (pred["scores"] > 0.5).sum().item()

#                 stats["num_detections"].append(num_detections)
#                 stats["num_gt_boxes"].append(num_gt)
#                 stats["high_conf_detections"].append(high_conf)

#     # Print statistics
#     print(
#         f"Average detections per image: {sum(stats['num_detections'])/len(stats['num_detections']):.2f}"
#     )
#     print(
#         f"Average GT boxes per image: {sum(stats['num_gt_boxes'])/len(stats['num_gt_boxes']):.2f}"
#     )
#     print(
#         f"Average high-confidence detections: {sum(stats['high_conf_detections'])/len(stats['high_conf_detections']):.2f}"
#     )


# if __name__ == "__main__":
#     evaluate_model()
