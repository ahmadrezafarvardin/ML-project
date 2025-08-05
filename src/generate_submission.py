# src/generate_submission.py
from inference import InferenceEngine
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import cv2
import csv


def generate_competition_csv(
    checkpoint_path="results/checkpoints/best_model.pth",
    nms_threshold=0.3,
    score_threshold=0.6,
    test_dir="dataset/test/images",
    output_csv="results/output.csv",
):
    engine = InferenceEngine(checkpoint_path, nms_threshold, score_threshold)
    test_dir = Path(test_dir)
    test_images = sorted(test_dir.glob("*.png"))  # sort for consistent order

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "x", "y", "width", "height"])

        for img_path in test_images:
            # Extract image_id (assumes filename is like 546.png)
            image_id = int(img_path.stem)

            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

            pred_boxes, pred_scores, pred_labels = engine.predict(img_tensor)

            for box in pred_boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                writer.writerow(
                    [image_id, float(x1), float(y1), float(width), float(height)]
                )

    print(f"Submission file saved to: {output_csv}")


if __name__ == "__main__":
    generate_competition_csv(
        checkpoint_path="results/checkpoints/best_model.pth",
        nms_threshold=0.3,
        score_threshold=0.6,
        test_dir="dataset/test/images",
        output_csv="results/output.csv",
    )
