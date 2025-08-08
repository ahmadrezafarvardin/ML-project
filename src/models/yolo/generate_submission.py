import json
import csv

# Path to YOLO JSON results
yolo_json_path = "results/yolo/yolo_predictions_images.json"
# Path to output CSV
output_csv_path = "results/yolo/output.csv"

with open(yolo_json_path, "r") as f:
    yolo_results = json.load(f)

with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "x", "y", "width", "height"])

    for image_name, data in yolo_results.items():
        # Remove extension and convert to int (if needed)
        image_id = int(image_name.split(".")[0])
        for pred in data["predictions"]:
            x, y, w, h = pred["bbox"]
            writer.writerow([image_id, x, y, w, h])

print(f"CSV saved to: {output_csv_path}")
