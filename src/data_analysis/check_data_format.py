# src/data_analysis/check_data_format.py
import json
from pathlib import Path


def check_annotation_format(dataset_path: str = "dataset"):
    """Check the actual format of annotations"""
    dataset_path = Path(dataset_path)

    # Check train labels
    label_dir = dataset_path / "train" / "labels"

    # Get first few JSON files
    json_files = list(label_dir.glob("*.json"))[:3]

    print("Checking annotation format...\n")

    for json_file in json_files:
        print(f"File: {json_file.name}")
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"Keys in JSON: {list(data.keys())}")

        if "annotations" in data:
            if data["annotations"]:
                print(f"First annotation: {data['annotations'][0]}")
                print(f"Annotation keys: {list(data['annotations'][0].keys())}")
            else:
                print("Empty annotations list")

        if "expression" in data:
            print(f"Expression: {data['expression']}")

        print("-" * 50)


if __name__ == "__main__":
    check_annotation_format()
