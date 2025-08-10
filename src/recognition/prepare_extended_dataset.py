# prepare_extended_dataset.py
import json
import shutil
from pathlib import Path
import random


def create_extended_training_set(dataset_path="dataset", num_val_to_use=100):
    """
    Create extended training set by using some validation samples
    """
    # Create extended dataset directory
    extended_path = Path("dataset_extended")
    extended_train = extended_path / "train"
    extended_train_images = extended_train / "images"
    extended_train_labels = extended_train / "labels"

    extended_val = extended_path / "valid"
    extended_val_images = extended_val / "images"
    extended_val_labels = extended_val / "labels"

    # Create directories
    for dir_path in [
        extended_train_images,
        extended_train_labels,
        extended_val_images,
        extended_val_labels,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy original training data
    original_train_images = Path(dataset_path) / "train" / "images"
    original_train_labels = Path(dataset_path) / "train" / "labels"

    for img_file in original_train_images.glob("*.png"):
        shutil.copy2(img_file, extended_train_images)

    for label_file in original_train_labels.glob("*.json"):
        shutil.copy2(label_file, extended_train_labels)

    # Get validation files
    val_images = list((Path(dataset_path) / "valid" / "images").glob("*.png"))
    val_labels = list((Path(dataset_path) / "valid" / "labels").glob("*.json"))

    # Filter valid samples
    valid_samples = []
    for label_file in val_labels:
        with open(label_file, "r") as f:
            data = json.load(f)

        if "expression" in data and data["expression"]:
            expression = data["expression"].replace("*", "x")
            # Check if expression contains only valid characters
            valid_chars = set("0123456789+-x/()")
            if all(c in valid_chars for c in expression):
                img_file = (
                    Path(dataset_path) / "valid" / "images" / (label_file.stem + ".png")
                )
                if img_file.exists():
                    valid_samples.append((img_file, label_file))

    print(f"Found {len(valid_samples)} valid validation samples")

    # Randomly select samples for training
    random.shuffle(valid_samples)
    train_samples = valid_samples[:num_val_to_use]
    val_samples = valid_samples[num_val_to_use:]

    # Copy selected samples to extended training set
    for img_file, label_file in train_samples:
        shutil.copy2(img_file, extended_train_images)
        shutil.copy2(label_file, extended_train_labels)

    # Copy remaining samples to extended validation set
    for img_file, label_file in val_samples:
        shutil.copy2(img_file, extended_val_images)
        shutil.copy2(label_file, extended_val_labels)

    print(
        f"Extended training set: {len(list(extended_train_images.glob('*.png')))} samples"
    )
    print(
        f"Extended validation set: {len(list(extended_val_images.glob('*.png')))} samples"
    )

    return str(extended_path)


if __name__ == "__main__":
    extended_dataset = create_extended_training_set(num_val_to_use=150)
    print(f"Created extended dataset at: {extended_dataset}")
