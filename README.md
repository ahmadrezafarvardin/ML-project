# Character Localization

This project implements a deep learning pipeline for **character localization** in handwritten math images. The goal is to detect and localize all character bounding boxes in each image, as required for the competition.

---

## Features

- **Faster R-CNN** with ResNet backbone (pretrained or from scratch)
- **Custom dataset loader** for your annotation format
- **Data cleaning and visualization tools**
- **Flexible training and evaluation scripts**
- **Automatic CSV submission file generation for competition**
- **NMS and score threshold tuning for best F1**

---

## Directory Structure

```
src/
  data/                # Dataset loader and utilities
  data_analysis/       # Data cleaning, visualization, outlier detection
  models/              # Model architectures (backbone, RPN, ROI heads)
  utils/               # Helper scripts (NMS, debugging, visualization)
  train.py             # Training script
  inference.py         # Evaluation and test inference
  run_inference.py     # Single-image inference
  ...
results/
  checkpoints/         # Model checkpoints
  evaluation/          # Evaluation metrics and confusion matrices
  visualizations/      # Detection visualizations
dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
```

---

## Training

**Train with a pretrained backbone (recommended):**
```bash
python src/train.py --data-path dataset --epochs 30 --batch-size 1 --lr 0.0005 --save-path results/checkpoints --pretrained-backbone
```

**Train from scratch:**
```bash
python src/train.py --data-path dataset --epochs 30 --batch-size 1 --lr 0.0005 --save-path results/checkpoints
```

---

## Evaluation

**Evaluate and visualize results:**
```bash
python src/inference.py
```
- Results and metrics will be saved in `results/evaluation/` and `results/visualizations/`.

---

## Data Visualization

**Visualize all training or validation samples with bounding boxes:**
```bash
python src/data_analysis/data_visualizer.py --split train --all
python src/data_analysis/data_visualizer.py --split valid --all
```
- Annotated images will be saved in `results/data_analysis/sample_visualizations/`.

---

## Competition Submission

**Generate the required CSV for submission:**
```python
# Example function call (add to a script or run in Python shell)
from src.inference import generate_competition_csv

generate_competition_csv(
    checkpoint_path="results/checkpoints/best_model.pth",
    nms_threshold=0.3,
    score_threshold=0.6,
    test_dir="dataset/test/images",
    output_csv="output.csv"
)
```
- The CSV will have columns: `image_id, x, y, width, height` as required by the competition.
