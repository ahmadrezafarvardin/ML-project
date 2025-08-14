# Mathematical Expression Recognition System

A comprehensive deep learning pipeline for detecting, localizing, and recognizing handwritten mathematical expressions using state-of-the-art computer vision techniques.

## 🎯 Project Overview

This project implements a complete pipeline for mathematical expression recognition, consisting of three main components:

1. **Character Detection & Localization** - Using YOLO and Faster R-CNN
2. **Character Clustering & Analysis** - Unsupervised learning for character grouping
3. **Expression Recognition** - End-to-end CRNN with CTC loss and character-based semi-supervised approaches

## 📊 Performance Summary

| Model | Task | Accuracy | Levenshtein Distance | Notes |
|-------|------|----------|---------------------|-------|
| YOLO v8 | Character Detection | 92.3% mAP | - | Best for real-time |
| Faster R-CNN | Character Localization | 89.7% mAP | - | Higher precision |
| Supervised CRNN | Expression Recognition | 54.35% | 0.83 | End-to-end |
| Semi-Supervised CRNN | Expression Recognition | ~60% (expected) | ~0.7 | With pseudo-labels |
| Semi-Supervised Character | Expression Recognition | 4.08% | 3.43 | Character-based (deprecated) |

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone <repository-url>
cd proj

# Create virtual environment
conda create -n ml_project python=3.10
conda activate ml_project

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights (if available)
# Place resnet50_model_gpu.pth in src/models/
```

### Dataset Structure

```
dataset/
├── train/
│   ├── images/      # Training images
│   └── labels/      # JSON annotations with bounding boxes
├── valid/
│   ├── images/      # Validation images
│   └── labels/      # JSON annotations
└── test/
    └── images/      # Test images (no labels)
```

### Complete Pipeline Execution

```bash
# 1. Data Analysis & Validation
python src/data_analysis/main_analysis.py --dataset dataset

# 2. Character Detection Training (YOLO)
python src/models/yolo/train_yolo.py \
    --data dataset_yolo \
    --epochs 100 \
    --batch 16

# 3. Character Clustering
python src/clustering/run_complete_pipeline.py \
    --model results/yolo/best.pt \
    --dataset dataset \
    --output results/clustering

# 4. Expression Recognition (CRNN)
python src/recognition/prepare_extended_dataset.py
python src/recognition/train_crnn.py \
    --dataset dataset_extended \
    --epochs 100 \
    --batch-size 16

# 5. Generate Final Predictions
python src/recognition/inference_tta.py \
    --model results/recognition/checkpoints/best_model.pth \
    --test-dir dataset/test/images \
    --output output.csv
```

## 📁 Project Structure

```
proj/
├── src/
│   ├── data_analysis/          # Data validation & visualization
│   ├── clustering/             # Character clustering algorithms
│   ├── models/                 # Model architectures
│   │   ├── yolo/              # YOLO implementation
│   │   ├── fasterrcnn/        # Faster R-CNN implementation
│   │   └── cascadercnn/       # Cascade R-CNN (experimental)
│   ├── recognition/            # CRNN expression recognition
│   ├── semiSL_recognition/     # Semi-supervised character approach
│   └── utils/                  # Helper utilities
├── dataset/                    # Original dataset
├── dataset_extended/           # Extended dataset for CRNN
├── dataset_yolo/              # YOLO format dataset
├── results/                    # Training outputs & checkpoints
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Component Details

### 1. Data Analysis Pipeline

Comprehensive data validation and visualization tools:

```bash
# Complete analysis with report generation
python src/data_analysis/main_analysis.py --dataset dataset

# Individual components
python src/data_analysis/data_validator.py      # Validate dataset integrity
python src/data_analysis/statistical_analyzer.py # Statistical analysis
python src/data_analysis/outlier_detector.py    # Detect outliers
python src/data_analysis/data_visualizer.py     # Visualize samples

# Clean dataset (remove outliers, fix annotations)
python src/data_analysis/data_cleaner.py --dataset dataset --output dataset_cleaned
```

### 2. Character Detection

#### YOLO v8 (Recommended for Speed)
```bash
# Prepare YOLO format dataset
python src/data/prepare_yolo_dataset.py --input dataset --output dataset_yolo

# Training
python src/models/yolo/train_yolo.py \
    --data dataset_yolo \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Inference
python src/models/yolo/inference_yolo.py \
    --weights results/yolo/best.pt \
    --source dataset/test/images \
    --conf-thres 0.25 \
    --save-txt

# Generate submission CSV
python src/models/yolo/generate_submission.py \
    --weights results/yolo/best.pt \
    --test-dir dataset/test/images \
    --output yolo_submission.csv
```

#### Faster R-CNN (Better Precision)
```bash
# Training with pretrained backbone
python src/models/fasterrcnn/train.py \
    --data-path dataset \
    --epochs 30 \
    --batch-size 2 \
    --lr 0.005 \
    --pretrained-backbone \
    --save-path results/fasterrcnn/checkpoints

# Training from scratch
python src/models/fasterrcnn/train.py \
    --data-path dataset \
    --epochs 50 \
    --batch-size 2 \
    --lr 0.001 \
    --save-path results/fasterrcnn/checkpoints

# Evaluation on validation set
python src/models/fasterrcnn/inference.py \
    --checkpoint results/fasterrcnn/checkpoints/best_model.pth \
    --data-path dataset \
    --split valid \
    --visualize

# Generate test predictions
python src/models/fasterrcnn/generate_submission.py \
    --checkpoint results/fasterrcnn/checkpoints/best_model.pth \
    --test-dir dataset/test/images \
    --output fasterrcnn_submission.csv \
    --nms-thresh 0.3 \
    --score-thresh 0.5
```

### 3. Character Clustering

Unsupervised clustering for character analysis:

```bash
# Complete clustering pipeline
python src/clustering/run_complete_pipeline.py \
    --model results/yolo/best.pt \
    --dataset dataset \
    --max-images 500 \
    --conf 0.25 \
    --output results/clustering

# Analyze clustering results
python src/clustering/analyse_scores.py \
    --results results/clustering/clustering_results.json

# Extract features only
python src/clustering/feature_extraction.py \
    --model results/yolo/best.pt \
    --dataset dataset \
    --feature-type all \
    --output results/clustering/features.npz
```

Features:
- Multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
- Various feature extractors (HOG, CNN, Statistical)
- Comprehensive evaluation metrics

### 4. Expression Recognition

#### A. CRNN Approaches (Recommended)

##### Supervised CRNN Training
```bash
# Prepare extended dataset
python src/recognition/prepare_extended_dataset.py \
    --input dataset \
    --output dataset_extended \
    --val-samples 150

# Train supervised CRNN
python src/recognition/train_crnn.py \
    --dataset dataset_extended \
    --output results/recognition \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --pretrained src/models/resnet50_model_gpu.pth
```

##### Semi-Supervised CRNN Training
```bash
# Option 1: Train from scratch with semi-supervision
python src/recognition/train_crnn_semi_supervised.py \
    --labeled-data dataset_extended \
    --unlabeled-data dataset/test/images \
    --output results/recognition/semi_supervised \
    --initial-epochs 50 \
    --semi-epochs 30 \
    --confidence 0.9 \
    --pseudo-label-weight 0.5

# Option 2: Fine-tune existing model with semi-supervision
python src/recognition/train_crnn_semi_supervised.py \
    --labeled-data dataset_extended \
    --unlabeled-data dataset/test/images \
    --initial-model results/recognition/checkpoints/best_model.pth \
    --output results/recognition/semi_supervised_finetuned \
    --initial-epochs 0 \
    --semi-epochs 30 \
    --confidence 0.95
```

##### CRNN Inference Options
```bash
# Standard inference
python src/recognition/inference.py \
    --model results/recognition/checkpoints/best_model.pth \
    --test-dir dataset/test/images \
    --output output.csv

# Test-time augmentation (recommended for better accuracy)
python src/recognition/inference_tta.py \
    --model results/recognition/checkpoints/best_model.pth \
    --test-dir dataset/test/images \
    --output output_tta.csv \
    --num-aug 4

# Evaluate on validation set
python src/recognition/inference.py \
    --model results/recognition/checkpoints/best_model.pth \
    --evaluate \
    --dataset dataset_extended
```

#### B. Character-Based Semi-Supervised Approach (Experimental/Deprecated)

This approach attempts to classify individual characters after detection:

```bash
# Extract character crops using YOLO
python src/semiSL_recognition/train_recognition.py \
    --dataset dataset \
    --output results/semiSL_recognition \
    --yolo-model results/yolo/best.pt \
    --clustering results/clustering/clustering_complete \
    --epochs 100 \
    --batch-size 32

# Evaluate and generate predictions
python src/semiSL_recognition/evaluate_and_predict.py \
    --evaluate \
    --predict \
    --model results/semiSL_recognition/character_classifier.pth \
    --yolo results/yolo/best.pt \
    --output semiSL_predictions.csv
```

**Note**: This approach achieved only 4% accuracy due to:
- Character alignment issues
- Limited real labeled data
- Error propagation in the pipeline

### 5. Visualization & Analysis

#### Generate Comprehensive Visualizations
```bash
# CRNN visualizations
python src/recognition/visualize_crnn.py \
    --model results/recognition/checkpoints/best_model.pth \
    --dataset dataset_extended \
    --samples 10

# Detection visualizations
python src/utils/inference_with_better_nms.py \
    --checkpoint results/fasterrcnn/checkpoints/best_model.pth \
    --image dataset/valid/images/sample.png \
    --nms-comparison

# Model comparison
python src/recognition/compare_models.py
```

#### Training Monitoring
```bash
# Monitor CRNN training
tensorboard --logdir results/recognition/logs

# Plot training curves
python src/utils/plot_training_curves.py \
    --history results/recognition/training_history.json
```

## 🎯 Output Formats

### Character Detection CSV
```csv
image_id,x,y,width,height
1,10,20,30,40
1,50,20,25,35
2,15,25,28,38
```

### Expression Recognition CSV
```csv
image_id,expression
1,2+3*4
2,(5-2)/3
3,7*8+9
```

## 🛠️ Advanced Usage

### Hyperparameter Tuning

```bash
# YOLO hyperparameter search
python src/models/yolo/train_yolo.py \
    --data dataset_yolo \
    --hyp src/models/yolo/hyp.yaml \
    --evolve 10

# CRNN learning rate finder
python src/recognition/lr_finder.py \
    --dataset dataset_extended \
    --min-lr 1e-6 \
    --max-lr 1e-1
```

### Ensemble Methods

```bash
# Combine multiple model predictions
python src/utils/ensemble_predictions.py \
    --predictions output1.csv output2.csv output3.csv \
    --weights 0.4 0.3 0.3 \
    --output ensemble_output.csv
```

### Data Augmentation

```bash
# Generate augmented training data
python src/data/augment_dataset.py \
    --input dataset/train \
    --output dataset_augmented \
    --augmentations rotate blur brightness \
    --factor 3
```

## 🐛 Troubleshooting

### Common Issues

1. **Low Detection Accuracy**
   - Lower confidence threshold (0.25 for YOLO, 0.5 for Faster R-CNN)
   - Increase training epochs
   - Use pretrained backbone
   - Check annotation quality with data_visualizer.py

2. **CRNN Training Issues**
   - Ensure using `dataset_extended` not original dataset
   - Reduce learning rate to 0.0001 or use lr finder
   - Enable gradient clipping (max_norm=5.0)
   - Check for NaN losses (use zero_infinity=True in CTC)

3. **Memory Errors**
   - Reduce batch size (minimum 8 for CRNN, 1 for Faster R-CNN)
   - Use gradient accumulation
   - Enable mixed precision training with --amp flag
   - Reduce image size for YOLO (416 instead of 640)

4. **Poor Expression Recognition**
   - Try semi-supervised training to leverage test data
   - Use test-time augmentation
   - Increase confidence threshold for pseudo-labels (0.95)
   - Check character set consistency ('x' vs '*')

5. **Dataset Issues**
   - Run data validation first
   - Fix bounding box coordinates (must be within image bounds)
   - Remove corrupted images
   - Balance class distribution

## 📚 Key Algorithms & Techniques

- **YOLO v8**: Single-stage detector with anchor-free design
- **Faster R-CNN**: Two-stage detector with Region Proposal Network
- **CRNN**: CNN (feature extraction) + RNN (sequence modeling) + CTC (alignment-free training)
- **Semi-Supervised Learning**: Self-training with pseudo-labels from high-confidence predictions
- **Test-Time Augmentation**: Multiple predictions with different augmentations, then voting
- **Clustering**: K-Means, DBSCAN, and Hierarchical clustering for character analysis

## 🔮 Future Improvements

1. **Attention Mechanisms**: Add attention layers to CRNN for better feature focus
2. **Beam Search Decoding**: Replace greedy CTC decoding for better accuracy
3. **Synthetic Data Generation**: Create artificial training samples
4. **Grammar Constraints**: Incorporate mathematical expression syntax rules
5. **Multi-Scale Detection**: Handle varying character sizes with FPN
6. **Knowledge Distillation**: Train smaller models from larger ones
7. **Active Learning**: Intelligently select samples for annotation

## 📄 License

This project is for educational and research purposes.
---

For detailed documentation on each component, refer to the individual README files in their respective directories.