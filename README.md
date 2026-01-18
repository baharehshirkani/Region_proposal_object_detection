# Region-Based Object Detection: Selective Search + CNN + SVM

A complete implementation of region-based object detection for **balloon detection**, combining the Selective Search algorithm with deep CNN features and SVM classification. This is an educational implementation of the approach described by Uijlings et al. (IJCV 2013), simplified to a single-class detection task.

## 📋 Overview

This project implements a complete object detection pipeline with five distinct stages:

1. **Region Proposal Generation** - Selective Search algorithm
2. **Sample Labeling** - IoU-based positive/negative classification
3. **Feature Extraction** - ResNet18 CNN embeddings
4. **Model Training** - Linear SVM classifier
5. **Inference & Post-processing** - NMS and visualization

### Key Features

- ✅ **Complete pipeline** from raw images to detections
- ✅ **Selective Search algorithm** with full implementation
- ✅ **Pre-trained ResNet18** for feature extraction
- ✅ **Configurable IoU thresholds** (tp/tn) for sample labeling
- ✅ **Multiple model variants** tested with different thresholds
- ✅ **Non-maximum suppression** for post-processing
- ✅ **Educational focus** with clear code documentation

---

## 🏗️ Architecture

### Core Algorithm: Selective Search

The Selective Search algorithm generates region proposals by hierarchical bottom-up merging of superpixels:

```
Image
  ↓
[Task 1] Felzenszwalb Segmentation
  ↓
[Tasks 2-7] Hierarchical Merging
  - Extract color/texture features per region
  - Find neighboring regions (spatial overlap)
  - Calculate multi-way similarity (color + texture + size + fill)
  - Iteratively merge highest-similarity pairs
  ↓
[Task 8] Convert to Bounding Boxes
  ↓
Region Proposals (~1500 per image)
```

**Key metrics for region similarity:**
- **Color Similarity**: 25-bin HSV histogram intersection
- **Texture Similarity**: Local Binary Pattern (LBP) with 8 orientations
- **Size Similarity**: Ratio of region sizes
- **Fill Similarity**: Ratio of bounding box overlaps

### Detection Pipeline

```
Image
  ↓
[Proposals] Generate ~1500 regions via Selective Search
  ↓
[Features] Extract 512-dim ResNet18 embeddings for each region
  ↓
[Classification] SVM decision per region
  ↓
[Post-processing] Non-Maximum Suppression (IoU=0.3)
  ↓
Detections with confidence scores
```

---

## 📁 Project Structure

```
Region_proposal_object_detection/
├── code/
│   ├── main.py                 # Simple baseline demo
│   ├── selective_search.py     # Core Selective Search algorithm
│   └── final_code.py           # Complete pipeline (5 modes)
│
├── data/balloon_dataset/       # Dataset and generated artifacts
│   ├── train/                  # 44 images + COCO annotations
│   ├── valid/                  # 9 images + COCO annotations
│   ├── test/                   # 5 images + COCO annotations
│   ├── proposals/              # Generated region proposals
│   ├── samples/                # Labeled sample crops (tp=0.75, tn=0.25)
│   ├── samples_tp80_tn20/      # Alternative variant (clean positives)
│   ├── samples_tp60_tn40/      # Alternative variant (hard negatives)
│   ├── features/               # ResNet18 embeddings
│   └── models/                 # Trained SVM models
│
└── results/                    # Visualization outputs
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scikit-image scikit-learn torch torchvision pillow matplotlib joblib
```

### Usage: 5 Pipeline Modes

#### Mode 1: Generate Region Proposals

```bash
python final_code.py --mode proposals \
  --data-root data/balloon_dataset \
  --out-dir data/balloon_dataset/proposals
```

Outputs: `proposals_train.json`, `proposals_valid.json`
- Contains ~1500 region proposals per image (x, y, width, height format)
- Generated via Selective Search with filtering (min size: 2000 pixels)

#### Mode 2: Create Labeled Samples

```bash
python final_code.py --mode samples \
  --data-root data/balloon_dataset \
  --proposals-dir data/balloon_dataset/proposals \
  --samples-dir data/balloon_dataset/samples \
  --tp 0.80 --tn 0.20
```

Outputs: `samples_train.json`, sample crops in `pos/` and `neg/` directories
- Labels regions as positive (IoU ≥ tp), negative (IoU ≤ tn), or ignored
- Creates 128×128 pixel crops of labeled regions
- `tp=0.80, tn=0.20` recommended (cleanest positives)

#### Mode 3: Extract CNN Features

```bash
python final_code.py --mode features \
  --data-root data/balloon_dataset \
  --samples-dir data/balloon_dataset/samples \
  --features-dir data/balloon_dataset/features
```

Outputs: `features_train_resnet18.npz`, `features_valid_resnet18.npz`
- Extracts 512-dimensional embeddings from ResNet18 (ImageNet pre-trained)
- Input crops resized to 224×224 pixels
- Processes in batches of 64 for efficiency

#### Mode 4: Train SVM Classifier

```bash
python final_code.py --mode train \
  --features-dir data/balloon_dataset/features \
  --model-out data/balloon_dataset/models/svm_resnet18.joblib
```

Outputs: `svm_resnet18.joblib`, `svm_resnet18.metrics.json`
- Trains LinearSVC with StandardScaler normalization
- Optional PCA dimensionality reduction (128 components)
- Produces precision, recall, AP metrics

#### Mode 5: Run Inference

```bash
python final_code.py --mode infer \
  --model data/balloon_dataset/models/svm_resnet18.joblib \
  --images data/balloon_dataset/test/*.jpg \
  --output-dir results/
```

Outputs: Detection JSON and visualizations
- Generates region proposals on test images
- Extracts ResNet18 features
- Runs SVM predictions with NMS post-processing
- Creates `*_det.jpg` files with bounding boxes drawn

---


## 📈 Performance Results

### Model Variants Comparison

#### Variant 1: tp=0.75, tn=0.25 (Default)
```
Training samples: 2153 | Validation samples: 619
Accuracy:  98.87%
Precision: 87.5%
Recall:    73.68%
AP:        0.9267
```

#### Variant 2: tp=0.80, tn=0.20 (BEST OVERALL)
```
Training samples: 3872 | Validation samples: 1178
Accuracy:  99.75%
Precision: 84.62%
Recall:    91.67%
AP:        0.9247
```
✅ **Recommended** - Highest recall, most balanced metrics, cleanest positives

#### Variant 3: tp=0.60, tn=0.40 (Hard Negatives)
```
Training samples: 5718 | Validation samples: 1758
Accuracy:  98.12%
Precision: 55.56%
Recall:    87.5%
AP:        0.7488
```
Trade-off: More training data but lower precision on hard cases

---

## 📖 References

**Original Paper:**
Uijlings, J. R., Van De Sande, K. E., Gevers, T., & Smeulders, A. W. (2013).
"Selective Search for Object Recognition"
*International Journal of Computer Vision*, 104(2), 154-171.

**Dataset:**
Balloon Detection dataset from Roboflow (CC BY 4.0 license)
- 58 total images split into train/valid/test
- COCO JSON format annotations
- Single class (balloon) with variable scales


