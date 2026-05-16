# Automated Blood Cell Classification System

> A complete two-phase pipeline for white blood cell classification using colour-guided segmentation, CNN transfer learning, hand-crafted texture features, and machine learning classifiers.

---

## Results at a glance

| Model | Accuracy | AUC-ROC | F1-Score | Kappa |
|---|---|---|---|---|
| **EfficientNetB0 + SVM** ★ | **98.75 %** | **0.9999** | **0.9875** | **0.9833** |
| ResNet50 + SVM | 93.75 % | 0.9929 | 0.9369 | 0.9167 |
| EfficientNetB0 + KNN | 93.75 % | 0.9943 | 0.9357 | 0.9167 |
| EfficientNetB0 + Random Forest | 93.13 % | 0.9915 | 0.9311 | 0.9083 |
| VGG16 / VGG19 + SVM | 92.50 % | 0.987 | 0.924 | 0.900 |

All 16 architecture × classifier combinations evaluated on 8 metrics — see [full table](#results).

---

## Table of contents

1. [Project overview](#project-overview)
2. [Dataset](#dataset)
3. [Pipeline — Phase I: Pre-processing](#phase-i--pre-processing)
4. [Pipeline — Phase II: Feature extraction, fusion & classification](#phase-ii--feature-extraction-fusion--classification)
5. [Results](#results)
6. [Repository structure](#repository-structure)
7. [Setup & usage](#setup--usage)
8. [Team](#team)

---

## Project overview

Manual white blood cell (WBC) differential counting from peripheral blood smears is laborious, subjective, and difficult to scale. This project implements a fully automated pipeline that:

- Enhances raw microscopic images using spatial and frequency-domain techniques (Phase I)
- Segments WBC nuclei using a colour-guided LAB-space Otsu pipeline with morphological post-processing (Phase II)
- Extracts rich feature representations by fusing deep CNN features (VGG16, VGG19, ResNet50, EfficientNetB0) with hand-crafted texture descriptors (LBP, GLCM)
- Classifies four WBC types with SVM, Random Forest, KNN, and Gradient Boosting after PCA dimensionality reduction

The best configuration — **EfficientNetB0 features + LBP + GLCM → PCA(100) → RBF-SVM** — achieves **98.75 % accuracy** and **0.9999 AUC-ROC** on a held-out test set.

---

## Dataset

**Blood Cells Image Dataset** — [Kaggle: paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

| Class | Type | Count | Key morphology |
|---|---|---|---|
| EOSINOPHIL | Granulocyte | ~3,117 | Bilobed nucleus · pink cytoplasmic granules |
| LYMPHOCYTE | Agranulocyte | ~3,094 | Large round nucleus · scant cytoplasm |
| MONOCYTE | Agranulocyte | ~3,055 | Kidney-shaped nucleus · largest WBC |
| NEUTROPHIL | Granulocyte | ~3,101 | Multi-lobed nucleus · pale granules |

- **Format:** 320 × 240 px colour JPEG
- **Total:** ~12,500 augmented images
- **Used in experiments:** 200 images per class (800 total) — increase `MAX_PER_CLASS` for production runs

> The dataset ZIP contains both `dataset-master/` and `dataset2-master/`. Images are in `dataset2-master/TRAIN/<CLASS>/`. The `find_class_root()` utility in the notebook handles this automatically.

---

## Phase I — Pre-processing

All spatial-domain functions are implemented **from scratch** — no built-in shortcut functions (`cv2.equalizeHist`, `scipy.ndimage.median_filter`, etc.) are used, per project requirements.

### Spatial domain

| Technique | Function | Effect |
|---|---|---|
| Grayscale conversion | `to_grayscale()` | Luminosity weighting: `0.2989R + 0.5870G + 0.1140B` |
| Histogram equalization | `histogram_equalization()` | Manual CDF-based LUT — redistributes intensities globally |
| Contrast stretching | `contrast_stretching()` | Linear map from `[min, max]` → `[0, 255]` |
| Gaussian smoothing | `gaussian_filter_manual()` | 5 × 5 kernel, σ = 1.0 — attenuates Gaussian noise |
| Median filtering | `median_filter_manual()` | 3 × 3 window — removes salt-and-pepper noise |

### Frequency domain

Both filters use 2D FFT → filter in frequency space → inverse FFT.

| Filter | Transfer function | Effect |
|---|---|---|
| Butterworth low-pass | `H = 1 / (1 + (D/D₀)^2n)` · D₀=30, n=2 | Retains structure, attenuates high-freq noise |
| Butterworth high-pass | `H = 1 / (1 + (D₀/D)^2n)` · D₀=30, n=2 | Emphasises edges and cell boundaries |

---

## Phase II — Feature extraction, fusion & classification

### Segmentation — colour-guided LAB Otsu

The segmentation pipeline exploits the LAB colour space to isolate WBC nuclei from red blood cells and background:

```
RGB image
    │
    ▼
Gaussian blur (5×5, σ=1.5)          ← noise stabilisation
    │
    ▼
Convert to LAB colour space
    │
    ▼
Extract b* channel                   ← WBC nuclei are blue → low b* values
                                       RBCs are pink/orange → high b* values
    │
    ▼
Gaussian blur on b* (7×7)           ← additional smoothing
    │
    ▼
Otsu threshold on b* (THRESH_INV)   ← auto threshold separates blue nucleus
    │
    ▼
Morphological opening (5×5 ellipse) ← removes thin noise speckles
    │
    ▼
Morphological closing (15×15 ellipse, ×2) ← fills holes inside nucleus
    │
    ▼
Connected components filtering      ← removes blobs < 300 px²
    │
    ▼
Binary mask (0 / 255)
```

This colour-guided approach outperforms greyscale Otsu alone because it uses the chromatic difference between blue WBC nuclei and pink/orange RBCs — a distinction that disappears in greyscale.

### Hand-crafted feature extraction

| Descriptor | Parameters | Output dim | Captures |
|---|---|---|---|
| LBP | P=8, R=1, uniform | 256 | Rotation-invariant micro-texture |
| GLCM | distances=[1,3], angles=[0°,45°,90°,135°], 6 Haralick props | 48 | Statistical spatial texture |
| **Combined** | | **304** | |

### CNN deep feature extraction (transfer learning)

| Architecture | Output dim | Reference |
|---|---|---|
| VGG16 | 512 | Simonyan & Zisserman, 2014 |
| VGG19 | 512 | Simonyan & Zisserman, 2014 |
| ResNet50 | 2,048 | He et al., 2016 |
| EfficientNetB0 | 1,280 | Tan & Le, 2019 |

All models loaded with ImageNet weights · classification head removed · `GlobalAveragePooling2D` appended.

### Feature fusion

```
CNN features (scaled)  ──┐
                          ├── np.hstack() ──► Fused vector
LBP + GLCM (scaled)   ──┘
```

Each feature set is independently `StandardScaler`-normalized before concatenation to equalise magnitudes.

### Dimensionality reduction

**PCA — 100 principal components** · retains ≥ 95 % of explained variance for all architectures.

### Classifiers

| Classifier | Configuration |
|---|---|
| SVM | RBF kernel · C=10 · gamma='scale' |
| Random Forest | 200 estimators |
| KNN | k=5 · Euclidean distance |
| Gradient Boosting | 100 estimators · lr=0.1 |

Train / test split: **80 % / 20 %** · stratified · `random_state=42`.

---

## Results

### Full metrics table — all 16 combinations

| Architecture + Classifier | Accuracy | Sensitivity | Specificity | Precision | Recall | F1 | AUC-ROC | Kappa |
|---|---|---|---|---|---|---|---|---|
| **EfficientNetB0 + SVM** ★ | **0.9875** | **0.9875** | **0.9958** | **0.9881** | **0.9875** | **0.9875** | **0.9999** | **0.9833** |
| ResNet50 + SVM | 0.9375 | 0.9375 | 0.9792 | 0.9384 | 0.9375 | 0.9369 | 0.9929 | 0.9167 |
| EfficientNetB0 + KNN | 0.9375 | 0.9375 | 0.9792 | 0.9414 | 0.9375 | 0.9357 | 0.9943 | 0.9167 |
| EfficientNetB0 + Random Forest | 0.9313 | 0.9312 | 0.9771 | 0.9334 | 0.9313 | 0.9311 | 0.9915 | 0.9083 |
| VGG16 + SVM | 0.9250 | 0.9250 | 0.9750 | 0.9253 | 0.9250 | 0.9243 | 0.9870 | 0.9000 |
| VGG19 + SVM | 0.9250 | 0.9250 | 0.9750 | 0.9262 | 0.9250 | 0.9250 | 0.9845 | 0.9000 |
| EfficientNetB0 + Gradient Boosting | 0.9125 | 0.9125 | 0.9708 | 0.9127 | 0.9125 | 0.9122 | 0.9906 | 0.8833 |
| ResNet50 + Random Forest | 0.8688 | 0.8687 | 0.9562 | 0.8680 | 0.8688 | 0.8662 | 0.9590 | 0.8250 |
| VGG16 + Gradient Boosting | 0.8313 | 0.8312 | 0.9437 | 0.8298 | 0.8313 | 0.8297 | 0.9591 | 0.7750 |
| ResNet50 + Gradient Boosting | 0.8250 | 0.8250 | 0.9417 | 0.8251 | 0.8250 | 0.8241 | 0.9637 | 0.7667 |
| VGG16 + Random Forest | 0.8187 | 0.8187 | 0.9396 | 0.8184 | 0.8187 | 0.8131 | 0.9470 | 0.7583 |
| VGG16 + KNN | 0.8000 | 0.8000 | 0.9333 | 0.7986 | 0.8000 | 0.7832 | 0.9434 | 0.7333 |
| VGG19 + Random Forest | 0.7875 | 0.7875 | 0.9292 | 0.7868 | 0.7875 | 0.7820 | 0.9381 | 0.7167 |
| VGG19 + Gradient Boosting | 0.7688 | 0.7687 | 0.9229 | 0.7684 | 0.7688 | 0.7656 | 0.9426 | 0.6917 |
| ResNet50 + KNN | 0.7375 | 0.7375 | 0.9125 | 0.7490 | 0.7375 | 0.7223 | 0.9070 | 0.6500 |
| VGG19 + KNN | 0.7125 | 0.7125 | 0.9042 | 0.7266 | 0.7125 | 0.6854 | 0.9161 | 0.6167 |

### Key findings

- **EfficientNetB0 dominates** — ranks 1st, 3rd, 4th, and 7th across all classifiers
- **SVM is the strongest classifier** — best within 3 of 4 architecture groups
- **Feature fusion consistently helps** — fused features outperform CNN-only for all architectures
- **All AUC-ROC values > 0.90** — even the weakest combination has strong discrimination ability
- **Top 30 PCA components** show gradual importance decay (PC3 ≈ 0.050 → PC92 ≈ 0.010), confirming that retaining 100 components was appropriate

---

## Repository structure

```
blood-cell-classification/
│
├── PHASE1.ipynb                  # Phase I — pre-processing notebook
├── PHASE2.ipynb            # Phase II — full pipeline notebook 
│
├── docs/
│   ├── Research_Report.docx      # Combined Phase I + II research paper
│   ├── Technical_Documentation.docx  # Full technical docs with code
│
└── README.md
```

---

## Setup & usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/blood-cell-classification.git
cd blood-cell-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy>=1.23
matplotlib>=3.5
scikit-learn>=1.1
scikit-image>=0.19
tensorflow>=2.10
opencv-python>=4.6
kagglehub>=0.2
pandas>=1.4
scipy>=1.9
```

### 3. Run in Google Colab (recommended)

1. Upload `PHASE2_FIXED.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — dataset downloads automatically via `kagglehub`
4. To use more images, change `MAX_PER_CLASS = 200` to a higher value

### 4. Adjust key parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `MAX_PER_CLASS` | Cell 13 | 200 | Images per class — increase for better accuracy |
| `N_COMPONENTS` | Cell 19 | 100 | PCA components retained |
| `min_area` | `segment_image()` | 300 px | Minimum nucleus blob size |
| `test_size` | Cell 23 | 0.2 | Train/test split ratio |

---

