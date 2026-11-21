# Lung Cancer Detection System Using Computer Vision and Deep Learning

**Academic & Research Prototype** — Thoracic CT image preprocessing, radiomic and convolutional feature extraction, Custom CNN vs. Transfer Learning (`VGG16`, `ResNet50`, `EfficientNetB0`), Grad-CAM explainability, and U-Net nodule segmentation.

> **Clinical Disclaimer:** This software is designed strictly for academic research and educational demonstration. It is **not** a clinically validated diagnostic device and must not be used for medical screening, diagnosis, or treatment decisions.

---

## 1. Project Architecture

```text
lung-cancer-detection/
├── src/                        # React + TypeScript + Tailwind + Chart.js Frontend
│   ├── components/             # Sidebar, TopBar, Chart.js visualizers
│   ├── pages/                  # Overview, Upload, Results, Performance, History
│   ├── services/               # REST API client
│   └── assets/images/          # Reference axial CT sample images
├── server.ts                   # Live Node/Express full-stack server (port 3000)
├── server/cvEngine.ts          # Real-time OpenCV-equivalent CV, DICOM parser & Grad-CAM
├── scripts/train_web_model.ts  # Patient-level 70/15/15 benchmark trainer & evaluator
├── backend/                    # Complete Python Flask & TensorFlow REST Backend
│   ├── app.py                  # Flask endpoints (/predict, /preprocess, /model-info, /health, /history)
│   ├── preprocessing.py        # OpenCV, PyDICOM, NiBabel, CLAHE, Canny, Otsu Lung ROI
│   ├── feature_extraction.py   # Histogram, GLCM, Contour Shape, HOG, Deep GAP features
│   ├── predict.py              # Keras model loader, architecture normalization, Grad-CAM
│   └── requirements.txt
├── training/                   # Deep Learning Training & Evaluation Scripts
│   ├── prepare_dataset.py      # Patient-level GroupShuffleSplit (70/15/15) preventing leakage
│   ├── train_cnn.py            # Custom 4-Block CNN with BatchNorm, Dropout, GAP
│   ├── train_resnet.py         # VGG16 & ResNet50 Transfer Learning + fine-tuning
│   ├── train_efficientnet.py   # EfficientNetB0 Transfer Learning + fine-tuning
│   ├── evaluate.py             # Confusion Matrix, Sensitivity, Specificity, ROC-AUC, Plots
│   ├── gradcam.py              # Standalone Grad-CAM side-by-side generator
│   └── segmentation.py         # Optional U-Net Lung Nodule Segmentation + Dice/IoU
└── results/
    └── evaluation_metrics.json # Generated evaluation output from held-out test split
```

---

## 2. Dataset Documentation & Leakage Prevention

Supported public research datasets:
1. **IQ-OTH/NCCD Lung Cancer Dataset**: Axial CT slices categorized into **Normal**, **Benign**, and **Malignant** (subject to dataset usage terms).
2. **LIDC-IDRI & LUNA16**: Thoracic CT volumes in DICOM/MHD format containing radiologist nodule annotations and masks suitable for U-Net segmentation (`training/segmentation.py`) and bounding-box detection (YOLO).

### Patient-Level Splitting Protocol (70% / 15% / 15%)
Adjacent axial CT slices from the same patient share near-identical anatomical backgrounds. Randomly splitting at the slice level causes severe **data leakage** and inflated accuracy. Both `training/prepare_dataset.py` (via `GroupShuffleSplit`) and `scripts/train_web_model.ts` strictly group slices by **Patient ID** before splitting:
- **70% Training**
- **15% Validation** (early stopping & learning rate scheduling)
- **15% Held-Out Testing** (never touched during training or model selection)

---

## 3. Step-by-Step Execution Instructions

### A. Live Web Dashboard (Node.js + Express + React)
1. Install Node dependencies:
   ```bash
   npm install
   ```
2. Train & evaluate the web inference weights on the patient-grouped cohort:
   ```bash
   npx tsx scripts/train_web_model.ts
   ```
3. Start the full-stack server on port `3000`:
   ```bash
   npm run dev
   ```

### B. Full Python / TensorFlow / OpenCV Training & Flask Pipeline
1. **Dataset Preparation**: Place `Normal/`, `Benign/`, `Malignant/` CT directories under `training/dataset/raw/` and run:
   ```bash
   python training/prepare_dataset.py --data-dir training/dataset/raw --out-dir training/dataset/splits
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. **Train Custom CNN & Transfer Learning Models** (CPU-compatible fallback built in):
   ```bash
   python training/train_cnn.py
   python training/train_resnet.py --arch vgg16
   python training/train_resnet.py --arch resnet50
   python training/train_efficientnet.py
   ```
4. **Evaluate on Held-Out Test Split**:
   ```bash
   python training/evaluate.py
   ```
   Generates `results/evaluation_metrics.json`, learning curves (`results/graphs/`), confusion matrices (`results/confusion_matrix/`), and ROC/model comparison charts (`results/model_comparison/`).
5. **Run Flask API**:
   ```bash
   python backend/app.py
   ```

---

## 4. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` (or `/api/health`) | Service status and loaded model architectures |
| `POST` | `/preprocess` (or `/api/preprocess`) | Returns 224×224 Denoised, CLAHE, Canny Edge, and Lung ROI PNGs |
| `POST` | `/predict` (or `/api/predict`) | Runs preprocessing, feature extraction, classification, Grad-CAM, and segmentation |
| `GET` | `/model-info` (or `/api/model-info`) | Returns architecture details, split summary, and test evaluation metrics |
| `GET` | `/history` (or `/api/history`) | Returns anonymized prediction history (no PHI stored) |
| `DELETE` | `/history/:id` (or `/api/history/:id`) | Deletes a specific prediction record |

---

## 5. Classification vs. Detection vs. Segmentation

- **Classification** (`Custom CNN`, `VGG16`, `ResNet50`, `EfficientNetB0`): Assigns a global slice category (`Normal`, `Benign`, `Malignant`). Grad-CAM visualizes spatial feature attention but does **not** guarantee exact tumor localization.
- **Detection** (`YOLO`, optional): Predicts bounding boxes `[x, y, w, h]` around suspected pulmonary nodules when trained on box-annotated corpora such as LUNA16.
- **Segmentation** (`U-Net` in `training/segmentation.py`): Generates pixel-level binary masks delineating nodule boundaries, evaluated using **Dice Coefficient** and **Intersection over Union (IoU)**.
