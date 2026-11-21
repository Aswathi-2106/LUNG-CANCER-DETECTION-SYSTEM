"""
OpenCV Medical CT Scan Preprocessing Pipeline
Supports PNG, JPEG, DICOM (.dcm), and NIfTI (.nii/.nii.gz) formats.
Preserves essential acquisition metadata while stripping patient PHI.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


TARGET_SIZE = (224, 224)


def apply_lung_window(hu_image: np.ndarray, window_center: float = -600.0, window_width: float = 1500.0) -> np.ndarray:
    """Convert Hounsfield Units (HU) to 8-bit grayscale using standard thoracic lung window."""
    lower = window_center - (window_width / 2.0)
    upper = window_center + (window_width / 2.0)
    windowed = np.clip(hu_image, lower, upper)
    scaled = ((windowed - lower) / max(1.0, (upper - lower))) * 255.0
    return scaled.astype(np.uint8)


def load_ct_image(file_path: str, slice_index: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Load a CT slice from PNG, JPEG, DICOM (.dcm), or NIfTI (.nii / .nii.gz).
    Returns an 8-bit grayscale numpy array and sanitized acquisition metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    metadata: Dict[str, object] = {"format": ext.lstrip(".").upper() or "IMAGE", "phi_stripped": True}

    if ext == ".dcm":
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required to load DICOM (.dcm) files.")
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
        hu_image = pixel_array * slope + intercept

        wc = getattr(ds, "WindowCenter", -600.0)
        ww = getattr(ds, "WindowWidth", 1500.0)
        wc = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
        ww = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)

        gray = apply_lung_window(hu_image, wc, ww)
        metadata.update({
            "format": "DICOM",
            "modality": str(getattr(ds, "Modality", "CT")),
            "slice_thickness_mm": str(getattr(ds, "SliceThickness", "unknown")),
            "window_center": wc,
            "window_width": ww,
            "original_shape": list(gray.shape),
        })
        return gray, metadata

    if file_path.lower().endswith((".nii", ".nii.gz")):
        if not HAS_NIBABEL:
            raise RuntimeError("nibabel is required to load NIfTI (.nii/.nii.gz) volumes.")
        nii = nib.load(file_path)
        vol = nii.get_fdata()
        mid = vol.shape[2] // 2 if slice_index is None else int(np.clip(slice_index, 0, vol.shape[2] - 1))
        slice_hu = vol[:, :, mid]
        gray = apply_lung_window(slice_hu)
        metadata.update({
            "format": "NIFTI",
            "slice_index": mid,
            "volume_shape": list(vol.shape),
            "voxel_zooms": [float(z) for z in nii.header.get_zooms()[:3]],
        })
        return gray, metadata

    # Standard 2D image (PNG, JPG, JPEG)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        pil_img = Image.open(file_path).convert("L")
        gray = np.array(pil_img, dtype=np.uint8)
    else:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
    metadata["original_shape"] = list(gray.shape)
    return gray, metadata


def extract_lung_roi(denoised_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment bilateral lung parenchyma from chest wall & scanner circle using Otsu thresholding,
    border flood-filling, and morphological hole closing to retain intra-parenchymal nodules.
    """
    h, w = denoised_gray.shape
    _, body_mask = cv2.threshold(denoised_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Flood-fill external air outside the thoracic body contour
    flood = body_mask.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    cv2.floodFill(flood, mask_ff, (w - 1, 0), 255)
    internal_air = cv2.bitwise_not(flood)

    # Morphologically close holes inside lung fields (so solid tumors remain inside ROI)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lung_roi = cv2.morphologyEx(internal_air, cv2.MORPH_CLOSE, kernel)

    # Fallback if threshold did not capture bilateral fields cleanly
    if np.count_nonzero(lung_roi) < 0.05 * h * w:
        lung_roi = np.zeros_like(denoised_gray)
        cv2.ellipse(lung_roi, (w // 2, h // 2), (int(w * 0.36), int(h * 0.32)), 0, 0, 360, 255, -1)

    roi_image = cv2.bitwise_and(denoised_gray, denoised_gray, mask=lung_roi)
    return lung_roi, roi_image


def preprocess_ct_pipeline(
    file_path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    filter_mode: str = "gaussian",
    clahe_clip: float = 2.0,
) -> Dict[str, object]:
    """
    Full 10-step OpenCV CT preprocessing pipeline:
    1. Load (PNG/JPEG/DICOM/NIfTI)
    2. Resize to 224x224
    3. Grayscale conversion
    4. Gaussian or Median noise removal
    5. CLAHE contrast enhancement
    6. Intensity normalization [0, 1]
    7. Otsu thresholding
    8. Lung ROI extraction
    9. Canny edge detection
    10. 3-channel RGB stack prepared for CNN / Transfer Learning
    """
    raw_gray, metadata = load_ct_image(file_path)
    resized = cv2.resize(raw_gray, target_size, interpolation=cv2.INTER_LINEAR)

    if filter_mode == "median":
        denoised = cv2.medianBlur(resized, 3)
    else:
        denoised = cv2.GaussianBlur(resized, (5, 5), sigmaX=1.0)

    clahe_op = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    clahe_img = clahe_op.apply(denoised)

    _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(clahe_img, threshold1=30, threshold2=90)
    lung_roi_mask, lung_roi_img = extract_lung_roi(denoised)

    normalized = clahe_img.astype(np.float32) / 255.0
    rgb_224 = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

    return {
        "metadata": metadata,
        "original_resized": resized,
        "denoised": denoised,
        "clahe": clahe_img,
        "otsu": otsu_thresh,
        "edges": edges,
        "lung_roi_mask": lung_roi_mask,
        "lung_roi_image": lung_roi_img,
        "normalized": normalized,
        "rgb_224": rgb_224,
    }


def augment_ct_slice(image_rgb: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Clinically conservative CT augmentation (small rotations, horizontal flip, slight zoom/brightness).
    Avoids unrealistic vertical flips or aggressive distortions that alter anatomical mediastinal priors.
    """
    rng = np.random.default_rng(seed)
    aug = image_rgb.copy()

    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)

    angle = float(rng.uniform(-10.0, 10.0))
    h, w = aug.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, float(rng.uniform(0.95, 1.05)))
    aug = cv2.warpAffine(aug, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-8.0, 8.0))
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return aug
