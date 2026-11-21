"""
Computer Vision & Deep Learning Feature Extraction Module
Extracts:
  - Traditional Radiomic Features: Intensity Histogram, GLCM Texture, Canny Edge/Contour Shape, HOG
  - Deep Learning Features: CNN / Transfer Learning Global Average Pooling (GAP) embeddings
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog


def extract_histogram_features(gray_img: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute first-order intensity statistics inside the Lung ROI."""
    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        pixels = gray_img[roi_mask > 0].astype(np.float32)
    else:
        pixels = gray_img.ravel().astype(np.float32)

    mean = float(np.mean(pixels))
    std = float(np.std(pixels)) + 1e-6
    centered = pixels - mean
    skewness = float(np.mean(centered ** 3) / (std ** 3))
    kurtosis = float(np.mean(centered ** 4) / (std ** 4) - 3.0)

    hist, _ = np.histogram(pixels, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist * 8.0 + 1e-12)))

    return {
        "hist_mean": mean,
        "hist_std": std,
        "hist_skewness": skewness,
        "hist_kurtosis": kurtosis,
        "hist_entropy": entropy,
    }


def extract_glcm_features(gray_img: np.ndarray) -> Dict[str, float]:
    """Compute multi-angle Gray-Level Co-occurrence Matrix (GLCM) texture descriptors."""
    quantized = (gray_img // 16).astype(np.uint8)  # 16 gray levels
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }


def extract_shape_and_edge_features(
    clahe_img: np.ndarray, edges: np.ndarray, roi_mask: np.ndarray
) -> Dict[str, float]:
    """Compute Canny edge density and focal lesion contour morphology inside lung ROI."""
    roi_pixels = max(1, int(np.count_nonzero(roi_mask)))
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    edge_density = float(np.count_nonzero(roi_edges) / roi_pixels)

    # Isolate candidate hyper-intense focal structures inside lung ROI
    mean_val = float(np.mean(clahe_img[roi_mask > 0])) if roi_pixels > 1 else 60.0
    std_val = float(np.std(clahe_img[roi_mask > 0])) if roi_pixels > 1 else 25.0
    thresh_val = max(125.0, mean_val + 1.45 * std_val)

    focal_mask = ((clahe_img >= thresh_val) & (roi_mask > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(focal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_areas: List[float] = []
    circularities: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30.0 <= area <= 4500.0:
            perim = max(1.0, cv2.arcLength(cnt, True))
            circ = min(1.0, float(4.0 * np.pi * area / (perim * perim)))
            valid_areas.append(float(area))
            circularities.append(circ)

    max_area_ratio = float(max(valid_areas) / roi_pixels) if valid_areas else 0.0
    primary_circ = float(circularities[int(np.argmax(valid_areas))]) if valid_areas else 0.0

    return {
        "edge_density": edge_density,
        "candidate_contour_count": float(len(valid_areas)),
        "max_lesion_area_ratio": max_area_ratio,
        "primary_circularity": primary_circ,
    }


def extract_hog_summary(clahe_img: np.ndarray) -> Dict[str, float]:
    """Extract Histogram of Oriented Gradients (HOG) structural summary."""
    hog_vec = hog(
        clahe_img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return {
        "hog_mean": float(np.mean(hog_vec)),
        "hog_std": float(np.std(hog_vec)),
        "hog_max": float(np.max(hog_vec)),
        "hog_l2_norm": float(np.linalg.norm(hog_vec)),
    }


def extract_deep_gap_features(rgb_224: np.ndarray, backbone_model) -> np.ndarray:
    """
    Extract Global Average Pooling (GAP) feature embeddings from a Keras CNN or Transfer Learning backbone.
    """
    batch = np.expand_dims(rgb_224.astype(np.float32), axis=0)
    embeddings = backbone_model.predict(batch, verbose=0)
    return embeddings.flatten()


def extract_all_traditional_features(preprocessed: Dict[str, object]) -> Dict[str, float]:
    """Combine Histogram, GLCM, Edge/Shape, and HOG descriptors into a unified dictionary."""
    clahe_img = preprocessed["clahe"]
    edges = preprocessed["edges"]
    roi_mask = preprocessed["lung_roi_mask"]

    features: Dict[str, float] = {}
    features.update(extract_histogram_features(clahe_img, roi_mask))
    features.update(extract_glcm_features(clahe_img))
    features.update(extract_shape_and_edge_features(clahe_img, edges, roi_mask))
    features.update(extract_hog_summary(clahe_img))
    return features
