"""
Model Loading, Architecture-Specific Normalization, Inference & Grad-CAM Prediction Module
"""

from __future__ import annotations

import base64
import io
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from preprocessing import preprocess_ct_pipeline
from feature_extraction import extract_all_traditional_features

CLASS_NAMES = ["Normal", "Benign", "Malignant"]

MODEL_NORMALIZERS = {
    "efficientnet_b0": tf.keras.applications.efficientnet.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "custom_cnn": lambda x: x.astype(np.float32) / 255.0,
}


def encode_array_to_png_data_url(img_array: np.ndarray) -> str:
    """Encode grayscale or RGB numpy image array to base64 PNG data URL."""
    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Locate the final 4D convolutional layer name for Grad-CAM computation."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub.name
    raise ValueError("No convolutional layer found in model for Grad-CAM.")


def compute_gradcam_overlay(
    model: tf.keras.Model,
    preprocessed_batch: np.ndarray,
    original_rgb: np.ndarray,
    class_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM heatmap and blended overlay using tf.GradientTape."""
    last_conv_name = find_last_conv_layer(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_batch, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(conv_map)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap_np, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_rgb, 0.58, heatmap_rgb, 0.42, 0)
    return heatmap_rgb, overlay


def load_saved_model(model_dir: str, model_id: str) -> Optional[tf.keras.Model]:
    """Load trained Keras model file (.keras or .h5) from model directory."""
    for ext in (".keras", ".h5"):
        candidate = os.path.join(model_dir, f"{model_id}{ext}")
        if os.path.exists(candidate):
            return tf.keras.models.load_model(candidate)
    best_path = os.path.join(model_dir, "best_model.keras")
    if os.path.exists(best_path):
        return tf.keras.models.load_model(best_path)
    return None


def predict_ct_scan(
    file_path: str,
    model_dir: str,
    model_id: str = "resnet50",
    filter_mode: str = "gaussian",
) -> Dict[str, object]:
    """Run full OpenCV preprocessing, feature extraction, Keras model inference, and Grad-CAM."""
    model = load_saved_model(model_dir, model_id)
    if model is None:
        raise FileNotFoundError(
            f"No trained model weights found in {model_dir}. Run training/train_resnet.py or training/train_efficientnet.py first."
        )

    prep = preprocess_ct_pipeline(file_path, filter_mode=filter_mode)
    trad_features = extract_all_traditional_features(prep)

    rgb_input = prep["rgb_224"].astype(np.float32)
    norm_fn = MODEL_NORMALIZERS.get(model_id, MODEL_NORMALIZERS["custom_cnn"])
    model_batch = np.expand_dims(norm_fn(rgb_input.copy()), axis=0)

    probs = model.predict(model_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    heatmap_rgb, overlay_rgb = compute_gradcam_overlay(model, model_batch, prep["rgb_224"], pred_idx)

    return {
        "model_used": model_id,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": {cls: round(float(probs[i]), 4) for i, cls in enumerate(CLASS_NAMES)},
        "metadata": prep["metadata"],
        "features": trad_features,
        "visualizations": {
            "original_png": encode_array_to_png_data_url(prep["original_resized"]),
            "denoised_png": encode_array_to_png_data_url(prep["denoised"]),
            "clahe_png": encode_array_to_png_data_url(prep["clahe"]),
            "edges_png": encode_array_to_png_data_url(prep["edges"]),
            "roi_mask_png": encode_array_to_png_data_url(prep["lung_roi_mask"]),
            "gradcam_heatmap_png": encode_array_to_png_data_url(heatmap_rgb),
            "gradcam_overlay_png": encode_array_to_png_data_url(overlay_rgb),
        },
        "disclaimer": (
            "Academic research output only. Grad-CAM visualizes model attention and does not confirm "
            "the presence or location of cancer."
        ),
    }
