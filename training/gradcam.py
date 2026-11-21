"""
Standalone Grad-CAM Visualization Generator
Produces side-by-side Original CT + Grad-CAM Overlay figures.
Note: Grad-CAM highlights image regions influencing the model prediction and does not
confirm the clinical presence or exact anatomical location of cancer.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from predict import CLASS_NAMES, MODEL_NORMALIZERS, compute_gradcam_overlay, load_saved_model
from preprocessing import preprocess_ct_pipeline


def generate_gradcam_report(image_path: str, model_dir: str, model_id: str, out_path: str) -> None:
    model = load_saved_model(model_dir, model_id)
    if model is None:
        raise FileNotFoundError(f"Model {model_id} not found in {model_dir}.")

    prep = preprocess_ct_pipeline(image_path)
    rgb = prep["rgb_224"]
    norm_fn = MODEL_NORMALIZERS.get(model_id, MODEL_NORMALIZERS["custom_cnn"])
    batch = np.expand_dims(norm_fn(rgb.astype(np.float32)), axis=0)

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    heatmap_rgb, overlay_rgb = compute_gradcam_overlay(model, batch, rgb, pred_idx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(rgb)
    axes[0].set_title("Preprocessed CT (CLAHE)")
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM Spatial Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Overlay ({CLASS_NAMES[pred_idx]}: {probs[pred_idx] * 100:.1f}%)")
    axes[2].axis("off")

    fig.suptitle(
        "Grad-CAM visualizes model attention and does not confirm presence or location of cancer.",
        fontsize=10,
        y=0.04,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", default="backend/model")
    parser.add_argument("--model-id", default="resnet50")
    parser.add_argument("--out", default="results/graphs/gradcam_output.png")
    args = parser.parse_args()
    generate_gradcam_report(args.image, args.model_dir, args.model_id, args.out)
