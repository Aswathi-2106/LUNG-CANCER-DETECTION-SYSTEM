"""
Patient-Level Test Evaluation & Comparative Visualization Suite
Generates:
  - Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, ROC-AUC
  - Classification report & Confusion Matrix
  - Training/Validation Accuracy & Loss Graphs -> results/graphs/
  - Confusion Matrix Heatmaps -> results/confusion_matrix/
  - CNN vs Transfer Learning Comparison & ROC Curves -> results/model_comparison/
  - Unified JSON Metrics -> results/evaluation_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from train_cnn import load_split_arrays

CLASS_NAMES = ["Normal", "Benign", "Malignant"]

MODEL_SPECS = [
    ("custom_cnn", "Custom CNN (4-Block Conv + GAP)", lambda x: x / 255.0),
    ("vgg16", "VGG16 Transfer Learning", tf.keras.applications.vgg16.preprocess_input),
    ("resnet50", "ResNet50 Transfer Learning", tf.keras.applications.resnet50.preprocess_input),
    ("efficientnet_b0", "EfficientNetB0 Transfer Learning", tf.keras.applications.efficientnet.preprocess_input),
]


def compute_specificity_per_class(cm: np.ndarray) -> List[float]:
    specs: List[float] = []
    total = np.sum(cm)
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = total - (tp + fp + fn)
        specs.append(float(tn / max(1, tn + fp)))
    return specs


def plot_learning_curves(history_csv: str, model_id: str, out_dir: str) -> List[Dict[str, float]]:
    df = pd.read_csv(history_csv)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df["accuracy"], label="Train Accuracy", color="#0284c7", linewidth=2)
    axes[0].plot(df["val_accuracy"], label="Val Accuracy", color="#0b1324", linestyle="--", linewidth=2)
    axes[0].set_title(f"{model_id} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["loss"], label="Train Loss", color="#0284c7", linewidth=2)
    axes[1].plot(df["val_loss"], label="Val Loss", color="#0b1324", linestyle="--", linewidth=2)
    axes[1].set_title(f"{model_id} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_id}_learning_curves.png"), dpi=160)
    plt.close(fig)

    return [
        {
            "epoch": int(i + 1),
            "trainLoss": round(float(row["loss"]), 4),
            "valLoss": round(float(row["val_loss"]), 4),
            "trainAccuracy": round(float(row["accuracy"]), 4),
            "valAccuracy": round(float(row["val_accuracy"]), 4),
        }
        for i, row in df.iterrows()
    ]


def plot_confusion_matrix(cm: np.ndarray, model_id: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix — {model_id}",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_id}_confusion_matrix.png"), dpi=160)
    plt.close(fig)


def evaluate_all_models(splits_dir: str, model_dir: str, results_dir: str) -> None:
    test_csv = os.path.join(splits_dir, "test.csv")
    graphs_dir = os.path.join(results_dir, "graphs")
    cm_dir = os.path.join(results_dir, "confusion_matrix")
    comp_dir = os.path.join(results_dir, "model_comparison")
    os.makedirs(comp_dir, exist_ok=True)

    summary_rows = []
    eval_models: Dict[str, object] = {}

    for model_id, display_name, norm_fn in MODEL_SPECS:
        model_path = os.path.join(model_dir, f"{model_id}.keras")
        if not os.path.exists(model_path):
            print(f"Skipping {model_id}: {model_path} not found.")
            continue

        x_test, y_test = load_split_arrays(test_csv, norm_fn, augment=False)
        model = tf.keras.models.load_model(model_path)
        probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        specs = compute_specificity_per_class(cm)
        macro_spec = float(np.mean(specs))

        print(f"\n=== Classification Report: {display_name} ===")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

        plot_confusion_matrix(cm, model_id, cm_dir)

        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        per_class = []
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        for c_idx, cls_name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, c_idx], probs[:, c_idx])
            roc_auc = float(auc(fpr, tpr))
            ax_roc.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

            tp = int(cm[c_idx, c_idx])
            fn = int(np.sum(cm[c_idx, :]) - tp)
            fp = int(np.sum(cm[:, c_idx]) - tp)
            c_prec = float(tp / max(1, tp + fp))
            c_rec = float(tp / max(1, tp + fn))
            c_f1 = float((2 * c_prec * c_rec) / max(1e-8, c_prec + c_rec))

            per_class.append({
                "className": cls_name,
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "f1Score": round(c_f1, 4),
                "sensitivity": round(c_rec, 4),
                "specificity": round(specs[c_idx], 4),
                "auc": round(roc_auc, 4),
                "support": int(tp + fn),
                "rocPoints": [{"fpr": round(float(x), 4), "tpr": round(float(y), 4)} for x, y in zip(fpr, tpr)],
            })

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2)
        ax_roc.set_title(f"ROC Curve — {display_name}")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f"{model_id}_roc_curve.png"), dpi=160)
        plt.close(fig_roc)

        hist_csv = os.path.join(model_dir, f"{model_id}_history.csv")
        history = plot_learning_curves(hist_csv, model_id, graphs_dir) if os.path.exists(hist_csv) else []
        macro_auc = float(np.mean([c["auc"] for c in per_class]))

        eval_models[model_id] = {
            "architecture": model_id,
            "displayName": display_name,
            "accuracy": round(acc, 4),
            "macroPrecision": round(prec, 4),
            "macroRecall": round(rec, 4),
            "macroF1": round(f1, 4),
            "macroSensitivity": round(rec, 4),
            "macroSpecificity": round(macro_spec, 4),
            "macroAuc": round(macro_auc, 4),
            "confusionMatrix": cm.tolist(),
            "perClassMetrics": per_class,
            "history": history,
        }
        summary_rows.append((model_id, acc, rec, macro_spec, f1, macro_auc))

    if summary_rows:
        comp_df = pd.DataFrame(
            summary_rows,
            columns=["Model", "Accuracy", "Sensitivity", "Specificity", "F1", "AUC"],
        )
        ax = comp_df.set_index("Model")[["Accuracy", "Sensitivity", "Specificity", "F1"]].plot(
            kind="bar", figsize=(9, 5), ylim=(0.5, 1.02)
        )
        ax.set_title("Custom CNN vs. Transfer Learning Comparison (Patient-Level Test Set)")
        ax.set_ylabel("Score")
        plt.xticks(rotation=15)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "model_comparison.png"), dpi=160)
        plt.close()

        best_id = max(eval_models.keys(), key=lambda k: eval_models[k]["macroF1"])
        with open(os.path.join(results_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"bestModelId": best_id, "models": eval_models}, fh, indent=2)
        print(f"\nSaved full evaluation report to {os.path.join(results_dir, 'evaluation_metrics.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="training/dataset/splits")
    parser.add_argument("--model-dir", default="backend/model")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    evaluate_all_models(args.splits_dir, args.model_dir, args.results_dir)
