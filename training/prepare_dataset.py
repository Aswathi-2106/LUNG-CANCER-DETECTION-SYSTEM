"""
Patient-Level Dataset Preparation & Leakage-Free 70/15/15 Splitter
Supports IQ-OTH/NCCD (Normal / Benign / Malignant) and LIDC-IDRI / LUNA16 slice exports.
Ensures all slices belonging to the same patient ID reside strictly in one split.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASS_DIRS = ["Normal", "Benign", "Malignant"]


def infer_patient_id(filename: str) -> str:
    """
    Extract patient identifier from filename conventions (e.g., 'Bengin case (12)_slice3.jpg' -> 'Bengin_case_12').
    Falls back to filename stem if no explicit case/patient pattern exists.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"^(.*?\(\d+\)|PAT_\w+_\d+|LIDC-IDRI-\d+)", stem, flags=re.IGNORECASE)
    if match:
        return re.sub(r"\s+", "_", match.group(1))
    # Strip trailing slice suffixes like _slice12 or _001
    cleaned = re.sub(r"[_-](slice|s)?\d+$", "", stem, flags=re.IGNORECASE)
    return cleaned or stem


def build_manifest(dataset_root: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label_idx, cls_name in enumerate(CLASS_DIRS):
        cls_dir = os.path.join(dataset_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
                continue
            fpath = os.path.join(cls_dir, fname)
            rows.append({
                "filepath": fpath,
                "label": cls_name,
                "label_idx": label_idx,
                "patient_id": f"{cls_name}_{infer_patient_id(fname)}",
            })
    if not rows:
        raise FileNotFoundError(
            f"No CT images found in {dataset_root}. Expected subdirectories: {CLASS_DIRS}."
        )
    return pd.DataFrame(rows)


def split_patient_level(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Perform patient-grouped 70% Train / 15% Validation / 15% Test split."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, df["label_idx"], groups=df["patient_id"]))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 0.15 / 0.85 ~= 0.1765 of remaining data for validation
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    train_idx, val_idx = next(
        gss_val.split(df_train_val, df_train_val["label_idx"], groups=df_train_val["patient_id"])
    )

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Assert zero patient leakage across splits
    p_tr = set(df_train["patient_id"])
    p_va = set(df_val["patient_id"])
    p_te = set(df_test["patient_id"])
    assert len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0, "Patient leakage detected!"

    return {"train": df_train, "val": df_val, "test": df_test}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare patient-level 70/15/15 CT dataset splits.")
    parser.add_argument("--data-dir", default="training/dataset/raw", help="Folder with Normal/Benign/Malignant dirs")
    parser.add_argument("--out-dir", default="training/dataset/splits", help="Output folder for CSV manifests")
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    splits = split_patient_level(manifest)
    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split_df in splits.items():
        out_csv = os.path.join(args.out_dir, f"{split_name}.csv")
        split_df.to_csv(out_csv, index=False)
        print(f"Saved {split_name}: {len(split_df)} slices, {split_df['patient_id'].nunique()} patients -> {out_csv}")
