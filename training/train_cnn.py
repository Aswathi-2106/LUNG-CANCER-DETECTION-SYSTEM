"""
Model 1: Custom Convolutional Neural Network (CNN) for Lung CT Classification
Includes Batch Normalization, Dropout, Global Average Pooling, EarlyStopping, ReduceLROnPlateau,
and automatic CPU-compatible fallback when GPU is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from preprocessing import augment_ct_slice, preprocess_ct_pipeline

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3


def configure_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return "/GPU:0"
    print("No GPU detected. Running training in CPU-compatible mode.")
    return "/CPU:0"


def load_split_arrays(
    csv_path: str,
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    images = []
    labels = df["label_idx"].to_numpy(dtype=np.int32)

    for idx, fpath in enumerate(df["filepath"]):
        prep = preprocess_ct_pipeline(fpath)
        rgb = prep["rgb_224"]
        if augment:
            rgb = augment_ct_slice(rgb, seed=idx + 100)
        images.append(preprocess_fn(rgb.astype(np.float32)))

    return np.asarray(images, dtype=np.float32), labels


def build_custom_cnn(input_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Custom_Lung_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_custom_cnn(splits_dir: str, out_dir: str, epochs: int = 40, batch_size: int = 16) -> None:
    device = configure_device()
    norm_fn = lambda x: x / 255.0

    x_train, y_train = load_split_arrays(os.path.join(splits_dir, "train.csv"), norm_fn, augment=True)
    x_val, y_val = load_split_arrays(os.path.join(splits_dir, "val.csv"), norm_fn, augment=False)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "custom_cnn.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    with tf.device(device):
        model = build_custom_cnn()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    pd.DataFrame(history.history).to_csv(os.path.join(out_dir, "custom_cnn_history.csv"), index=False)
    print(f"Saved Custom CNN model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="training/dataset/splits")
    parser.add_argument("--out-dir", default="backend/model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_custom_cnn(args.splits_dir, args.out_dir, args.epochs, args.batch_size)
