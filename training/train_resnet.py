"""
Model 2A & 2B: Transfer Learning with VGG16 and ResNet50
Uses ImageNet pretrained weights, frozen-base head warmup, selective fine-tuning,
EarlyStopping, ReduceLROnPlateau, and architecture-specific input normalization.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from train_cnn import configure_device, load_split_arrays


def build_transfer_model(arch: str, num_classes: int = 3) -> tuple[tf.keras.Model, tf.keras.Model, callable]:
    if arch == "vgg16":
        base = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        norm_fn = tf.keras.applications.vgg16.preprocess_input
    elif arch == "resnet50":
        base = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        norm_fn = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError(f"Unsupported architecture for train_resnet.py: {arch}")

    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{arch}_lung_classifier")
    return model, base, norm_fn


def train_transfer_architecture(
    arch: str,
    splits_dir: str,
    out_dir: str,
    warmup_epochs: int = 12,
    finetune_epochs: int = 25,
    batch_size: int = 16,
) -> None:
    device = configure_device()
    model, base, norm_fn = build_transfer_model(arch)

    x_train, y_train = load_split_arrays(os.path.join(splits_dir, "train.csv"), norm_fn, augment=True)
    x_val, y_val = load_split_arrays(os.path.join(splits_dir, "val.csv"), norm_fn, augment=False)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{arch}.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    with tf.device(device):
        # Phase 1: Train classification head with frozen ImageNet backbone
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h1 = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=warmup_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Phase 2: Unfreeze top layers for selective fine-tuning
        base.trainable = True
        unfreeze_from = max(0, len(base.layers) - (4 if arch == "vgg16" else 20))
        for layer in base.layers[:unfreeze_from]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=finetune_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    combined = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in set(h1.history) | set(h2.history)}
    pd.DataFrame(combined).to_csv(os.path.join(out_dir, f"{arch}_history.csv"), index=False)
    print(f"Saved fine-tuned {arch} model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["vgg16", "resnet50"], default="resnet50")
    parser.add_argument("--splits-dir", default="training/dataset/splits")
    parser.add_argument("--out-dir", default="backend/model")
    parser.add_argument("--warmup-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_transfer_architecture(
        args.arch,
        args.splits_dir,
        args.out_dir,
        args.warmup_epochs,
        args.finetune_epochs,
        args.batch_size,
    )
