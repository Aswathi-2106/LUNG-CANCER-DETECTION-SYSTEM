"""
Model 2C: EfficientNetB0 Transfer Learning & Fine-Tuning
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from train_cnn import configure_device, load_split_arrays


def train_efficientnet(
    splits_dir: str,
    out_dir: str,
    warmup_epochs: int = 12,
    finetune_epochs: int = 25,
    batch_size: int = 16,
) -> None:
    device = configure_device()
    norm_fn = tf.keras.applications.efficientnet.preprocess_input

    x_train, y_train = load_split_arrays(os.path.join(splits_dir, "train.csv"), norm_fn, augment=True)
    x_val, y_val = load_split_arrays(os.path.join(splits_dir, "val.csv"), norm_fn, augment=False)

    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="efficientnet_b0_lung_classifier")

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "efficientnet_b0.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    with tf.device(device):
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

        # Unfreeze top 25 layers (keep BatchNorm frozen for stability)
        base.trainable = True
        for layer in base.layers[:-25]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
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
    pd.DataFrame(combined).to_csv(os.path.join(out_dir, "efficientnet_b0_history.csv"), index=False)
    print(f"Saved fine-tuned EfficientNetB0 model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="training/dataset/splits")
    parser.add_argument("--out-dir", default="backend/model")
    parser.add_argument("--warmup-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_efficientnet(
        args.splits_dir,
        args.out_dir,
        args.warmup_epochs,
        args.finetune_epochs,
        args.batch_size,
    )
