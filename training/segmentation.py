"""
Optional Lung Nodule Segmentation Module (U-Net) & Detection Guidance
Computes pixel-level lung nodule segmentation masks, Dice Coefficient, and IoU.
Important:
  - Classification models alone (CNN / ResNet50 / EfficientNetB0) do NOT perform precise tumor localization.
  - Pixel-level U-Net training requires paired binary nodule masks (e.g., LIDC-IDRI / LUNA16 annotations).
  - Bounding-box object detection (e.g., YOLOv8) requires annotated lesion bounding boxes.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred >= 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * bce + 0.5 * (1.0 - dice_coefficient(y_true, y_pred))


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape: Tuple[int, int, int] = (224, 224, 1)) -> tf.keras.Model:
    """Standard 4-stage U-Net architecture for binary lung nodule mask segmentation."""
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b = conv_block(p3, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    model = models.Model(inputs, outputs, name="Lung_Nodule_UNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dice_bce_loss,
        metrics=[dice_coefficient, iou_score],
    )
    return model


def overlay_segmentation_mask(ct_gray_224: np.ndarray, pred_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Highlight predicted nodule segmentation contour on top of the original CT slice."""
    rgb = cv2.cvtColor(ct_gray_224.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    binary = (pred_mask.squeeze() >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = rgb.copy()
    overlay[binary > 0] = (0.55 * overlay[binary > 0] + 0.45 * np.array([239, 68, 68])).astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, (245, 158, 11), 2)
    return overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or verify U-Net lung nodule segmentation architecture.")
    parser.add_argument("--summary", action="store_true", help="Print Keras U-Net layer summary")
    args = parser.parse_args()

    unet = build_unet()
    if args.summary:
        unet.summary()
    else:
        print(f"Initialized U-Net model '{unet.name}' with {unet.count_params():,} trainable parameters.")
