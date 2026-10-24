"""Train the CNN on CIFAR-10 and save the model + training curves.

Usage:
    python src/train.py --epochs 10

CIFAR-10 downloads automatically on first run (via Keras). Pixel values are
scaled from 0-255 to 0-1. Light augmentation (horizontal flips + small
shifts) is applied so the model generalises instead of memorising.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from model import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # --- Load data (downloads automatically the first time) ---
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # --- Augmentation: flips + small shifts, applied on the fly ---
    datagen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=len(x_train) // args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # --- Report + save ---
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model.save(os.path.join(RESULTS_DIR, "cnn_model.keras"))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "history.png"))
    print(f"Saved model + curves to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
