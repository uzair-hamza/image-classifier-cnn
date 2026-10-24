"""Evaluate the trained model: test accuracy, report, confusion matrix.

Usage:
    python src/evaluate.py

Requires results/cnn_model.keras (created by train.py).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from model import CLASS_NAMES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    model_path = os.path.join(RESULTS_DIR, "cnn_model.keras")
    if not os.path.exists(model_path):
        raise SystemExit("No trained model found. Run: python src/train.py")

    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.reshape(-1)

    model = keras.models.load_model(model_path)
    preds = model.predict(x_test, verbose=0).argmax(axis=1)

    acc = (preds == y_test).mean()
    print(f"Test accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (rows = true, columns = predicted)")
    plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(10), CLASS_NAMES)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"Saved confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
