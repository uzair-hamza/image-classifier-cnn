"""Predict one image with the trained model.

Usage:
    python src/predict.py path/to/image.png

The image is resized to 32x32 (CIFAR-10 size) and the top-3 predicted
classes are printed. Requires results/cnn_model.keras (from train.py).
"""

import os
import sys

import numpy as np
from tensorflow import keras

from model import CLASS_NAMES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/image.png")

    model_path = os.path.join(RESULTS_DIR, "cnn_model.keras")
    if not os.path.exists(model_path):
        raise SystemExit("No trained model found. Run: python src/train.py")

    img = keras.utils.load_img(sys.argv[1], target_size=(32, 32))
    x = keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]
    print(f"Predicted: {CLASS_NAMES[top3[0]]} ({probs[top3[0]] * 100:.1f}%)")
    print("Top 3:")
    for i in top3:
        print(f"  {CLASS_NAMES[i]}: {probs[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
