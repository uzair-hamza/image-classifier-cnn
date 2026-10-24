"""CNN architecture for CIFAR-10 image classification.

Shared by train.py, evaluate.py and predict.py so all three always use the
exact same model. Input images are 32x32 RGB, output is one of 10 classes.
"""

from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_model() -> keras.Model:
    """Build a small CNN: two conv blocks, then a dense head."""
    model = keras.Sequential([
        # --- Conv block 1: learn edges/textures at 32 filters ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # --- Conv block 2: learn bigger patterns at 64 filters ---
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # --- Dense head: classify from the learned features ---
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
