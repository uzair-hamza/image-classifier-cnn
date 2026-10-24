# Image Classifier with CNN (CIFAR-10)

A convolutional neural network (CNN) that classifies 32×32 colour images into 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Built by **Muhammad Uzair Hamza** — BSc Computer Systems Engineering student, UET Peshawar.
This project is part of my machine learning / computer vision learning journey.

## Dataset

**CIFAR-10** — 60,000 32×32 colour images (50,000 training + 10,000 test).
Official page: https://www.cs.toronto.edu/~kriz/cifar.html

The code downloads CIFAR-10 automatically on first run (via Keras), so you do
not need to download anything by hand.

## Project structure

```
image-classifier-cnn/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py      # CNN architecture (shared by train / evaluate / predict)
│   ├── train.py      # Trains the model, saves it + training curves
│   ├── evaluate.py   # Test accuracy + confusion matrix
│   └── predict.py    # Predict one image from the command line
└── results/          # Created by train.py (accuracy plots, report)
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --epochs 10
```

This trains the CNN, prints train/test accuracy, saves the model to
`results/cnn_model.keras`, and saves training curves to `results/history.png`.

## Evaluate

```bash
python src/evaluate.py
```

Prints test accuracy and a classification report, and saves a confusion
matrix to `results/confusion_matrix.png`.

## Predict one image

```bash
python src/predict.py path/to/image.png
```

Prints the predicted class and the top-3 probabilities.

## How the model works

- Input: 32×32×3 image, pixel values scaled to 0–1.
- Conv block 1: 32 filters (3×3) → ReLU → 32 filters (3×3) → ReLU → max-pool → dropout.
- Conv block 2: 64 filters (3×3) → ReLU → 64 filters (3×3) → ReLU → max-pool → dropout.
- Dense: 128 units → ReLU → dropout → 10 units (softmax).
- Loss: sparse categorical crossentropy. Optimizer: Adam.
- Data augmentation (random flips/shifts) is used during training so the
  model generalises instead of memorising the training set.

## What I learned

- Why CNNs beat plain dense networks on images: shared weights (convolution)
  learn local patterns (edges, textures) anywhere in the image.
- Why pooling helps: it shrinks the image and makes the model less sensitive
  to small shifts.
- Why dropout + augmentation matter: without them the model memorises the
  training images and fails on new ones (overfitting).
- How to read a confusion matrix: it shows *which* classes get mixed up
  (for example cat vs dog), not just one overall accuracy number.

## My results

Trained for 20 epochs on CIFAR-10 (50,000 training images, batch size 32,
Adam optimizer, data augmentation on):

- Test accuracy: **72.16%** (10,000 test images)
- Weighted precision: 0.73, recall: 0.72, F1-score: 0.72
- Training curves: `results/history.png`
- Confusion matrix: `results/confusion_matrix.png`

Notes: the first 5-epoch run reached only 61.49% — training longer (20 epochs)
raised accuracy by over 10 points. More epochs, a deeper network, or transfer
learning would push it higher still. The trained model file
(`results/cnn_model.keras`) is not committed; run `train.py` to recreate it,
then `predict.py` works on any image.

## Next ideas

- Try a deeper network or transfer learning (e.g. MobileNetV2).
- Tune learning rate, batch size, and augmentation and record what changes.
- Test the model on my own photos (resized to 32×32).
