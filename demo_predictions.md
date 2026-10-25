# Demo predictions (v2 model)

Sample predictions on the first 5 CIFAR-10 test images, using the improved
(v2) model. These are examples only — the measured result is 76.24% test
accuracy on all 10,000 test images (see `eval_v2.log` and the README).

| Image | True class | Predicted class | Confidence |
|-------|------------|-----------------|------------|
| 1 | cat | cat | 73.2% |
| 2 | ship | ship | 99.7% |
| 3 | ship | ship | 72.3% |
| 4 | airplane | airplane | 81.3% |
| 5 | frog | frog | 99.8% |

Run it yourself with `src/predict.py` after training (`src/train_v2.py`).
The v2 model file is not committed; run the training script to recreate it.
