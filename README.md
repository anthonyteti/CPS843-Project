# Static ASL Alphabet Recognition (CPS843)

Pivoted project that compares a classical computer vision pipeline (HOG + PCA + Linear SVM) against a lightweight CNN on the Sign Language MNIST dataset (28x28 grayscale handshapes for the ASL alphabet, excluding `J` and `Z`).

## Quick start

1. **Download the dataset (user action).**
   - From Kaggle grab `sign_mnist_train.csv` and `sign_mnist_test.csv` (a few MB each).
   - Place them under `data/raw/sign_mnist/`.
   - No further preprocessing is required; the scripts load the CSVs directly.

2. **Create a virtualenv and install dependencies.**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Run the pipelines.**
   - Classical HOG + PCA + Linear SVM:
     ```bash
     python src/classical/train_hog_svm.py --train_csv data/raw/sign_mnist/sign_mnist_train.csv ^
        --test_csv data/raw/sign_mnist/sign_mnist_test.csv --pca_dims 20 50 100 200
     ```
     Outputs metrics, confusion matrix, and a pickled model in `results/classical/`.

   - CNN (small/medium/large variants, with optional augmentation):
     ```bash
     python src/cnn/train_sign_cnn.py --train_csv data/raw/sign_mnist/sign_mnist_train.csv ^
        --test_csv data/raw/sign_mnist/sign_mnist_test.csv --variant medium --augment
     ```
     Saves checkpoints to `results/checkpoints/` and training curves + metrics to `results/plots/`.

   - Evaluate a saved CNN checkpoint on the held-out test split:
     ```bash
     python src/cnn/eval_sign_cnn.py --checkpoint results/checkpoints/sign_cnn.pt
     ```

## Repo layout
```
data/
  raw/sign_mnist/         # place sign_mnist_train.csv & sign_mnist_test.csv here (user-provided)
results/
  classical/              # HOG+SVM artifacts and plots
  checkpoints/            # CNN checkpoints
  plots/                  # CNN training curves
  cnn_eval/               # CNN evaluation-time confusion matrices
src/
  classical/train_hog_svm.py
  cnn/models.py
  cnn/train_sign_cnn.py
  cnn/eval_sign_cnn.py
  sign_mnist_dataset.py   # shared dataset + dataloader utilities
  utils.py
report/
  notes.md / drafts for CPS843 write-up
old_project/
  src/                    # original video-based dataset + training code
  scripts/                # helper scripts for WLASL/MS-ASL
  video_data/             # previous data/raw, data/frames, data/splits
  results_video/          # historical checkpoints/plots
  resources/              # WLASL metadata, MS-ASL zip, etc.
```

Legacy video-word files now live entirely under `old_project/`, so the repo root only contains Sign Language MNIST assets.

## Experiments to run
| Experiment | How to reproduce | Outputs |
|------------|------------------|---------|
| Classical vs CNN baseline | `train_hog_svm.py` (PCA=100) vs `train_sign_cnn.py --variant medium --augment` | accuracies + confusion matrices |
| PCA dimension sweep | `train_hog_svm.py --pca_dims 20 50 100 200` | `results/classical/hog_svm_metrics.json` |
| CNN capacity sweep | `train_sign_cnn.py --variant small/medium/large` | training curves, `*_metrics.json`, checkpoints |
| Augmentation study | run CNN with and without `--augment` | compare metrics JSON + accuracy plots |

Use the generated metrics/plots directly in the report (section 5 of `pivot_plan.md`).
