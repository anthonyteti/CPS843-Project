import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sign_mnist_dataset import SIGN_MNIST_LETTERS, load_sign_mnist_csv, train_val_split
from utils import ensure_dir, set_seed


def compute_hog_batch(
    images: np.ndarray,
    orientations: int,
    pixels_per_cell: Tuple[int, int],
    cells_per_block: Tuple[int, int],
) -> np.ndarray:
    """Extract HOG features from a batch of images.
    
    L2-Hys normalization is more robust to illumination changes than L2 alone.
    transform_sqrt applies power-law compression to reduce sensitivity to shadows.
    """
    feats: List[np.ndarray] = []
    for img in images:
        feat = hog(
            img / 255.0,  # Normalize to [0, 1]
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
        )
        feats.append(feat)
    return np.stack(feats, axis=0)


def plot_confusion(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HOG + PCA + Linear SVM on Sign Language MNIST.")
    parser.add_argument("--train_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_train.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_test.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/classical"))
    parser.add_argument("--pca_dims", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels_per_cell", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--cells_per_block", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--C", type=float, default=1.0, help="Linear SVM C parameter.")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_images, train_labels = load_sign_mnist_csv(args.train_csv)
    test_images, test_labels = load_sign_mnist_csv(args.test_csv)
    
    # Split training set into train/val at the data level (not feature level)
    # to get realistic validation performance estimates
    train_idx, val_idx = train_val_split(len(train_images), val_ratio=args.val_ratio, seed=args.seed)

    train_images_split = train_images[train_idx]
    val_images = train_images[val_idx]
    y_train = train_labels[train_idx]
    y_val = train_labels[val_idx]
    y_test = test_labels

    hog_kwargs = {
        "orientations": args.orientations,
        "pixels_per_cell": tuple(args.pixels_per_cell),
        "cells_per_block": tuple(args.cells_per_block),
    }
    print("Extracting HOG features...")
    X_train_hog = compute_hog_batch(train_images_split, **hog_kwargs)
    X_val_hog = compute_hog_batch(val_images, **hog_kwargs)
    X_test_hog = compute_hog_batch(test_images, **hog_kwargs)

    # Standardize features to zero mean, unit variance for better SVM convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_hog)
    X_val = scaler.transform(X_val_hog)
    X_test = scaler.transform(X_test_hog)

    results: List[Dict] = []
    best_run: Dict | None = None

    feature_dim = X_train.shape[1]
    valid_dims = [dim for dim in args.pca_dims if dim <= feature_dim]
    if not valid_dims:
        raise ValueError(f"All requested PCA dims are greater than feature dimension {feature_dim}.")
    if len(valid_dims) != len(args.pca_dims):
        skipped = sorted(set(args.pca_dims) - set(valid_dims))
        print(f"Skipping PCA dims larger than feature size {feature_dim}: {skipped}")

    # Sweep over PCA dimensions to find the best accuracy vs. complexity trade-off
    for dim in valid_dims:
        print(f"Training Linear SVM with PCA dim={dim}...")
        pca = PCA(n_components=dim, random_state=args.seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)

        clf = LinearSVC(C=args.C, max_iter=args.max_iter)
        start = time.perf_counter()
        clf.fit(X_train_pca, y_train)
        train_time = time.perf_counter() - start

        train_acc = accuracy_score(y_train, clf.predict(X_train_pca))
        val_acc = accuracy_score(y_val, clf.predict(X_val_pca))
        test_acc = accuracy_score(y_test, clf.predict(X_test_pca))
        print(
            f"PCA {dim}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
            f"time={train_time:.2f}s"
        )
        run_info = {
            "pca_dim": dim,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_time_sec": train_time,
            "pca_explained_var": float(np.sum(pca.explained_variance_ratio_)),
        }
        results.append(run_info)
        if best_run is None or val_acc > best_run["val_acc"]:
            best_run = {
                **run_info,
                "model": clf,
                "pca": pca,
            }

    metrics_path = args.output_dir / "hog_svm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if best_run is None:
        print("No runs were completed.")
        return

    model_dir = ensure_dir(args.output_dir / "artifacts")
    with (model_dir / f"best_hog_svm_pca{best_run['pca_dim']}.pkl").open("wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "pca": best_run["pca"],
                "svm": best_run["model"],
                "hog_params": hog_kwargs,
                "pca_dim": best_run["pca_dim"],
            },
            f,
        )

    # Confusion matrix on test set
    best_pca = best_run["pca"]
    best_model = best_run["model"]
    X_test_best = best_pca.transform(X_test)
    y_pred = best_model.predict(X_test_best)
    cm = confusion_matrix(y_test, y_pred)
    conf_path = args.output_dir / "hog_svm_confusion.png"
    plot_confusion(cm, SIGN_MNIST_LETTERS, conf_path)
    print(f"Saved confusion matrix to {conf_path}")

    report_path = args.output_dir / "hog_svm_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Classification report (test set)\n")
        f.write(classification_report(y_test, y_pred, target_names=SIGN_MNIST_LETTERS))
    print(f"Saved classification report to {report_path}")


if __name__ == "__main__":
    main()
