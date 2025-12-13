import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from cnn.models import SignLanguageCNN, count_parameters  # type: ignore  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_pca_accuracy() -> None:
    data_path = PROJECT_ROOT / "results" / "classical" / "hog_svm_metrics.json"
    data = load_json(data_path)
    dims = [entry["pca_dim"] for entry in data]
    test_acc = [entry["test_acc"] * 100 for entry in data]

    plt.figure(figsize=(6, 4))
    plt.plot(dims, test_acc, marker="o")
    plt.xlabel("PCA dimension")
    plt.ylabel("Test accuracy (%)")
    plt.title("HOG+SVM accuracy vs PCA dimension")
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = PROJECT_ROOT / "results" / "classical" / "pca_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PCA accuracy plot to {out_path}")


def plot_cnn_capacity() -> None:
    runs: List[Tuple[str, str, bool]] = [
        ("small_aug", "small", True),
        ("medium_aug", "medium", True),
        ("large_aug", "large", True),
        ("small_noaug", "small", False),
        ("medium_noaug", "medium", False),
        ("large_noaug", "large", False),
    ]
    params = []
    accuracies = []
    labels = []
    for run_name, variant, augment in runs:
        metrics_path = PROJECT_ROOT / "results" / "plots" / run_name / f"{variant}_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        model = SignLanguageCNN(num_classes=24, variant=variant)
        params.append(count_parameters(model))
        accuracies.append(metrics["test_acc"] * 100)
        labels.append(f"{variant}{' + aug' if augment else ' (no aug)'}")

    plt.figure(figsize=(7, 4))
    plt.scatter(params, accuracies, color="tab:blue")
    for x, y, label in zip(params, accuracies, labels):
        plt.text(x, y + 1, label, ha="center", fontsize=8)
    plt.xlabel("Parameters")
    plt.ylabel("Test accuracy (%)")
    plt.title("CNN capacity vs accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = PROJECT_ROOT / "results" / "plots" / "cnn_capacity.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved CNN capacity plot to {out_path}")


def main() -> None:
    plot_pca_accuracy()
    plot_cnn_capacity()


if __name__ == "__main__":
    main()
