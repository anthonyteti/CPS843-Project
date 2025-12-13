import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cnn.models import SignLanguageCNN
from sign_mnist_dataset import SIGN_MNIST_LETTERS, build_sign_mnist_dataloaders
from utils import ensure_dir, get_device, set_seed


def plot_confusion(cm: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("CNN Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(SIGN_MNIST_LETTERS))
    plt.xticks(tick_marks, SIGN_MNIST_LETTERS, rotation=90)
    plt.yticks(tick_marks, SIGN_MNIST_LETTERS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Load a saved checkpoint and evaluate on test set.
    
    Returns metrics dict with loss, accuracy, and confusion matrix.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    num_classes = len(label_to_idx)
    model = SignLanguageCNN(num_classes=num_classes, variant=checkpoint["variant"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    cm = confusion_matrix(labels, preds)
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "confusion_matrix": cm,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Sign Language CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_train.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_test.csv"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("results/cnn_eval"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    _, _, loader, _ = build_sign_mnist_dataloaders(
        args.train_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=args.image_size,
        augment=False,
    )

    metrics = evaluate_checkpoint(args.checkpoint, loader, device)
    cm = metrics.pop("confusion_matrix")

    ensure_dir(args.output_dir)
    metrics_path = args.output_dir / "cnn_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved metrics to {metrics_path}")
    plot_confusion(cm, args.output_dir / "cnn_confusion.png")


if __name__ == "__main__":
    main()
