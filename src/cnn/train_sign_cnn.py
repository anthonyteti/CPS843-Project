import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cnn.models import SignLanguageCNN, count_parameters
from sign_mnist_dataset import build_sign_mnist_dataloaders
from utils import ensure_dir, get_device, save_training_plot, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on a dataset without computing gradients."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sign Language CNN on Sign Language MNIST.")
    parser.add_argument("--train_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_train.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/raw/sign_mnist/sign_mnist_test.csv"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variant", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="Enable rotation/translation augmentation.")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=Path, default=Path("results/checkpoints/sign_cnn.pt"))
    parser.add_argument("--plots_dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.val_ratio >= 0.5:
        raise ValueError("val_ratio should be between 0 and 0.5 to leave enough data for training.")

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, label_to_idx = build_sign_mnist_dataloaders(
        args.train_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=args.image_size,
        augment=args.augment,
    )
    num_classes = len(label_to_idx)
    model = SignLanguageCNN(num_classes=num_classes, variant=args.variant)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_state: Dict | None = None
    patience_counter = 0

    # Track metrics for plotting learning curves
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Early stopping: save best model and stop if no improvement for `patience` epochs
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {
                # state_dict tensors share storage with the live model.
                "model_state": deepcopy(model.state_dict()),
                "variant": args.variant,
                "label_to_idx": label_to_idx,
                "val_acc": best_val_acc,
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        print("Training did not produce a valid checkpoint.")
        return

    ensure_dir(args.checkpoint_path.parent)
    torch.save(best_state, args.checkpoint_path)
    print(f"Saved best checkpoint to {args.checkpoint_path}")

    ensure_dir(args.plots_dir)
    loss_plot = args.plots_dir / f"{args.variant}_loss.png"
    acc_plot = args.plots_dir / f"{args.variant}_acc.png"
    save_training_plot(train_losses, val_losses, ylabel="Loss", save_path=loss_plot, title="CNN Loss")
    save_training_plot(train_accs, val_accs, ylabel="Accuracy", save_path=acc_plot, title="CNN Accuracy")

    # Evaluate the best checkpoint (not the final epoch) on test set
    model.load_state_dict(best_state["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    metrics = {
        "variant": args.variant,
        "augment": args.augment,
        "image_size": args.image_size,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_epochs": len(train_losses),
    }
    metrics_path = args.plots_dir / f"{args.variant}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
