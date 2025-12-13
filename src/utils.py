import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all libraries.
    
    Disables cuDNN benchmarking for deterministic behavior (slight speed cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_split_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def save_training_plot(
    train_metric: List[float],
    val_metric: List[float],
    ylabel: str,
    save_path: str | Path,
    title: str = "",
) -> None:
    """Save a simple train/val curve plot."""
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, label="train")
    plt.plot(epochs, val_metric, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    ensure_dir(Path(save_path).parent)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def count_correct(preds: torch.Tensor, labels: torch.Tensor) -> int:
    """Utility to count correct predictions."""
    predicted = torch.argmax(preds, dim=1)
    return int((predicted == labels).sum().item())


def class_weights_from_counts(class_counts: List[int]) -> torch.Tensor:
    """Compute inverse frequency weights for imbalanced classes."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.sum() * len(class_counts)
    return weights


def compute_class_weights_from_csv(
    csv_path: str | Path, label_to_idx: Dict[str, int]
) -> torch.Tensor:
    """Load split CSV and derive class weights aligned with label_to_idx."""
    df = pd.read_csv(csv_path)
    counts = [0 for _ in range(len(label_to_idx))]
    for label in df["label"]:
        idx = label_to_idx[label]
        counts[idx] += 1
    return class_weights_from_counts(counts)
