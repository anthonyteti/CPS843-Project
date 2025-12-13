from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

SIGN_MNIST_LETTERS: List[str] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]
LABEL_TO_LETTER: Dict[int, str] = {idx: letter for idx, letter in enumerate(SIGN_MNIST_LETTERS)}
LETTER_TO_LABEL: Dict[str, int] = {letter: idx for idx, letter in LABEL_TO_LETTER.items()}


def letter_from_index(class_idx: int) -> str:
    return LABEL_TO_LETTER[class_idx]


def index_from_letter(letter: str) -> int:
    return LETTER_TO_LABEL[letter]


def load_sign_mnist_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Sign Language MNIST CSV file into numpy arrays (N, 28, 28) and labels."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"CSV at {csv_path} is missing a 'label' column.")
    labels = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)
    images = pixels.reshape(-1, 28, 28)
    return images, labels


def train_val_split(
    num_samples: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/val indices."""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def sign_mnist_transforms(
    image_size: int = 32,
    train: bool = True,
    augment: bool = False,
) -> T.Compose:
    """Default transforms for Sign Language MNIST (grayscale).
    
    Augmentation includes rotations, translations, and scaling to improve
    generalization. Only applied during training if augment=True.
    """
    ops: List = []
    if image_size != 28:
        ops.append(T.Resize((image_size, image_size), antialias=True))
    
    # Data augmentation: geometric perturbations to simulate natural variation
    if train and augment:
        ops.extend(
            [
                T.RandomRotation(degrees=10),  # ±10° rotation
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # ±10% translation
                    scale=(0.9, 1.1),  # 0.9x to 1.1x scaling
                ),
            ]
        )
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),  # Map [0, 1] → [-1, 1]
        ]
    )
    return T.Compose(ops)


class SignLanguageMNIST(Dataset):
    """Torch dataset wrapping Sign Language MNIST arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[T.Compose] = None,
    ) -> None:
        assert len(images) == len(labels), "Images/labels must have same length."
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])
        pil_image = Image.fromarray(image)
        if self.transform is None:
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        else:
            tensor = self.transform(pil_image)
        return tensor, label


@dataclass
class SignMNISTDataModule:
    """Helper that keeps datasets and label metadata together."""

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    label_to_idx: Dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)


def _remap_labels(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Remap dataset's sparse label IDs (missing J/Z) to contiguous indices.
    
    Sign Language MNIST excludes J and Z (dynamic signs), so labels have gaps.
    We remap to 0-23 to avoid index out-of-bounds errors in cross-entropy loss.
    """
    unique_raw = sorted(set(train_labels.tolist()) | set(test_labels.tolist()))
    if len(unique_raw) != len(SIGN_MNIST_LETTERS):
        raise ValueError(
            f"Expected {len(SIGN_MNIST_LETTERS)} classes but found {len(unique_raw)} unique labels."
        )
    raw_to_idx = {raw: idx for idx, raw in enumerate(unique_raw)}
    train_mapped = np.array([raw_to_idx[int(lbl)] for lbl in train_labels], dtype=np.int64)
    test_mapped = np.array([raw_to_idx[int(lbl)] for lbl in test_labels], dtype=np.int64)
    return train_mapped, test_mapped, raw_to_idx


def build_sign_mnist_datasets(
    train_csv: str | Path,
    test_csv: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_size: int = 32,
    augment: bool = False,
) -> SignMNISTDataModule:
    train_images, train_labels_raw = load_sign_mnist_csv(train_csv)
    test_images, test_labels_raw = load_sign_mnist_csv(test_csv)
    train_labels, test_labels, _ = _remap_labels(train_labels_raw, test_labels_raw)
    train_idx, val_idx = train_val_split(len(train_images), val_ratio=val_ratio, seed=seed)
    train_transform = sign_mnist_transforms(image_size=image_size, train=True, augment=augment)
    eval_transform = sign_mnist_transforms(image_size=image_size, train=False, augment=False)

    train_subset = SignLanguageMNIST(train_images[train_idx], train_labels[train_idx], transform=train_transform)
    val_subset = SignLanguageMNIST(train_images[val_idx], train_labels[val_idx], transform=eval_transform)
    test_dataset = SignLanguageMNIST(test_images, test_labels, transform=eval_transform)
    label_to_idx = {letter: idx for idx, letter in LABEL_TO_LETTER.items()}
    return SignMNISTDataModule(
        train_dataset=train_subset,
        val_dataset=val_subset,
        test_dataset=test_dataset,
        label_to_idx=dict(LETTER_TO_LABEL),
    )


def build_sign_mnist_dataloaders(
    train_csv: str | Path,
    test_csv: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_size: int = 32,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    data_module = build_sign_mnist_datasets(
        train_csv=train_csv,
        test_csv=test_csv,
        val_ratio=val_ratio,
        seed=seed,
        image_size=image_size,
        augment=augment,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        data_module.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        data_module.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, data_module.label_to_idx
