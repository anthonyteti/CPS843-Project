from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CNNConfig:
    """Configuration for CNN variants with different capacities.
    
    Dropout increases with model size to match regularization to capacity.
    """
    name: str
    conv_channels: Tuple[int, ...]  # Number of filters in each conv stage
    fc_dim: int  # Hidden units in fully-connected layer
    dropout: float


CNN_VARIANTS: Dict[str, CNNConfig] = {
    "small": CNNConfig(name="small", conv_channels=(16, 32), fc_dim=64, dropout=0.2),
    "medium": CNNConfig(name="medium", conv_channels=(32, 64), fc_dim=128, dropout=0.3),
    "large": CNNConfig(name="large", conv_channels=(64, 128, 128), fc_dim=256, dropout=0.4),
}


class SignLanguageCNN(nn.Module):
    """Simple CNN for 28x28 grayscale inputs (Sign Language MNIST).
    
    Architecture: Conv-BN-ReLU-MaxPool blocks + AdaptiveAvgPool + FC classifier
    BatchNorm stabilizes training, AdaptiveAvgPool makes the model flexible to input sizes.
    """

    def __init__(
        self,
        num_classes: int = 24,
        variant: str = "medium",
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if variant not in CNN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(CNN_VARIANTS.keys())}.")
        cfg = CNN_VARIANTS[variant]
        
        # Build conv layers: each stage is Conv-BN-ReLU-MaxPool
        layers = []
        prev_channels = in_channels
        for out_channels in cfg.conv_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # Halves spatial resolution
                ]
            )
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling allows variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Two-layer classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, cfg.fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
