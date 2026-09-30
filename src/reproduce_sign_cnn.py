"""Run the corrected large augmented CNN with a recorded, fixed protocol."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sign_mnist_dataset import train_val_split


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/reproduction"
    output.mkdir(parents=True, exist_ok=True)
    train_path = root / "data/raw/sign_mnist/sign_mnist_train.csv"
    test_path = root / "data/raw/sign_mnist/sign_mnist_test.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if train.shape != (27455, 785) or test.shape != (7172, 785):
        raise ValueError("Unexpected Sign Language MNIST dataset dimensions")
    train_indices, val_indices = train_val_split(len(train), val_ratio=0.1, seed=42)
    # Hash pixel rows only: overlap would be a stronger warning than equal labels.
    train_hashes = set(pd.util.hash_pandas_object(train.drop(columns="label"), index=False))
    test_hashes = set(pd.util.hash_pandas_object(test.drop(columns="label"), index=False))
    overlap = len(train_hashes & test_hashes)
    if overlap:
        raise ValueError(f"Found {overlap} identical train/test image hashes")
    pd.DataFrame({"row_index": np.r_[train_indices, val_indices],
                  "split": ["train"] * len(train_indices) + ["validation"] * len(val_indices)}).to_csv(
                      output / "split_indices.csv", index=False)
    checkpoint = output / "sign_cnn_large_aug.pt"
    commands = [
        [sys.executable, "-u", "src/cnn/train_sign_cnn.py", "--variant", "large",
         "--image_size", "48", "--augment", "--epochs", "20", "--seed", "42",
         "--num_workers", "0", "--checkpoint_path", str(checkpoint.relative_to(root)), "--plots_dir", str(output.relative_to(root))],
        [sys.executable, "-u", "src/cnn/eval_sign_cnn.py", "--checkpoint", str(checkpoint.relative_to(root)),
         "--image_size", "48", "--seed", "42", "--num_workers", "0", "--output_dir", str(output.relative_to(root))],
    ]
    manifest = {
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib"]},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_thread_limit": 4,
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, test_path]},
        "n_train": len(train_indices), "n_validation": len(val_indices), "n_test": len(test),
        "exact_pixel_hash_overlap_train_test": overlap,
        "commands": [[Path(c[0]).name] + c[1:] for c in commands],
        "protocol": "One seed (42), large CNN, augmentation, 48px, max 20 epochs, patience 5; select by validation accuracy.",
        "limitations": ["One run does not measure across-seed uncertainty.",
                        "Static benchmark images do not establish real-world signing accuracy.",
                        "This is an image split, not a verified signer-disjoint split."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Bound CPU parallelism and avoid Windows worker processes duplicating data.
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for i, command in enumerate(commands):
        with (output / ("training.log" if i == 0 else "evaluation.log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print((output / "cnn_eval_metrics.json").read_text())


if __name__ == "__main__":
    main()
