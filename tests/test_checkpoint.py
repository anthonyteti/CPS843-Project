import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnn import train_sign_cnn as trainer


class CheckpointTests(unittest.TestCase):
    def test_best_epoch_survives_later_parameter_updates(self):
        model = torch.nn.Linear(1, 1, bias=False)
        epoch = 0

        def train(*args):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                model.weight.fill_(epoch)
            return 1.0, 0.5

        observed_test_weights = []
        validation = iter([0.9, 0.6])

        def evaluate(model, loader, *args):
            if loader == "test":
                observed_test_weights.append(model.weight.item())
                return 1.0, 0.8
            return 1.0, next(validation)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            argv = ["train", "--epochs", "2", "--checkpoint_path", str(checkpoint),
                    "--plots_dir", tmp]
            with patch.object(sys, "argv", argv), \
                 patch.object(trainer, "get_device", return_value=torch.device("cpu")), \
                 patch.object(trainer, "build_sign_mnist_dataloaders", return_value=("train", "val", "test", {0: 0})), \
                 patch.object(trainer, "SignLanguageCNN", return_value=model), \
                 patch.object(trainer, "train_one_epoch", side_effect=train), \
                 patch.object(trainer, "evaluate", side_effect=evaluate), \
                 patch.object(trainer, "save_training_plot"):
                trainer.main()
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["model_state"]["weight"].item(), 1.0)
            self.assertEqual(saved["val_acc"], 0.9)
            self.assertEqual(observed_test_weights, [1.0])


if __name__ == "__main__":
    unittest.main()
