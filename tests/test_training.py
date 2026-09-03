import unittest

import torch

from deepae.model import DeepAutoencoder
from deepae.training import nonlinear_cg_batch, reconstruction_metrics


class TrainingTests(unittest.TestCase):
    def test_conjugate_gradient_reduces_cross_entropy(self) -> None:
        torch.manual_seed(0)
        data = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
        ).repeat(8, 1)
        model = DeepAutoencoder([4, 3, 2])
        before = reconstruction_metrics(model, data)["cross_entropy_per_example"]
        nonlinear_cg_batch(model, data, iterations=3)
        after = reconstruction_metrics(model, data)["cross_entropy_per_example"]
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
