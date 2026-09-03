import unittest

import torch

from deepae.model import DeepAutoencoder
from deepae.rbm import RestrictedBoltzmannMachine


class DeepAutoencoderTests(unittest.TestCase):
    def test_unrolling_transposes_generative_weights(self) -> None:
        first = RestrictedBoltzmannMachine(4, 3, seed=1)
        second = RestrictedBoltzmannMachine(3, 2, hidden_type="gaussian", seed=2)
        model = DeepAutoencoder.from_rbms([first, second])

        torch.testing.assert_close(model.encoder[0].weight, first.weights.T)
        torch.testing.assert_close(model.encoder[1].weight, second.weights.T)
        torch.testing.assert_close(model.decoder[0].weight, second.weights)
        torch.testing.assert_close(model.decoder[1].weight, first.weights)
        self.assertEqual(model.encode(torch.rand(5, 4)).shape, (5, 2))
        self.assertEqual(model(torch.rand(5, 4)).shape, (5, 4))

    def test_reconstruction_is_in_unit_interval(self) -> None:
        output = DeepAutoencoder([4, 3, 2])(torch.randn(8, 4))
        self.assertTrue(bool(torch.all(output >= 0)))
        self.assertTrue(bool(torch.all(output <= 1)))


if __name__ == "__main__":
    unittest.main()
