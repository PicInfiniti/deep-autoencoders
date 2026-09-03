import unittest

import torch

from deep_autoencoders.rbm import RestrictedBoltzmannMachine, pretrain_stack


class RestrictedBoltzmannMachineTests(unittest.TestCase):
    def test_cd1_updates_parameters(self) -> None:
        rbm = RestrictedBoltzmannMachine(visible_units=4, hidden_units=3, seed=7)
        before = rbm.weights.clone()
        error = rbm.cd1_step(
            torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
            learning_rate=0.1,
            momentum=0.5,
            weight_decay=0.0002,
        )
        self.assertGreater(error, 0.0)
        self.assertFalse(torch.equal(before, rbm.weights))

    def test_top_rbm_is_gaussian_and_transforms_deterministically(self) -> None:
        data = torch.rand(20, 4, generator=torch.Generator().manual_seed(1))
        rbms, histories = pretrain_stack(
            data,
            [4, 3, 2],
            epochs=1,
            batch_size=5,
            device="cpu",
            seed=1,
            verbose=False,
        )
        self.assertEqual([rbm.hidden_type for rbm in rbms], ["bernoulli", "gaussian"])
        self.assertEqual(len(histories), 2)
        self.assertEqual(rbms[-1].transform(rbms[0].transform(data)).shape, (20, 2))


if __name__ == "__main__":
    unittest.main()
