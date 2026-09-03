# Deep Autoencoders

A from-scratch PyTorch implementation of the deep autoencoder introduced by
Geoffrey Hinton and Ruslan Salakhutdinov. The network learns a nonlinear,
low-dimensional representation of MNIST in two stages:

1. greedily pretrain each encoder layer as a restricted Boltzmann machine
   (RBM) using one-step contrastive divergence (CD-1), and
2. unroll the RBM stack into a symmetric autoencoder and fine-tune all weights
   by backpropagating reconstruction cross-entropy.

The implementation uses PyTorch for tensor computation and NumPy for PyTorch's
array interoperability. MNIST downloading, checksum validation, IDX parsing,
tests, metrics, and command-line handling otherwise use Python's standard
library.

## Paper

This repository implements and cites:

> Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of
> data with neural networks. *Science, 313*(5786), 504–507.
> [https://doi.org/10.1126/science.1127647](https://doi.org/10.1126/science.1127647)

The paper is available through the DOI above. The authors' original MATLAB
implementation is available from [Geoffrey Hinton's University of Toronto
page](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html). A local
personal-use PDF can be kept as `Deep-Autoencoders.pdf`; it is intentionally
excluded from Git to respect its redistribution terms.

```bibtex
@article{hinton2006reducing,
  author  = {Hinton, Geoffrey E. and Salakhutdinov, Ruslan R.},
  title   = {Reducing the Dimensionality of Data with Neural Networks},
  journal = {Science},
  volume  = {313},
  number  = {5786},
  pages   = {504--507},
  year    = {2006},
  doi     = {10.1126/science.1127647}
}
```

## What is reproduced

The `paper` preset follows the published MNIST setup and its accompanying
reference code:

| Component | Setting |
|---|---|
| Encoder | 784-1000-500-250-30 |
| Decoder | symmetric 30-250-500-1000-784 |
| Activations | logistic, except for the linear 30-unit code |
| Pretraining | one RBM per encoder layer, CD-1, 50 epochs |
| Intermediate RBMs | Bernoulli hidden units, learning rate 0.1 |
| Code RBM | unit-variance Gaussian hidden units, learning rate 0.001 |
| Regularization | weight decay 0.0002 |
| Momentum | 0.5 through epoch 5, then 0.9 |
| Fine-tuning | reconstruction cross-entropy, 200 epochs |
| Evaluation | cross-entropy and squared reconstruction error per example |

The original MATLAB program fine-tuned with Carl Rasmussen's nonlinear
conjugate-gradient routine, performing three line searches on each 1,000-image
super-batch. This implementation supplies a native Polak-Ribiere nonlinear-CG
optimizer with three Armijo line searches per super-batch. It preserves the
paper's optimization scheme but is not intended to be bit-for-bit identical to
MATLAB. The `quick` preset uses Adam and a smaller network for convenient local
experimentation.

## Setup

Python 3.10 or newer is required. With a virtual environment active:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

PyTorch is pinned in `requirements.txt`; the configured official wheel index
installs a CPU build. To use CUDA, install the matching PyTorch wheel for your
CUDA version and keep the version pin synchronized.

## Download MNIST

```bash
python -m deep_autoencoders download
```

This downloads and verifies the four MNIST files under `data/mnist/`. The data
directory is ignored by Git.

## Train

Run a manageable local experiment first:

```bash
python -m deep_autoencoders train --preset quick --pca
```

Run the full architecture and schedule reported by the paper:

```bash
python -m deep_autoencoders train --preset paper --pca
```

The paper preset is computationally expensive, especially on CPU. A short
end-to-end smoke run can be launched with:

```bash
python -m deep_autoencoders train \
  --layers 784,64,32,10 \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --train-limit 1000 \
  --test-limit 200
```

Training writes `autoencoder.pt` and `metrics.json` under `runs/mnist/`. Both
the output directory and checkpoints are ignored by Git.

Evaluate a saved model:

```bash
python -m deep_autoencoders evaluate runs/mnist/autoencoder.pt
```

Use `python -m deep_autoencoders train --help` to override architecture,
epochs, batch sizes, optimizer, learning rate, device, and dataset limits.

## Tests

```bash
python -m unittest discover -v
```

The tests cover IDX parsing, CD-1 parameter updates, Bernoulli/Gaussian RBM
stacking, symmetric weight unrolling, output bounds, and nonlinear-CG
fine-tuning.

## Repository layout

```text
deep_autoencoders/
  data.py       MNIST download and IDX parsing
  rbm.py        Bernoulli/Gaussian RBMs and greedy CD-1 pretraining
  model.py      symmetric autoencoder and RBM weight unrolling
  training.py   fine-tuning, nonlinear CG, metrics, and PCA
  cli.py        download, train, and evaluate commands
tests/          dependency-free unit tests
```

## License

MIT. See [`LICENSE`](LICENSE).
