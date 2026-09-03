"""Command-line interface for the MNIST paper reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import download_mnist, load_mnist
from .model import DeepAutoencoder
from .rbm import pretrain_stack
from .training import (
    fine_tune,
    pca_reconstruction_metrics,
    reconstruction_metrics,
    resolve_device,
    seed_everything,
)

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "layers": [784, 256, 128, 30],
        "pretrain_epochs": 5,
        "finetune_epochs": 20,
        "pretrain_batch_size": 100,
        "finetune_batch_size": 256,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    "paper": {
        "layers": [784, 1000, 500, 250, 30],
        "pretrain_epochs": 50,
        "finetune_epochs": 200,
        "pretrain_batch_size": 100,
        "finetune_batch_size": 1000,
        "optimizer": "cg",
        "learning_rate": 0.001,
    },
}


def _parse_layers(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if len(result) < 2 or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("layers must contain at least two positive integers")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepae",
        description="Reproduce Hinton and Salakhutdinov's deep autoencoder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="download and verify MNIST")
    download_parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))

    train_parser = subparsers.add_parser("train", help="pretrain RBMs and fine-tune the autoencoder")
    train_parser.add_argument("--preset", choices=PRESETS, default="quick")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/mnist"))
    train_parser.add_argument("--layers", type=_parse_layers)
    train_parser.add_argument("--pretrain-epochs", type=int)
    train_parser.add_argument("--finetune-epochs", type=int)
    train_parser.add_argument("--pretrain-batch-size", type=int)
    train_parser.add_argument("--finetune-batch-size", type=int)
    train_parser.add_argument("--optimizer", choices=("cg", "adam", "sgd"))
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--train-limit", type=int)
    train_parser.add_argument("--test-limit", type=int)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--pca", action="store_true", help="also evaluate a PCA baseline")
    train_parser.add_argument("--no-download", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a saved checkpoint")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--test-limit", type=int)
    return parser


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(PRESETS[args.preset])
    for name in (
        "layers",
        "pretrain_epochs",
        "finetune_epochs",
        "pretrain_batch_size",
        "finetune_batch_size",
        "optimizer",
        "learning_rate",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    return config


def _train(args: argparse.Namespace) -> None:
    config = _effective_config(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    train_images, _, test_images, _ = load_mnist(
        args.data_dir, download=not args.no_download
    )
    if args.train_limit is not None:
        train_images = train_images[: args.train_limit]
    if args.test_limit is not None:
        test_images = test_images[: args.test_limit]
    if config["layers"][0] != train_images.shape[1]:
        raise ValueError(
            f"input layer has {config['layers'][0]} units, but MNIST has {train_images.shape[1]} pixels"
        )

    # The reference code randomizes once, then reuses fixed mini-batches.
    generator = torch.Generator().manual_seed(args.seed)
    train_images = train_images[torch.randperm(len(train_images), generator=generator)]

    rbms, pretraining_history = pretrain_stack(
        train_images,
        config["layers"],
        epochs=config["pretrain_epochs"],
        batch_size=config["pretrain_batch_size"],
        device=device,
        seed=args.seed,
    )
    model = DeepAutoencoder.from_rbms(rbms)
    finetuning_history = fine_tune(
        model,
        train_images,
        test_images,
        epochs=config["finetune_epochs"],
        batch_size=config["finetune_batch_size"],
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        device=device,
        seed=args.seed,
    )
    final_metrics = reconstruction_metrics(
        model, test_images, batch_size=config["finetune_batch_size"], device=device
    )

    results: dict[str, Any] = {
        "config": config,
        "seed": args.seed,
        "device": str(device),
        "train_examples": len(train_images),
        "test_examples": len(test_images),
        "pretraining_history": pretraining_history,
        "finetuning_history": finetuning_history,
        "test_metrics": final_metrics,
    }
    if args.pca:
        print("Fitting PCA baseline...", flush=True)
        results["pca"] = pca_reconstruction_metrics(
            train_images, test_images, config["layers"][-1]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "autoencoder.pt"
    torch.save(
        {
            "architecture": config["layers"],
            "model_state_dict": model.cpu().state_dict(),
            "rbm_state_dicts": [rbm.state_dict() for rbm in rbms],
        },
        checkpoint_path,
    )
    results_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {results_path}")


def _evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DeepAutoencoder(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    _, _, test_images, _ = load_mnist(args.data_dir, download=False)
    if args.test_limit is not None:
        test_images = test_images[: args.test_limit]
    metrics = reconstruction_metrics(model, test_images, device=device)
    print(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "download":
        download_mnist(args.data_dir)
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)


if __name__ == "__main__":
    main()
