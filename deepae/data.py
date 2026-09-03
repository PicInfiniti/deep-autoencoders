"""Dependency-free MNIST download and IDX parsing."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import torch

_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(root: str | Path) -> None:
    """Download the four canonical MNIST gzip files and verify their checksums."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in _FILES.items():
        destination = root / filename
        if destination.exists() and _md5(destination) == expected_md5:
            print(f"Using {destination}")
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {_BASE_URL}/{filename}")
        try:
            with urllib.request.urlopen(f"{_BASE_URL}/{filename}") as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if _md5(temporary) != expected_md5:
                raise RuntimeError(f"checksum mismatch for {filename}")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic number in {path}: {magic}")
        payload = stream.read()
    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(f"expected {expected} image bytes in {path}, found {len(payload)}")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(count, -1).float() / 255.0


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic number in {path}: {magic}")
        payload = stream.read()
    if len(payload) != count:
        raise ValueError(f"expected {count} labels in {path}, found {len(payload)}")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).long()


def load_mnist(
    root: str | Path,
    *,
    download: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train images, train labels, test images, and test labels."""
    root = Path(root)
    if download:
        download_mnist(root)
    missing = [name for name in _FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"MNIST is missing {missing}; run `python -m deepae download`"
        )

    train_images = _read_images(root / "train-images-idx3-ubyte.gz")
    train_labels = _read_labels(root / "train-labels-idx1-ubyte.gz")
    test_images = _read_images(root / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_labels(root / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not match")
    return train_images, train_labels, test_images, test_labels
