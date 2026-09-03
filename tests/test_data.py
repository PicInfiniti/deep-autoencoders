import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from deepae.data import _read_images, _read_labels


class MnistParsingTests(unittest.TestCase):
    def test_idx_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images.gz"
            label_path = root / "labels.gz"
            with gzip.open(image_path, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 2, 2, 2))
                stream.write(bytes([0, 64, 128, 255, 255, 128, 64, 0]))
            with gzip.open(label_path, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 2))
                stream.write(bytes([3, 7]))

            images = _read_images(image_path)
            labels = _read_labels(label_path)
            self.assertEqual(images.shape, (2, 4))
            self.assertEqual(labels.tolist(), [3, 7])
            torch.testing.assert_close(images[0, -1], torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
