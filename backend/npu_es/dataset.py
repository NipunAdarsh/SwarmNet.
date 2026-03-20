"""
MNIST dataset loader — pure numpy, no external ML dependencies.

Downloads the MNIST handwritten digit dataset from the official mirror
and parses the IDX binary format into numpy arrays.
"""

import gzip
import logging
import os
import struct
import urllib.request

import numpy as np

logger = logging.getLogger("npu-es.dataset")

# Official MNIST mirror (Yann LeCun's site is sometimes down)
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, dest: str) -> None:
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        return
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Downloaded %s (%.1f KB)", os.path.basename(dest), os.path.getsize(dest) / 1024)


def _parse_idx_images(path: str) -> np.ndarray:
    """Parse IDX image file into numpy array of shape (N, 784), float32, [0,1]."""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return data


def _parse_idx_labels(path: str) -> np.ndarray:
    """Parse IDX label file into numpy array of shape (N,), int64."""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return data


def download_mnist(data_dir: str | None = None) -> str:
    """Download MNIST dataset files to ``data_dir``.

    Returns the directory path containing the downloaded files.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    os.makedirs(data_dir, exist_ok=True)

    for key, filename in _FILES.items():
        url = f"{_MIRROR}/{filename}"
        dest = os.path.join(data_dir, filename)
        _download_file(url, dest)

    return data_dir


def load_mnist(
    data_dir: str | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset as numpy arrays.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the ``.gz`` files. If None, downloads first.
    max_train : int, optional
        Cap the number of training samples (useful for quick experiments).
    max_test : int, optional
        Cap the number of test samples.

    Returns
    -------
    X_train : np.ndarray, shape (N, 784), float32
    y_train : np.ndarray, shape (N,), int64
    X_test  : np.ndarray, shape (M, 784), float32
    y_test  : np.ndarray, shape (M,), int64
    """
    data_dir = download_mnist(data_dir)

    X_train = _parse_idx_images(os.path.join(data_dir, _FILES["train_images"]))
    y_train = _parse_idx_labels(os.path.join(data_dir, _FILES["train_labels"]))
    X_test = _parse_idx_images(os.path.join(data_dir, _FILES["test_images"]))
    y_test = _parse_idx_labels(os.path.join(data_dir, _FILES["test_labels"]))

    if max_train:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]
    if max_test:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]

    logger.info(
        "MNIST loaded — train: %d samples, test: %d samples",
        len(X_train), len(X_test),
    )
    return X_train, y_train, X_test, y_test
