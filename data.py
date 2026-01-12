"""Data loading utilities."""
import torch
import os
import urllib.request
import zipfile
from typing import Iterator


def load_enwik8() -> bytes:
    """Download and load enwik8 dataset."""
    filepath = "enwik8.txt"

    if not os.path.exists(filepath):
        print("Downloading enwik8...")
        url = "http://mattmahoney.net/dc/enwik8.zip"
        urllib.request.urlretrieve(url, "enwik8.zip")

        with zipfile.ZipFile("enwik8.zip", 'r') as z:
            z.extractall()

        # Rename extracted file
        if os.path.exists("enwik8") and not os.path.exists(filepath):
            os.rename("enwik8", filepath)

        # Cleanup
        if os.path.exists("enwik8.zip"):
            os.remove("enwik8.zip")

        print(f"Downloaded {os.path.getsize(filepath):,} bytes")

    with open(filepath, 'rb') as f:
        return f.read()


def load_data(path: str) -> bytes:
    """Load dataset from path or download enwik8."""
    if path == "enwik8":
        return load_enwik8()

    with open(path, 'rb') as f:
        return f.read()


def create_batches(
    data: bytes,
    batch_size: int,
    seq_length: int,
    shuffle: bool = True
) -> Iterator[torch.Tensor]:
    """
    Create training batches from raw bytes.

    Each batch is [batch_size, seq_length + 1] to allow
    both forward (input[:-1] -> target[1:]) and
    backward (input[1:] -> target[:-1]) training.
    """
    # Convert bytes to tensor
    tokens = torch.tensor(list(data), dtype=torch.long)
    n_tokens = len(tokens)

    # Each sequence needs seq_length + 1 tokens
    tokens_per_seq = seq_length + 1
    n_sequences = n_tokens // tokens_per_seq

    # Trim to complete sequences and reshape
    tokens = tokens[:n_sequences * tokens_per_seq]
    tokens = tokens.reshape(n_sequences, tokens_per_seq)

    # Shuffle if requested
    if shuffle:
        perm = torch.randperm(n_sequences)
        tokens = tokens[perm]

    # Yield batches
    for i in range(0, n_sequences, batch_size):
        batch = tokens[i:i + batch_size]
        if len(batch) == batch_size:  # Skip incomplete final batch
            yield batch


def split_data(data: bytes, train_ratio: float = 0.9) -> tuple[bytes, bytes]:
    """Split data into train and validation sets."""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]
