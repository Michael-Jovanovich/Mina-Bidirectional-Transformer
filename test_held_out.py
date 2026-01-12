"""Held-out test for Mina - verify correction BPC is real.

This script is intentionally simple and auditable.
It loads a trained model and tests on data NOT used in training.

CRITICAL: No true targets (Y) are used in any forward pass.
Y is ONLY used to compute the final BPC metric (comparing output to ground truth).
"""
import math
import torch
import torch.nn.functional as F

from model import BidirectionalTransformer, TransformerConfig, Direction

LN2 = math.log(2)


def download_test_data() -> bytes:
    """Download enwik9 and extract a held-out portion NOT in enwik8."""
    import urllib.request
    import zipfile
    import os
    from pathlib import Path

    # enwik8 is first 100M bytes of enwik9
    # We'll use bytes 100M-101M as held-out test (never seen in training)
    cache_path = Path("enwik9_test_holdout.bin")

    if cache_path.exists():
        print(f"Loading cached held-out data from {cache_path}")
        return cache_path.read_bytes()

    print("Downloading enwik9 for held-out test data...")
    url = "http://mattmahoney.net/dc/enwik9.zip"
    zip_path = "enwik9.zip"

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting held-out portion (bytes 100M-101M, never seen in training)...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('enwik9') as f:
            # Skip first 100M (that's enwik8, used for training)
            f.read(100_000_000)
            # Read next 1M as held-out test
            held_out = f.read(1_000_000)

    cache_path.write_bytes(held_out)
    print(f"Saved held-out data to {cache_path}")
    return held_out


@torch.no_grad()
def inference_forward_only(model: BidirectionalTransformer, x: torch.Tensor) -> torch.Tensor:
    """Forward pass only. Input: X, Output: logits. No Y involved."""
    output = model(x, Direction.FORWARD)
    return output.logits


@torch.no_grad()
def inference_full_pipeline(model: BidirectionalTransformer, x: torch.Tensor) -> torch.Tensor:
    """Full 3-pass pipeline. Input: X, Output: refined logits. No Y involved.

    Step 1: Forward pass on X -> predictions
    Step 2: Retrodiction pass on predictions -> hidden states
    Step 3: Correction pass on X with retrodiction hidden states -> refined logits
    """
    # Step 1: Forward
    fwd_output = model(x, Direction.FORWARD)
    predictions = fwd_output.logits.argmax(dim=-1)  # No Y here

    # Step 2: Retrodiction on predictions (NOT on true Y)
    retro_output = model(predictions, Direction.RETRODICTION)
    # Note: retro_output.hidden_states is just from processing predictions
    # No Y involved

    # Step 3: Correction
    corr_output = model(
        x,
        Direction.CORRECTION,
        retrodiction_hidden_states=retro_output.hidden_states,  # From step 2, no Y
    )

    return corr_output.logits


def compute_bpc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute bits-per-character. This is where Y is used - ONLY for evaluation."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    return loss.item() / LN2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\n=== Loading trained model ===")
    checkpoint_path = "mina_0.1_30m_enwik8.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config from checkpoint if available, otherwise use default
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Loaded config from checkpoint: embed_dim={config.embed_dim}, layers={config.n_shared_layers}+{config.n_correction_layers}")
    else:
        config = TransformerConfig()

    model = BidirectionalTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model parameters: {model.count_parameters():,}")

    # Load held-out test data
    print("\n=== Loading held-out test data ===")
    test_data = download_test_data()
    print(f"Test data size: {len(test_data):,} bytes")
    print("This data is from enwik9 bytes 100M-101M, NOT used in training (enwik8 is 0-100M)")

    # Create test sequences
    seq_len = config.seq_length
    batch_size = 32
    n_sequences = 100  # Test on 100 sequences

    print(f"\n=== Running inference on {n_sequences} held-out sequences ===")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")

    forward_bpcs = []
    correction_bpcs = []
    total_sequences_tested = 0

    for i in range(0, n_sequences, batch_size):
        batch_end = min(i + batch_size, n_sequences)
        actual_batch_size = batch_end - i

        # Create batch of sequences from held-out data
        sequences = []
        for j in range(actual_batch_size):
            start = (i + j) * seq_len
            seq_bytes = test_data[start:start + seq_len + 1]
            if len(seq_bytes) < seq_len + 1:
                break
            sequences.append(list(seq_bytes))

        if not sequences:
            break

        total_sequences_tested += len(sequences)
        tokens = torch.tensor(sequences, dtype=torch.long, device=device)

        # Split into input (X) and target (Y)
        x = tokens[:, :-1]  # Input context
        y = tokens[:, 1:]   # Target (ONLY used for BPC computation, not inference)

        # Forward only
        fwd_logits = inference_forward_only(model, x)
        fwd_bpc = compute_bpc(fwd_logits, y)
        forward_bpcs.append(fwd_bpc)

        # Full pipeline (forward -> retrodiction -> correction)
        corr_logits = inference_full_pipeline(model, x)
        corr_bpc = compute_bpc(corr_logits, y)
        correction_bpcs.append(corr_bpc)

        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i//batch_size}: fwd_bpc={fwd_bpc:.3f}, corr_bpc={corr_bpc:.3f}")

    # Final results
    avg_forward_bpc = sum(forward_bpcs) / len(forward_bpcs)
    avg_correction_bpc = sum(correction_bpcs) / len(correction_bpcs)
    gap = avg_forward_bpc - avg_correction_bpc

    print("\n" + "=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"Data source: enwik9 bytes 100M-101M (NOT in training set)")
    print(f"Sequences tested: {total_sequences_tested}")
    print(f"Sequence length: {seq_len}")
    print()
    print(f"Forward-only BPC:     {avg_forward_bpc:.4f}")
    print(f"Full pipeline BPC:    {avg_correction_bpc:.4f}")
    print(f"Improvement (gap):    {gap:.4f} BPC")
    print(f"Improvement ratio:    {(1 - avg_correction_bpc/avg_forward_bpc)*100:.1f}%")
    print("=" * 60)

    print("\nVERIFICATION:")
    print("- Y (true targets) was ONLY used in compute_bpc() for final evaluation")
    print("- inference_forward_only() sees only X")
    print("- inference_full_pipeline() sees only X and derived hidden states")
    print("- No Y flowed into any model forward pass")


if __name__ == "__main__":
    main()
