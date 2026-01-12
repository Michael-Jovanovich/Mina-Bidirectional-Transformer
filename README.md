# Mina: A Bidirectional Transformer That Corrects Its Own Predictions

A 30.6M parameter model achieving **0.22 BPC** on held-out data—down from 1.53 BPC using forward-only prediction. **94% of token positions improve** when the model can reconsider them.

## The Architecture

The model runs three passes:

1. **Forward**: Predict the next token (standard causal transformer)
2. **Retrodiction**: Run predictions *backward* through an anti-causal transformer
3. **Correction**: Predict again with cross-attention to retrodiction hidden states

```
Forward:     X → [8 shared layers, causal] → logits
Retrodiction: P → [8 shared layers, anti-causal] → hidden states
Correction:  X → [4 correction layers] → refined logits
                    ↑ cross-attention to retrodiction
```

**Key insight**: Retrodiction never sees true targets—only forward's predictions. Correction learns to read retrodiction's "confusion signals" about whether predictions make sense.

## Results

### Held-Out Test (enwik9 data never seen in training)

| Metric | Forward-Only | With Correction |
|--------|--------------|-----------------|
| BPC | 1.53 | 0.22 |
| Positions Improved | — | 94.1% |

The 1.31 BPC gap quantifies "hindsight advantage"—how much better you could predict each token with future context.

## Pretrained Model

Download the pretrained checkpoint from [Releases](../../releases):

| Model | Params | Dataset | Correction BPC | Download |
|-------|--------|---------|----------------|----------|
| Mina 0.1 | 30.6M | enwik8 | 0.22 | [mina_0.1_30m_enwik8.tar.gz](../../releases/download/v0.1/mina_0.1_30m_enwik8.tar.gz) |

```bash
# Download and extract
tar -xzf mina_0.1_30m_enwik8.tar.gz

# Test pretrained model
python test_held_out.py --checkpoint mina_0.1_30m_enwik8.pt
```

## Quick Start (Train from Scratch)

```bash
# Install dependencies
pip install torch

# Download enwik8 (100MB)
python -c "from data import download_enwik8; download_enwik8()"

# Train
python train.py --epochs 20

# Monitor training (optional, opens GUI)
python monitor.py
```

Training takes ~23 minutes per epoch on RTX 5090. You should see:
- Improvement rate warming from ~75% to ~94% in first few epochs
- Correction BPC dropping much faster than forward BPC

## Testing on Held-Out Data

```bash
# Test on enwik9 data never seen during training
python test_held_out.py --checkpoint best_model.pt
```

## Files

| File | Description |
|------|-------------|
| `model.py` | Bidirectional 1.0 architecture (30.6M params) |
| `train.py` | Training loop with gradient accumulation |
| `data.py` | enwik8 data loading |
| `test_held_out.py` | Held-out evaluation on enwik9 |
| `monitor.py` | Tkinter GUI for live training metrics |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (tested on 2.5)
- ~6GB VRAM (batch_size=32)

## The Compute Tradeoff

- **3 passes per prediction** (~3x compute vs single-pass)
- **7x improvement** in BPC (1.53 → 0.22)
- Compute-adjusted, still net positive

## Acknowledgments

This research was a collaboration with Claude (Opus 4.5) via Claude Code. The architecture emerged from iterative discussion. Claude wrote the implementation code.
