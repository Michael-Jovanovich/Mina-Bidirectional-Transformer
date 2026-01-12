"""Training script for Mina (Bidirectional 1.0).

Correction has full access to retrodiction's hidden states via cross-attention,
enabling it to read confusion signals about forward's predictions.
"""
import argparse
import json
import math
import threading
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from data import create_batches, load_data, split_data
from monitor import MonitorGUI, TrainingMonitor, check_stop_requested
from model import BidirectionalTransformer, Direction, TransformerConfig

LN2 = math.log(2)


def collect_gradient_stats(model: BidirectionalTransformer) -> dict[str, float]:
    """Collect gradient norms by component to identify where parameters are most needed."""
    stats = {}

    # Embeddings
    for name in ['token_embed', 'pos_embed', 'direction_embed']:
        param = getattr(model, name).weight
        if param.grad is not None:
            stats[f'embed/{name}'] = param.grad.norm().item()

    # Directional blocks (shared for forward/retrodiction)
    for i, block in enumerate(model.blocks):
        # Attention
        attn_grad = 0.0
        for pname in ['qkv', 'proj']:
            p = getattr(block.attn, pname).weight
            if p.grad is not None:
                attn_grad += p.grad.norm().item() ** 2
        stats[f'block/{i}/attn'] = attn_grad ** 0.5

        # MLP
        mlp_grad = 0.0
        for p in block.mlp.parameters():
            if p.grad is not None:
                mlp_grad += p.grad.norm().item() ** 2
        stats[f'block/{i}/mlp'] = mlp_grad ** 0.5

    # Correction blocks
    for i, block in enumerate(model.correction_blocks):
        # Self-attention
        self_attn_grad = 0.0
        for pname in ['qkv', 'proj']:
            p = getattr(block.self_attn, pname).weight
            if p.grad is not None:
                self_attn_grad += p.grad.norm().item() ** 2
        stats[f'corr/{i}/self_attn'] = self_attn_grad ** 0.5

        # Cross-attention (the key component!)
        cross_attn_grad = 0.0
        for pname in ['q_proj', 'kv_proj', 'out_proj']:
            p = getattr(block.cross_attn, pname).weight
            if p.grad is not None:
                cross_attn_grad += p.grad.norm().item() ** 2
        stats[f'corr/{i}/cross_attn'] = cross_attn_grad ** 0.5

        # MLP
        mlp_grad = 0.0
        for p in block.mlp.parameters():
            if p.grad is not None:
                mlp_grad += p.grad.norm().item() ** 2
        stats[f'corr/{i}/mlp'] = mlp_grad ** 0.5

    # Output head
    if model.output.weight.grad is not None:
        stats['output/lm_head'] = model.output.weight.grad.norm().item()

    return stats


def summarize_gradient_stats(all_stats: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize gradient stats across batches."""
    summary = defaultdict(list)
    for stats in all_stats:
        for k, v in stats.items():
            summary[k].append(v)

    result = {}
    for k, values in summary.items():
        t = torch.tensor(values)
        result[k] = {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'max': t.max().item(),
        }
    return result


def compute_warmth_per_position(retrodiction_loss: torch.Tensor) -> torch.Tensor:
    """Convert per-position retrodiction loss to warmth (high loss = cold)."""
    return torch.exp(-retrodiction_loss.clamp(max=20.0)).clamp(min=1e-8)


def compute_per_position_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Compute cross-entropy loss per position."""
    B, T = targets.shape
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction='none'
    ).reshape(B, T)


def train_step(
    model: BidirectionalTransformer,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    improvement_margin: float = 0.05,
    warmth_pred_weight: float = 0.1,
    improvement_weight: float = 0.1,
    use_warmth_embedding: bool = False,
    accumulate: int = 1,
    step_idx: int = 0,
) -> dict[str, float]:
    """Training step with cross-attention to retrodiction hidden states.

    Key design: Retrodiction runs on FORWARD'S PREDICTIONS, not true tokens.
    This prevents information leakage and creates true "proofreading" behavior:
    - Retrodiction analyzes what forward actually produced
    - Warmth indicates how well retrodiction understands forward's output
    - Low warmth = forward likely made an error (retrodiction confused by bad prediction)
    """
    # zero_grad is called after optimizer.step() for accumulation
    if step_idx % accumulate == 0:
        optimizer.zero_grad()
    vocab_size = model.config.vocab_size

    fwd_inputs = tokens[:, :-1]
    fwd_targets = tokens[:, 1:]
    retro_targets = tokens[:, :-1]

    # Phase 1: Forward pass
    fwd_output = model(fwd_inputs, Direction.FORWARD)
    fwd_loss_per_pos = compute_per_position_loss(fwd_output.logits, fwd_targets, vocab_size)
    fwd_loss = fwd_loss_per_pos.mean()

    # Get forward's predictions (no gradient through this)
    with torch.no_grad():
        fwd_preds = fwd_output.logits.argmax(dim=-1)

    # Phase 2: Retrodiction pass on forward's predictions (true proofreading)
    retro_output = model(fwd_preds, Direction.RETRODICTION)
    retro_loss_per_pos = compute_per_position_loss(retro_output.logits, retro_targets, vocab_size)
    retro_loss = retro_loss_per_pos.mean()

    with torch.no_grad():
        warmth = compute_warmth_per_position(retro_loss_per_pos)

    # Phase 3: Correction with cross-attention to retrodiction hidden states
    corr_output = model(
        fwd_inputs,
        Direction.CORRECTION,
        warmth=warmth if use_warmth_embedding else None,
        retrodiction_hidden_states=retro_output.hidden_states,
    )
    corr_loss_per_pos = compute_per_position_loss(corr_output.logits, fwd_targets, vocab_size)
    corr_loss = corr_loss_per_pos.mean()

    # Auxiliary losses
    warmth_pred_loss = F.mse_loss(corr_output.warmth_pred, warmth)

    improvement_per_pos = fwd_loss_per_pos - corr_loss_per_pos
    coldness = 1.0 - warmth
    raw_improvement_loss = F.softplus(-improvement_per_pos + improvement_margin)
    improvement_loss = (raw_improvement_loss * coldness).sum() / coldness.sum()

    total_loss = (
        fwd_loss + retro_loss + corr_loss
        + warmth_pred_weight * warmth_pred_loss
        + improvement_weight * improvement_loss
    )

    # Scale loss for accumulation
    (total_loss / accumulate).backward()
    
    # Collect gradient stats if profiling (before they get clipped/cleared)
    grad_stats = None
    if hasattr(model, '_collect_grad_stats') and model._collect_grad_stats:
        grad_stats = collect_gradient_stats(model)
    
    # Only step optimizer every accumulate batches
    if (step_idx + 1) % accumulate == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        improvement_rate = (improvement_per_pos > 0).float().mean().item()
        mean_improvement = improvement_per_pos.mean().item()
        cold_mask = warmth < 0.5
        if cold_mask.any():
            cold_improvement_rate = (improvement_per_pos[cold_mask] > 0).float().mean().item()
        else:
            cold_improvement_rate = 0.0

    return {
        'forward': fwd_loss.item(),
        'backward': retro_loss.item(),
        'correction': corr_loss.item(),
        'warmth_pred': warmth_pred_loss.item(),
        'improvement': improvement_loss.item(),
        'improvement_rate': improvement_rate,
        'mean_improvement': mean_improvement,
        'cold_improvement_rate': cold_improvement_rate,
        'grad_stats': grad_stats,
    }


@torch.no_grad()
def validate(
    model: BidirectionalTransformer,
    val_data: bytes,
    config: TransformerConfig,
    batch_size: int,
    use_warmth_embedding: bool = False,
) -> dict[str, float]:
    """Validate model on held-out data."""
    model.eval()
    device = next(model.parameters()).device
    vocab_size = config.vocab_size

    totals = {
        'forward': 0.0,
        'backward': 0.0,
        'correction': 0.0,
        'warmth_pred': 0.0,
        'improvement_rate': 0.0,
        'mean_improvement': 0.0,
    }
    n_batches = 0

    for tokens in create_batches(val_data, batch_size, config.seq_length, shuffle=False):
        tokens = tokens.to(device)

        fwd_inputs = tokens[:, :-1]
        fwd_targets = tokens[:, 1:]
        retro_targets = tokens[:, :-1]

        # Forward pass
        fwd_output = model(fwd_inputs, Direction.FORWARD)
        fwd_loss_per_pos = compute_per_position_loss(fwd_output.logits, fwd_targets, vocab_size)
        totals['forward'] += fwd_loss_per_pos.mean().item()

        # Retrodiction on forward's predictions (true proofreading)
        fwd_preds = fwd_output.logits.argmax(dim=-1)
        retro_output = model(fwd_preds, Direction.RETRODICTION)
        retro_loss_per_pos = compute_per_position_loss(retro_output.logits, retro_targets, vocab_size)
        totals['backward'] += retro_loss_per_pos.mean().item()

        warmth = compute_warmth_per_position(retro_loss_per_pos)

        corr_output = model(
            fwd_inputs,
            Direction.CORRECTION,
            warmth=warmth if use_warmth_embedding else None,
            retrodiction_hidden_states=retro_output.hidden_states,
        )
        corr_loss_per_pos = compute_per_position_loss(corr_output.logits, fwd_targets, vocab_size)
        totals['correction'] += corr_loss_per_pos.mean().item()
        totals['warmth_pred'] += F.mse_loss(corr_output.warmth_pred, warmth).item()

        improvement_per_pos = fwd_loss_per_pos - corr_loss_per_pos
        totals['improvement_rate'] += (improvement_per_pos > 0).float().mean().item()
        totals['mean_improvement'] += improvement_per_pos.mean().item()

        n_batches += 1

    model.train()

    if n_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(
    path: str,
    model: BidirectionalTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TransformerConfig,
    val_metrics: dict[str, float] | None = None,
) -> None:
    """Save model checkpoint."""
    checkpoint: dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if val_metrics is not None:
        checkpoint['val_metrics'] = val_metrics
    torch.save(checkpoint, path)


def train(
    config: TransformerConfig,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 3e-4,
    accumulate: int = 1,
    data_path: str = "enwik8",
    use_monitor: bool = True,
    improvement_margin: float = 0.05,
    warmth_pred_weight: float = 0.1,
    improvement_weight: float = 0.1,
    checkpoint_path: str | None = None,
    use_warmth_embedding: bool = False,
    gradient_profile: bool = False,
) -> None:
    """Main training loop for Mina with cross-attention to retrodiction."""
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Mode: MINA (Bidirectional 1.0) (asymmetric architecture)")

    model = BidirectionalTransformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    if accumulate > 1:
        print(f"Gradient accumulation: {accumulate} steps (effective batch: {batch_size * accumulate})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_epoch = 0
    best_val_loss = float('inf')

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'val_metrics' in ckpt:
            best_val_loss = ckpt['val_metrics'].get('forward', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    train_data, val_data = split_data(data)
    print(f"Train: {len(train_data):,} bytes, Val: {len(val_data):,} bytes")

    print("Settings:")
    print("  Cross-attention to retrodiction: ENABLED")
    warmth_status = 'ENABLED' if use_warmth_embedding else 'DISABLED'
    print(f"  Warmth embedding: {warmth_status}")
    print(f"  improvement_margin: {improvement_margin}")
    print(f"  warmth_pred_weight: {warmth_pred_weight}")
    print(f"  improvement_weight: {improvement_weight}")

    monitor = None
    if use_monitor:
        log_file = "training_log.json"
        history_file = "training_history.jsonl"
        monitor = TrainingMonitor(log_path=log_file, history_path=history_file)
        print(f"Logging to: {log_file}, {history_file}")
        threading.Thread(
            target=lambda: MonitorGUI(log_path=log_file).run(),
            daemon=True,
        ).start()

    # Total effective batches (optimizer steps, not micro-batches)
    micro_batches_per_epoch = len(train_data) // ((config.seq_length + 1) * batch_size)
    total_batches = micro_batches_per_epoch // accumulate
    stop_requested = False
    
    # Gradient profiling
    gradient_stats_all = [] if gradient_profile else None
    if gradient_profile:
        print("Gradient profiling: ENABLED (will save to gradient_profile.json)")
        model._collect_grad_stats = True
    else:
        model._collect_grad_stats = False

    for epoch in range(start_epoch, epochs):
        if stop_requested:
            break

        epoch_start = time.time()
        epoch_metrics = {
            'fwd': 0.0,
            'retro': 0.0,
            'corr': 0.0,
            'warmth_pred': 0.0,
            'improvement': 0.0,
            'improvement_rate': 0.0,
            'mean_improvement': 0.0,
        }
        n_batches = 0

        for batch_idx, tokens in enumerate(create_batches(train_data, batch_size, config.seq_length)):
            if batch_idx % 10 == 0 and check_stop_requested():
                print("\n>>> Stop requested")
                stop_requested = True
                break

            tokens = tokens.to(device)
            losses = train_step(
                model,
                optimizer,
                tokens,
                improvement_margin=improvement_margin,
                warmth_pred_weight=warmth_pred_weight,
                improvement_weight=improvement_weight,
                use_warmth_embedding=use_warmth_embedding,
                accumulate=accumulate,
                step_idx=batch_idx,
            )

            epoch_metrics['fwd'] += losses['forward']
            epoch_metrics['retro'] += losses['backward']
            epoch_metrics['corr'] += losses['correction']
            epoch_metrics['warmth_pred'] += losses['warmth_pred']
            epoch_metrics['improvement'] += losses['improvement']
            epoch_metrics['improvement_rate'] += losses['improvement_rate']
            epoch_metrics['mean_improvement'] += losses['mean_improvement']
            n_batches += 1
            
            # Collect gradient stats
            if gradient_profile and losses.get('grad_stats'):
                gradient_stats_all.append(losses['grad_stats'])

            if monitor:
                monitor.update(
                    epoch=epoch + 1,
                    batch=batch_idx // accumulate,  # Effective batch (optimizer step)
                    total_batches=total_batches,
                    fwd_loss=losses['forward'],
                    bwd_loss=losses['backward'],
                    corr_loss=losses['correction'],
                    warmth_pred_loss=losses['warmth_pred'],
                    improvement_loss=losses['improvement'],
                    improvement_rate=losses['improvement_rate'],
                    tokens=tokens.numel(),
                )

            # Log every 100 effective batches (optimizer steps)
            if batch_idx % (100 * accumulate) == 0:
                effective_batch = batch_idx // accumulate
                print(
                    f"  Batch {effective_batch}: "
                    f"fwd={losses['forward']:.4f}, "
                    f"corr={losses['correction']:.4f}, "
                    f"impr_rate={losses['improvement_rate']:.1%}",
                    flush=True
                )

        if stop_requested:
            print("Training stopped by user. Saving checkpoint...")
            save_checkpoint('checkpoint_stopped.pt', model, optimizer, epoch, config)
            print("  Saved to checkpoint_stopped.pt")
            break

        epoch_time = time.time() - epoch_start
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        val_metrics = validate(model, val_data, config, batch_size, use_warmth_embedding)

        if monitor:
            monitor.update_validation(val_metrics)

        fwd_bpc = val_metrics['forward'] / LN2
        corr_bpc = val_metrics['correction'] / LN2

        print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)", flush=True)
        print(f"  Train - fwd: {epoch_metrics['fwd']:.4f}, corr: {epoch_metrics['corr']:.4f}")
        print(f"        - improvement_rate: {epoch_metrics['improvement_rate']:.1%}")
        print(f"        - mean_improvement: {epoch_metrics['mean_improvement']:.4f}")
        print(f"  Val   - fwd: {val_metrics['forward']:.4f} (BPC: {fwd_bpc:.3f})")
        print(f"        - corr: {val_metrics['correction']:.4f} (BPC: {corr_bpc:.3f})")
        print(f"        - GAP: {fwd_bpc - corr_bpc:.3f} BPC")
        print(f"        - improvement_rate: {val_metrics['improvement_rate']:.1%}", flush=True)

        if val_metrics['forward'] < best_val_loss:
            best_val_loss = val_metrics['forward']
            save_checkpoint('best_model.pt', model, optimizer, epoch, config, val_metrics)
            print("  Saved best model")

        save_checkpoint('checkpoint.pt', model, optimizer, epoch, config, val_metrics)
        print()
    
    # Save gradient profile if collected
    if gradient_profile and gradient_stats_all:
        print()
        print("Saving gradient profile...")
        summary = summarize_gradient_stats(gradient_stats_all)

        # Sort by mean gradient (descending) to show where gradients are highest
        sorted_components = sorted(summary.items(), key=lambda x: -x[1]['mean'])

        print()
        print("Gradient Profile (top components by mean gradient norm):")
        print(f"{'Component':<30} {'Mean':>10} {'Std':>10} {'Max':>10}")
        print("-" * 62)
        for name, stats in sorted_components[:15]:
            print(f"{name:<30} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['max']:>10.4f}")

        with open('gradient_profile.json', 'w') as f:
            json.dump({
                'summary': summary,
                'n_samples': len(gradient_stats_all),
            }, f, indent=2)
        print()
        print("Full profile saved to gradient_profile.json")


def main() -> None:
    parser = argparse.ArgumentParser(description='Train Mina (Bidirectional 1.0)')

    # Data and training
    parser.add_argument('--data', default='enwik8')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--accumulate', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_shared_layers', type=int, default=8, help='Shared forward/retro layers')
    parser.add_argument('--n_correction_layers', type=int, default=4, help='Correction-specific layers')
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--no-enhanced-direction', action='store_true', help='Disable enhanced direction embedding')

    # Loss weights
    parser.add_argument('--improvement-margin', type=float, default=0.05)
    parser.add_argument('--warmth-pred-weight', type=float, default=0.1)
    parser.add_argument('--improvement-weight', type=float, default=0.1)

    # Flags
    parser.add_argument('--no-monitor', action='store_true')
    parser.add_argument(
        '--use-warmth-embedding',
        action='store_true',
        help='Also add warmth embedding (for ablation)',
    )
    parser.add_argument(
        '--gradient-profile',
        action='store_true',
        help='Collect gradient stats to help decide where to add parameters',
    )

    args = parser.parse_args()

    config = TransformerConfig(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_shared_layers=args.n_shared_layers,
        n_correction_layers=args.n_correction_layers,
        seq_length=args.seq_length,
        enhanced_direction_embed=not args.no_enhanced_direction,
    )

    train(
        config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        accumulate=args.accumulate,
        data_path=args.data,
        use_monitor=not args.no_monitor,
        improvement_margin=args.improvement_margin,
        warmth_pred_weight=args.warmth_pred_weight,
        improvement_weight=args.improvement_weight,
        checkpoint_path=args.checkpoint,
        use_warmth_embedding=args.use_warmth_embedding,
        gradient_profile=args.gradient_profile,
    )


if __name__ == '__main__':
    main()
