"""Mina: Bidirectional Transformer with Self-Correction (Architecture 1.0)

Three-pass architecture:
1. Forward: Predict next token (causal attention)
2. Retrodiction: Reconstruct input from predictions (anti-causal attention)
3. Correction: Refine predictions using cross-attention to retrodiction hidden states

Key design choices:
- Asymmetric depth: 10 shared layers, 5 correction layers
- Shared weights between forward and retrodiction passes
- Cross-attention uses last N hidden states from retrodiction (N = correction layers)
"""
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Direction(IntEnum):
    FORWARD = 0
    RETRODICTION = 1
    CORRECTION = 2


class ForwardOutput(NamedTuple):
    logits: torch.Tensor
    hidden_states: list[torch.Tensor]


class CorrectionOutput(NamedTuple):
    logits: torch.Tensor
    warmth_pred: torch.Tensor


@dataclass
class TransformerConfig:
    vocab_size: int = 256
    embed_dim: int = 384
    n_heads: int = 6
    n_shared_layers: int = 8      # Shared forward/retrodiction layers (increased!)
    n_correction_layers: int = 4  # Correction-specific layers (reduced)
    seq_length: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.1
    device: str = "cuda"
    enhanced_direction_embed: bool = True  # Use learned transform for direction


class EnhancedDirectionEmbedding(nn.Module):
    """Direction embedding with learned transformation.

    Gradient profile showed direction_embed has very high gradients (0.365 mean, 1.57 max).
    A simple 3-way embedding isn't expressive enough. This adds a learned transform.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(3, embed_dim)
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    @property
    def weight(self):
        """Return transformed embeddings for compatibility with Mina interface."""
        return self.transform(self.embed.weight)

    def forward(self, direction: int) -> torch.Tensor:
        return self.transform(self.embed.weight[direction])


class DirectionalSelfAttention(nn.Module):
    """Self-attention with direction-dependent masking (causal or anti-causal)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.seq_length = config.seq_length

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        causal = torch.tril(torch.ones(config.seq_length, config.seq_length))
        self.register_buffer("causal_mask", causal.view(1, 1, config.seq_length, config.seq_length))

        anticausal = torch.triu(torch.ones(config.seq_length, config.seq_length))
        self.register_buffer("anticausal_mask", anticausal.view(1, 1, config.seq_length, config.seq_length))

    def forward(self, x: torch.Tensor, direction: Direction) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        mask = self.anticausal_mask[:, :, :T, :T] if direction == Direction.RETRODICTION else self.causal_mask[:, :, :T, :T]
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention from correction to retrodiction hidden states."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.kv_proj = nn.Linear(config.embed_dim, 2 * config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k, v = self.kv_proj(context).chunk(2, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_dim = config.embed_dim * config.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectionalBlock(nn.Module):
    """Transformer block for forward/retrodiction passes."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = DirectionalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, direction: Direction) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), direction)
        x = x + self.mlp(self.ln2(x))
        return x


class CorrectionBlock(nn.Module):
    """Transformer block for correction pass with cross-attention to retrodiction."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.self_attn = DirectionalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ln_context = nn.LayerNorm(config.embed_dim)
        self.cross_attn = CrossAttention(config)
        self.ln3 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, retrodiction_hidden: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), Direction.CORRECTION)
        x = x + self.cross_attn(self.ln2(x), self.ln_context(retrodiction_hidden))
        x = x + self.mlp(self.ln3(x))
        return x


class BidirectionalTransformer(nn.Module):
    """Mina: Bidirectional transformer with self-correction.

    Based on gradient profile analysis:
    - More shared layers (8) for forward/retrodiction processing
    - Fewer correction layers (4) since cross-attention has adequate capacity
    - Enhanced direction embedding with learned transformation

    Correction cross-attends to the LAST n_correction_layers hidden states
    from the shared blocks, giving access to the most processed representations.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.seq_length, config.embed_dim)

        # Enhanced or simple direction embedding based on config
        if config.enhanced_direction_embed:
            self.direction_embed = EnhancedDirectionEmbedding(config.embed_dim, config.dropout)
        else:
            self.direction_embed = nn.Embedding(3, config.embed_dim)

        # Warmth embedding (optional, kept for ablation)
        self.warmth_embed = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.Tanh()
        )

        self.warmth_predictor = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(config.dropout)

        # Shared forward/retrodiction blocks (MORE layers based on gradient analysis)
        self.blocks = nn.ModuleList([
            DirectionalBlock(config) for _ in range(config.n_shared_layers)
        ])

        # Correction blocks with cross-attention (FEWER layers)
        self.correction_blocks = nn.ModuleList([
            CorrectionBlock(config) for _ in range(config.n_correction_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size)

        self.apply(self._init_weights)

        # Print architecture summary
        print(f"Architecture: {config.n_shared_layers} shared layers, {config.n_correction_layers} correction layers")
        print(f"  Correction cross-attends to last {config.n_correction_layers} shared hidden states")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward_direction(self, tokens: torch.Tensor, direction: Direction) -> ForwardOutput:
        """Forward/retrodiction pass returning hidden states for cross-attention."""
        B, T = tokens.shape

        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos) + self.direction_embed.weight[direction]
        x = self.dropout(x)

        hidden_states = []
        for block in self.blocks:
            x = block(x, direction)
            hidden_states.append(x)

        logits = self.output(self.ln_f(x))
        return ForwardOutput(logits=logits, hidden_states=hidden_states)

    def forward_correction(
        self,
        tokens: torch.Tensor,
        retrodiction_hidden_states: list[torch.Tensor],
        warmth: torch.Tensor | None = None,
    ) -> CorrectionOutput:
        """Correction pass with cross-attention to retrodiction hidden states.

        Uses the LAST n_correction_layers hidden states from retrodiction,
        which contain the most processed information.
        """
        B, T = tokens.shape

        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos) + self.direction_embed.weight[Direction.CORRECTION]

        if warmth is not None:
            x = x + self.warmth_embed(warmth.unsqueeze(-1))

        x = self.dropout(x)

        # Use last n_correction_layers hidden states from retrodiction
        # This gives correction access to the most processed representations
        n_shared = len(retrodiction_hidden_states)
        n_corr = len(self.correction_blocks)
        start_idx = n_shared - n_corr  # e.g., 8 shared - 4 corr = start at index 4

        for i, block in enumerate(self.correction_blocks):
            retro_hidden = retrodiction_hidden_states[start_idx + i]
            x = block(x, retro_hidden)

        x = self.ln_f(x)
        logits = self.output(x)
        warmth_pred = self.warmth_predictor(x).squeeze(-1)

        return CorrectionOutput(logits=logits, warmth_pred=warmth_pred)

    def forward(
        self,
        tokens: torch.Tensor,
        direction: Direction,
        warmth: torch.Tensor | None = None,
        retrodiction_hidden_states: list[torch.Tensor] | None = None,
    ) -> ForwardOutput | CorrectionOutput:
        """Unified forward pass."""
        if direction == Direction.CORRECTION:
            if retrodiction_hidden_states is None:
                raise ValueError("Correction mode requires retrodiction_hidden_states")
            return self.forward_correction(tokens, retrodiction_hidden_states, warmth)
        return self.forward_direction(tokens, direction)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
