"""
Projector modules that bridge the speech encoder output space to the LLM
input embedding space.

Each projector takes (B, T_speech, D_encoder) and outputs (B, T_proj, D_llm).
Some projectors also downsample the time axis.

Projector lineup:
    - mlp: 2-layer MLP, no time downsampling
    - concat: stack k frames + 2-layer MLP (SLAM-LLM style "linear")
    - conv1d: non-overlapping Conv1d + 2-layer MLP (SLAM-LLM style "cov1d-linear")
    - transformer: standard transformer encoder layers + linear projection
    - qformer: BLIP-2 Q-Former with learnable queries (fixed output length)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP projector (no downsampling)
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """Two-layer MLP with ReLU. No time downsampling."""

    def __init__(self, encoder_dim: int, llm_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj(x), lengths


# ---------------------------------------------------------------------------
# Concat (stack) projector — SLAM-LLM "linear"
# ---------------------------------------------------------------------------

class ConcatProjector(nn.Module):
    """Concatenate k consecutive frames then project via 2-layer MLP.

    This is SLAM-LLM's EncoderProjectorConcat. Deterministic k× downsampling.
    Truncates trailing frames that don't fill a complete group.

    Args:
        encoder_dim: Encoder hidden dimension.
        llm_dim: LLM embedding dimension.
        downsample_factor: Number of frames to concatenate (k).
        hidden_dim: MLP hidden dimension.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_factor: int = 5,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.k = downsample_factor
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim * self.k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        k = self.k

        # Truncate trailing frames (SLAM-LLM style)
        discard = t % k
        if discard > 0:
            x = x[:, :-discard, :]

        x = x.contiguous().view(b, -1, d * k)
        x = self.mlp(x)

        lengths = lengths // k
        return x, lengths


# ---------------------------------------------------------------------------
# Conv1d projector — SLAM-LLM "cov1d-linear"
# ---------------------------------------------------------------------------

class Conv1dProjector(nn.Module):
    """Non-overlapping 1D convolution + 2-layer MLP.

    This is SLAM-LLM's EncoderProjectorCov1d. Uses kernel_size=stride for
    clean non-overlapping downsampling.

    Args:
        encoder_dim: Encoder hidden dimension.
        llm_dim: LLM embedding dimension.
        downsample_factor: Conv stride and kernel size (k).
        hidden_dim: MLP hidden dimension.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_factor: int = 5,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.k = downsample_factor
        self.conv = nn.Conv1d(
            encoder_dim, encoder_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        )
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, D) → (B, D, T) → conv → (B, D, T') → (B, T', D)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)

        lengths = (lengths + self.k - 1) // self.k
        return x, lengths


# ---------------------------------------------------------------------------
# Transformer projector
# ---------------------------------------------------------------------------

class TransformerProjector(nn.Module):
    """Standard transformer encoder layers + linear projection.

    Unlike Q-Former, this preserves the input sequence length (no fixed queries).
    Use with CTC compression or a conv/concat pre-downsampler if the sequence
    is too long for the LLM context.

    Optionally downsamples via strided pooling before the transformer.

    Args:
        encoder_dim: Encoder hidden dimension (input).
        llm_dim: LLM embedding dimension (output).
        d_model: Internal transformer dimension. If different from encoder_dim,
            a linear projection is applied first.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads (must divide d_model evenly).
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        downsample_factor: If > 1, apply average pooling before transformer.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        d_model: int | None = None,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        d_model = d_model or encoder_dim

        # Project encoder_dim → d_model if they differ
        if d_model != encoder_dim:
            self.input_proj = nn.Linear(encoder_dim, d_model)
        else:
            self.input_proj = None

        if downsample_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
        else:
            self.pool = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project to d_model if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Optional downsampling
        if self.pool is not None:
            x = x.transpose(1, 2)  # (B, D, T)
            x = self.pool(x)
            x = x.transpose(1, 2)  # (B, T', D)
            lengths = (lengths + self.downsample_factor - 1) // self.downsample_factor

        # Build padding mask: True for padded positions
        max_len = x.size(1)
        pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(self.proj(x))

        return x, lengths


# ---------------------------------------------------------------------------
# Q-Former projector (BLIP-2 style)
# ---------------------------------------------------------------------------

class QFormerProjector(nn.Module):
    """BLIP-2 Q-Former with learnable query vectors.

    Produces a fixed-length output regardless of input length, which is
    very LLM-context-friendly. Uses cross-attention from learnable queries
    to encoder hidden states.

    Args:
        encoder_dim: Encoder hidden dimension.
        llm_dim: LLM embedding dimension.
        num_queries: Number of learnable query vectors (= output length).
        num_layers: Number of Q-Former layers.
        hidden_dim: Q-Former hidden dimension.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_queries: int = 64,
        num_layers: int = 6,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.num_queries = num_queries

        from transformers import Blip2QFormerConfig, Blip2QFormerModel

        config = Blip2QFormerConfig()
        config.encoder_hidden_size = encoder_dim
        config.hidden_size = hidden_dim
        config.num_hidden_layers = num_layers
        config.num_attention_heads = 12
        config.intermediate_size = hidden_dim * 4

        self.query = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

        self.qformer = Blip2QFormerModel(config)
        self.proj = nn.Linear(hidden_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        # Build encoder attention mask from lengths
        max_len = x.size(1)
        encoder_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        encoder_mask = encoder_mask.long()

        query = self.query.expand(B, -1, -1)

        output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=encoder_mask,
            return_dict=True,
        )

        projected = self.norm(self.proj(output.last_hidden_state))

        # Fixed output length = num_queries for all samples
        out_lengths = torch.full((B,), self.num_queries, dtype=torch.long, device=x.device)
        return projected, out_lengths


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PROJECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLPProjector,
    "concat": ConcatProjector,
    "conv1d": Conv1dProjector,
    "transformer": TransformerProjector,
    "qformer": QFormerProjector,
}


def build_projector(name: str, encoder_dim: int, llm_dim: int, **kwargs) -> nn.Module:
    """Build a projector by name.

    Args:
        name: One of 'mlp', 'concat', 'conv1d', 'transformer', 'qformer'.
        encoder_dim: Output dim of the speech encoder.
        llm_dim: Input embedding dim of the LLM.
        **kwargs: Extra args forwarded to the projector constructor.

    Returns:
        Projector module.
    """
    if name not in PROJECTOR_REGISTRY:
        raise ValueError(f"Unknown projector '{name}'. Choose from {list(PROJECTOR_REGISTRY)}")
    return PROJECTOR_REGISTRY[name](encoder_dim, llm_dim, **kwargs)