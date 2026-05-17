"""
Implementation of Conformer from torchaudio but with Relative Positional
MultiheadAttention instead of regular MultiheadAttention. Batch-first throughout.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from st.models.attention import (
    RelPositionMultiHeadAttention,
    RelPositionalEncoding,
)


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Returns (B, T) bool mask where True = padding."""
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(nn.Module):
    """Conformer convolution module. Input/output: (B, T, D)."""

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels, num_channels, depthwise_kernel_size,
                stride=1, padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels, bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(num_channels, input_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d -> back to (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class ConformerLayer(nn.Module):
    """Single Conformer block. Batch-first: (B, T, D) throughout.

    Macaron FFN -> [conv] -> attention -> [conv] -> FFN -> LN.
    `convolution_first` flips the conv module to before attention.
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        pos_bias_u: Optional[nn.Parameter] = None,
        pos_bias_v: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = RelPositionMultiHeadAttention(
            n_head=num_attention_heads,
            n_feat=input_dim,
            dropout_rate=dropout,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, D)
            attn_mask: (B, T1, T2) bool, True = masked
            pos_emb:   (1, 2T-1, D) relative positional embedding

        Returns:
            (B, T, D)
        """
        # Macaron FFN 1 (half-residual)
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual

        # Optional conv-first ordering
        if self.convolution_first:
            x = x + self.conv_module(x)

        # Self-attention with relative PE
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x, key=x, value=x,
            mask=attn_mask,
            pos_emb=pos_emb,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        # Conv module (default ordering)
        if not self.convolution_first:
            x = x + self.conv_module(x)

        # Macaron FFN 2 (half-residual)
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        return self.final_layer_norm(x)


class Conformer(nn.Module):
    """Conformer encoder with shared relative positional encoding.

    Args:
        input_dim:                  feature dim (also model dim).
        num_heads:                  attention heads per layer.
        ffn_dim:                    hidden dim of FFN modules.
        num_layers:                 number of Conformer blocks.
        depthwise_conv_kernel_size: kernel size for depthwise conv.
        dropout:                    dropout rate.
        use_group_norm:             GroupNorm instead of BatchNorm in conv.
        convolution_first:          apply conv before attention in each layer.
        max_len:                    max sequence length for relative PE buffer.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        max_len: int = 8192,
    ):
        super().__init__()

        d_k = input_dim // num_heads

        # Shared learnable biases for relative PE (Transformer-XL eq. matrices c, d).
        # Shared across layers, matching the Conformer paper.
        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, d_k))

        self.pos_enc = RelPositionalEncoding(
            d_model=input_dim,
            dropout_rate=dropout,
            max_len=max_len,
            xscale=math.sqrt(input_dim),
        )

        self.conformer_layers = nn.ModuleList([
            ConformerLayer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                num_attention_heads=num_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
                convolution_first=convolution_first,
                pos_bias_u=self.pos_bias_u,
                pos_bias_v=self.pos_bias_v,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       (B, T, input_dim)
            lengths: (B,) valid frame counts

        Returns:
            output:  (B, T, input_dim)
            lengths: (B,) — unchanged
        """
        padding_mask = _lengths_to_padding_mask(lengths)         # (B, T)
        attn_mask = padding_mask.unsqueeze(1)                    # (B, 1, T), broadcasts over queries

        x, pos_emb = self.pos_enc(x)                             # x scaled by sqrt(d), pos_emb: (1, 2T-1, D)

        for layer in self.conformer_layers:
            x = layer(x, attn_mask, pos_emb)

        return x, lengths