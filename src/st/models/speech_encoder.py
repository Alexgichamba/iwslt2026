"""
Speech encoder built on torchaudio.models.Conformer.

Supports:
- Standalone CTC pretraining for ASR
- Extraction of hidden states for downstream projector → LLM pipeline
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class ConvSubsampler(nn.Module):
    """Convolutional subsampling (factor 4x) applied before the conformer.

    Takes raw log-mel features and reduces the time dimension by 4x while
    projecting to the conformer input dimension. This is standard practice
    to keep conformer self-attention tractable.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 2x stride twice on freq axis: input_dim // 4
        self.linear = nn.Linear(32 * (input_dim // 4), output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F) log-mel features
            lengths: (B,) original lengths

        Returns:
            out: (B, T', D) subsampled features
            lengths: (B,) subsampled lengths
        """
        # (B, T, F) → (B, 1, T, F) for Conv2d
        x = x.unsqueeze(1)
        x = self.conv(x)  # (B, 32, T//4, F//4)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)  # (B, T//4, 32 * F//4)
        x = self.linear(x)  # (B, T//4, D)

        # Update lengths for the 4x subsampling
        lengths = ((lengths - 1) // 2 + 1)
        lengths = ((lengths - 1) // 2 + 1)

        return x, lengths


class SpeechEncoder(nn.Module):
    """Conformer-based speech encoder with optional CTC head.

    Architecture:
        mel features → ConvSubsampler (4x) → Conformer blocks → hidden states
                                                                  ↓ (optional)
                                                                CTC head → logits

    Args:
        input_dim: Number of mel-frequency bins (e.g. 80).
        encoder_dim: Hidden dimension of the conformer.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward dimension.
        num_layers: Number of conformer blocks.
        depthwise_conv_kernel_size: Kernel size for depthwise convolution.
        dropout: Dropout rate.
        vocab_size: If provided, attaches a CTC projection head.
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 12,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        self.subsampler = ConvSubsampler(input_dim, encoder_dim)

        self.conformer = torchaudio.models.Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        self.ctc_head = nn.Linear(encoder_dim, vocab_size) if vocab_size else None

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, F) log-mel spectrogram
            lengths: (B,) frame counts per utterance

        Returns:
            dict with keys:
                - "hidden_states": (B, T', D) encoder output
                - "lengths": (B,) output lengths
                - "ctc_logits": (B, T', V) only if ctc_head exists
        """
        x, lengths = self.subsampler(features, lengths)
        x, lengths = self.conformer(x, lengths)

        out = {"hidden_states": x, "lengths": lengths}

        if self.ctc_head is not None:
            out["ctc_logits"] = self.ctc_head(x)

        return out

    def get_output_dim(self) -> int:
        return self.encoder_dim

    def freeze(self) -> None:
        """Freeze all encoder parameters (for stage 2+ training)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
