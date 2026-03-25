"""
CTC-based adaptive compression.

Sits between the encoder and projector. Uses CTC predictions to merge
consecutive frames predicted as the same token, producing a
content-adaptive compression of the encoder hidden states.

Flow:
    encoder hidden_states + CTC logits → CTCCompressor → compressed hidden_states

Three merge strategies:
    - avg: equal weight to all frames in a CTC segment
    - weighted: weight by CTC posterior probability
    - softmax: softmax-normalized CTC posteriors as weights
"""

from __future__ import annotations

from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCCompressor(nn.Module):
    """Compress encoder hidden states using CTC predictions.

    Groups consecutive frames with the same CTC prediction (after greedy
    decoding) and merges them using a weighted combination of the hidden states.

    Args:
        strategy: Merge strategy — 'avg', 'weighted', or 'softmax'.
        blank_id: CTC blank token index (default 0).
        remove_blanks: If True, frames predicted as blank are removed entirely.
            If False, consecutive blanks are merged into a single frame.
    """

    def __init__(
        self,
        strategy: str = "avg",
        blank_id: int = 0,
        remove_blanks: bool = False,
    ):
        super().__init__()
        assert strategy in ("avg", "weighted", "softmax"), \
            f"Unknown strategy '{strategy}'. Choose from: avg, weighted, softmax"
        self.strategy = strategy
        self.blank_id = blank_id
        self.remove_blanks = remove_blanks

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctc_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D) encoder hidden states
            ctc_logits: (B, T, V) CTC logits (pre-softmax)
            lengths: (B,) valid lengths per utterance

        Returns:
            compressed: (B, T', D) compressed hidden states (padded)
            new_lengths: (B,) compressed lengths
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Greedy decode
        ctc_probs = ctc_logits.softmax(dim=-1)  # (B, T, V)
        predictions = ctc_logits.argmax(dim=-1)  # (B, T)

        # Find CTC segments per utterance
        all_segments = []  # list of list of (token_id, count)
        new_lengths = []

        for b in range(B):
            seq_len = lengths[b].item()
            pred_seq = predictions[b, :seq_len].tolist()

            # Group consecutive same predictions: [(token_id, count), ...]
            segments = [(token_id, sum(1 for _ in group))
                        for token_id, group in groupby(pred_seq)]

            if self.remove_blanks:
                segments = [(tid, cnt) for tid, cnt in segments if tid != self.blank_id]

            # Handle edge case: all blanks removed
            if not segments:
                segments = [(self.blank_id, 1)]

            all_segments.append(segments)
            new_lengths.append(len(segments))

        new_max_len = max(new_lengths)

        # Build weight matrix: (B, T, T') where T' is compressed length
        # compressed[b, :, t'] = sum over t of weights[b, t, t'] * hidden[b, t, :]
        compressed = torch.zeros(B, new_max_len, D, dtype=dtype, device=device)

        for b in range(B):
            segments = all_segments[b]
            frame_idx = 0
            pred_seq = predictions[b].tolist()
            seq_len = lengths[b].item()

            # Walk through original sequence, matching to segments
            orig_idx = 0
            for seg_idx, (token_id, count) in enumerate(self._iter_with_blanks(
                pred_seq[:seq_len], segments, self.remove_blanks, self.blank_id
            )):
                if orig_idx + count > seq_len:
                    count = seq_len - orig_idx

                segment_hidden = hidden_states[b, orig_idx:orig_idx + count]  # (count, D)

                if self.strategy == "avg":
                    compressed[b, seg_idx] = segment_hidden.mean(dim=0)

                elif self.strategy == "weighted":
                    segment_probs = ctc_probs[b, orig_idx:orig_idx + count, token_id]  # (count,)
                    weights = segment_probs / (segment_probs.sum() + 1e-10)
                    compressed[b, seg_idx] = (weights.unsqueeze(-1) * segment_hidden).sum(dim=0)

                elif self.strategy == "softmax":
                    segment_probs = ctc_probs[b, orig_idx:orig_idx + count, token_id]
                    weights = F.softmax(segment_probs, dim=0)
                    compressed[b, seg_idx] = (weights.unsqueeze(-1) * segment_hidden).sum(dim=0)

                orig_idx += count

        new_lengths = torch.tensor(new_lengths, dtype=torch.long, device=device)
        return compressed, new_lengths

    @staticmethod
    def _iter_with_blanks(pred_seq, segments, remove_blanks, blank_id):
        """Iterate through the original prediction sequence, yielding
        (token_id, count) for each group — including blank groups if
        remove_blanks is True (we need to skip them in the original but
        still advance the original index)."""
        for token_id, group in groupby(pred_seq):
            count = sum(1 for _ in group)
            if remove_blanks and token_id == blank_id:
                # Yield a sentinel so the caller can advance orig_idx
                # but seg_idx doesn't advance (handled by caller)
                yield (token_id, count)
            else:
                yield (token_id, count)


class CTCCompressorV2(nn.Module):
    """Cleaner implementation using matrix multiplication.

    Builds a (B, T, T') weight matrix and does a single batched matmul.
    More memory but faster and fully differentiable.

    Args:
        strategy: Merge strategy — 'avg', 'weighted', or 'softmax'.
        blank_id: CTC blank token index.
        remove_blanks: If True, blank segments are excluded.
    """

    def __init__(
        self,
        strategy: str = "avg",
        blank_id: int = 0,
        remove_blanks: bool = True,
    ):
        super().__init__()
        assert strategy in ("avg", "weighted", "softmax")
        self.strategy = strategy
        self.blank_id = blank_id
        self.remove_blanks = remove_blanks

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctc_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        ctc_probs = ctc_logits.softmax(dim=-1)
        predictions = ctc_logits.argmax(dim=-1)

        # Build segments per utterance
        batch_segments = []
        new_lengths = []

        for b in range(B):
            seq_len = lengths[b].item()
            pred_seq = predictions[b, :seq_len].tolist()

            segments = []
            offset = 0
            for token_id, group in groupby(pred_seq):
                count = sum(1 for _ in group)
                if self.remove_blanks and token_id == self.blank_id:
                    offset += count
                    continue
                segments.append((token_id, offset, count))
                offset += count

            if not segments:
                # All blanks — keep one frame
                segments = [(self.blank_id, 0, 1)]

            batch_segments.append(segments)
            new_lengths.append(len(segments))

        new_max_len = max(new_lengths)

        # Build weight matrix (B, T, T')
        weights = torch.zeros(B, T, new_max_len, dtype=dtype, device=device)

        for b in range(B):
            for seg_idx, (token_id, offset, count) in enumerate(batch_segments[b]):
                if self.strategy == "avg":
                    weights[b, offset:offset + count, seg_idx] = 1.0 / count

                elif self.strategy == "weighted":
                    seg_probs = ctc_probs[b, offset:offset + count, token_id]
                    w = seg_probs / (seg_probs.sum() + 1e-10)
                    weights[b, offset:offset + count, seg_idx] = w

                elif self.strategy == "softmax":
                    seg_probs = ctc_probs[b, offset:offset + count, token_id]
                    w = F.softmax(seg_probs, dim=0)
                    weights[b, offset:offset + count, seg_idx] = w

        # Batched matmul: (B, T, T')^T @ (B, T, D) = (B, T', D)
        compressed = torch.bmm(weights.transpose(1, 2), hidden_states)
        new_lengths = torch.tensor(new_lengths, dtype=torch.long, device=device)

        return compressed, new_lengths


def build_ctc_compressor(config: dict | None) -> CTCCompressorV2 | None:
    """Build a CTC compressor from config, or None if disabled.

    Config format:
        ctc_compress:
            strategy: avg       # avg, weighted, or softmax
            remove_blanks: true
        # or
        ctc_compress: null      # disabled
    """
    if config is None:
        return None

    return CTCCompressorV2(
        strategy=config.get("strategy", "avg"),
        blank_id=config.get("blank_id", 0),
        remove_blanks=config.get("remove_blanks", True),
    )