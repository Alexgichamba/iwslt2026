"""
Collator for SpeechAura training.

Intentionally simple — the collator does NOT know about the encoder,
compressor, or LLM token format. It just:
  1. Pads mel features
  2. Tokenizes target text → target_ids
  3. Optionally encodes CTC labels

Sequence assembly (input_ids, labels, audio placeholders) happens inside
SpeechAura.forward() after encoding, when actual post-compression lengths
are known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class AuraCollator:
    """Collator for SpeechAura batches.

    Args:
        tokenizer:         Aura tokenizer (PreTrainedTokenizerFast).
        vocab:             Optional char->id CTC vocab. If provided, also
                           returns ctc_labels and ctc_label_lengths.
        max_target_tokens: Drop samples whose target exceeds this token count.
    """

    tokenizer:         Any
    vocab:             dict[str, int] | None = None
    max_target_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Args:
            batch: List of dicts from SpeechDataset.__getitem__().

        Returns:
            Dict with tensors + language list, or None if all samples dropped.
        """
        # 1. Tokenize targets, drop samples that are too long
        keep: list[int] = []
        target_ids_list: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            ids = self.tokenizer.encode(b["text"], add_special_tokens=False)
            if len(ids) > self.max_target_tokens:
                log.debug(
                    f"Dropping sample {b.get('audio_id', i)}: "
                    f"target {len(ids)} tokens > max_target_tokens={self.max_target_tokens}"
                )
                continue
            target_ids_list.append(torch.tensor(ids, dtype=torch.long))
            keep.append(i)

        if not keep:
            return None

        # 2. Pad mel features
        mel_lens = torch.tensor([batch[i]["mel_len"] for i in keep], dtype=torch.long)
        max_mel  = int(mel_lens.max().item())
        mel_pad  = torch.zeros(len(keep), max_mel, 80)
        for j, i in enumerate(keep):
            b = batch[i]
            mel_pad[j, : b["mel_len"]] = b["mel"]

        # 3. Pad target token ids
        target_lens = torch.tensor([t.size(0) for t in target_ids_list], dtype=torch.long)
        max_target  = int(target_lens.max().item())
        target_pad  = torch.zeros(len(keep), max_target, dtype=torch.long)
        for j, t in enumerate(target_ids_list):
            target_pad[j, : t.size(0)] = t

        out: dict[str, Any] = {
            "audio_features": mel_pad,               # (B, T_mel, 80)
            "audio_lengths":  mel_lens,               # (B,)
            "target_ids":     target_pad,             # (B, L_target)
            "target_lengths": target_lens,            # (B,)
            "language":       [batch[i]["language"] for i in keep],
        }

        # 4. Optional CTC labels
        if self.vocab is not None:
            ctc_list:    list[torch.Tensor] = []
            ctc_lengths: list[int]           = []
            for i in keep:
                text    = batch[i]["text"]
                encoded = [self.vocab[c] for c in text if c in self.vocab]
                ctc_list.append(torch.tensor(encoded, dtype=torch.long))
                ctc_lengths.append(len(encoded))

            max_ctc = max(len(t) for t in ctc_list) if ctc_list else 0
            ctc_pad = torch.zeros(len(keep), max_ctc, dtype=torch.long)
            for j, lab in enumerate(ctc_list):
                ctc_pad[j, : lab.size(0)] = lab

            out["ctc_labels"]        = ctc_pad
            out["ctc_label_lengths"] = torch.tensor(ctc_lengths, dtype=torch.long)

        return out