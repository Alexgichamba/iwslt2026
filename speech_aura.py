"""
speech_aura.py — Speech-LLM: Conformer Encoder + Projector + Aura-1B

Wires your pretrained SpeechEncoder to the Aura-1B model (custom LLaMA for
African language translation: Hausa, Igbo, Yoruba, Bemba, English).

Architecture:
    audio → SpeechEncoder (4x subsample + Conformer) → Projector (MLP)
         → embed scatter → Aura-1B decoder → logits

Usage:
    # Training
    python speech_aura.py train \
        --encoder_config configs/encoder/conformer_base.yaml \
        --encoder_ckpt checkpoints/encoder_base/encoder_final.pt \
        --aura_ckpt /path/to/Aura-1B/model.pt \
        --aura_tokenizer /path/to/Aura-1B/tokenizer.json \
        --index /ocean/projects/cis250145p/shared/datasets/ASR_INDEX.csv \
        --task transcribe

    # Inference
    python speech_aura.py infer \
        --encoder_config configs/encoder/conformer_base.yaml \
        --encoder_ckpt checkpoints/encoder_base/encoder_final.pt \
        --aura_ckpt /path/to/Aura-1B/model.pt \
        --aura_tokenizer /path/to/Aura-1B/tokenizer.json \
        --checkpoint runs/speech_aura/checkpoint_step10000 \
        --audio_path test.wav \
        --language igbo --task transcribe
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# 1. SPEECH ENCODER (same as speech_llm.py)
# ============================================================================

class ConvSubsampler(nn.Module):
    """4x convolutional subsampling on mel features."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(32 * (input_dim // 4), output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.linear(x)
        lengths = ((lengths - 1) // 2 + 1)
        lengths = ((lengths - 1) // 2 + 1)
        return x, lengths


class SpeechEncoder(nn.Module):
    """Conformer encoder with optional CTC head."""
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        x, lengths = self.subsampler(features, lengths)
        x, lengths = self.conformer(x, lengths)
        out = {"hidden_states": x, "lengths": lengths}
        if self.ctc_head is not None:
            out["ctc_logits"] = self.ctc_head(x)
        return out

    def get_output_dim(self) -> int:
        return self.encoder_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


# ============================================================================
# 2. AUDIO PROJECTOR
# ============================================================================

class AudioProjector(nn.Module):
    """3-layer MLP projector: encoder_dim → expand → llm_hidden_size."""
    def __init__(self, encoder_dim: int, llm_hidden_size: int):
        super().__init__()
        expand_dim = llm_hidden_size * 2
        self.proj1 = nn.Linear(encoder_dim, expand_dim)
        self.act1 = nn.GELU()
        self.proj2 = nn.Linear(expand_dim, expand_dim)
        self.act2 = nn.GELU()
        self.proj3 = nn.Linear(expand_dim, llm_hidden_size)

        nn.init.zeros_(self.proj3.weight)
        nn.init.zeros_(self.proj3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.proj1(x))
        x = self.act2(self.proj2(x))
        return self.proj3(x)


class TransformerProjector(nn.Module):
    """Transformer projector: bidirectional self-attention over audio frames.

    Architecture:
        encoder_dim → linear up-project → TransformerEncoder → linear → llm_hidden_size

    Cross-frame attention lets the projector merge redundant frames,
    emphasize phoneme boundaries, and produce richer representations
    than a per-frame MLP.

    Args:
        encoder_dim: Input dimension from speech encoder.
        llm_hidden_size: Output dimension matching the LLM.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward dimension (default: 4x hidden).
        dropout: Dropout rate.
    """
    def __init__(
        self,
        encoder_dim: int,
        llm_hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = llm_hidden_size
        ffn_dim = ffn_dim or llm_hidden_size * 4

        # Up-project from encoder dim to transformer dim
        self.input_proj = nn.Linear(encoder_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

        # Transformer encoder layers (bidirectional, no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(self.d_model, llm_hidden_size)

        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, encoder_dim) audio frame features
            lengths: (B,) actual frame counts (for padding mask)

        Returns:
            (B, T, llm_hidden_size) projected features
        """
        x = self.input_norm(self.input_proj(x))

        # Build padding mask if lengths provided
        # TransformerEncoder expects: True = ignore this position
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            src_key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_proj(x)
        return x


def build_projector(
    projector_type: str,
    encoder_dim: int,
    llm_hidden_size: int,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """Build projector by type name."""
    if projector_type == "mlp":
        proj = AudioProjector(encoder_dim, llm_hidden_size)
    elif projector_type == "transformer":
        proj = TransformerProjector(
            encoder_dim, llm_hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown projector type: {projector_type}. Choose 'mlp' or 'transformer'.")

    n_params = sum(p.numel() for p in proj.parameters())
    log.info(f"  Projector ({projector_type}): {n_params:,} params")
    return proj


# ============================================================================
# 3. SPEECH-AURA MODEL
# ============================================================================

# Special tokens using Aura's reserved slots
AUDIO_PLACEHOLDER_ID = 11  # <|reserved_special_token_6|> — marks audio embedding positions
TRANSCRIPT_START_ID = 12   # <|reserved_special_token_7|> — marks where transcript begins

# Language code mapping (matches Aura's LANG_MAP)
LANG_MAP = {
    "bemba": 2,   # <|bem_Latn|>
    "bem": 2,
    "yoruba": 3,  # <|yor_Latn|>
    "yor": 3,
    "hausa": 4,   # <|hau_Latn|>
    "hau": 4,
    "igbo": 5,    # <|ibo_Latn|>
    "ibo": 5,
    "english": 10, # <|eng_Latn|>
    "eng": 10,
}


class SpeechAura(nn.Module):
    """
    Conformer encoder + MLP projector + Aura-1B decoder.

    During forward:
        1. Encode audio → (B, T', encoder_dim)
        2. Project → (B, T', aura_hidden=1280)
        3. Build input_ids: [BOS, LANG_TAG, audio_placeholders, ...]
        4. Embed input_ids, masked_scatter audio features into placeholders
        5. Run through Aura decoder
        6. Compute CE loss on target tokens
    """

    def __init__(
        self,
        encoder: SpeechEncoder,
        aura_ckpt: str,
        aura_tokenizer: str,
        aura_size: str = "1b",
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
        freeze_projector: bool = False,
        projector_ckpt: str | None = None,
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        projector_type: str = "mlp",
        projector_layers: int = 2,
        projector_heads: int = 8,
    ):
        super().__init__()
        from transformers import PreTrainedTokenizerFast

        # --- Encoder ---
        self.encoder = encoder
        if freeze_encoder:
            self.encoder.freeze()

        # --- Tokenizer ---
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=aura_tokenizer)
        special_tokens_dict = {}
        if self.tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = "<|end_of_text|>"
        if self.tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = "<|end_of_text|>"
        if self.tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = "<|begin_of_text|>"
        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.bos_id = self.tokenizer.bos_token_id or 0
        self.eos_id = self.tokenizer.eos_token_id or 1
        self.audio_token_id = AUDIO_PLACEHOLDER_ID

        # --- Aura LLM ---
        from llama3 import LlamaTransformer
        from model_factory import model_presets

        config = model_presets['llama-iwslt'][aura_size]
        self.llm = LlamaTransformer(config)

        # Load pretrained weights
        checkpoint = torch.load(aura_ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.llm.load_state_dict(state_dict)
        log.info(f"Loaded Aura-{aura_size} from {aura_ckpt}")

        del checkpoint, state_dict
        import gc; gc.collect()

        self.llm_hidden = config.dim  # 1280 for 1b

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # --- LoRA ---
        if lora_rank > 0 and freeze_llm:
            self._apply_lora(lora_rank, lora_alpha, lora_target_modules or ["q_proj", "v_proj"])

        # --- Projector (always trainable) ---
        self.projector = build_projector(
            projector_type=projector_type,
            encoder_dim=encoder.get_output_dim(),
            llm_hidden_size=self.llm_hidden,
            num_layers=projector_layers,
            num_heads=projector_heads,
        )

        # Load pretrained projector if provided
        if projector_ckpt:
            state = torch.load(projector_ckpt, map_location="cpu", weights_only=True)
            self.projector.load_state_dict(state)
            log.info(f"  Loaded projector from {projector_ckpt}")

        # Freeze projector if requested (for stage 2: LoRA-only training)
        if freeze_projector:
            for param in self.projector.parameters():
                param.requires_grad = False
            log.info(f"  Projector frozen")

        log.info(f"SpeechAura initialized:")
        log.info(f"  Encoder dim: {encoder.get_output_dim()}")
        log.info(f"  Aura hidden: {self.llm_hidden}")
        log.info(f"  Vocab size:  {config.vocab_size}")
        log.info(f"  LoRA rank:   {lora_rank}")
        log.info(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _apply_lora(self, rank: int, alpha: int, target_modules: list[str]):
        """Apply LoRA adapters to specified LLM modules."""
        self._lora_layers = nn.ModuleDict()
        scale = alpha / rank
        for name, module in self.llm.named_modules():
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                safe_name = name.replace(".", "_")
                lora_a = nn.Linear(module.in_features, rank, bias=False)
                lora_b = nn.Linear(rank, module.out_features, bias=False)
                nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_b.weight)
                self._lora_layers[f"{safe_name}_a"] = lora_a
                self._lora_layers[f"{safe_name}_b"] = lora_b

                original_forward = module.forward
                def make_lora_forward(orig, la, lb, s):
                    def lora_forward(x):
                        return orig(x) + lb(la(x.to(la.weight.dtype))) * s
                    return lora_forward
                module.forward = make_lora_forward(original_forward, lora_a, lora_b, scale)
        log.info(f"  LoRA applied to: {target_modules}")

    def _get_embed_layer(self):
        return self.llm.model.embed_tokens

    def _encode_audio(self, features, feature_lengths):
        enc_out = self.encoder(features, feature_lengths)
        hidden = enc_out["hidden_states"]
        lengths = enc_out["lengths"]
        # TransformerProjector accepts lengths for padding mask; MLP ignores it
        if isinstance(self.projector, TransformerProjector):
            projected = self.projector(hidden, lengths)
        else:
            projected = self.projector(hidden)
        return projected, lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 1. Encode audio
        audio_embeds, audio_lens = self._encode_audio(audio_features, audio_lengths)

        # 2. Get text embeddings
        embed_layer = self._get_embed_layer()
        inputs_embeds = embed_layer(input_ids)

        # 3. Build audio mask and scatter
        audio_mask = (input_ids == self.audio_token_id)
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

        all_audio = []
        for i in range(audio_embeds.size(0)):
            all_audio.append(audio_embeds[i, : audio_lens[i]])
        all_audio_cat = torch.cat(all_audio, dim=0)

        n_placeholders = audio_mask.sum().item()
        n_audio = all_audio_cat.size(0)
        if n_placeholders != n_audio:
            if n_audio > n_placeholders:
                all_audio_cat = all_audio_cat[:n_placeholders]
            else:
                pad = torch.zeros(
                    n_placeholders - n_audio, all_audio_cat.size(-1),
                    device=all_audio_cat.device, dtype=all_audio_cat.dtype,
                )
                all_audio_cat = torch.cat([all_audio_cat, pad], dim=0)

        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask_3d, all_audio_cat.to(inputs_embeds.dtype)
        )

        # 4. Forward through Aura (uses inputs_embeds directly)
        # Aura's forward expects input_ids, so we need to bypass embed_tokens
        # and inject our embeds. We do this by temporarily replacing the embedding.
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Run through transformer layers directly
        h = inputs_embeds
        for layer in self.llm.model.layers:
            h = layer(h, position_ids=position_ids)
        h = self.llm.model.norm(h)
        logits = self.llm.lm_head(h)
        logits = logits.float()

        # 5. Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        target_lang: str = "eng",
        max_new_tokens: int = 256,
    ) -> str:
        """Generate text from audio input using KV-cached decoding."""
        from kvcache import KVcache

        audio_embeds, audio_lens = self._encode_audio(audio_features, audio_lengths)
        n_audio_tokens = audio_lens[0].item()

        # Build input: [BOS, LANG_TAG, audio_placeholders, TRANSCRIPT_START]
        lang_id = LANG_MAP.get(target_lang, LANG_MAP["eng"])
        prefix_ids = [self.bos_id, lang_id] + [self.audio_token_id] * n_audio_tokens + [TRANSCRIPT_START_ID]
        input_ids = torch.tensor([prefix_ids], device=audio_embeds.device)

        # Embed and scatter audio
        embed_layer = self._get_embed_layer()
        inputs_embeds = embed_layer(input_ids)

        audio_mask = (input_ids == self.audio_token_id)
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
        audio_flat = audio_embeds[0, :n_audio_tokens].to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask_3d, audio_flat)

        # Prefill with KV cache
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        cache = KVcache(self.llm.config.n_layers)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(input_ids.device.type == "cuda")):
            h = inputs_embeds
            for layer in self.llm.model.layers:
                h = layer(h, position_ids=position_ids, use_cache=True, cache=cache)
            h = self.llm.model.norm(h)
            logits = self.llm.lm_head(h)
            logits = logits.float()

        # Autoregressive decoding with KV cache
        generated = []
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        # Debug: show top predictions after prefill
        top5 = logits[0, -1].topk(5)
        log.debug(f"generate() first token prediction: {next_token.item()} "
                  f"(eos={self.eos_id}), top5: "
                  + ", ".join(f"{self.tokenizer.decode([t.item()])!r}({t.item()})" for t in top5.indices))

        for step in range(max_new_tokens):
            if next_token.item() == self.eos_id:
                break
            generated.append(next_token.item())

            # Feed single token with cached context
            next_embed = embed_layer(next_token)
            pos = torch.tensor([[seq_len + step]], device=input_ids.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(input_ids.device.type == "cuda")):
                h = next_embed
                for layer in self.llm.model.layers:
                    h = layer(h, position_ids=pos, use_cache=True, cache=cache)
                h = self.llm.model.norm(h)
                logits = self.llm.lm_head(h)
                logits = logits.float()
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================================
# 4. DATASET
# ============================================================================

class SpeechDataset(Dataset):
    """Reads CSV index with: audio_id, path, transcript, language, split, source, ..., duration"""

    def __init__(
        self,
        index_path: str,
        split: str = "train",
        languages: list[str] | None = None,
        max_audio_sec: float = 20.0,
        sample_rate: int = 16000,
        task: str = "transcribe",
    ):
        self.entries = []
        with open(index_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "") != split:
                    continue
                lang = row.get("language", "")
                if languages and lang not in languages:
                    continue
                dur = row.get("duration", "")
                if dur and dur.replace(".", "", 1).isdigit():
                    if float(dur) > max_audio_sec:
                        continue
                self.entries.append(row)

        self.max_frames = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate
        self.task = task

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=80,
        )

        # Duration tracking for batch sampler
        self.durations = []
        for e in self.entries:
            dur = e.get("duration", "")
            if dur and dur.replace(".", "", 1).isdigit():
                self.durations.append(float(dur))
            else:
                self.durations.append(max_audio_sec)

        lang_counts = {}
        for e in self.entries:
            lang = e.get("language", "?")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        log.info(f"Loaded {len(self.entries)} examples from {index_path} [split={split}]")
        log.info(f"  Language distribution: {lang_counts}")
        log.info(f"  Duration stats: mean={sum(self.durations)/max(len(self.durations),1):.1f}s, "
                 f"max={max(self.durations) if self.durations else 0:.1f}s")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = entry.get("path", entry.get("audio_path", ""))

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        waveform = torch.from_numpy(data)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > self.max_frames:
            waveform = waveform[:self.max_frames]

        mel = self.mel_transform(waveform)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = (mel + 4.0) / 4.0
        mel = mel.T  # (T, 80)

        target = entry.get("transcript", "")
        language = entry.get("language", "")

        return {
            "mel": mel,
            "mel_length": mel.size(0),
            "target": target,
            "language": language,
        }


def collate_fn(batch, tokenizer, bos_id, eos_id, audio_token_id, task="transcribe", max_seq_len=1024):
    """Build batches with [BOS, LANG_TAG, audio_placeholders, TRANSCRIPT_START, target_tokens, EOS]."""

    mel_lengths = torch.tensor([b["mel_length"] for b in batch])
    max_mel = mel_lengths.max().item()
    mel_batch = torch.zeros(len(batch), max_mel, 80)
    for i, b in enumerate(batch):
        mel_batch[i, :b["mel_length"]] = b["mel"]

    # Encoder output lengths (4x subsampling)
    enc_lengths = ((mel_lengths - 1) // 2 + 1)
    enc_lengths = ((enc_lengths - 1) // 2 + 1)

    all_input_ids = []
    all_labels = []
    for i, b in enumerate(batch):
        n_audio = enc_lengths[i].item()
        lang_id = LANG_MAP.get(b["language"], LANG_MAP.get("eng", 10))

        # Tokenize target
        target_ids = tokenizer.encode(b["target"], add_special_tokens=False)

        # Input: [BOS, LANG, audio..., TRANSCRIPT_START, target..., EOS]
        ids = [bos_id, lang_id] + [audio_token_id] * n_audio + [TRANSCRIPT_START_ID] + target_ids + [eos_id]
        ids = torch.tensor(ids, dtype=torch.long)

        # Labels: -100 for [BOS, LANG, audio..., TRANSCRIPT_START], then target_ids + EOS
        prompt_len = 2 + n_audio + 1  # BOS + LANG + audio + TRANSCRIPT_START
        labels = ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(ids)
        all_labels.append(labels)

    # Skip sequences exceeding max_seq_len
    filtered_ids, filtered_labels, filtered_mels, filtered_mel_lengths = [], [], [], []
    for i in range(len(all_input_ids)):
        if all_input_ids[i].size(0) > max_seq_len:
            continue
        filtered_ids.append(all_input_ids[i])
        filtered_labels.append(all_labels[i])
        filtered_mels.append(mel_batch[i])
        filtered_mel_lengths.append(mel_lengths[i])

    if not filtered_ids:
        return None

    all_input_ids = filtered_ids
    all_labels = filtered_labels
    mel_batch = torch.stack(filtered_mels)
    mel_lengths = torch.stack(filtered_mel_lengths)

    # Pad
    max_seq = max(ids.size(0) for ids in all_input_ids)
    pad_id = eos_id
    input_ids_padded = torch.full((len(all_input_ids), max_seq), pad_id, dtype=torch.long)
    labels_padded = torch.full((len(all_input_ids), max_seq), -100, dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(all_input_ids, all_labels)):
        input_ids_padded[i, :ids.size(0)] = ids
        labels_padded[i, :lab.size(0)] = lab

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "audio_features": mel_batch,
        "audio_lengths": mel_lengths,
    }


# ============================================================================
# 5. TRAINING
# ============================================================================

def load_encoder(config_path: str, checkpoint_path: str | None = None) -> SpeechEncoder:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    encoder = SpeechEncoder(
        input_dim=cfg.get("input_dim", 80),
        encoder_dim=cfg.get("encoder_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        ffn_dim=cfg.get("ffn_dim", 2048),
        num_layers=cfg.get("num_layers", 12),
        depthwise_conv_kernel_size=cfg.get("depthwise_conv_kernel_size", 31),
        dropout=cfg.get("dropout", 0.1),
    )
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "encoder" in state:
            state = state["encoder"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith("ctc_head")}
        encoder.load_state_dict(state, strict=False)
        log.info(f"Loaded encoder from {checkpoint_path}")
    return encoder


@torch.no_grad()
def evaluate(model, val_loader, device, task="transcribe", max_batches=50, generate_samples=20):
    from st.utils.metrics import compute_wer, compute_bleu, compute_chrf

    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in val_loader:
        if batch is None:
            continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                audio_features=batch["audio_features"],
                audio_lengths=batch["audio_lengths"],
            )
        total_loss += outputs["loss"].item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    results = {"loss": total_loss / max(n_batches, 1)}
    torch.cuda.empty_cache()

    # Generation-based metrics
    if generate_samples > 0:
        predictions, references = [], []
        val_ds = val_loader.dataset
        for idx in range(min(generate_samples, len(val_ds))):
            sample = val_ds[idx]
            mel = sample["mel"].unsqueeze(0).to(device)
            mel_len = torch.tensor([sample["mel_length"]], device=device)
            try:
                pred = model.generate(
                    audio_features=mel,
                    audio_lengths=mel_len,
                    target_lang=sample["language"],
                    max_new_tokens=64,
                )
                predictions.append(pred.strip())
                references.append(sample["target"].strip())
            except Exception as e:
                log.warning(f"Generation failed for sample {idx}: {e}")
                continue
            finally:
                # Free KV cache memory after each generation
                del mel, mel_len
                torch.cuda.empty_cache()

        if predictions:
            if task == "transcribe":
                results["wer"] = compute_wer(predictions, references)
            else:
                results["bleu"] = compute_bleu(predictions, references)["bleu"]
                results["chrf"] = compute_chrf(predictions, references)["chrf"]
            for i in range(min(3, len(predictions))):
                log.info(f"  [sample {i}] ref: {references[i][:80]}")
                log.info(f"  [sample {i}] hyp: {predictions[i][:80]}")

    # Final cleanup before returning to training
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return results


def save_checkpoint(model, optimizer, step, output_dir):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
    if hasattr(model, "_lora_layers"):
        torch.save(model._lora_layers.state_dict(), os.path.join(ckpt_dir, "lora.pt"))
    # Save full LLM if it was unfrozen (any LLM param requires grad)
    llm_trainable = any(p.requires_grad for p in model.llm.parameters())
    if llm_trainable:
        torch.save(model.llm.state_dict(), os.path.join(ckpt_dir, "llm.pt"))
        log.info(f"  Saved full LLM weights")
    config = {"encoder_dim": model.encoder.get_output_dim(), "llm_hidden": model.llm_hidden}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Saved checkpoint to {ckpt_dir}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(args.encoder_config, args.encoder_ckpt)
    # If lora_rank > 0, LLM must be frozen (LoRA trains instead)
    # If lora_rank == 0 and --unfreeze_llm, full fine-tune the LLM
    freeze_llm = not args.unfreeze_llm
    if args.lora_rank > 0 and not freeze_llm:
        log.warning("LoRA requires frozen LLM — setting freeze_llm=True")
        freeze_llm = True

    model = SpeechAura(
        encoder=encoder,
        aura_ckpt=args.aura_ckpt,
        aura_tokenizer=args.aura_tokenizer,
        aura_size=args.aura_size,
        freeze_encoder=True,
        freeze_llm=freeze_llm,
        freeze_projector=args.freeze_projector,
        projector_ckpt=args.projector_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        projector_type=args.projector_type,
        projector_layers=args.projector_layers,
        projector_heads=args.projector_heads,
    )
    model = model.to(device)

    languages = args.languages.split(",") if args.languages else None

    train_ds = SpeechDataset(args.index, split=args.train_split, languages=languages,
                              max_audio_sec=args.max_audio_sec, task=args.task)
    val_ds = None
    if args.val_split:
        val_ds = SpeechDataset(args.index, split=args.val_split, languages=languages,
                                max_audio_sec=args.max_audio_sec, task=args.task)

    def collate_wrapper(batch):
        return collate_fn(batch, model.tokenizer, model.bos_id, model.eos_id,
                          model.audio_token_id, args.task, args.max_seq_len)

    # Duration-based batch sampler
    from st.utils.samplers import DurationBucketSampler

    class FastDurationBucketSampler(DurationBucketSampler):
        def __init__(self, *a, max_batch_size=64, **kw):
            self.max_batch_size = max_batch_size
            super().__init__(*a, **kw)
        def _get_durations(self):
            return self.dataset.durations
        def _create_all_batches(self):
            all_batches = []
            for bucket in self.buckets:
                batch, batch_duration = [], 0.0
                for idx in bucket:
                    dur = self.durations[idx]
                    if batch and (batch_duration + dur > self.target_duration
                                  or len(batch) >= self.max_batch_size):
                        all_batches.append(batch)
                        batch, batch_duration = [], 0.0
                    batch.append(idx)
                    batch_duration += dur
                if batch:
                    all_batches.append(batch)
            return all_batches

    train_sampler = FastDurationBucketSampler(
        dataset=train_ds, target_duration=args.max_batch_duration,
        max_batch_size=args.max_batch_size, shuffle=True, shuffle_buckets=True,
    )
    log.info(f"  Duration sampler: {len(train_sampler)} batches, target={args.max_batch_duration}s")

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                               num_workers=args.num_workers, collate_fn=collate_wrapper, pin_memory=True)
    val_loader = None
    if val_ds:
        val_sampler = FastDurationBucketSampler(
            dataset=val_ds, target_duration=args.max_batch_duration,
            max_batch_size=args.max_batch_size, shuffle=False, shuffle_buckets=False,
        )
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                                 num_workers=args.num_workers, collate_fn=collate_wrapper, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    from st.utils.schedulers import build_scheduler
    scheduler = build_scheduler(
        name="cosine_warmup_restarts", optimizer=optimizer, total_steps=args.max_steps,
        max_lr=args.lr, min_lr=args.min_lr,
        first_cycle_steps=args.first_cycle_steps or args.max_steps,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
    )

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name or os.path.basename(args.output_dir),
                       config=vars(args))
            log.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            use_wandb = False

    model.train()
    global_step, epoch = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc="Training", unit="step", dynamic_ncols=True)
    running_loss, loss_count = 0.0, 0

    log.info(f"Starting training for {args.max_steps} steps")
    oom_cooldown = 0
    optimizer.zero_grad()

    while global_step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if batch is None:
                continue

            # Skip batches during OOM cooldown
            if oom_cooldown > 0:
                oom_cooldown -= 1
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            cur_bs = batch["audio_features"].size(0)
            cur_dur = batch["audio_lengths"].sum().item() * 0.01
            seq_len = batch["input_ids"].size(1)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        audio_features=batch["audio_features"],
                        audio_lengths=batch["audio_lengths"],
                    )
                    loss = outputs["loss"] / args.grad_accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM at step {global_step}: bs={cur_bs}, seq={seq_len} — skipping")
                # Aggressive cleanup to prevent OOM cascade
                if hasattr(outputs, "loss"):
                    del outputs
                del batch
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                import gc; gc.collect()
                torch.cuda.empty_cache()
                # Skip a few batches to let memory settle
                oom_cooldown = 3
                continue

            step_loss = loss.item() * args.grad_accum
            running_loss += step_loss
            loss_count += 1

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            scheduler.step()
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{step_loss:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                             bs=cur_bs, dur=f"{cur_dur:.0f}s", epoch=epoch)

            if global_step % args.log_every == 0:
                avg_loss = running_loss / loss_count
                lr = optimizer.param_groups[0]["lr"]
                log.info(f"step {global_step}/{args.max_steps} | loss {step_loss:.4f} | "
                         f"avg_loss {avg_loss:.4f} | lr {lr:.2e} | bs {cur_bs} | dur {cur_dur:.0f}s")
                if use_wandb:
                    wandb.log({"train/loss": step_loss, "train/avg_loss": avg_loss,
                               "train/lr": lr, "train/batch_size": cur_bs,
                               "train/batch_duration_sec": cur_dur, "train/epoch": epoch}, step=global_step)
                running_loss, loss_count = 0.0, 0

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir)

            if val_loader and global_step % args.eval_every == 0:
                torch.cuda.empty_cache()
                metrics = evaluate(model, val_loader, device, task=args.task)
                log.info(f"step {global_step} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                model.train()

            if global_step >= args.max_steps:
                break

    pbar.close()
    if val_loader:
        metrics = evaluate(model, val_loader, device, task=args.task)
        log.info(f"Final | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    save_checkpoint(model, optimizer, global_step, args.output_dir)
    if use_wandb:
        wandb.finish()
    log.info("Training complete!")


# ============================================================================
# 6. INFERENCE
# ============================================================================

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(args.encoder_config, args.encoder_ckpt)
    model = SpeechAura(
        encoder=encoder,
        aura_ckpt=args.aura_ckpt,
        aura_tokenizer=args.aura_tokenizer,
        aura_size=args.aura_size,
    )

    # Load checkpoint
    ckpt_dir = args.checkpoint
    model.projector.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "projector.pt"), map_location="cpu", weights_only=True)
    )
    lora_path = os.path.join(ckpt_dir, "lora.pt")
    if os.path.exists(lora_path) and hasattr(model, "_lora_layers"):
        model._lora_layers.load_state_dict(
            torch.load(lora_path, map_location="cpu", weights_only=True)
        )

    model = model.to(device)
    model.eval()

    data, sr = sf.read(args.audio_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80,
    )
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = (mel + 4.0) / 4.0
    mel = mel.T.unsqueeze(0).to(device)
    mel_len = torch.tensor([mel.size(1)], device=device)

    output = model.generate(
        audio_features=mel,
        audio_lengths=mel_len,
        target_lang=args.language,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{'='*60}")
    print(f"Audio: {args.audio_path}")
    print(f"Language: {args.language}")
    print(f"{'='*60}")
    print(f"Output: {output}")
    print(f"{'='*60}\n")


# ============================================================================
# 7. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Speech-Aura: Conformer + Aura-1B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    tp = subparsers.add_parser("train")
    tp.add_argument("--encoder_config", required=True)
    tp.add_argument("--encoder_ckpt", default=None)
    tp.add_argument("--aura_ckpt", required=True, help="Path to Aura model.pt")
    tp.add_argument("--aura_tokenizer", required=True, help="Path to Aura tokenizer.json")
    tp.add_argument("--aura_size", default="1b", choices=["124m", "500m", "978m", "1b", "2b"])
    tp.add_argument("--index", required=True)
    tp.add_argument("--train_split", default="train")
    tp.add_argument("--val_split", default=None)
    tp.add_argument("--languages", default=None)
    tp.add_argument("--max_audio_sec", type=float, default=20.0)
    tp.add_argument("--output_dir", default="runs/speech_aura")
    tp.add_argument("--task", default="transcribe", choices=["translate", "transcribe"])
    tp.add_argument("--max_batch_duration", type=float, default=120.0)
    tp.add_argument("--max_batch_size", type=int, default=64)
    tp.add_argument("--max_seq_len", type=int, default=1024)
    tp.add_argument("--grad_accum", type=int, default=8)
    tp.add_argument("--lr", type=float, default=2e-4)
    tp.add_argument("--min_lr", type=float, default=1e-6)
    tp.add_argument("--warmup_steps", type=int, default=1000)
    tp.add_argument("--first_cycle_steps", type=int, default=None)
    tp.add_argument("--gamma", type=float, default=1.0, help="Max LR decay factor after each cycle")
    tp.add_argument("--max_steps", type=int, default=50000)
    tp.add_argument("--lora_rank", type=int, default=8)
    tp.add_argument("--lora_alpha", type=int, default=32)
    tp.add_argument("--projector_type", default="mlp", choices=["mlp", "transformer"])
    tp.add_argument("--projector_layers", type=int, default=2, help="Num layers for transformer projector")
    tp.add_argument("--projector_heads", type=int, default=8, help="Num attention heads for transformer projector")
    tp.add_argument("--projector_ckpt", default=None, help="Path to pretrained projector.pt (for stage 2 training)")
    tp.add_argument("--freeze_projector", action="store_true", help="Freeze projector (for LoRA-only stage 2)")
    tp.add_argument("--unfreeze_llm", action="store_true", help="Full fine-tune the LLM (no LoRA, trains all 1B params)")
    tp.add_argument("--log_every", type=int, default=100)
    tp.add_argument("--save_every", type=int, default=5000)
    tp.add_argument("--eval_every", type=int, default=5000)
    tp.add_argument("--num_workers", type=int, default=4)
    tp.add_argument("--no_wandb", action="store_true")
    tp.add_argument("--wandb_project", default="iwslt2026")
    tp.add_argument("--wandb_run_name", default=None)

    # --- Infer ---
    ip = subparsers.add_parser("infer")
    ip.add_argument("--encoder_config", required=True)
    ip.add_argument("--encoder_ckpt", default=None)
    ip.add_argument("--aura_ckpt", required=True)
    ip.add_argument("--aura_tokenizer", required=True)
    ip.add_argument("--aura_size", default="1b")
    ip.add_argument("--checkpoint", required=True)
    ip.add_argument("--audio_path", required=True)
    ip.add_argument("--language", required=True)
    ip.add_argument("--task", default="transcribe")
    ip.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)


if __name__ == "__main__":
    main()