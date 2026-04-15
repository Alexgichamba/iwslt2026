"""
speech_llm.py — Single-file Speech-LLM: Conformer Encoder + Projector + Qwen3 LLM

Wires your pretrained SpeechEncoder to any Qwen model (e.g. Qwen/Qwen3-1.7B)
using the same masked_scatter fusion pattern as Qwen3-ASR.

Architecture:
    audio → SpeechEncoder (4x subsample + Conformer) → Projector (MLP) → embed scatter → Qwen3 decoder → logits

Usage:
    # Training (LoRA fine-tune)
    python speech_llm.py train \
        --encoder_ckpt checkpoints/encoder/encoder_step50000.pt \
        --encoder_config configs/encoder/conformer_base.yaml \
        --llm_name Qwen/Qwen3-1.7B \
        --train_index data/ast_train.tsv \
        --val_index data/ast_val.tsv \
        --output_dir runs/speech_llm_v1 \
        --max_steps 20000

    # Inference
    python speech_llm.py infer \
        --checkpoint runs/speech_llm_v1/checkpoint_step20000 \
        --audio_path test.wav \
        --task translate  # or "transcribe"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import torchaudio  # only used for MelSpectrogram transform + Conformer model
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# 1. SPEECH ENCODER (your existing code, inlined for self-containment)
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        x, lengths = self.subsampler(features, lengths)
        x, lengths = self.conformer(x, lengths)
        out = {"hidden_states": x, "lengths": lengths}
        if self.ctc_head is not None:
            out["ctc_logits"] = self.ctc_head(x)
        return out

    def get_output_dim(self) -> int:
        return self.encoder_dim

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


# ============================================================================
# 2. AUDIO PROJECTOR
# ============================================================================

class AudioProjector(nn.Module):
    """2-layer MLP projector: encoder_dim → llm_hidden_size.

    Same pattern as Qwen3-ASR (proj1 → GELU → proj2).
    """

    def __init__(self, encoder_dim: int, llm_hidden_size: int):
        super().__init__()
        self.proj1 = nn.Linear(encoder_dim, llm_hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(x)))


# ============================================================================
# 3. SPEECH-LLM MODEL
# ============================================================================

# Special token that marks where audio embeddings go in the input sequence.
AUDIO_PLACEHOLDER = "<|audio|>"


class SpeechLLM(nn.Module):
    """
    Conformer encoder + MLP projector + Qwen3 decoder.

    During forward:
        1. Encode audio → (B, T', encoder_dim)
        2. Project → (B, T', llm_hidden)
        3. Embed input_ids (which contain <|audio|> placeholders)
        4. masked_scatter audio features into placeholder positions
        5. Run through the LLM decoder
        6. Compute CE loss on label tokens
    """

    def __init__(
        self,
        encoder: SpeechEncoder,
        llm_name: str = "Qwen/Qwen3-1.7B",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        freeze_encoder: bool = True,
        freeze_llm: bool = True,  # frozen base, LoRA trains
        audio_token_id: int | None = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # --- Encoder ---
        self.encoder = encoder
        if freeze_encoder:
            self.encoder.freeze()

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)

        # Add audio placeholder token if not already present
        if AUDIO_PLACEHOLDER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_PLACEHOLDER]})
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)

        # --- LLM ---
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Resize embeddings to accommodate the new audio token
        self.llm.resize_token_embeddings(len(self.tokenizer))

        llm_hidden = self.llm.config.hidden_size

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # --- LoRA (lightweight) ---
        if lora_rank > 0 and freeze_llm:
            self._apply_lora(lora_rank, lora_alpha, lora_target_modules or ["q_proj", "v_proj"])

        # --- Projector (always trainable) ---
        self.projector = AudioProjector(encoder.get_output_dim(), llm_hidden)

        log.info(f"SpeechLLM initialized:")
        log.info(f"  Encoder dim: {encoder.get_output_dim()}")
        log.info(f"  LLM hidden:  {llm_hidden}")
        log.info(f"  Audio token:  {AUDIO_PLACEHOLDER} → id {self.audio_token_id}")
        log.info(f"  LoRA rank:    {lora_rank}")
        log.info(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _apply_lora(self, rank: int, alpha: int, target_modules: list[str]):
        """Apply LoRA adapters to specified LLM modules."""
        try:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, config)
            log.info(f"  LoRA applied to: {target_modules}")
        except ImportError:
            log.warning("peft not installed — applying manual LoRA")
            self._apply_manual_lora(rank, alpha, target_modules)

    def _apply_manual_lora(self, rank: int, alpha: int, target_modules: list[str]):
        """Fallback: manual LoRA without peft library."""
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

                # Monkey-patch forward
                original_forward = module.forward

                def make_lora_forward(orig, la, lb, s):
                    def lora_forward(x):
                        return orig(x) + lb(la(x.to(la.weight.dtype))) * s
                    return lora_forward

                module.forward = make_lora_forward(original_forward, lora_a, lora_b, scale)

    def _encode_audio(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run audio through encoder + projector."""
        enc_out = self.encoder(features, feature_lengths)
        hidden = enc_out["hidden_states"]   # (B, T', encoder_dim)
        lengths = enc_out["lengths"]        # (B,)
        projected = self.projector(hidden)  # (B, T', llm_hidden)
        return projected, lengths

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, S) with <|audio|> placeholders
        attention_mask: torch.Tensor,     # (B, S)
        labels: torch.Tensor,            # (B, S) with -100 for non-target tokens
        audio_features: torch.Tensor,     # (B, T_mel, 80) mel spectrograms
        audio_lengths: torch.Tensor,      # (B,) mel frame counts
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with masked_scatter audio fusion.

        Returns:
            dict with "loss" and "logits"
        """
        # 1. Encode audio
        audio_embeds, audio_lens = self._encode_audio(audio_features, audio_lengths)
        # audio_embeds: (B, T_enc, llm_hidden)

        # 2. Get text embeddings from the LLM's embedding layer
        # get_input_embeddings() works regardless of peft wrapping
        embed_layer = self.llm.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)  # (B, S, llm_hidden)

        # 3. Build audio mask and scatter
        audio_mask = (input_ids == self.audio_token_id)  # (B, S)
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

        # Flatten all audio embeddings across the batch for scatter
        # We need exactly as many audio vectors as there are placeholder tokens
        all_audio = []
        for i in range(audio_embeds.size(0)):
            all_audio.append(audio_embeds[i, : audio_lens[i]])  # (T_enc_i, D)
        all_audio_cat = torch.cat(all_audio, dim=0)  # (total_audio_tokens, D)

        n_placeholders = audio_mask.sum().item()
        n_audio = all_audio_cat.size(0)
        if n_placeholders != n_audio:
            log.warning(
                f"Placeholder count ({n_placeholders}) != audio token count ({n_audio}). "
                f"Truncating/padding audio to match."
            )
            if n_audio > n_placeholders:
                all_audio_cat = all_audio_cat[:n_placeholders]
            else:
                pad = torch.zeros(
                    n_placeholders - n_audio,
                    all_audio_cat.size(-1),
                    device=all_audio_cat.device,
                    dtype=all_audio_cat.dtype,
                )
                all_audio_cat = torch.cat([all_audio_cat, pad], dim=0)

        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask_3d, all_audio_cat.to(inputs_embeds.dtype)
        )

        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,     # (1, T_mel, 80)
        audio_lengths: torch.Tensor,      # (1,)
        prompt: str = "Transcribe the following speech.",
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate text from audio input."""
        # Encode audio
        audio_embeds, audio_lens = self._encode_audio(audio_features, audio_lengths)
        n_audio_tokens = audio_lens[0].item()

        # Build input using Qwen3 chat format — include empty think block
        audio_placeholders = AUDIO_PLACEHOLDER * n_audio_tokens
        full_prompt = (
            f"<|im_start|>system\n{prompt}<|im_end|>\n"
            f"<|im_start|>user\n{audio_placeholders}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>"
        )

        encoded = self.tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(audio_embeds.device)
        attention_mask = encoded["attention_mask"].to(audio_embeds.device)

        # Embed and scatter
        embed_layer = self.llm.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)
        audio_mask = (input_ids == self.audio_token_id)
        audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

        audio_flat = audio_embeds[0, :n_audio_tokens].to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask_3d, audio_flat)

        # Get EOS token ids (im_end + eos)
        eos_ids = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.eos_token_id,
        ]
        eos_ids = [x for x in eos_ids if x is not None]

        # Generate — use sampling as recommended by Qwen3
        gen_out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            eos_token_id=eos_ids,
            **generate_kwargs,
        )

        # When using inputs_embeds (not input_ids), generate() returns ONLY
        # the newly generated tokens, not the input. No slicing needed.
        generated_ids = gen_out[0]
        
        # Debug: log raw token ids
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        log.debug(f"generate() produced {len(generated_ids)} tokens: {generated_ids[:30].tolist()}")
        log.debug(f"generate() raw text: {raw_text[:200]}")
        
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip any thinking block if present
        if "<think>" in text:
            # Take everything after </think>
            parts = text.split("</think>")
            text = parts[-1] if len(parts) > 1 else text

        return text.strip()


# ============================================================================
# 4. DATASET
# ============================================================================

class SpeechTranslationDataset(Dataset):
    """
    Reads a CSV index file with columns:
        audio_id, path, transcript, language, split, source, speaker_id, sample_rate, duration
        (and optionally: translation, src_language, tgt_language for AST index)

    Uses the `split` column to filter rows (e.g. "train", "dev", "test").
    Handles missing sample_rate/duration gracefully.

    Args:
        index_path: Path to the CSV index file.
        split: Which split to use (matches the `split` column).
        languages: If provided, only include rows with these language codes.
    """

    def __init__(
        self,
        index_path: str,
        tokenizer,
        audio_token_id: int,
        split: str = "train",
        languages: list[str] | None = None,
        max_audio_sec: float = 30.0,
        sample_rate: int = 16000,
        task: str = "translate",  # "translate" or "transcribe"
    ):
        import csv

        self.entries = []
        with open(index_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split
                if row.get("split", "") != split:
                    continue
                # Filter by language if specified
                lang = row.get("language", row.get("src_language", ""))
                if languages and lang not in languages:
                    continue
                # Filter out entries longer than max_audio_sec if duration is available
                dur = row.get("duration", "")
                if dur and dur.replace(".", "", 1).isdigit():
                    if float(dur) > max_audio_sec:
                        continue
                self.entries.append(row)

        self.tokenizer = tokenizer
        self.audio_token_id = audio_token_id
        self.max_frames = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate
        self.task = task

        # Mel transform (at target sample rate)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        )

        lang_counts = {}
        for e in self.entries:
            lang = e.get("language", e.get("src_language", "?"))
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Extract durations for batch sampler (fall back to max if missing)
        self.durations = []
        for e in self.entries:
            dur = e.get("duration", "")
            if dur and dur.replace(".", "", 1).isdigit():
                self.durations.append(float(dur))
            else:
                self.durations.append(max_audio_sec)  # conservative fallback

        log.info(f"Loaded {len(self.entries)} examples from {index_path} [split={split}]")
        log.info(f"  Language distribution: {lang_counts}")
        log.info(f"  Duration stats: mean={sum(self.durations)/len(self.durations):.1f}s, "
                 f"max={max(self.durations):.1f}s")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Audio path: try "path" first (your ASR index), then "audio_path" (AST index)
        audio_path = entry.get("path", entry.get("audio_path", ""))

        # Load audio via soundfile (avoids torchcodec/libnvrtc dependency)
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # mono
        waveform = torch.from_numpy(data)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > self.max_frames:
            waveform = waveform[: self.max_frames]

        # Mel spectrogram → log mel
        mel = self.mel_transform(waveform)  # (80, T)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = (mel + 4.0) / 4.0  # rough normalization
        mel = mel.T  # (T, 80)

        # Target text
        if self.task == "translate":
            target = entry.get("translation", entry.get("transcript", ""))
        else:
            target = entry.get("transcript", "")

        src_lang = entry.get("language", entry.get("src_language", ""))

        return {
            "mel": mel,
            "mel_length": mel.size(0),
            "target": target,
            "src_language": src_lang,
        }


class DurationBatchSampler(torch.utils.data.Sampler):
    """Kept as alias — use DurationBucketSampler from st.utils.samplers instead."""
    pass  # unused, kept for reference


def collate_fn(batch, tokenizer, audio_token_id, task="translate", max_seq_len=1024):
    """
    Collate function that:
    1. Pads mel features
    2. Builds input_ids with audio placeholders
    3. Builds labels with -100 masking on prompt + audio tokens
    """
    # Pad mel features
    mel_lengths = torch.tensor([b["mel_length"] for b in batch])
    max_mel = mel_lengths.max().item()
    mel_batch = torch.zeros(len(batch), max_mel, 80)
    for i, b in enumerate(batch):
        mel_batch[i, : b["mel_length"]] = b["mel"]

    # Compute encoder output lengths (4x subsampling)
    enc_lengths = ((mel_lengths - 1) // 2 + 1)
    enc_lengths = ((enc_lengths - 1) // 2 + 1)

    # Build text sequences
    all_input_ids = []
    all_labels = []
    for i, b in enumerate(batch):
        n_audio = enc_lengths[i].item()

        # Audio placeholder tokens
        audio_str = AUDIO_PLACEHOLDER * n_audio

        # Build using Qwen3 chat template format
        # System message with task instruction — include language for disambiguation
        if task == "translate":
            system_msg = f"Translate the following {b['src_language']} speech to English."
        else:
            system_msg = f"Transcribe the following {b['src_language']} speech."

        target = b["target"]

        # Format with empty thinking block — teaches the model to skip thinking
        # This matches Qwen3's non-thinking output: <think>\n\n</think>content
        full_text = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{audio_str}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>{target}<|im_end|>"
        )

        encoded = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"].squeeze(0)

        # Labels: -100 for everything before the target (including the empty think block)
        prompt_part = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{audio_str}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>"
        )
        prompt_encoded = tokenizer(prompt_part, add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].size(1)

        labels = ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(ids)
        all_labels.append(labels)

    # Skip any sequences that exceed max_seq_len (avoids corrupted training signal)
    filtered_ids = []
    filtered_labels = []
    filtered_mels = []
    filtered_mel_lengths = []
    for i in range(len(all_input_ids)):
        if all_input_ids[i].size(0) > max_seq_len:
            log.debug(f"Skipping sample with {all_input_ids[i].size(0)} tokens (max_seq_len={max_seq_len})")
            continue
        filtered_ids.append(all_input_ids[i])
        filtered_labels.append(all_labels[i])
        filtered_mels.append(mel_batch[i])
        filtered_mel_lengths.append(mel_lengths[i])

    if not filtered_ids:
        # Entire batch exceeded max_seq_len — return a dummy batch that the training loop can skip
        return None

    all_input_ids = filtered_ids
    all_labels = filtered_labels
    mel_batch = torch.stack(filtered_mels)
    mel_lengths = torch.stack(filtered_mel_lengths)

    # Pad sequences
    max_seq = max(ids.size(0) for ids in all_input_ids)
    input_ids_padded = torch.full((len(batch), max_seq), tokenizer.pad_token_id or 0, dtype=torch.long)
    labels_padded = torch.full((len(batch), max_seq), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_seq, dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(all_input_ids, all_labels)):
        input_ids_padded[i, : ids.size(0)] = ids
        labels_padded[i, : lab.size(0)] = lab
        attention_mask[i, : ids.size(0)] = 1

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "audio_features": mel_batch,
        "audio_lengths": mel_lengths,
    }


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def load_encoder(config_path: str, checkpoint_path: str | None = None) -> SpeechEncoder:
    """Load encoder from config + optional pretrained checkpoint."""
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
        # Handle different checkpoint formats
        if "encoder" in state:
            state = state["encoder"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        # Strip "encoder." prefix if present
        state = {k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v for k, v in state.items()}
        # Remove CTC head weights if present (we don't need them)
        state = {k: v for k, v in state.items() if not k.startswith("ctc_head")}
        encoder.load_state_dict(state, strict=False)
        log.info(f"Loaded encoder from {checkpoint_path}")

    return encoder


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    encoder = load_encoder(args.encoder_config, args.encoder_ckpt)

    # Build model
    model = SpeechLLM(
        encoder=encoder,
        llm_name=args.llm_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        freeze_encoder=True,
        freeze_llm=True,
    )
    model = model.to(device)

    # Parse languages filter
    languages = args.languages.split(",") if args.languages else None

    # Dataset — single CSV, split by the `split` column
    train_ds = SpeechTranslationDataset(
        args.index,
        model.tokenizer,
        model.audio_token_id,
        split=args.train_split,
        languages=languages,
        max_audio_sec=args.max_audio_sec,
        task=args.task,
    )
    val_ds = None
    if args.val_split:
        val_ds = SpeechTranslationDataset(
            args.index,
            model.tokenizer,
            model.audio_token_id,
            split=args.val_split,
            languages=languages,
            max_audio_sec=args.max_audio_sec,
            task=args.task,
        )

    def collate_wrapper(batch):
        return collate_fn(batch, model.tokenizer, model.audio_token_id, args.task, max_seq_len=args.max_seq_len)

    # Duration-based batch sampler: caps total audio seconds per batch
    # instead of fixed instance count, preventing OOM from long-utterance batches
    from st.utils.samplers import DurationBucketSampler

    # Patch the dataset to expose .samples for the sampler's _get_durations,
    # but override _get_durations to use our pre-extracted CSV durations instead
    # of calling torchaudio.info on 2M files
    class FastDurationBucketSampler(DurationBucketSampler):
        """DurationBucketSampler that uses pre-extracted durations from CSV
        and caps max instances per batch."""
        def __init__(self, *args, max_batch_size=64, **kwargs):
            self.max_batch_size = max_batch_size
            super().__init__(*args, **kwargs)

        def _get_durations(self):
            return self.dataset.durations

        def _create_all_batches(self):
            """Override to also enforce max_batch_size."""
            all_batches = []
            for bucket in self.buckets:
                batch = []
                batch_duration = 0.0
                for idx in bucket:
                    duration = self.durations[idx]
                    if batch and (batch_duration + duration > self.target_duration
                                  or len(batch) >= self.max_batch_size):
                        all_batches.append(batch)
                        batch = []
                        batch_duration = 0.0
                    batch.append(idx)
                    batch_duration += duration
                if batch:
                    all_batches.append(batch)
            return all_batches

    train_sampler = FastDurationBucketSampler(
        dataset=train_ds,
        target_duration=args.max_batch_duration,
        max_batch_size=args.max_batch_size,
        shuffle=True,
        shuffle_buckets=True,
    )
    log.info(f"  Duration sampler: {len(train_sampler)} batches, "
             f"target_duration={args.max_batch_duration}s")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = FastDurationBucketSampler(
            dataset=val_ds,
            target_duration=args.max_batch_duration,
            shuffle=False,
            shuffle_buckets=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=True,
        )

    # Optimizer — only trainable params (projector + LoRA)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # LR schedule: cosine annealing with warmup (from st.utils.schedulers)
    from st.utils.schedulers import build_scheduler
    scheduler = build_scheduler(
        name="cosine_warmup_restarts",
        optimizer=optimizer,
        total_steps=args.max_steps,
        max_lr=args.lr,
        min_lr=args.min_lr,
        first_cycle_steps=args.first_cycle_steps or args.max_steps,
        cycle_mult=args.cycle_mult,
        gamma=args.gamma,
        warmup_steps=args.warmup_steps,
    )

    # Mixed precision: bf16 autocast is used in the forward pass.
    # GradScaler is NOT needed for bf16 (same exponent range as fp32).

    # W&B logging
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                config={
                    "encoder_config": args.encoder_config,
                    "encoder_ckpt": args.encoder_ckpt,
                    "llm_name": args.llm_name,
                    "task": args.task,
                    "languages": args.languages,
                    "max_batch_duration": args.max_batch_duration,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": args.max_steps,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "max_audio_sec": args.max_audio_sec,
                    "train_examples": len(train_ds),
                    "val_examples": len(val_ds) if val_ds else 0,
                    "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
            )
            log.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            use_wandb = False

    # Training
    model.train()
    global_step = 0
    epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Starting training for {args.max_steps} steps")
    log.info(f"  Max batch duration: {args.max_batch_duration}s")
    log.info(f"  Grad accum: {args.grad_accum}")

    # Progress bar
    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc="Training", unit="step", dynamic_ncols=True)
    running_loss = 0.0
    loss_count = 0

    while global_step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if batch is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            cur_batch_size = batch["audio_features"].size(0)
            cur_batch_duration = batch["audio_lengths"].sum().item() * 0.01  # mel frames × 10ms

            # Skip excessively long sequences that would OOM on logits
            seq_len = batch["input_ids"].size(1)
            mem_estimate_gb = cur_batch_size * seq_len * 151670 * 4 / 1e9  # float32 logits
            if mem_estimate_gb > 40:  # conservative threshold for 80GB GPU
                log.warning(
                    f"Skipping batch: bs={cur_batch_size}, seq_len={seq_len}, "
                    f"estimated logits={mem_estimate_gb:.1f}GB"
                )
                continue

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        audio_features=batch["audio_features"],
                        audio_lengths=batch["audio_lengths"],
                    )
                    loss = outputs["loss"] / args.grad_accum

                loss.backward()

            except torch.cuda.OutOfMemoryError:
                log.warning(
                    f"OOM at step {global_step}: bs={cur_batch_size}, seq_len={seq_len}, "
                    f"dur={cur_batch_duration:.0f}s — skipping batch"
                )
                torch.cuda.empty_cache()
                optimizer.zero_grad()
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
            pbar.set_postfix(
                loss=f"{step_loss:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                bs=cur_batch_size, dur=f"{cur_batch_duration:.0f}s", epoch=epoch,
            )

            if global_step % args.log_every == 0:
                avg_loss = running_loss / loss_count
                current_lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"step {global_step}/{args.max_steps} | "
                    f"loss {step_loss:.4f} | avg_loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | bs {cur_batch_size} | dur {cur_batch_duration:.0f}s"
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/avg_loss": avg_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/batch_size": cur_batch_size,
                        "train/batch_duration_sec": cur_batch_duration,
                    }, step=global_step)
                running_loss = 0.0
                loss_count = 0

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir)

            if val_loader is not None and global_step % args.eval_every == 0:
                torch.cuda.empty_cache()
                val_metrics = evaluate(model, val_loader, device, task=args.task)
                metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                log.info(f"step {global_step} | {metrics_str}")
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                model.train()

            if global_step >= args.max_steps:
                break

    pbar.close()
    # Final eval
    if val_loader is not None:
        val_metrics = evaluate(model, val_loader, device, task=args.task)
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        log.info(f"Final eval | {metrics_str}")
        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
    save_checkpoint(model, optimizer, global_step, args.output_dir)
    if use_wandb:
        wandb.finish()
    log.info("Training complete!")


@torch.no_grad()
def evaluate(
    model: SpeechLLM,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    generate_samples: int = 20,
    task: str = "transcribe",
) -> dict[str, float]:
    """Run validation: compute loss + generation-based metrics (WER/BLEU/chrF).

    Args:
        max_batches: Max batches for loss computation.
        generate_samples: Number of individual samples to generate for metrics.
        task: "transcribe" (computes WER) or "translate" (computes BLEU/chrF).

    Returns:
        dict with "loss" and metric scores.
    """
    from st.utils.metrics import compute_wer, compute_bleu, compute_chrf

    model.eval()

    # --- Loss ---
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                audio_features=batch["audio_features"],
                audio_lengths=batch["audio_lengths"],
            )
        total_loss += outputs["loss"].item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    results = {"loss": total_loss / max(n_batches, 1)}

    # Free memory before generation
    torch.cuda.empty_cache()

    # --- Generation-based metrics on a small sample ---
    if generate_samples > 0:
        predictions = []
        references = []
        val_ds = val_loader.dataset
        indices = list(range(min(generate_samples, len(val_ds))))

        for idx in indices:
            sample = val_ds[idx]
            mel = sample["mel"].unsqueeze(0).to(device)  # (1, T, 80)
            mel_len = torch.tensor([sample["mel_length"]], device=device)
            ref = sample["target"]

            if task == "translate":
                prompt = f"Translate the following {sample['src_language']} speech to English."
            else:
                prompt = f"Transcribe the following {sample['src_language']} speech."

            try:
                pred = model.generate(
                    audio_features=mel,
                    audio_lengths=mel_len,
                    prompt=prompt,
                    max_new_tokens=128,
                )
                predictions.append(pred.strip())
                references.append(ref.strip())
            except Exception as e:
                log.warning(f"Generation failed for sample {idx}: {e}")
                continue

        if predictions:
            if task == "transcribe":
                results["wer"] = compute_wer(predictions, references)
            else:
                bleu = compute_bleu(predictions, references)
                chrf = compute_chrf(predictions, references)
                results["bleu"] = bleu["bleu"]
                results["chrf"] = chrf["chrf"]

            # Log a few examples
            for i in range(min(3, len(predictions))):
                log.info(f"  [sample {i}] ref: {references[i][:80]}")
                log.info(f"  [sample {i}] hyp: {predictions[i][:80]}")

    return results


def save_checkpoint(model: SpeechLLM, optimizer, step: int, output_dir: str):
    """Save projector weights + LoRA weights (not the full LLM)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save projector
    torch.save(model.projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))

    # Save LoRA weights if using peft
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(os.path.join(ckpt_dir, "lora_adapter"))
    elif hasattr(model, "_lora_layers"):
        torch.save(model._lora_layers.state_dict(), os.path.join(ckpt_dir, "lora_manual.pt"))

    # Save config
    config = {
        "encoder_dim": model.encoder.get_output_dim(),
        "audio_token_id": model.audio_token_id,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"Saved checkpoint to {ckpt_dir}")


# ============================================================================
# 6. INFERENCE
# ============================================================================

def infer(args):
    """Run inference on a single audio file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = args.checkpoint
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)

    # Load encoder
    encoder = load_encoder(args.encoder_config, args.encoder_ckpt)

    # Build model (no LoRA during init — we'll load the adapter)
    model = SpeechLLM(
        encoder=encoder,
        llm_name=args.llm_name,
        lora_rank=0,  # will load separately
        freeze_encoder=True,
        freeze_llm=True,
    )

    # Load projector
    model.projector.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "projector.pt"), map_location="cpu", weights_only=True)
    )

    # Load LoRA adapter if present
    lora_dir = os.path.join(ckpt_dir, "lora_adapter")
    if os.path.exists(lora_dir):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
        log.info("Loaded LoRA adapter")
    elif os.path.exists(os.path.join(ckpt_dir, "lora_manual.pt")):
        log.warning("Manual LoRA loading not implemented for inference — using base LLM only")

    model = model.to(device)
    model.eval()

    # Load audio
    data, sr = sf.read(args.audio_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    )
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = (mel + 4.0) / 4.0
    mel = mel.T.unsqueeze(0).to(device)  # (1, T, 80)
    mel_len = torch.tensor([mel.size(1)], device=device)

    # Generate
    lang = args.language
    if args.task == "translate":
        prompt = f"Translate the following {lang} speech to English."
    else:
        prompt = f"Transcribe the following {lang} speech."

    output = model.generate(
        audio_features=mel,
        audio_lengths=mel_len,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{'='*60}")
    print(f"Audio: {args.audio_path}")
    print(f"Task:  {args.task}")
    print(f"{'='*60}")
    print(f"Output: {output}")
    print(f"{'='*60}\n")


def debug(args):
    """Debug the generation pipeline — test with untrained model to verify tokens flow correctly."""
    import logging as _logging
    _logging.getLogger(__name__).setLevel(_logging.DEBUG)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(args.encoder_config, args.encoder_ckpt)
    model = SpeechLLM(
        encoder=encoder,
        llm_name=args.llm_name,
        lora_rank=args.lora_rank,
        freeze_encoder=True,
        freeze_llm=True,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt_dir = args.checkpoint
        projector_path = os.path.join(ckpt_dir, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(
                torch.load(projector_path, map_location="cpu", weights_only=True)
            )
            log.info(f"Loaded projector from {projector_path}")
        lora_dir = os.path.join(ckpt_dir, "lora_adapter")
        if os.path.exists(lora_dir):
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
            log.info(f"Loaded LoRA adapter from {lora_dir}")

    model = model.to(device)
    model.eval()

    # Load audio
    data, sr = sf.read(args.audio_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    )
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = (mel + 4.0) / 4.0
    mel = mel.T.unsqueeze(0).to(device)
    mel_len = torch.tensor([mel.size(1)], device=device)

    # Test 1: Check encoder output
    audio_embeds, audio_lens = model._encode_audio(mel, mel_len)
    print(f"\n{'='*60}")
    print(f"Audio: {args.audio_path}")
    print(f"Mel frames: {mel.size(1)}, Encoder tokens: {audio_lens[0].item()}")
    print(f"Audio embed shape: {audio_embeds.shape}, dtype: {audio_embeds.dtype}")
    print(f"Audio embed stats: mean={audio_embeds[0].mean():.4f}, std={audio_embeds[0].std():.4f}")

    # Test 2: Check prompt tokenization
    n_audio = audio_lens[0].item()
    audio_placeholders = AUDIO_PLACEHOLDER * n_audio
    prompt_text = (
        f"<|im_start|>system\nTranscribe the following speech.<|im_end|>\n"
        f"<|im_start|>user\n{audio_placeholders}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    encoded = model.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    audio_mask = (input_ids == model.audio_token_id)
    print(f"\nPrompt tokens: {input_ids.size(1)}")
    print(f"Audio placeholders in prompt: {audio_mask.sum().item()}")
    print(f"Expected audio tokens: {n_audio}")
    print(f"Match: {audio_mask.sum().item() == n_audio}")
    
    # Show first/last few tokens
    ids = input_ids[0].tolist()
    print(f"\nFirst 20 token ids: {ids[:20]}")
    print(f"First 20 decoded: {model.tokenizer.decode(ids[:20])}")
    print(f"Last 10 token ids: {ids[-10:]}")
    print(f"Last 10 decoded: {model.tokenizer.decode(ids[-10:])}")

    # Test 3: Try generation
    print(f"\n--- Generation test ---")
    output = model.generate(
        audio_features=mel,
        audio_lengths=mel_len,
        prompt="Transcribe the following speech.",
        max_new_tokens=64,
    )
    print(f"Generated: '{output}'")
    
    # Test 4: Try teacher-forced forward pass to check loss
    print(f"\n--- Forward pass test ---")
    target = "This is a test transcription."
    full_text = (
        f"<|im_start|>system\nTranscribe the following speech.<|im_end|>\n"
        f"<|im_start|>user\n{audio_placeholders}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>{target}<|im_end|>"
    )
    enc = model.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_enc = model.tokenizer(
        f"<|im_start|>system\nTranscribe the following speech.<|im_end|>\n"
        f"<|im_start|>user\n{audio_placeholders}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>",
        add_special_tokens=False, return_tensors="pt"
    ).to(device)
    labels = enc["input_ids"].clone()
    labels[0, :prompt_enc["input_ids"].size(1)] = -100
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            audio_features=mel,
            audio_lengths=mel_len,
        )
    print(f"Forward loss: {out['loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")
    
    # Check what the model predicts at the first target position
    first_target_pos = prompt_enc["input_ids"].size(1)
    top_tokens = out["logits"][0, first_target_pos - 1].topk(10)
    print(f"\nTop 10 predicted tokens at first target position:")
    for score, tid in zip(top_tokens.values, top_tokens.indices):
        print(f"  {model.tokenizer.decode([tid.item()])!r} (id={tid.item()}, score={score.item():.2f})")
    
    print(f"{'='*60}\n")


# ============================================================================
# 7. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Speech-LLM: Conformer + Qwen3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    train_p = subparsers.add_parser("train", help="Train the speech-LLM")
    train_p.add_argument("--encoder_config", required=True, help="Encoder YAML config")
    train_p.add_argument("--encoder_ckpt", default=None, help="Pretrained encoder checkpoint")
    train_p.add_argument("--llm_name", default="Qwen/Qwen3-1.7B", help="HuggingFace LLM name")
    train_p.add_argument("--index", required=True, help="CSV index file (with split column)")
    train_p.add_argument("--train_split", default="train", help="Value of split column for training")
    train_p.add_argument("--val_split", default=None, help="Value of split column for validation (e.g. 'dev')")
    train_p.add_argument("--languages", default=None, help="Comma-separated language codes to include (e.g. 'igbo,hausa,yoruba')")
    train_p.add_argument("--max_audio_sec", type=float, default=30.0, help="Max audio duration in seconds")
    train_p.add_argument("--output_dir", default="runs/speech_llm", help="Output directory")
    train_p.add_argument("--task", default="translate", choices=["translate", "transcribe"])
    train_p.add_argument("--max_batch_duration", type=float, default=120.0, help="Max total audio seconds per batch")
    train_p.add_argument("--max_batch_size", type=int, default=64, help="Max instances per batch (caps short-utterance batches)")
    train_p.add_argument("--max_seq_len", type=int, default=1024, help="Max token sequence length (skips outliers that cause OOM)")
    train_p.add_argument("--batch_size", type=int, default=None, help="(Deprecated) Ignored when using duration batching")
    train_p.add_argument("--grad_accum", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    train_p.add_argument("--min_lr", type=float, default=1e-6, help="Min LR for cosine schedule")
    train_p.add_argument("--warmup_steps", type=int, default=500)
    train_p.add_argument("--first_cycle_steps", type=int, default=None, help="Steps in first cycle (default: max_steps)")
    train_p.add_argument("--cycle_mult", type=float, default=1.0, help="Cycle length multiplier after each restart")
    train_p.add_argument("--gamma", type=float, default=1.0, help="Max LR decay factor after each cycle")
    train_p.add_argument("--max_steps", type=int, default=20000)
    train_p.add_argument("--lora_rank", type=int, default=16)
    train_p.add_argument("--lora_alpha", type=int, default=32)
    train_p.add_argument("--log_every", type=int, default=50)
    train_p.add_argument("--save_every", type=int, default=2000)
    train_p.add_argument("--eval_every", type=int, default=500, help="Run validation every N steps")
    train_p.add_argument("--num_workers", type=int, default=4)
    train_p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    train_p.add_argument("--wandb_project", default="speech-llm", help="W&B project name")
    train_p.add_argument("--wandb_run_name", default=None, help="W&B run name (default: output_dir basename)")

    # --- Infer ---
    infer_p = subparsers.add_parser("infer", help="Run inference")
    infer_p.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    infer_p.add_argument("--encoder_config", required=True, help="Encoder YAML config")
    infer_p.add_argument("--encoder_ckpt", default=None, help="Pretrained encoder checkpoint")
    infer_p.add_argument("--llm_name", default="Qwen/Qwen3-1.7B")
    infer_p.add_argument("--audio_path", required=True, help="Audio file path")
    infer_p.add_argument("--language", required=True, help="Source language (e.g. 'igbo', 'bemba')")
    infer_p.add_argument("--task", default="translate", choices=["translate", "transcribe"])
    infer_p.add_argument("--max_new_tokens", type=int, default=256)

    # --- Debug ---
    debug_p = subparsers.add_parser("debug", help="Debug generation pipeline")
    debug_p.add_argument("--encoder_config", required=True)
    debug_p.add_argument("--encoder_ckpt", default=None)
    debug_p.add_argument("--llm_name", default="Qwen/Qwen3-0.6B")
    debug_p.add_argument("--audio_path", required=True, help="Any audio file to test with")
    debug_p.add_argument("--lora_rank", type=int, default=0, help="Set >0 to test with LoRA")
    debug_p.add_argument("--checkpoint", default=None, help="Checkpoint dir to load projector + LoRA from")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "debug":
        debug(args)


if __name__ == "__main__":
    main()