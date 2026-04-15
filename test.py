# Add temporarily at the end of speech_aura.py and run it
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


# Load model with checkpoint
encoder = load_encoder("configs/encoder/conformer_base.yaml", "checkpoints/encoder_base/encoder_final.pt")
model = SpeechAura(
    encoder=encoder,
    aura_ckpt="Aura-1B/model.pt",
    aura_tokenizer="Aura-1B/tokenizer.json",
    lora_rank=8,
    lora_alpha=32,
    projector_type="transformer",
    projector_layers=2,
    projector_heads=8
)
model.projector.load_state_dict(
    torch.load("runs/speech_aura_transformer_proj_zero_init/checkpoint_step4000/projector.pt", map_location="cpu")
)
# model._lora_layers.load_state_dict(
#     torch.load("runs/speech_aura/checkpoint_step10000/lora.pt", map_location="cpu")
# )
model = model.to("cuda").eval()

device="cuda"

# ============================================================================
# AUDIO ABLATION TEST
# ============================================================================

# Load a real audio sample from the val set
val_entries = []
with open("/ocean/projects/cis250145p/shared/datasets/ASR_INDEX.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("split") == "dev" and row.get("language") in ("igbo", "yoruba", "hausa", "bemba"):
            val_entries.append(row)
            if len(val_entries) >= 5:
                break

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80,
)

for entry in val_entries:
    path = entry.get("path", entry.get("audio_path", ""))
    ref = entry.get("transcript", "")
    lang = entry.get("language", "")

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = (mel + 4.0) / 4.0
    mel = mel.T.unsqueeze(0).to(device)
    mel_len = torch.tensor([mel.size(1)], device=device)

    # Encode audio
    with torch.no_grad():
        audio_embeds, audio_lens = model._encode_audio(mel, mel_len)
    n_audio = audio_lens[0].item()

    # Build input_ids with teacher-forced target (same as training)
    lang_id = LANG_MAP.get(lang, 10)
    target_ids = model.tokenizer.encode(ref, add_special_tokens=False)
    ids = [model.bos_id, lang_id] + [model.audio_token_id] * n_audio + [TRANSCRIPT_START_ID] + target_ids + [model.eos_id]
    input_ids = torch.tensor([ids], device=device)

    prompt_len = 2 + n_audio + 1
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    embed_layer = model._get_embed_layer()
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    def run_forward(embeds):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = embeds
            for layer in model.llm.model.layers:
                h = layer(h, position_ids=position_ids)
            h = model.llm.model.norm(h)
            logits = model.llm.lm_head(h).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss.item()

    # Condition 1: Normal (real audio scattered in)
    inputs_embeds = embed_layer(input_ids)
    audio_mask = (input_ids == model.audio_token_id)
    audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
    audio_flat = audio_embeds[0, :n_audio].to(inputs_embeds.dtype)
    normal_embeds = inputs_embeds.masked_scatter(audio_mask_3d, audio_flat)
    loss_normal = run_forward(normal_embeds)

    # Condition 2: Zeroed audio
    zero_embeds = embed_layer(input_ids).clone()
    zero_embeds = zero_embeds.masked_fill(audio_mask_3d, 0.0)
    loss_zero = run_forward(zero_embeds)

    # Condition 3: Random noise at text embedding scale
    rand_embeds = embed_layer(input_ids).clone()
    noise = torch.randn(n_audio, model.llm_hidden, device=device) * 0.5
    rand_embeds = rand_embeds.masked_scatter(audio_mask_3d, noise.to(rand_embeds.dtype))
    loss_rand = run_forward(rand_embeds)

    print(f"\n[{lang}] ref: {ref[:80]}")
    print(f"  Normal audio loss: {loss_normal:.4f}")
    print(f"  Zeroed audio loss: {loss_zero:.4f}")
    print(f"  Random audio loss: {loss_rand:.4f}")

print("\n" + "="*60)
print("If all 3 losses are similar, the model is ignoring audio")
print("and just copying from teacher-forced text tokens.")
print("="*60)


# ============================================================================
# TEACHER-FORCED GREEDY vs KV-CACHE GENERATE
# ============================================================================

entry = val_entries[0]
path = entry.get("path", entry.get("audio_path", ""))
ref = entry.get("transcript", "")
lang = entry.get("language", "")

data, sr = sf.read(path, dtype="float32")
if data.ndim > 1:
    data = data[:, 0]
waveform = torch.from_numpy(data)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
mel = mel_transform(waveform)
mel = torch.clamp(mel, min=1e-10).log10()
mel = (mel + 4.0) / 4.0
mel = mel.T.unsqueeze(0).to(device)
mel_len = torch.tensor([mel.size(1)], device=device)

# Method 1: Full forward pass, greedy decode from logits (training path)
with torch.no_grad():
    audio_embeds, audio_lens = model._encode_audio(mel, mel_len)
n_audio = audio_lens[0].item()
lang_id = LANG_MAP.get(lang, 10)
target_ids = model.tokenizer.encode(ref, add_special_tokens=False)
ids = [model.bos_id, lang_id] + [model.audio_token_id] * n_audio + [TRANSCRIPT_START_ID] + target_ids + [model.eos_id]
input_ids = torch.tensor([ids], device=device)

embed_layer = model._get_embed_layer()
inputs_embeds = embed_layer(input_ids)
audio_mask = (input_ids == model.audio_token_id)
audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
audio_flat = audio_embeds[0, :n_audio].to(inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(audio_mask_3d, audio_flat)

batch_size, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    h = inputs_embeds
    for layer in model.llm.model.layers:
        h = layer(h, position_ids=position_ids)
    h = model.llm.model.norm(h)
    logits = model.llm.lm_head(h).float()

# Extract greedy predictions starting after TRANSCRIPT_START
prompt_len = 2 + n_audio + 1
pred_ids = logits[0, prompt_len-1:-1].argmax(dim=-1)  # predict next token at each position
pred_text = model.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)

# Method 2: Autoregressive generate (inference path)
with torch.no_grad():
    gen_text = model.generate(mel, mel_len, target_lang=lang, max_new_tokens=64)

print(f"\nref:              {ref[:100]}")
print(f"teacher greedy:   {pred_text[:100]}")
print(f"AR generate:      {gen_text[:100]}")
print(f"\nIf teacher greedy is good but AR generate is garbage,")
print(f"the bug is in the generate() KV cache / decode loop.")


# Method 3: Debug AR generate step by step
from kvcache import KVcache

with torch.no_grad():
    audio_embeds, audio_lens = model._encode_audio(mel, mel_len)
n_audio = audio_lens[0].item()
lang_id = LANG_MAP.get(lang, 10)
prefix_ids = [model.bos_id, lang_id] + [model.audio_token_id] * n_audio + [TRANSCRIPT_START_ID]
input_ids = torch.tensor([prefix_ids], device=device)

embed_layer = model._get_embed_layer()
inputs_embeds = embed_layer(input_ids)
audio_mask = (input_ids == model.audio_token_id)
audio_mask_3d = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
audio_flat = audio_embeds[0, :n_audio].to(inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(audio_mask_3d, audio_flat)

batch_size, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
cache = KVcache(model.llm.config.n_layers)

# Prefill
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    h = inputs_embeds
    for layer in model.llm.model.layers:
        h = layer(h, position_ids=position_ids, use_cache=True, cache=cache)
    h = model.llm.model.norm(h)
    logits_kv = model.llm.lm_head(h).float()

# Also run WITHOUT KV cache for comparison (same input, no cache)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    h2 = inputs_embeds
    for layer in model.llm.model.layers:
        h2 = layer(h2, position_ids=position_ids)
    h2 = model.llm.model.norm(h2)
    logits_no_kv = model.llm.lm_head(h2).float()

# Compare last position predictions
top5_kv = logits_kv[0, -1].topk(5)
top5_no_kv = logits_no_kv[0, -1].topk(5)

print(f"\nPrefill last position (should predict first transcript token):")
print(f"  WITH KV cache:    {[(model.tokenizer.decode([t.item()]), f'{p.item():.2f}') for t, p in zip(top5_kv.indices, top5_kv.values)]}")
print(f"  WITHOUT KV cache: {[(model.tokenizer.decode([t.item()]), f'{p.item():.2f}') for t, p in zip(top5_no_kv.indices, top5_no_kv.values)]}")

# Check: do the logits match?
diff = (logits_kv[0, -1] - logits_no_kv[0, -1]).abs().max().item()
print(f"  Max logit difference: {diff:.6f}")

# Check what token the model predicts first
first_token_kv = top5_kv.indices[0].item()
first_token_no_kv = top5_no_kv.indices[0].item()
print(f"\n  KV first token: {first_token_kv} = {model.tokenizer.decode([first_token_kv])!r} (eos={model.eos_id})")
print(f"  No-KV first token: {first_token_no_kv} = {model.tokenizer.decode([first_token_no_kv])!r}")


dummy_audio = torch.randn(1, 200, 80, device=device)
dummy_lens = torch.tensor([200], device=device)
with torch.no_grad():
    proj_out, _ = model._encode_audio(dummy_audio, dummy_lens)
print(f"Projector norms: mean={proj_out.norm(dim=-1).mean().item():.4f}")