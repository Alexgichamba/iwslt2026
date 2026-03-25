"""
Inference: load a trained SpeechLLM and translate audio files.

Usage:
    python -m st.inference.generate \
        --config configs/experiment/train_st.yaml \
        --encoder_ckpt checkpoints/encoder/encoder_final.pt \
        --projector_ckpt checkpoints/st/st_epoch10.pt \
        --lora_path checkpoints/st/lora_epoch10 \
        --audio_paths audio1.wav audio2.wav
"""

from __future__ import annotations

import argparse
import logging

import torch
import torchaudio

from st.models.speech_encoder import SpeechEncoder
from st.models.llm_wrapper import LLMWrapper
from st.models.speech_llm import SpeechLLM
from st.utils.audio import build_feature_extractor
from st.utils.config import load_config

logger = logging.getLogger(__name__)


def load_model(
    config: dict,
    encoder_ckpt: str,
    projector_ckpt: str,
    lora_path: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> SpeechLLM:
    """Load the full model from saved checkpoints."""
    encoder_cfg = config["encoder"]
    projector_cfg = config["projector"]
    llm_cfg = config["llm"]

    # Encoder
    enc_state = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    saved_cfg = enc_state.get("config", {})

    encoder = SpeechEncoder(
        input_dim=saved_cfg.get("input_dim", encoder_cfg.get("input_dim", 80)),
        encoder_dim=saved_cfg.get("encoder_dim", encoder_cfg.get("encoder_dim", 512)),
        num_heads=saved_cfg.get("num_heads", encoder_cfg.get("num_heads", 8)),
        ffn_dim=saved_cfg.get("ffn_dim", encoder_cfg.get("ffn_dim", 2048)),
        num_layers=saved_cfg.get("num_layers", encoder_cfg.get("num_layers", 12)),
        depthwise_conv_kernel_size=saved_cfg.get(
            "depthwise_conv_kernel_size",
            encoder_cfg.get("depthwise_conv_kernel_size", 31),
        ),
        vocab_size=None,
    )
    state = enc_state["model_state_dict"]
    state = {k: v for k, v in state.items() if not k.startswith("ctc_head.")}
    encoder.load_state_dict(state, strict=False)
    encoder.freeze()

    # LLM
    llm = LLMWrapper(
        model_name_or_path=llm_cfg["model_name_or_path"],
        torch_dtype=getattr(torch, llm_cfg.get("dtype", "bfloat16")),
        freeze=True,
    )

    # Load LoRA if applicable
    if lora_path is not None:
        from peft import PeftModel
        llm.model = PeftModel.from_pretrained(llm.model, lora_path)
        logger.info(f"LoRA loaded from {lora_path}")

    # Full model
    model = SpeechLLM(
        encoder=encoder,
        projector_name=projector_cfg.get("name", "conv"),
        llm=llm,
        projector_kwargs=projector_cfg.get("kwargs", {}),
    )

    # Load projector weights
    proj_state = torch.load(projector_ckpt, map_location=device, weights_only=False)
    model.projector.load_state_dict(proj_state["projector_state_dict"])

    return model.to(device).eval()


def translate_files(
    model: SpeechLLM,
    audio_paths: list[str],
    sample_rate: int = 16000,
    n_mels: int = 80,
    device: torch.device = torch.device("cpu"),
    **generate_kwargs,
) -> list[str]:
    """Translate a list of audio files."""
    feature_extractor = build_feature_extractor(sample_rate=sample_rate, n_mels=n_mels)

    features_list = []
    lengths = []

    for path in audio_paths:
        wav, sr = torchaudio.load(path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        feat = feature_extractor(wav.squeeze(0))  # (n_mels, T)
        feat = feat.transpose(0, 1)  # (T, n_mels)
        features_list.append(feat)
        lengths.append(feat.size(0))

    # Pad
    max_len = max(lengths)
    padded = torch.zeros(len(features_list), max_len, n_mels)
    for i, feat in enumerate(features_list):
        padded[i, : feat.size(0)] = feat

    padded = padded.to(device)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return model.translate(padded, length_tensor, **generate_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--projector_ckpt", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--audio_paths", nargs="+", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, args.encoder_ckpt, args.projector_ckpt, args.lora_path, device)

    translations = translate_files(
        model,
        args.audio_paths,
        sample_rate=config.get("data", {}).get("sample_rate", 16000),
        n_mels=config.get("encoder", {}).get("input_dim", 80),
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    for path, translation in zip(args.audio_paths, translations):
        print(f"{path}: {translation}")


if __name__ == "__main__":
    main()
