"""
Stage 2/3: Train speech encoder → projector → LLM for ASR, ST, or CoT-ST.

Unified training script controlled by `data.task`:
    - asr:    speech → transcript (WER metric)
    - st:     speech → translation (BLEU metric)
    - cot_st: speech → "<transcript>...</transcript> <translation>...</translation>" (BLEU+WER)

Step-based training with gradient accumulation, resume, CTC compression.

Usage:
    python -m st.training.train_st --config configs/experiment/train_asr.yaml
    python -m st.training.train_st --config configs/experiment/train_st_stage3.yaml
    python -m st.training.train_st --config configs/experiment/train_st_stage3_cot.yaml \
        --resume_from checkpoints/st/st_step10000.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from st.models.speech_encoder import SpeechEncoder
from st.models.llm_wrapper import LLMWrapper
from st.models.speech_llm import SpeechLLM
from st.data import SpeechDataset, BalancedSampler, build_dataset
from st.utils.audio import build_feature_extractor
from st.utils.config import load_config
from st.utils.schedulers import build_scheduler

logger = logging.getLogger(__name__)

VALID_TASKS = ("asr", "st", "cot_st")


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

class SimpleCollator:
    """Collator for ASR and standard ST.

    Target is a raw text string (transcript for ASR, translation for ST).
    No prompt template — the speech embeddings are the only prefix.

    Args:
        feature_extractor: Callable (waveform → log-mel).
        tokenizer: HuggingFace tokenizer from the LLM.
        max_target_length: Max tokens for the target sequence.
    """

    def __init__(self, feature_extractor, tokenizer, max_target_length: int = 256):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        waveforms = [item["waveform"] for item in batch]
        texts = [item["text"] for item in batch]

        features = []
        lengths = []
        for wav in waveforms:
            feat = self.feature_extractor(wav).T  # (T, n_mels)
            features.append(feat)
            lengths.append(feat.size(0))

        padded_features = pad_sequence(features, batch_first=True)
        feature_lengths = torch.tensor(lengths, dtype=torch.long)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        return {
            "features": padded_features,
            "feature_lengths": feature_lengths,
            "text_input_ids": tokenized["input_ids"],
            "text_attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone(),
        }


class CoTSTCollator:
    """Collator for Chain-of-Thought speech translation.

    Target: "<transcript> {src} </transcript> <translation> {en} </translation>"

    Args:
        feature_extractor: Callable (waveform → log-mel).
        tokenizer: HuggingFace tokenizer from the LLM.
        max_target_length: Max tokens for the full CoT target.
    """

    def __init__(self, feature_extractor, tokenizer, max_target_length: int = 512):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        waveforms = [item["waveform"] for item in batch]
        translations = [item["text"] for item in batch]
        transcripts = [item.get("transcript", "") for item in batch]

        features = []
        lengths = []
        for wav in waveforms:
            feat = self.feature_extractor(wav).T
            features.append(feat)
            lengths.append(feat.size(0))

        padded_features = pad_sequence(features, batch_first=True)
        feature_lengths = torch.tensor(lengths, dtype=torch.long)

        targets = []
        for transcript, translation in zip(transcripts, translations):
            targets.append(
                f"<transcript> {transcript} </transcript> "
                f"<translation> {translation} </translation>"
            )

        tokenized = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        return {
            "features": padded_features,
            "feature_lengths": feature_lengths,
            "text_input_ids": tokenized["input_ids"],
            "text_attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone(),
        }


# ---------------------------------------------------------------------------
# CoT dataset wrapper
# ---------------------------------------------------------------------------

class CoTSpeechDataset(torch.utils.data.Dataset):
    """Wraps a SpeechDataset to also return transcript alongside translation."""

    def __init__(self, base_dataset: SpeechDataset, lowercase: bool = False):
        self.base = base_dataset
        self.lowercase = lowercase

    @property
    def entries(self):
        return self.base.entries

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        transcript = self.base.entries[idx].get("transcript", "")
        if self.lowercase:
            transcript = transcript.lower()
        item["transcript"] = transcript
        return item


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _extract_text(raw: str, task: str, field: str = "translation") -> str:
    """Extract the relevant text from a raw model output or reference.

    For asr/st: return as-is.
    For cot_st: extract content between the relevant tags.
    """
    if task == "cot_st":
        tag = f"<{field}>"
        end_tag = f"</{field}>"
        if tag in raw:
            return raw.split(tag)[-1].split(end_tag)[0].strip()
    return raw.strip()


@torch.no_grad()
def validate(
    model: SpeechLLM,
    loader: DataLoader,
    device: torch.device,
    task: str = "st",
    step: int = 0,
    output_dir: Path | None = None,
    max_new_tokens: int = 128,
    max_val_batches: int | None = None,
) -> dict[str, float]:
    """Run validation with task-appropriate metrics.

    Args:
        max_val_batches: If set, only run this many batches (for speed).
            E.g. with batch_size=4, max_val_batches=50 → 200 samples.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_refs = []

    for batch_idx, batch in enumerate(loader):
        if max_val_batches is not None and batch_idx >= max_val_batches:
            break

        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            features=features,
            feature_lengths=feature_lengths,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )
        total_loss += out["loss"].item()
        num_batches += 1

        # Greedy decoding (no beam search — much faster)
        generations = model.translate(
            features, feature_lengths,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )

        for i, raw_pred in enumerate(generations):
            # For ASR: extract transcript; for ST: extract translation; for CoT: extract translation
            if task == "asr":
                pred = raw_pred.strip()
            elif task == "cot_st":
                pred = _extract_text(raw_pred, "cot_st", "translation")
            else:
                pred = raw_pred.strip()
            all_preds.append(pred)

            # Decode reference
            ref_ids = labels[i][labels[i] != -100].tolist()
            ref_text = model.llm.tokenizer.decode(ref_ids, skip_special_tokens=True)
            if task == "cot_st":
                ref_text = _extract_text(ref_text, "cot_st", "translation")
            all_refs.append(ref_text)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = {"val/loss": avg_loss}

    # Task-specific metrics
    if task == "asr":
        from st.utils.metrics import compute_wer
        wer = compute_wer(all_preds, all_refs) if all_refs else 0.0
        metrics["val/wer"] = wer
    else:
        from st.utils.metrics import compute_bleu
        bleu = compute_bleu(all_preds, all_refs)["bleu"] if all_refs else 0.0
        metrics["val/bleu"] = bleu

    # For CoT, also compute WER on the transcript part
    if task == "cot_st":
        from st.utils.metrics import compute_wer
        cot_transcript_preds = []
        cot_transcript_refs = []
        for i, raw_pred in enumerate(generations):
            cot_transcript_preds.append(_extract_text(raw_pred, "cot_st", "transcript"))
            ref_ids = labels[i][labels[i] != -100].tolist()
            ref_text = model.llm.tokenizer.decode(ref_ids, skip_special_tokens=True)
            cot_transcript_refs.append(_extract_text(ref_text, "cot_st", "transcript"))
        if cot_transcript_refs:
            metrics["val/transcript_wer"] = compute_wer(cot_transcript_preds, cot_transcript_refs)

    # Save predictions CSV
    if output_dir is not None:
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "prediction"])
            for ref, pred in zip(all_refs, all_preds):
                writer.writerow([ref, pred])
        logger.info(f"Val predictions saved to {csv_path} ({len(all_preds)} samples)")

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infinite_loader(loader: DataLoader):
    while True:
        yield from loader


def load_pretrained_encoder(
    config: dict, checkpoint_path: str, device: torch.device,
    keep_ctc_head: bool = False,
) -> SpeechEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    saved_vocab = ckpt.get("vocab", {})

    vocab_size = len(saved_vocab) if (keep_ctc_head and saved_vocab) else None

    encoder = SpeechEncoder(
        input_dim=saved_cfg.get("input_dim", config.get("input_dim", 80)),
        encoder_dim=saved_cfg.get("encoder_dim", config.get("encoder_dim", 512)),
        num_heads=saved_cfg.get("num_heads", config.get("num_heads", 8)),
        ffn_dim=saved_cfg.get("ffn_dim", config.get("ffn_dim", 2048)),
        num_layers=saved_cfg.get("num_layers", config.get("num_layers", 12)),
        depthwise_conv_kernel_size=saved_cfg.get(
            "depthwise_conv_kernel_size",
            config.get("depthwise_conv_kernel_size", 31),
        ),
        dropout=saved_cfg.get("dropout", config.get("dropout", 0.1)),
        vocab_size=vocab_size,
    )

    state = ckpt["model_state_dict"]
    if not keep_ctc_head:
        state = {k: v for k, v in state.items() if not k.startswith("ctc_head.")}
    encoder.load_state_dict(state, strict=False)
    return encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/st")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    encoder_cfg = config.get("encoder", {})
    projector_cfg = config.get("projector", {})
    llm_cfg = config.get("llm", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Task ---
    task = data_cfg.get("task", "st")
    assert task in VALID_TASKS, f"Unknown task '{task}'. Choose from {VALID_TASKS}"
    logger.info(f"Task: {task}")

    lowercase = data_cfg.get("lowercase", False)
    ctc_compress_cfg = config.get("ctc_compress", None)

    # --- Build components ---
    encoder = load_pretrained_encoder(
        encoder_cfg, encoder_cfg["checkpoint_path"], device,
        keep_ctc_head=(ctc_compress_cfg is not None),
    )
    encoder.freeze()
    logger.info(f"Encoder loaded and frozen (ctc_head={'kept' if ctc_compress_cfg else 'removed'}).")

    from st.models.ctc_compressor import build_ctc_compressor
    ctc_compressor = build_ctc_compressor(ctc_compress_cfg)
    if ctc_compressor is not None:
        logger.info(f"CTC compressor: strategy={ctc_compress_cfg.get('strategy', 'avg')}")

    lora_config = llm_cfg.get("lora", None)
    llm = LLMWrapper(
        model_name_or_path=llm_cfg["model_name_or_path"],
        torch_dtype=getattr(torch, llm_cfg.get("dtype", "bfloat16")),
        lora_config=lora_config,
        freeze=llm_cfg.get("freeze", True),
    )
    logger.info(f"LLM loaded: {llm_cfg['model_name_or_path']}")

    model = SpeechLLM(
        encoder=encoder,
        projector_name=projector_cfg.get("name", "conv1d"),
        llm=llm,
        projector_kwargs=projector_cfg.get("kwargs", {}),
        ctc_compressor=ctc_compressor,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

    # --- Data ---
    index_paths = data_cfg["index_paths"]
    if isinstance(index_paths, str):
        index_paths = [index_paths]

    feature_extractor = build_feature_extractor(
        sample_rate=data_cfg.get("sample_rate", 16000),
        n_mels=encoder_cfg.get("input_dim", 80),
    )

    # Determine text_column and collator based on task
    if task == "asr":
        text_column = "transcript"
        collator = SimpleCollator(
            feature_extractor=feature_extractor,
            tokenizer=llm.tokenizer,
            max_target_length=data_cfg.get("max_target_length", 256),
        )
    elif task == "st":
        text_column = "translation"
        collator = SimpleCollator(
            feature_extractor=feature_extractor,
            tokenizer=llm.tokenizer,
            max_target_length=data_cfg.get("max_target_length", 256),
        )
    elif task == "cot_st":
        text_column = "translation"  # base dataset loads translation; CoT wrapper adds transcript
        collator = CoTSTCollator(
            feature_extractor=feature_extractor,
            tokenizer=llm.tokenizer,
            max_target_length=data_cfg.get("max_target_length", 512),
        )

    # Build datasets
    def _build_split(split: str):
        ds = build_dataset(
            index_paths=index_paths,
            target_sample_rate=data_cfg.get("sample_rate", 16000),
            text_column=text_column,
            split=split,
            languages=data_cfg.get("languages", None),
            sources=data_cfg.get("sources", None),
            max_duration=data_cfg.get("max_duration", 30.0),
            min_duration=data_cfg.get("min_duration", 0.1),
            lowercase=lowercase,
        )
        if task == "cot_st":
            ds = CoTSpeechDataset(ds, lowercase=lowercase)
        return ds

    dataset = _build_split(data_cfg.get("split", "train"))
    val_dataset = _build_split("dev")

    # --- Sampler ---
    sampler = None
    shuffle = True
    balance_by = data_cfg.get("balance_by", None)
    sampler_target = dataset.base if hasattr(dataset, "base") else dataset
    if balance_by and isinstance(sampler_target, SpeechDataset):
        sampler = BalancedSampler(
            sampler_target,
            group_by=balance_by,
            samples_per_group=data_cfg.get("samples_per_group", None),
        )
        shuffle = False

    batch_size = training_cfg.get("batch_size", 4)
    grad_accum = training_cfg.get("grad_accum_steps", 1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=training_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )
    logger.info(f"Train: {len(dataset)} samples, Val: {len(val_dataset)} samples")

    # --- Step budget ---
    total_steps = training_cfg["total_steps"]
    save_every = training_cfg.get("save_every_steps", 5000)
    log_every = training_cfg.get("log_every_steps", 100)
    eval_every = training_cfg.get("eval_every_steps", 5000)

    micro_steps_per_epoch = len(loader)
    optim_steps_per_epoch = micro_steps_per_epoch / grad_accum
    equiv_epochs = total_steps / optim_steps_per_epoch if optim_steps_per_epoch > 0 else 0
    logger.info(
        f"Training for {total_steps:,} optimizer steps "
        f"(~{equiv_epochs:.1f} epochs, "
        f"grad_accum={grad_accum}, "
        f"effective_batch={batch_size * grad_accum})"
    )

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.get("lr", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    sched_cfg = training_cfg.get("scheduler", {})
    scheduler = build_scheduler(
        name=sched_cfg.get("name", "none"),
        optimizer=optimizer,
        total_steps=total_steps,
        **{k: v for k, v in sched_cfg.items() if k != "name"},
    )
    if scheduler is not None:
        logger.info(f"Scheduler: {sched_cfg['name']}")

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        lora_dir = Path(args.resume_from).parent / f"lora_step{start_step}"
        if lora_config is not None and lora_dir.exists():
            from peft import PeftModel
            model.llm.model = PeftModel.from_pretrained(model.llm.model, str(lora_dir))
            logger.info(f"Resumed LoRA from {lora_dir}")
        logger.info(f"Resumed from {args.resume_from} at step {start_step}")

    # --- wandb ---
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg.get("project", "iwslt2026-st"),
            name=wandb_cfg.get("name", f"train-{task}"),
            tags=wandb_cfg.get("tags", []),
            config={
                **config,
                "_task": task,
                "_optim_steps_per_epoch": round(optim_steps_per_epoch, 1),
                "_equiv_epochs": round(equiv_epochs, 2),
                "_trainable_params": trainable,
                "_total_params": total_params,
                "_resumed_from_step": start_step,
            },
            save_code=False,
            resume="allow" if start_step > 0 else None,
        )
    else:
        wandb.init(mode="disabled")

    # --- Training loop ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    log_micro_steps = 0
    optim_step = start_step
    t_start = time.time()

    remaining_steps = total_steps - start_step
    pbar = tqdm(total=remaining_steps, desc=f"Training ({task})", unit="step")

    for micro_step, batch in enumerate(infinite_loader(loader), start=1):
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            features=features,
            feature_lengths=feature_lengths,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )

        loss = out["loss"] / grad_accum
        loss.backward()
        running_loss += out["loss"].item()
        log_micro_steps += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            optim_step += 1
            pbar.update(1)

            # --- Logging ---
            if optim_step % log_every == 0:
                avg_loss = running_loss / log_micro_steps
                elapsed = time.time() - t_start
                samples_sec = (log_micro_steps * batch_size) / elapsed
                current_epoch = optim_step / optim_steps_per_epoch if optim_steps_per_epoch > 0 else 0

                pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"], epoch=current_epoch)

                log_dict = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": current_epoch,
                    "train/samples_per_sec": samples_sec,
                }
                # Log compression ratio if available
                if "compression_ratio" in out:
                    log_dict["train/compression_ratio"] = out["compression_ratio"]

                if wandb.run is not None:
                    wandb.log(log_dict, step=optim_step)

                running_loss = 0.0
                log_micro_steps = 0
                t_start = time.time()

            # --- Validation ---
            if optim_step % eval_every == 0:
                val_metrics = validate(
                    model, val_loader, device, task=task,
                    step=optim_step, output_dir=output_dir,
                    max_val_batches=training_cfg.get("max_val_batches", 25),
                )
                # Log with task-appropriate metric name
                metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                logger.info(f"Step {optim_step:,} — {metric_str}")
                if wandb.run is not None:
                    wandb.log(val_metrics, step=optim_step)

            # --- Checkpoint ---
            if optim_step % save_every == 0:
                current_epoch = optim_step / optim_steps_per_epoch if optim_steps_per_epoch > 0 else 0
                ckpt_path = output_dir / f"st_step{optim_step}.pt"
                save_dict = {
                    "step": optim_step,
                    "epoch": current_epoch,
                    "task": task,
                    "projector_state_dict": model.projector.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "config": config,
                }
                if lora_config is not None:
                    model.llm.model.save_pretrained(output_dir / f"lora_step{optim_step}")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved: {ckpt_path} (epoch ~{current_epoch:.2f})")

            if optim_step >= total_steps:
                break

    pbar.close()

    # --- Save final ---
    final_epoch = total_steps / optim_steps_per_epoch if optim_steps_per_epoch > 0 else 0
    ckpt_path = output_dir / "st_final.pt"
    save_dict = {
        "step": total_steps,
        "epoch": final_epoch,
        "task": task,
        "projector_state_dict": model.projector.state_dict(),
        "config": config,
    }
    if lora_config is not None:
        model.llm.model.save_pretrained(output_dir / "lora_final")
    torch.save(save_dict, ckpt_path)
    logger.info(f"Training complete. Final checkpoint saved ({total_steps:,} steps, ~{final_epoch:.1f} epochs)")

    wandb.finish()


if __name__ == "__main__":
    main()