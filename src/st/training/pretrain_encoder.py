"""
Stage 1: Pretrain speech encoder with CTC loss.

Step-based training with resume support. Specify total_steps, save_every_steps,
log_every_steps, eval_every_steps. Epoch count is derived and reported.

Usage:
    python -m st.training.pretrain_encoder --config configs/experiment/pretrain_ctc.yaml
    python -m st.training.pretrain_encoder --config configs/experiment/pretrain_ctc.yaml \
        --resume_from checkpoints/encoder/encoder_step50000.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from st.models.speech_encoder import SpeechEncoder
from st.data import SpeechDataset, ASRCollator, BalancedSampler, build_vocab_from_index, build_dataset
from st.utils.audio import build_feature_extractor
from st.utils.config import load_config
from st.utils.schedulers import build_scheduler

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate(
    model: SpeechEncoder,
    loader: DataLoader,
    device: torch.device,
    vocab: dict[str, int],
    step: int = 0,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Run validation: compute CTC loss and greedy WER. Saves predictions CSV."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    total_loss = 0.0
    num_batches = 0

    idx_to_char = {v: k for k, v in vocab.items()}
    all_preds = []
    all_refs = []

    for batch in loader:
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        out = model(features, feature_lengths)
        log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

        total_loss += loss.item()
        num_batches += 1

        preds = out["ctc_logits"].argmax(dim=-1)
        for i in range(preds.size(0)):
            pred_ids = preds[i, : out["lengths"][i]].tolist()
            decoded = []
            prev = -1
            for idx in pred_ids:
                if idx != 0 and idx != prev:
                    decoded.append(idx_to_char.get(idx, ""))
                prev = idx
            all_preds.append("".join(decoded))

            ref_ids = labels[i, : label_lengths[i]].tolist()
            all_refs.append("".join(idx_to_char.get(idx, "") for idx in ref_ids))

    avg_loss = total_loss / max(num_batches, 1)

    from st.utils.metrics import compute_wer
    wer = compute_wer(all_preds, all_refs) if all_refs else 0.0

    if output_dir is not None:
        import csv
        from jiwer import wer as jiwer_wer
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "prediction", "wer"])
            for ref, pred in zip(all_refs, all_preds):
                if ref.strip():
                    sample_wer = jiwer_wer(ref, pred) if pred.strip() else 1.0
                else:
                    sample_wer = 0.0 if not pred.strip() else 1.0
                writer.writerow([ref, pred, f"{sample_wer:.4f}"])
        n_empty = sum(1 for p in all_preds if not p.strip())
        logger.info(
            f"Val predictions saved to {csv_path} "
            f"({len(all_preds)} total, {n_empty} empty predictions)"
        )

    model.train()
    return {"val/ctc_loss": avg_loss, "val/wer": wer}


def infinite_loader(loader: DataLoader):
    """Wrap a DataLoader to restart automatically — yields batches forever."""
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/encoder")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    encoder_cfg = config.get("encoder", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    index_paths = data_cfg["index_paths"]
    if isinstance(index_paths, str):
        index_paths = [index_paths]

    lowercase = data_cfg.get("lowercase", False)

    dataset = build_dataset(
        index_paths=index_paths,
        target_sample_rate=data_cfg.get("sample_rate", 16000),
        text_column="transcript",
        split=data_cfg.get("split", "train"),
        languages=data_cfg.get("languages", None),
        sources=data_cfg.get("sources", None),
        max_duration=data_cfg.get("max_duration", 30.0),
        min_duration=data_cfg.get("min_duration", 0.1),
        lowercase=lowercase,
    )

    # Build vocab
    vocab = build_vocab_from_index(
        index_paths[0],
        text_column="transcript",
        split=data_cfg.get("split", "train"),
        languages=data_cfg.get("languages", None),
        lowercase=lowercase,
    )
    for extra_path in index_paths[1:]:
        extra_vocab = build_vocab_from_index(
            extra_path,
            text_column="transcript",
            split=data_cfg.get("split", "train"),
            languages=data_cfg.get("languages", None),
            lowercase=lowercase,
        )
        for char in extra_vocab:
            if char not in vocab:
                vocab[char] = len(vocab)

    logger.info(f"Vocabulary size: {len(vocab)}")

    # --- Model ---
    model = SpeechEncoder(
        input_dim=encoder_cfg.get("input_dim", 80),
        encoder_dim=encoder_cfg.get("encoder_dim", 512),
        num_heads=encoder_cfg.get("num_heads", 8),
        ffn_dim=encoder_cfg.get("ffn_dim", 2048),
        num_layers=encoder_cfg.get("num_layers", 12),
        depthwise_conv_kernel_size=encoder_cfg.get("depthwise_conv_kernel_size", 31),
        dropout=encoder_cfg.get("dropout", 0.1),
        vocab_size=len(vocab),
    ).to(device)

    logger.info(f"Encoder parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- DataLoader ---
    feature_extractor = build_feature_extractor(
        sample_rate=data_cfg.get("sample_rate", 16000),
        n_mels=encoder_cfg.get("input_dim", 80),
    )
    collator = ASRCollator(feature_extractor=feature_extractor, vocab=vocab)

    sampler = None
    shuffle = True
    balance_by = data_cfg.get("balance_by", None)
    if balance_by and isinstance(dataset, SpeechDataset):
        sampler = BalancedSampler(
            dataset,
            group_by=balance_by,
            samples_per_group=data_cfg.get("samples_per_group", None),
        )
        shuffle = False

    batch_size = training_cfg.get("batch_size", 16)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=training_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )

    # --- Val DataLoader ---
    val_dataset = build_dataset(
        index_paths=index_paths,
        target_sample_rate=data_cfg.get("sample_rate", 16000),
        text_column="transcript",
        split="dev",
        languages=data_cfg.get("languages", None),
        sources=data_cfg.get("sources", None),
        max_duration=data_cfg.get("max_duration", 30.0),
        min_duration=data_cfg.get("min_duration", 0.1),
        lowercase=lowercase,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # --- Step budget ---
    total_steps = training_cfg["total_steps"]
    save_every = training_cfg.get("save_every_steps", 10000)
    log_every = training_cfg.get("log_every_steps", 100)
    eval_every = training_cfg.get("eval_every_steps", 5000)

    steps_per_epoch = len(loader)
    equiv_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    logger.info(
        f"Training for {total_steps:,} steps "
        f"(~{equiv_epochs:.1f} epochs, {steps_per_epoch:,} steps/epoch, "
        f"batch_size={batch_size})"
    )

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from {args.resume_from} at step {start_step}")

    # --- wandb ---
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg.get("project", "iwslt2026-st"),
            name=wandb_cfg.get("name", "pretrain-encoder"),
            tags=wandb_cfg.get("tags", []),
            config={
                **config,
                "_steps_per_epoch": steps_per_epoch,
                "_equiv_epochs": round(equiv_epochs, 2),
                "_resumed_from_step": start_step,
            },
            save_code=False,
            resume="allow" if start_step > 0 else None,
        )
    else:
        wandb.init(mode="disabled")

    # --- Training loop (step-based) ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    model.train()

    running_loss = 0.0
    log_steps = 0
    t_start = time.time()

    remaining_steps = total_steps - start_step
    pbar = tqdm(
        total=remaining_steps,
        desc="Training",
        unit="step",
        initial=0,
    )

    for step_offset, batch in enumerate(infinite_loader(loader), start=1):
        step = start_step + step_offset

        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        out = model(features, feature_lengths)
        log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        log_steps += 1
        pbar.update(1)

        # --- Logging ---
        if step % log_every == 0:
            avg_loss = running_loss / log_steps
            elapsed = time.time() - t_start
            samples_sec = (log_steps * batch_size) / elapsed
            current_epoch = step / steps_per_epoch if steps_per_epoch > 0 else 0

            pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"], epoch=current_epoch)

            if wandb.run is not None:
                wandb.log({
                    "train/ctc_loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": current_epoch,
                    "train/samples_per_sec": samples_sec,
                }, step=step)

            running_loss = 0.0
            log_steps = 0
            t_start = time.time()

        # --- Validation ---
        if step % eval_every == 0:
            val_metrics = validate(model, val_loader, device, vocab, step=step, output_dir=output_dir)
            logger.info(
                f"Step {step:,} — val_loss: {val_metrics['val/ctc_loss']:.4f}, "
                f"val_wer: {val_metrics['val/wer']:.4f}"
            )
            if wandb.run is not None:
                wandb.log(val_metrics, step=step)

        # --- Checkpoint ---
        if step % save_every == 0:
            current_epoch = step / steps_per_epoch if steps_per_epoch > 0 else 0
            ckpt_path = output_dir / f"encoder_step{step}.pt"
            torch.save({
                "step": step,
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "vocab": vocab,
                "config": encoder_cfg,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path} (epoch ~{current_epoch:.2f})")

        if step >= total_steps:
            break

    pbar.close()

    # --- Save final ---
    final_epoch = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    torch.save({
        "step": total_steps,
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": encoder_cfg,
    }, output_dir / "encoder_final.pt")
    logger.info(f"Training complete. Final checkpoint saved ({total_steps:,} steps, ~{final_epoch:.1f} epochs)")

    wandb.finish()


if __name__ == "__main__":
    main()