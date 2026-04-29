"""
SpeechDataset: reads ASR/AST index CSVs and returns mel + text pairs.

Index CSV columns (ASR):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

Index CSV columns (AST, superset):
    ... + translation, src_language, tgt_language

The dataset is intentionally dumb — it just reads audio and returns mel
features + text strings. All tokenization happens in the collator so the
dataset stays reusable across training stages.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Default mel transform settings (25ms window, 10ms hop, 80 bins @ 16kHz)
_DEFAULT_SR      = 16000
_DEFAULT_N_MELS  = 80
_DEFAULT_N_FFT   = 400
_DEFAULT_HOP     = 160


class SpeechDataset(Dataset):
    """Speech dataset backed by an ASR/AST index CSV.

    Args:
        index_path:       Path to CSV index file.
        split:            Value of the 'split' column to keep (e.g. "train", "dev").
                          None = keep all rows.
        languages:        Keep only these language codes. None = keep all.
        sources:          Keep only these source names. None = keep all.
        text_column:      Column to use as the primary text target.
                          "transcript" for ASR, "translation" for AST.
        max_duration:     Drop utterances longer than this (seconds).
        min_duration:     Drop utterances shorter than this (seconds).
        sample_rate:      Resample all audio to this rate.
        lowercase:        Lowercase all text targets.
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = "train",
        task: str = "asr",
        languages: list[str] | None = None,
        src_languages: list[str] | None = None,
        tgt_languages: list[str] | None = None,
        sources: list[str] | None = None,
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        sample_rate: int = _DEFAULT_SR,
        lowercase: bool = False,
    ):
        if task not in ("asr", "cot"):
            raise ValueError(f"task must be 'asr' or 'cot', got '{task}'")

        self.task = task
        self.sample_rate = sample_rate
        self.lowercase   = lowercase
        self.max_frames  = int(max_duration * sample_rate)

        effective_src = src_languages if src_languages is not None else languages

        self.entries: list[dict[str, str]] = self._load_index(
            index_path, split, task, effective_src, tgt_languages,
            sources, min_duration, max_duration,
        )

        # Pre-extract durations for DurationBucketSampler (avoids probing files at runtime)
        self.durations: list[float] = [
            float(e["duration"]) if e.get("duration") else max_duration
            for e in self.entries
        ]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=_DEFAULT_N_FFT,
            hop_length=_DEFAULT_HOP,
            n_mels=_DEFAULT_N_MELS,
        )

        # Log summary
        lang_counts: dict[str, int] = {}
        for e in self.entries:
            lang = e.get("language") or e.get("src_language") or "?"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total_hours = sum(self.durations) / 3600
        log.info(
            f"SpeechDataset: {len(self.entries)} examples from {index_path} "
            f"[split={split}, {total_hours:.1f}h]"
        )
        log.info(f"  Languages: {lang_counts}")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_index(
        path: str | Path,
        split: str | None,
        task: str,
        src_languages: list[str] | None,
        tgt_languages: list[str] | None,
        sources: list[str] | None,
        min_duration: float,
        max_duration: float,
    ) -> list[dict[str, str]]:
        src_set    = set(src_languages) if src_languages else None
        tgt_set    = set(tgt_languages) if tgt_languages else None
        source_set = set(sources) if sources else None

        entries = []
        skipped_no_dur = 0
        skipped_no_translation = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Duration filter
                dur_str = row.get("duration", "").strip()
                if not dur_str:
                    skipped_no_dur += 1
                    continue
                dur = float(dur_str)
                if dur < min_duration or dur > max_duration:
                    continue

                # Split filter
                if split is not None and row.get("split", "") != split:
                    continue

                # Source-language filter
                if src_set is not None:
                    src_lang = row.get("language") or row.get("src_language") or ""
                    if src_lang not in src_set:
                        continue

                # Target-language filter (CoT only — ASR has no tgt_language column)
                if tgt_set is not None and task == "cot":
                    if row.get("tgt_language", "") not in tgt_set:
                        continue

                # Source filter
                if source_set is not None and row.get("source", "") not in source_set:
                    continue

                # CoT requires both transcript and translation populated
                if task == "cot":
                    transcript  = row.get("transcript",  "").strip()
                    translation = row.get("translation", "").strip()
                    if not transcript or not translation:
                        skipped_no_translation += 1
                        continue

                entries.append(row)

        if skipped_no_dur:
            log.warning(f"Skipped {skipped_no_dur} rows with missing duration in {path}")
        if skipped_no_translation:
            log.warning(
                f"Skipped {skipped_no_translation} rows missing transcript/translation "
                f"(required for task=cot)"
            )
        return entries

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to 10 neighbors on load failure (corrupt / missing audio)
        for offset in range(10):
            actual = (idx + offset) % len(self.entries)
            try:
                return self._load_sample(actual)
            except Exception as exc:
                if offset == 0:
                    e = self.entries[actual]
                    log.warning(
                        f"Failed to load {e.get('audio_id', '?')} "
                        f"({e.get('path', '?')}): {exc}"
                    )
        raise RuntimeError(f"Could not load any sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        entry      = self.entries[idx]
        audio_path = entry.get("path") or entry.get("audio_path") or ""

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        waveform = torch.from_numpy(data)

        # Use index sample_rate when available (avoids soundfile header re-read)
        sr_str = entry.get("sample_rate", "").strip()
        if sr_str:
            sr = int(float(sr_str))

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)

        if waveform.size(0) > self.max_frames:
            waveform = waveform[: self.max_frames]

        # Log-mel spectrogram
        mel = self.mel_transform(waveform)               # (80, T)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = mel.T                                       # (T, 80)

        # Source language (always present in both ASR and AST CSVs)
        src_language = entry.get("language") or entry.get("src_language") or ""

        transcript = entry.get("transcript", "")
        if self.lowercase:
            transcript = transcript.lower()

        if self.task == "asr":
            translation  = ""
            tgt_language = src_language
        else:  # cot
            translation = entry.get("translation", "")
            if self.lowercase:
                translation = translation.lower()
            tgt_language = entry.get("tgt_language", "english")

        return {
            "audio_id":     entry.get("audio_id", ""),
            "mel":          mel,
            "mel_len":      mel.size(0),
            "transcript":   transcript,
            "translation":  translation,
            "src_language": src_language,
            "tgt_language": tgt_language,
            "task":         self.task,
            "source":       entry.get("source", ""),
        }
