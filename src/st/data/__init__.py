"""
Dataset and collation for speech translation training.

Index format (CSV):
    ASR index: audio_id, path, text, language, split, source, speaker_id, sample_rate, duration
    AST index: audio_id, path, text, language, split, source, speaker_id, sample_rate, duration, translation

Paths in the index are absolute (e.g. /ocean/projects/.../audio.wav).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import soundfile as sf
import torchaudio.functional as F
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeechDataset(Dataset):
    """Speech dataset backed by an ASR/AST index CSV.

    The index CSV must have at least: audio_id, path, text, language, split,
    source, duration. For ST mode, it should also have a 'translation' column.

    Audio paths in the index are absolute, so no audio_root is needed.

    Args:
        index_path: Path to the index CSV.
        target_sample_rate: Resample all audio to this rate.
        text_column: Column to use as the text target ('text' for ASR,
            'translation' for ST).
        split: If set, only load rows matching this split value.
        languages: If set, only load rows matching these language codes.
        sources: If set, only load rows matching these source names.
        max_duration: Skip utterances longer than this (seconds).
        min_duration: Skip utterances shorter than this (seconds).
    """

    def __init__(
        self,
        index_path: str | Path,
        target_sample_rate: int = 16000,
        text_column: str = "text",
        split: str | None = None,
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        lowercase: bool = False,
    ):
        self.target_sample_rate = target_sample_rate
        self.text_column = text_column
        self.lowercase = lowercase

        self.entries = self._load_index(
            index_path, split, languages, sources, min_duration, max_duration
        )
        logger.info(
            f"Loaded {len(self.entries)} entries from {index_path} "
            f"(split={split}, langs={languages}, sources={sources}, "
            f"dur=[{min_duration}, {max_duration}]s)"
        )

    @staticmethod
    def _load_index(
        path: str | Path,
        split: str | None,
        languages: list[str] | None,
        sources: list[str] | None,
        min_duration: float,
        max_duration: float,
    ) -> list[dict[str, str]]:
        entries = []
        lang_set = set(languages) if languages else None
        source_set = set(sources) if sources else None

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            skipped_no_meta = 0
            for row in reader:
                # --- filters ---
                # Require both sample_rate and duration to be present
                dur_str = row.get("duration", "").strip()
                sr_str = row.get("sample_rate", "").strip()
                if not dur_str or not sr_str:
                    skipped_no_meta += 1
                    continue

                dur = float(dur_str)
                if dur < min_duration or dur > max_duration:
                    continue

                if split is not None and row.get("split", "") != split:
                    continue

                # Language filter: supports both 'language' (ASR) and 'src_language' (AST)
                if lang_set is not None:
                    lang = row.get("language", "") or row.get("src_language", "")
                    if lang not in lang_set:
                        continue

                if source_set is not None and row.get("source", "") not in source_set:
                    continue

                # Must have non-empty transcript
                if not row.get("transcript", "").strip():
                    continue

                entries.append(row)

        if skipped_no_meta > 0:
            logger.warning(
                f"Skipped {skipped_no_meta} entries missing sample_rate or duration in {path}"
            )

        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Try loading the requested sample; on failure, skip to the next valid one
        for offset in range(10):  # try up to 10 neighbors
            actual_idx = (idx + offset) % len(self.entries)
            try:
                return self._load_sample(actual_idx)
            except Exception as e:
                if offset == 0:
                    entry = self.entries[actual_idx]
                    logger.warning(f"Failed to load {entry.get('audio_id', '?')} "
                                   f"({entry.get('path', '?')}): {e}")
        # Should never reach here, but just in case
        raise RuntimeError(f"Could not load any sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        audio_path = entry["path"]

        # Load audio with soundfile
        data, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        # Use sample_rate from the index if available
        expected_sr = entry.get("sample_rate", "").strip()
        if expected_sr:
            sr = int(float(expected_sr))

        # Resample if needed
        if sr != self.target_sample_rate:
            waveform = F.resample(waveform, sr, self.target_sample_rate)

        # Mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resolve text column
        text = entry.get(self.text_column, "")
        if self.lowercase:
            text = text.lower()

        language = entry.get("language", "") or entry.get("src_language", "")

        return {
            "audio_id": entry.get("audio_id", ""),
            "waveform": waveform.squeeze(0),
            "text": text,
            "language": language,
            "source": entry.get("source", ""),
            "speaker_id": entry.get("speaker_id", ""),
        }


def build_dataset(
    index_paths: str | Path | list[str | Path],
    target_sample_rate: int = 16000,
    text_column: str = "text",
    split: str | None = None,
    languages: list[str] | None = None,
    sources: list[str] | None = None,
    max_duration: float = 30.0,
    min_duration: float = 0.1,
    lowercase: bool = False,
) -> Dataset:
    """Build a dataset from one or more index CSVs.

    If multiple index paths are given, they are concatenated into a single
    ConcatDataset. This lets you combine e.g. BembaSpeech + NaijaVoices +
    Common Voice indexes into one training set.

    Args:
        index_paths: Single path or list of paths to index CSVs.
        (remaining args forwarded to SpeechDataset)

    Returns:
        A Dataset (SpeechDataset if single index, ConcatDataset if multiple).
    """
    if isinstance(index_paths, (str, Path)):
        index_paths = [index_paths]

    datasets = []
    for path in index_paths:
        ds = SpeechDataset(
            index_path=path,
            target_sample_rate=target_sample_rate,
            text_column=text_column,
            split=split,
            languages=languages,
            sources=sources,
            max_duration=max_duration,
            min_duration=min_duration,
            lowercase=lowercase,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Vocabulary (for CTC)
# ---------------------------------------------------------------------------

def build_vocab_from_index(
    index_path: str | Path,
    text_column: str = "text",
    split: str | None = None,
    languages: list[str] | None = None,
    lowercase: bool = False,
) -> dict[str, int]:
    """Build a character-level vocabulary from an index CSV.

    Index 0 is reserved for CTC blank. Characters are sorted for
    deterministic ordering.
    """
    chars: set[str] = set()
    lang_set = set(languages) if languages else None

    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row.get("split", "") != split:
                continue
            lang = row.get("language", "") or row.get("src_language", "")
            if lang_set is not None and lang not in lang_set:
                continue
            text = row.get(text_column, "")
            if lowercase:
                text = text.lower()
            chars.update(text)

    vocab = {"<blank>": 0}
    for i, c in enumerate(sorted(chars)):
        vocab[c] = i + 1

    logger.info(f"Built vocab: {len(vocab)} tokens (incl. blank) from {index_path}")
    return vocab


def build_vocab_from_datasets(datasets: list[SpeechDataset]) -> dict[str, int]:
    """Build a character vocabulary from one or more loaded SpeechDatasets.

    Useful when combining multiple indexes — ensures a unified vocab.
    """
    chars: set[str] = set()
    for ds in datasets:
        for entry in ds.entries:
            chars.update(entry.get("text", ""))

    vocab = {"<blank>": 0}
    for i, c in enumerate(sorted(chars)):
        vocab[c] = i + 1

    logger.info(f"Built vocab: {len(vocab)} tokens (incl. blank) from {len(datasets)} datasets")
    return vocab


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

def _extract_and_pad_features(
    waveforms: list[torch.Tensor],
    feature_extractor: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared logic: extract log-mel features and pad to batch.

    Returns:
        features: (B, T_max, n_mels)
        lengths: (B,)
    """
    features = []
    lengths = []
    for wav in waveforms:
        feat = feature_extractor(wav)  # (n_mels, T)
        feat = feat.transpose(0, 1)    # (T, n_mels)
        features.append(feat)
        lengths.append(feat.size(0))

    max_len = max(lengths)
    n_mels = features[0].size(1)
    padded = torch.zeros(len(features), max_len, n_mels)
    for i, feat in enumerate(features):
        padded[i, : feat.size(0)] = feat

    return padded, torch.tensor(lengths, dtype=torch.long)


@dataclass
class ASRCollator:
    """Collator for CTC pretraining.

    Extracts log-mel features, pads, and encodes transcripts as integer
    label sequences for CTC loss.

    Args:
        feature_extractor: Callable (waveform → log-mel), e.g. from
            build_feature_extractor().
        vocab: Character-to-index mapping. Index 0 = CTC blank.
    """

    feature_extractor: Any
    vocab: dict[str, int]

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        waveforms = [item["waveform"] for item in batch]
        texts = [item["text"] for item in batch]

        features, lengths = _extract_and_pad_features(waveforms, self.feature_extractor)

        # Encode text → CTC integer labels
        labels = []
        label_lengths = []
        for text in texts:
            encoded = [self.vocab[c] for c in text if c in self.vocab]
            labels.append(torch.tensor(encoded, dtype=torch.long))
            label_lengths.append(len(encoded))

        max_label_len = max(label_lengths) if label_lengths else 0
        padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.size(0)] = lab

        return {
            "features": features,
            "feature_lengths": lengths,
            "labels": padded_labels,
            "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
        }


@dataclass
class STCollator:
    """Collator for speech translation training.

    Extracts log-mel features and tokenizes the translation target for
    teacher-forced LLM training.

    The prompt_template can include a {language} placeholder which gets
    filled per-sample, e.g.:
        "Translate the following {language} speech to English:\\n"

    Args:
        feature_extractor: Callable (waveform → log-mel).
        tokenizer: HuggingFace tokenizer from the LLM.
        max_target_length: Max tokens for the target sequence.
        prompt_template: Template string. Use {language} for per-sample language.
    """

    feature_extractor: Any
    tokenizer: Any
    max_target_length: int = 256
    prompt_template: str = "Translate the following {language} speech to English:\n"

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        waveforms = [item["waveform"] for item in batch]
        texts = [item["text"] for item in batch]
        languages = [item.get("language", "unknown") for item in batch]

        features, lengths = _extract_and_pad_features(waveforms, self.feature_extractor)

        # Build per-sample prompts with language filled in
        targets = []
        for text, lang in zip(texts, languages):
            prompt = self.prompt_template.format(language=lang)
            targets.append(prompt + text)

        tokenized = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        return {
            "features": features,
            "feature_lengths": lengths,
            "text_input_ids": tokenized["input_ids"],
            "text_attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone(),
        }


# ---------------------------------------------------------------------------
# Sampler for balanced language/source sampling
# ---------------------------------------------------------------------------

class BalancedSampler(torch.utils.data.Sampler):
    """Samples indices such that each language (or source) is roughly equally
    represented in each epoch.

    Useful when your combined index is heavily skewed — e.g. 200k Hausa
    rows but only 20k Bemba. This upsamples minority groups per epoch.

    Args:
        dataset: A SpeechDataset.
        group_by: Column to balance on ('language' or 'source').
        samples_per_group: Number of samples per group per epoch. If None,
            uses the size of the largest group.
    """

    def __init__(
        self,
        dataset: SpeechDataset,
        group_by: str = "language",
        samples_per_group: int | None = None,
    ):
        self.dataset = dataset
        self.group_by = group_by

        # Build group → indices mapping
        self.groups: dict[str, list[int]] = {}
        for i, entry in enumerate(dataset.entries):
            # Resolve column aliases for language
            if group_by == "language":
                key = entry.get("language", "") or entry.get("src_language", "") or "unknown"
            else:
                key = entry.get(group_by, "unknown")
            self.groups.setdefault(key, []).append(i)

        if samples_per_group is None:
            samples_per_group = max(len(v) for v in self.groups.values())
        self.samples_per_group = samples_per_group

        logger.info(
            f"BalancedSampler: {len(self.groups)} groups by '{group_by}', "
            f"{samples_per_group} samples/group/epoch, "
            f"groups: {{{', '.join(f'{k}: {len(v)}' for k, v in self.groups.items())}}}"
        )

    def __iter__(self):
        indices = []
        g = torch.Generator()
        g.manual_seed(torch.randint(0, 2**32, (1,)).item())

        for group_indices in self.groups.values():
            n = len(group_indices)
            t = torch.tensor(group_indices, dtype=torch.long)

            if n >= self.samples_per_group:
                # Subsample
                perm = torch.randperm(n, generator=g)[: self.samples_per_group]
                indices.append(t[perm])
            else:
                # Upsample: full pass + random remainder
                full_passes = self.samples_per_group // n
                remainder = self.samples_per_group % n
                parts = [t[torch.randperm(n, generator=g)] for _ in range(full_passes)]
                if remainder > 0:
                    parts.append(t[torch.randperm(n, generator=g)[:remainder]])
                indices.append(torch.cat(parts))

        # Shuffle everything together
        all_indices = torch.cat(indices)
        all_indices = all_indices[torch.randperm(len(all_indices), generator=g)]
        return iter(all_indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_group * len(self.groups)