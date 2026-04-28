"""
Duration-based bucket sampler.

Groups audio samples by duration into buckets of similar length, then
builds batches that cap total audio seconds per batch rather than instance
count. This prevents OOM from variable-length sequences while keeping
padding overhead low.

Requires the dataset to expose a .durations list (pre-extracted from CSV).
"""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler


class DurationBucketSampler(Sampler):
    """Batch sampler that caps total audio duration per batch.

    Args:
        dataset:              Dataset with a `.durations` attribute (list[float]).
        target_duration:      Max total audio seconds per batch.
        max_batch_size:       Hard cap on instances per batch (prevents huge
                              batches from many short utterances).
        bucket_width:         Multiplier — samples within a bucket have durations
                              spanning at most [min, min * bucket_width].
        shuffle:              Shuffle within each bucket before batching.
        shuffle_buckets:      Shuffle bucket order each epoch.
        drop_last:            Drop the final incomplete batch per bucket.
    """

    def __init__(
        self,
        dataset,
        target_duration: float = 120.0,
        max_batch_size: int = 64,
        bucket_width: float = 1.5,
        shuffle: bool = True,
        shuffle_buckets: bool = True,
        drop_last: bool = False,
    ):
        self.dataset          = dataset
        self.target_duration  = target_duration
        self.max_batch_size   = max_batch_size
        self.bucket_width     = bucket_width
        self.shuffle          = shuffle
        self.shuffle_buckets  = shuffle_buckets
        self.drop_last        = drop_last

        self.durations: list[float] = dataset.durations

        self._buckets   = self._make_buckets()
        self._all_batches = self._make_batches(self._buckets)

    # ------------------------------------------------------------------

    def _make_buckets(self) -> list[list[int]]:
        """Sort indices by duration and group into width-bounded buckets."""
        indexed = sorted(enumerate(self.durations), key=lambda x: x[1])
        buckets: list[list[int]] = []
        current: list[int]       = []
        bucket_min: float | None = None

        for idx, dur in indexed:
            if bucket_min is None:
                current    = [idx]
                bucket_min = dur
            elif dur <= bucket_min * self.bucket_width:
                current.append(idx)
            else:
                buckets.append(current)
                current    = [idx]
                bucket_min = dur

        if current:
            buckets.append(current)

        return buckets

    def _make_batches(self, buckets: list[list[int]]) -> list[list[int]]:
        """Build batches from buckets respecting target_duration and max_batch_size."""
        all_batches: list[list[int]] = []

        for bucket in buckets:
            batch: list[int]  = []
            batch_dur: float  = 0.0

            for idx in bucket:
                dur = self.durations[idx]
                # Flush if either cap would be exceeded
                if batch and (
                    batch_dur + dur > self.target_duration
                    or len(batch) >= self.max_batch_size
                ):
                    all_batches.append(batch)
                    batch, batch_dur = [], 0.0

                batch.append(idx)
                batch_dur += dur

            if batch and not self.drop_last:
                all_batches.append(batch)

        return all_batches

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle within each bucket, then rebuild batches fresh each epoch
        buckets = [b.copy() for b in self._buckets]
        if self.shuffle:
            for b in buckets:
                random.shuffle(b)
        
        batches = self._make_batches(buckets)
        
        if self.shuffle_buckets:
            random.shuffle(batches)
        
        yield from batches

    def __len__(self) -> int:
        return len(self._all_batches)
