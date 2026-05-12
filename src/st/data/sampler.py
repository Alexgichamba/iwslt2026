"""
Duration-based bucket sampler with synchronized bucket selection for DDP.

Groups audio samples by duration into buckets of similar length, then
builds batches that cap total audio seconds per batch rather than instance
count. This prevents OOM from variable-length sequences while keeping
padding overhead low.

Synchronized bucketing (from Zelasko et al. 2025):
  - All DDP ranks use the same seed for BUCKET ORDER selection
    → all ranks draw from the same duration range each step
    → eliminates tail-worker effect from one rank drawing 30s audio
       while another draws 2s audio
  - Per-rank seed used for SAMPLE ORDER within each bucket
    → each rank still sees completely different audio files
    → true data parallelism is preserved

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
        shuffle:              Shuffle samples within each bucket before batching.
        shuffle_buckets:      Shuffle bucket order each epoch.
        drop_last:            Drop the final incomplete batch per bucket.
        rank:                 DDP rank (0 for single GPU). Used for per-rank
                              sample-order seed.
        world_size:           DDP world size (1 for single GPU).
        seed:                 Base seed for the shared bucket-order RNG.
                              All ranks must use the same value.
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
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset          = dataset
        self.target_duration  = target_duration
        self.max_batch_size   = max_batch_size
        self.bucket_width     = bucket_width
        self.shuffle          = shuffle
        self.shuffle_buckets  = shuffle_buckets
        self.drop_last        = drop_last
        self.rank             = rank
        self.world_size       = world_size
        self.seed             = seed

        self.durations: list[float] = dataset.durations

        # Two separate RNGs — this is the key change
        # Bucket-order RNG: shared seed, same on all ranks
        self._bucket_rng = random.Random(seed)
        # Sample-order RNG: per-rank seed, different on all ranks
        self._sample_rng = random.Random(seed + rank * 31337)

        self._buckets     = self._make_buckets()
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
        # Step 1: Shuffle SAMPLES within each bucket using the PER-RANK RNG.
        # Different ranks get different samples from the same bucket.
        buckets = [b.copy() for b in self._buckets]
        if self.shuffle:
            for b in buckets:
                self._sample_rng.shuffle(b)

        # Step 2: Build batches from the shuffled buckets.
        batches = self._make_batches(buckets)

        # Step 3: Shuffle BUCKET ORDER using the SHARED RNG.
        # All ranks pick the same bucket order → same duration range each step
        # → no tail-worker effect.
        if self.shuffle_buckets:
            self._bucket_rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return len(self._all_batches)

    def set_epoch(self, epoch: int) -> None:
        """Call this at the start of each epoch (like DistributedSampler).
        Advances both RNGs deterministically so each epoch has different
        shuffling but all ranks stay synchronized on bucket order."""
        self._bucket_rng = random.Random(self.seed + epoch)
        self._sample_rng = random.Random(self.seed + epoch + self.rank * 31337)