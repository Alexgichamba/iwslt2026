"""
Duration-based bucket sampler with synchronized batches for DDP.

Groups samples by duration into buckets, then builds batches that cap total
audio seconds per batch (not instance count). Prevents OOM from variable-
length sequences while keeping padding overhead low.

Synchronized batches across ranks (Zelasko et al. 2025, ASRU):
  - All ranks use the SAME shared RNG to shuffle samples within buckets
    and to shuffle bucket order → all ranks build an IDENTICAL global
    batch list.
  - Each rank then takes batches[rank::world_size] as its slice.
    → At step k, all ranks process audio from the same duration range
      (eliminating the tail-worker effect).
    → Each rank sees a disjoint subset of the data (true data parallelism).
    → Every sample is seen exactly once per epoch across all ranks combined.

This is a pre-built-list approximation of the paper's streaming-buffer
approach. It gives up determinism-via-streaming and the concurrent-bucketing
startup speedup, but achieves the same tail-worker fix and data-parallelism
properties with a simpler mechanism.

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
        rank:                 DDP rank (0 for single GPU). Used to slice the
                              shared global batch list.
        world_size:           DDP world size (1 for single GPU).
        seed:                 Shared seed — must be identical on all ranks.
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

        # Single shared RNG — same seed on all ranks → all ranks build
        # the same global batch list, then slice by rank.
        self._rng = random.Random(seed)

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
        # Step 1: Shuffle SAMPLES within each bucket using the SHARED RNG.
        # All ranks produce identical bucket contents.
        buckets = [b.copy() for b in self._buckets]
        if self.shuffle:
            for b in buckets:
                self._rng.shuffle(b)

        # Step 2: Build batches from the (identically) shuffled buckets.
        batches = self._make_batches(buckets)

        # Step 3: Shuffle BATCH ORDER using the SHARED RNG.
        # All ranks see the same order → at step k all ranks pull from
        # the same duration range (paper's tail-worker fix).
        if self.shuffle_buckets:
            self._rng.shuffle(batches)

        # Step 4: Slice for this rank. With identical global lists across
        # ranks and an interleaved slice, every sample is covered exactly
        # once per epoch and no two ranks see the same batch.
        batches = batches[self.rank::self.world_size]
        yield from batches

    def __len__(self) -> int:
        # Per-rank batch count. Use ceil-style division so rank 0 (which gets
        # any remainder) doesn't under-report.
        total = len(self._all_batches)
        return (total + self.world_size - 1 - self.rank) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch (like DistributedSampler).
        Re-seeds the shared RNG deterministically so each epoch shuffles
        differently but all ranks stay synchronized."""
        self._rng = random.Random(self.seed + epoch)