from torch.utils.data import Sampler
import numpy as np
import random
from typing import List, Iterator
import torchaudio
import os

class DurationBucketSampler(Sampler):
    """
    Sampler that creates batches based on total audio duration with automatic bucketing.
    """
    
    def __init__(
        self, 
        dataset,
        target_duration: float = 300,  # 5 minutes default
        bucket_length_multiplier: float = 1.5,
        shuffle: bool = True,
        shuffle_buckets: bool = True
    ):
        self.dataset = dataset
        self.target_duration = target_duration
        self.bucket_length_multiplier = bucket_length_multiplier
        self.shuffle = shuffle
        self.shuffle_buckets = shuffle_buckets
        
        # Cache audio durations
        self.durations = self._get_durations()
        
        # Create natural length-based buckets
        self.buckets = self._create_natural_buckets()
        
        # Precalculate all batches to get exact count
        self.all_batches = self._create_all_batches()
    
    
        
    def _get_durations(self):
        """Get duration for each audio sample"""
        durations = []
        for idx in range(len(self.dataset)):
            audio_path = self.dataset.samples[idx][0]
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            durations.append(duration)
        return durations
    
    def _create_natural_buckets(self):
        """Create buckets based on natural length groupings"""
        # Sort indices by duration
        indices = list(range(len(self.dataset)))
        indices_and_durations = [(idx, self.durations[idx]) for idx in indices]
        sorted_indices_and_durations = sorted(indices_and_durations, key=lambda x: x[1])
        
        # Create buckets with samples of similar length
        buckets = []
        current_bucket = []
        min_length = None
        
        for idx, duration in sorted_indices_and_durations:
            if min_length is None:
                # Start a new bucket
                current_bucket = [idx]
                min_length = duration
            elif duration > min_length * self.bucket_length_multiplier:
                # This sample is too different in length, start a new bucket
                buckets.append(current_bucket)
                current_bucket = [idx]
                min_length = duration
            else:
                # Add to current bucket
                current_bucket.append(idx)
        
        # Add the last bucket if not empty
        if current_bucket:
            buckets.append(current_bucket)
            
        return buckets
    
    def _create_all_batches(self):
        """Precalculate all batches based on buckets and target duration"""
        all_batches = []
        
        for bucket in self.buckets:
            batch = []
            batch_duration = 0.0
            
            for idx in bucket:
                duration = self.durations[idx]
                
                # If adding this sample would exceed target duration, add batch to list
                if batch_duration + duration > self.target_duration and batch:
                    all_batches.append(batch)
                    batch = []
                    batch_duration = 0.0
                    
                # Add current sample to batch
                batch.append(idx)
                batch_duration += duration
                
            # Add the last batch if not empty
            if batch:
                all_batches.append(batch)
                
        return all_batches
        
    def __iter__(self) -> Iterator[List[int]]:
        # Create a copy to avoid modifying the original
        batches = [batch.copy() for batch in self.all_batches]
        
        # Shuffle the batches if needed
        if self.shuffle_buckets:
            random.shuffle(batches)
            
        # Optionally shuffle within each batch
        if self.shuffle:
            for batch in batches:
                random.shuffle(batch)
        
        # Yield each batch
        for batch in batches:
            yield batch
                
    def __len__(self) -> int:
        """Return the exact number of batches"""
        return len(self.all_batches)