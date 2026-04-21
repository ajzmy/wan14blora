"""
Distributed sampler for GRPO training.

DistributedGRPOSampler:
  - Each iteration yields the same indices on all ranks (synchronized by seed)
  - Supports K-repeat for group advantage computation:
    each unique sample is repeated K times so we can compute
    per-prompt advantages within a group
"""

import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedGRPOSampler(Sampler):
    """
    Distributed sampler where all ranks get the SAME sample indices each step.

    For GRPO with group advantage:
      - batch_size=1, num_generations=K
      - Each step, all 16 GPUs get the SAME sample
      - Each GPU generates 1 video → 16 videos total per sample
      - Advantages computed across the 16 videos

    For GRPO with K-repeat on single GPU:
      - k_repeat > 1 means each sample index is repeated K times in the batch
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=True):
        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.num_samples = len(dataset)
        if self.drop_last:
            self.total_size = self.num_samples
        else:
            self.total_size = self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        """
        All ranks iterate over the SAME shuffled indices.
        This ensures all GPUs process the same prompt for group advantage.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class DistributedKRepeatSampler(Sampler):
    """
    Sampler that distributes unique samples across ranks,
    with each sample repeated K times per rank for group advantage.

    Total samples per step = num_replicas * K
    Each rank gets K copies of one unique sample.
    """

    def __init__(self, dataset, k_repeat=4, num_replicas=None, rank=None,
                 shuffle=True, seed=0):
        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.k_repeat = k_repeat
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Number of unique samples per epoch
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        """
        Each rank gets a different subset of dataset indices.
        Each index is then repeated K times.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Pad to make evenly divisible
        if len(indices) < self.total_size:
            indices += indices[:self.total_size - len(indices)]
        indices = indices[:self.total_size]

        # Subsample for this rank
        rank_indices = indices[self.rank:self.total_size:self.num_replicas]

        # Repeat each index K times
        repeated = []
        for idx in rank_indices:
            repeated.extend([idx] * self.k_repeat)

        return iter(repeated)

    def __len__(self):
        return self.num_samples * self.k_repeat
