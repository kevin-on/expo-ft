"""Deterministic replay indices, shared across learner hosts without an RPC."""
from collections.abc import Sequence

import numpy as np


class DistributedReplaySampler:
    """Choose global minibatches first, then return this host's column slice.

    Each candidate entry is either a row count (all rows in ``range(count)``)
    or an ordered array of eligible row indices for one robot's replay. All
    hosts must have the same candidates, seed and completed-update counter.
    There is no mutable RNG/cursor to checkpoint. Reuse across UTD minibatches
    is allowed; replacement within a minibatch is used only when N < B.
    """

    def __init__(self, *, seed: int, global_batch_size: int, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid learner rank/world size')
        if global_batch_size < 1 or global_batch_size % world_size:
            raise ValueError('Global batch size must be divisible by learner host count')
        self.seed = seed
        self.global_batch_size = global_batch_size
        self.local_batch_size = global_batch_size // world_size
        self.rank = rank

    def sample(self, candidates: Sequence, *, update_step: int, num_batches: int,
               stream: str = 'critic'):
        """Return (minibatches, local_batch, [robot, row]), or None if empty."""
        if update_step < 0 or num_batches < 1:
            raise ValueError('Invalid update step or minibatch count')
        stream_id = {'critic': 0, 'actor': 1}[stream]
        counts = np.asarray([c if np.isscalar(c) else len(c) for c in candidates], dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError('Candidate counts must be nonnegative')
        total = int(counts.sum())
        if total == 0:
            return None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(update_step), stream_id]))
        begin = self.rank * self.local_batch_size
        selected = np.empty((num_batches, self.local_batch_size), dtype=np.int64)
        for minibatch in range(num_batches):
            indices = rng.choice(total, self.global_batch_size,
                                 replace=total < self.global_batch_size)
            selected[minibatch] = indices[begin:begin + self.local_batch_size]
        ends = np.cumsum(counts)
        robots = np.searchsorted(ends, selected, side='right')
        rows = selected - np.r_[0, ends[:-1]][robots]
        for robot, candidate in enumerate(candidates):
            if not np.isscalar(candidate):
                mask = robots == robot
                rows[mask] = np.asarray(candidate)[rows[mask]]
        return np.stack((robots, rows), axis=-1)
