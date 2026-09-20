"""BatchProcessor: fetches and mixes training batches from replay buffers."""
import jax
import numpy as np

from expo_ft.agents import restore_replay_buffer
from expo_ft.data.replay_buffer import PiReplayBuffer
from expo_ft.utils.train_utils import clear_batch, combine_batches


class BatchProcessor:
    """Builds critic and optional actor batches for one gradient update.

    Modes:
    - online only (offline_ratio=0): all samples from the online replay buffer
    - mixed (0 < offline_ratio < 1): shuffled online + offline critic batches
    - RTCLearner (use_dagger_hil_sampling): critic batch from replay; actor from HIL chunks
    """
    def __init__(
        self,
        replay_buffer: PiReplayBuffer,
        offline_replay_buffer: PiReplayBuffer,
        data_sharding,
        batch_size: int,
        utd_ratio: int,
        offline_ratio: float,
        actor_success_only: bool,
        use_dagger_hil_sampling: bool,  # True for RTCLearner: actor batch from HIL chunks only
        dataset=None,
        replay_buffers=None,
    ):
        if dataset is not None:
            # offline_ratio=0: seed demos into the online replay buffer only.
            if offline_ratio == 0 or use_dagger_hil_sampling:
                replay_buffer.insert_dataset(dataset)
            if offline_ratio != 0:
                offline_replay_buffer.insert_dataset(dataset)

        self.replay_buffer = replay_buffer
        self.offline_replay_buffer = offline_replay_buffer
        self.data_sharding = data_sharding
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.offline_ratio = offline_ratio
        self.actor_success_only = actor_success_only
        self.use_dagger_hil_sampling = use_dagger_hil_sampling
        self.replay_buffers = replay_buffers
        self._robot_rng = np.random.default_rng(replay_buffer._seed)
        self._robot_batches = {}

        replay_batch_multiplier = 1.0 if use_dagger_hil_sampling else (1 - offline_ratio)
        self.replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(batch_size * utd_ratio * replay_batch_multiplier),
            },
            data_sharding=data_sharding,
        )
        if replay_buffers is not None:
            self.replay_iterator = self._online_batches(
                int(batch_size * utd_ratio * replay_batch_multiplier)
            )

        self.offline_iterator = None
        if offline_ratio > 0 and not use_dagger_hil_sampling:
            self.offline_iterator = offline_replay_buffer.get_iterator(
                sample_args={
                    "batch_size": int(batch_size * utd_ratio * offline_ratio),
                },
                data_sharding=data_sharding,
            )

        self.hil_iterator = None
        if use_dagger_hil_sampling:
            self.hil_iterator = replay_buffer.get_iterator(
                sample_args={"batch_size": batch_size, "hil_only": True},
                data_sharding=data_sharding,
            )
            if replay_buffers is not None:
                self.hil_iterator = self._online_batches(batch_size, hil_only=True)

        if actor_success_only and not use_dagger_hil_sampling:
            self._offline_actor_bs = int(batch_size * offline_ratio)
            self._online_actor_bs = batch_size - self._offline_actor_bs

        self._ep_buffer_start = replay_buffer._insert_index

    def insert_transition(self, transition_dict):
        self.replay_buffer.insert(transition_dict)

    def on_episode_start(self):
        self._ep_buffer_start = self.replay_buffer._insert_index

    def on_episode_done(self, success):
        if success:
            self.replay_buffer.mark_episode_success(
                self._ep_buffer_start, self.replay_buffer._insert_index
            )
        self._ep_buffer_start = self.replay_buffer._insert_index

    def restore(self, checkpoint_dir, up_to_step=None):
        """Restore replay buffer from disk and rebuild success marks."""
        restore_replay_buffer(checkpoint_dir, self.replay_buffer, up_to_step=up_to_step)
        self.replay_buffer.restore_success_marks()

    def next_batch(self, combine_rng):
        """Return (critic_batch, actor_batch, new_rng) for one update step."""
        if self.replay_buffers is not None and self.offline_ratio == 1 and not self.use_dagger_hil_sampling:
            batch = next(self.offline_iterator)
            new_rng = combine_rng
        elif self.use_dagger_hil_sampling or self.offline_ratio == 0:
            batch = next(self.replay_iterator)
            new_rng = combine_rng
        else:
            online_batch = next(self.replay_iterator)
            offline_batch = next(self.offline_iterator)
            shuffle_key, new_rng = jax.random.split(combine_rng)
            batch = combine_batches(online_batch, offline_batch, rng=shuffle_key)
            clear_batch(online_batch)
            clear_batch(offline_batch)

        batch = self.replay_buffer.apply_data_sharding(batch, self.data_sharding)

        actor_batch = None
        if self.use_dagger_hil_sampling:
            actor_batch = next(self.hil_iterator)
            actor_batch = self.replay_buffer.apply_data_sharding(actor_batch, self.data_sharding)
        elif self.actor_success_only:
            actor_batch = self._sample_success_actor_batch(new_rng)
            if actor_batch is not None:
                new_rng_parts = jax.random.split(new_rng)
                new_rng = new_rng_parts[0]
                actor_batch = self.replay_buffer.apply_data_sharding(
                    actor_batch, self.data_sharding
                )

        return batch, actor_batch, new_rng

    def _sample_buffer(self, buffer, batch_size, **sample_kwargs):
        if buffer is self.replay_buffer and self.replay_buffers is not None:
            if batch_size not in self._robot_batches:
                self._robot_batches[batch_size] = buffer.allocate_sample_batch(batch_size)
            raw = sample_robot_buffers(self.replay_buffers, batch_size, self._robot_rng,
                                       out=self._robot_batches[batch_size], **sample_kwargs)
        else:
            raw = buffer.sample_jax(batch_size, **sample_kwargs)
        if raw is None:
            return None
        return buffer._convert_to_openpi_format(raw)

    def _online_batches(self, batch_size, **sample_kwargs):
        # Sample only when requested: no prefetch from the previous policy round.
        while True:
            yield self._sample_buffer(self.replay_buffer, batch_size, **sample_kwargs)

    def _sample_success_actor_batch(self, rng):
        if self.offline_ratio == 0:
            return self._sample_buffer(self.replay_buffer, self.batch_size, success_only=True)

        online_part = self._sample_buffer(
            self.replay_buffer, self._online_actor_bs, success_only=True
        )
        if online_part is not None:
            offline_part = self._sample_buffer(
                self.offline_replay_buffer, self._offline_actor_bs, success_only=True
            )
            if offline_part is not None:
                shuffle_key, _ = jax.random.split(rng)
                return combine_batches(online_part, offline_part, rng=shuffle_key)

        return self._sample_buffer(
            self.offline_replay_buffer, self.batch_size, success_only=True
        )


def sample_robot_buffers(buffers, batch_size, rng, *, hil_only=False, success_only=False, out=None):
    """Sample uniformly over eligible rows, building each n-step chunk locally."""
    counts = []
    for buffer in buffers:
        end = max(0, len(buffer) - buffer._replan_steps)
        key = "hil_chunk" if hil_only else "is_success" if success_only else None
        counts.append(np.count_nonzero(buffer.dataset_dict[key][:end]) if key else end)
    if not sum(counts) or batch_size == 0:
        if success_only:
            return None
        raise ValueError("No eligible robot replay samples available")
    allocation = rng.multinomial(batch_size, np.asarray(counts) / sum(counts))
    if out is None:
        out = buffers[0].allocate_sample_batch(batch_size)
    start = 0
    for buffer, n in zip(buffers, allocation):
        if n:
            stop = start + int(n)
            buffer.sample_jax(int(n), hil_only=hil_only, success_only=success_only,
                              out={key: value[start:stop] for key, value in out.items()})
            start = stop
    # The permutation copies the completed batch out of the reusable host arrays.
    order = rng.permutation(batch_size)
    return jax.tree.map(lambda value: value[order], out)
