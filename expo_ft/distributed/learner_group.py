"""Coordination inside a JAX learner job; WAN transport remains on rank zero.

Small control results and complete recorded rounds are broadcast once. Replay
sampling remains local on each host. All ranks execute updates and checkpoints
in the same order. Launch with srun --kill-on-bad-exit=1 so a failed rank cannot
leave peers waiting indefinitely in a collective.
"""
import os

import jax
import numpy as np

from .channel import encode, decode


def initialize_learner(flags):
    count = int(os.environ.get('EXPO_PROCESS_COUNT', '1'))
    if count == 1:
        return
    if flags.split_role != 'learner' or flags.fsdp_devices != 1:
        raise ValueError('Multi-host execution requires split learner and fsdp_devices=1')
    if flags.offline_ratio != 0 or flags.num_robot != 2:
        raise ValueError('Multi-host learner currently supports two robots and offline_ratio=0 (demo-seeded replay)')
    jax.distributed.initialize(
        coordinator_address=os.environ['EXPO_COORDINATOR'], num_processes=count,
        process_id=int(os.environ['EXPO_PROCESS_ID']),
        local_device_ids=list(range(int(os.environ.get('EXPO_LOCAL_DEVICE_COUNT', '4')))),
        initialization_timeout=180,
    )


def local_value(value):
    """Fetch one local replica without gathering the same parameters from peers."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if not value.is_fully_replicated:
            raise ValueError('Expected a replicated learner value (fsdp_devices=1)')
        value = value.addressable_shards[0].data
    return np.asarray(jax.device_get(value))


def replicate(value, sharding):
    """Replicate a local scalar/key over the global mesh on every process."""
    if sharding is None or sharding.is_fully_addressable:
        return jax.device_put(value, sharding)
    return jax.jit(lambda x: x, in_shardings=sharding, out_shardings=sharding)(value)


class LearnerGroup:
    def __init__(self):
        self.size = jax.process_count()
        self.leader = jax.process_index() == 0

    def call(self, function):
        """Run an external operation only on rank zero; share result or error.

        A fixed transfer shape avoids recompiling for every episode length.
        The payload uses the same array encoding as the existing local channel.
        No files or extra model snapshots are created between learner nodes.
        """
        if self.size == 1:
            return function()
        from jax.experimental import multihost_utils
        result, failure, payload = None, None, b''
        if self.leader:
            try:
                result = function()
                payload = encode({'result': result})
            except Exception as exc:
                failure = exc
                payload = encode({'error': f'{type(exc).__name__}: {exc}'})
        broadcast = multihost_utils.broadcast_one_to_all
        # uint32 words work with JAX's default x64-disabled mode.
        length = len(payload)
        words = broadcast(np.array([length >> 32, length & 0xffffffff], np.uint32),
                          is_source=self.leader)
        length = (int(words[0]) << 32) | int(words[1])
        chunk_size = 1024 if length <= 1024 else 4 * 1024**2
        received = bytearray() if not self.leader else None
        chunk = np.zeros(chunk_size, np.uint8)
        for offset in range(0, length, chunk_size):
            size = min(chunk_size, length - offset)
            if self.leader:
                chunk[:size] = np.frombuffer(payload, np.uint8, count=size, offset=offset)
            # broadcast_one_to_all uses psum, which promotes uint8 to uint32.
            # Restore bytes before extending the serialized message.
            data = np.asarray(broadcast(chunk, is_source=self.leader), dtype=np.uint8)
            if received is not None:
                received.extend(memoryview(data)[:size])
        if not self.leader:
            message = decode(received)
            if 'error' in message:
                raise RuntimeError('Learner leader failed: ' + message['error'])
            return message['result']
        if failure is not None:
            raise failure
        return result

    def barrier(self, name):
        if self.size > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(name)
