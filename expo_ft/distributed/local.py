"""The split runners on one host, sharing the learner's GPU0 parameter buffers.

Only delivery changes: records use an in-process mailbox and policies use array
references. Replay, sampling, round barriers, UI, persistence and resume belong
to runner.py. There is no second model build, sidecar, relay or snapshot payload.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import dataclasses
import threading
import time


class LocalPeerError(RuntimeError):
    """The supervisor reports the original role exception after both exit."""


class LocalSession:
    """Two addressed mailboxes, consumed with the split Channel interface."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.condition = threading.Condition()
        self.messages = [{}, {}]
        self.closed = [False, False]
        self.aborted = False
        self.failure = None

    def endpoint(self, role):
        return LocalChannel(self, role)

    def abort(self, error):
        with self.condition:
            self.aborted = True
            if self.failure is None and error is not None and not isinstance(error, LocalPeerError):
                self.failure = error
            self.condition.notify_all()

    def check(self):
        if self.aborted:
            raise LocalPeerError('Colocated peer stopped') from self.failure


class LocalChannel:
    def __init__(self, session, role):
        self.session, self.role = session, role
        self.aborting = False

    def check_session(self):
        with self.session.condition:
            self.session.check()
            if self.session.closed[1 - self.role]:
                raise LocalPeerError('Colocated peer closed')

    def send(self, topic, key, value):
        if topic == 'abort':
            self.aborting = True
            self.session.abort(None)
            return
        # Match Channel.send ownership: subsequent producer mutation cannot
        # alter recorded observations/actions. No serialization or disk staging.
        self.send_reference(topic, key, deepcopy(value))

    def send_reference(self, topic, key, value):
        with self.session.condition:
            self.check_session()
            if self.session.closed[self.role]:
                raise RuntimeError('Colocated channel closed')
            messages = self.session.messages[1 - self.role]
            if (topic, key) in messages:
                raise ValueError('Duplicate local message: {} {}'.format(topic, key))
            messages[topic, key] = value
            self.session.condition.notify_all()

    def receive(self, topic, key, timeout=None, *, check=None):
        deadline = time.monotonic() + (self.session.timeout if timeout is None else timeout)
        while True:
            # Do not call application callbacks while holding the mailbox lock.
            if check is not None:
                check()
            with self.session.condition:
                self.session.check()
                messages = self.session.messages[self.role]
                if (topic, key) in messages:
                    return messages[topic, key]
                if self.session.closed[1 - self.role]:
                    raise LocalPeerError('Colocated peer closed before {} {}'.format(topic, key))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Waiting for local {} {}'.format(topic, key))
                self.session.condition.wait(min(.02, remaining))

    def release(self, topic, key):
        with self.session.condition:
            self.session.messages[self.role].pop((topic, key), None)

    def flush(self):
        with self.session.condition:
            if not self.aborting:
                self.session.check()

    def wait_sent(self, topic, key, timeout=None):
        # Publishing already transfers ownership to the peer's mailbox.
        self.flush()

    def close(self):
        with self.session.condition:
            self.session.closed[self.role] = True
            self.session.messages[self.role].clear()
            self.session.condition.notify_all()


def device_replica(value, device):
    """Select an existing full replica; never silently copy/gather parameters."""
    import jax

    if not isinstance(value, jax.Array):
        return value
    if not value.is_fully_replicated:
        raise ValueError('Colocated inference requires replicated parameters (fsdp_devices=1)')
    for shard in value.addressable_shards:
        if shard.device == device and shard.data.shape == value.shape:
            return shard.data
    raise ValueError('No complete parameter replica on the inference device')


def inference_view(learner, rng):
    """Lightweight inference state, with the exact same GPU buffers as learner.

    Its RNG is independent and it has no optimizer state. A view is only sampled
    before round_finished; subsequent updates may donate/invalidate its buffers.
    The next version replaces the view before sampling resumes.
    """
    import jax

    device = learner.actor.infer_device
    local = lambda tree: jax.tree.map(lambda value: device_replica(value, device), tree)
    actor_state = dataclasses.replace(
        learner.actor_train_state, params=local(learner.actor_train_state.params),
        step=local(learner.actor_train_state.step), opt_state=(), ema_params=None)
    def state_view(state):
        return state.replace(params=local(state.params), step=local(state.step), opt_state=())
    encoder = state_view(learner.batch_encoder)
    edit = state_view(learner.edit_actor)
    critic = state_view(learner.target_critic)
    return learner.replace(
        rng=jax.device_put(rng, learner.actor.infer_sharding),
        actor_train_state=actor_state, batch_encoder=encoder, edit_actor=edit,
        target_critic=critic, critic=None, temp=None, target_actor_params=None,
        rollout_cache=False,
        _infer_cache={'actor_train_state': actor_state, 'batch_encoder_params': encoder.params,
                      'edit_actor_params': edit.params, 'target_critic_params': critic.params})


class LocalPolicyExchange:
    def __init__(self, sender, receiver, session, seed):
        self.sender, self.receiver = sender, receiver
        self.session, self.seed = session, seed

    def publish(self, learner, contract, version):
        import jax
        from .protocol import key

        started = time.monotonic()
        view = inference_view(learner, jax.random.fold_in(jax.random.PRNGKey(self.seed), 7351))
        # Dispatch is asynchronous; the ready barrier must cover every policy leaf.
        jax.block_until_ready(view._infer_cache)
        self.sender.send_reference('policy', key(self.session, version),
                                   (contract, version, view))
        return {'policy_bytes': 0, 'local_policy_seconds': time.monotonic() - started}

    def install(self, previous, contract, version, *, check=None):
        from .protocol import key

        received_contract, received_version, view = self.receiver.receive(
            'policy', key(self.session, version), check=check)
        started = time.monotonic()
        if received_contract != contract or received_version != version:
            raise ValueError('Local policy identity/version mismatch')
        if previous is not None:
            view = view.replace(rng=previous.rng)
        return view, {'install_seconds': time.monotonic() - started}


def run_colocated(flags, agent, buffers, batch_processor, checkpoint_manager, checkpoint_dir,
                  save_checkpoint, start_step, resuming, replicated_sharding, mirror_robot,
                  *, env_factory=None):
    """Run the unmodified split round protocol, inference/TUI on the main thread."""
    import jax
    from .policy import identity
    from .protocol import task_contract
    from .runner import run_inference, run_learner

    if jax.process_count() != 1 or flags.fsdp_devices != 1:
        raise ValueError('Colocated mode requires one process and fsdp_devices=1')
    if agent.rollout_cache:
        raise ValueError('Build the colocated learner with rollout_cache=False')
    session = LocalSession(flags.split_timeout)
    learner_channel, inference_channel = session.endpoint(0), session.endpoint(1)
    policy = LocalPolicyExchange(learner_channel, inference_channel, flags.split_session, flags.seed)
    contract = identity(agent, task_contract(flags, mirror_robot))

    def learn():
        try:
            return run_learner(
                flags, agent, buffers, batch_processor, checkpoint_manager, checkpoint_dir,
                save_checkpoint, start_step, resuming, replicated_sharding, mirror_robot,
                channel=learner_channel, policy_exchange=policy, contract=contract)
        except BaseException as error:
            session.abort(error)
            raise
        finally:
            learner_channel.close()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-learner') as worker:
        future = worker.submit(learn)
        try:
            run_inference(flags, env_factory=env_factory, channel=inference_channel,
                          policy_exchange=policy, contract=contract)
        except BaseException as error:
            session.abort(error)
        finally:
            inference_channel.close()
        try:
            result = future.result()
        except BaseException as error:
            session.abort(error)
            if session.failure is None:
                raise
    if session.failure is not None:
        raise session.failure
    return result
