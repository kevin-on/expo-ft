"""Placement helpers shared by learner variants."""
import jax


def place_update_rng(learner, agent):
    """Restore inference's single-device RNG to the training mesh, preserving FSDP.

    Only RNG placement changes. Parameter arrays, optimizer state and RNG values
    are preserved, including the alias when learner and agent are the same object.
    """
    if learner.data_sharding is None or learner.data_sharding.num_devices == 1:
        return learner, agent
    replicated = jax.sharding.NamedSharding(
        learner.data_sharding.mesh, jax.sharding.PartitionSpec()
    )
    same_agent = agent is learner
    learner = learner.replace(rng=jax.device_put(learner.rng, replicated))
    agent = learner if same_agent else agent.replace(rng=jax.device_put(agent.rng, replicated))
    return learner, agent
