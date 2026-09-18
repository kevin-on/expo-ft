"""Rollout client for RL training / eval.

Dials the learner/eval host (which listens) and serves environment operations over
that connection — a direct app-owned socket (no SSH tunnel). We dial because the
robot workstation can't accept inbound connections while the learner/eval host can.
Supports operations: create_env, reset, step, get_observation, get_info_for_step, render.
"""

import asyncio
import dataclasses
import logging
import os
import socket
import time
from typing import Dict, Any, Optional

import numpy as np
import websockets
import websockets.asyncio.client as _client
from openpi_client import msgpack_numpy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import tyro

def load_task_config(config_path: Optional[str]):
    """Load task config from a FILE PATH (or a dotted module path), like config_flags.

    A .py path is loaded by location, not turned into a dotted module name: config file
    names carry speed factors ("hang0.5speed.py") and a dot in the stem is a package
    separator to __import__, so the dotted route cannot import them at all.
    """
    if config_path is None:
        return None

    if config_path.endswith('.py'):
        import importlib.util
        path = os.path.abspath(config_path)
        if not os.path.exists(path):
            raise ImportError(f"Task config '{config_path}' does not exist (looked at {path})")
        # A name, not a module path: the stem may hold dots, and nothing imports it back.
        spec = importlib.util.spec_from_file_location(
            f"_task_config_{os.path.splitext(os.path.basename(path))[0]}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to load task config from '{config_path}': {e}")
    else:
        try:
            module = __import__(config_path.replace('/', '.'), fromlist=['get_config'])
        except Exception as e:
            raise ImportError(f"Failed to load task config from '{config_path}': {e}")

    return module.get_config()


@dataclasses.dataclass
class Args:
    """Configuration arguments for the rollout client.

    Dials the learner/eval host (which listens) and serves environment operations
    over that connection. A direct app-owned socket (no SSH tunnel), so TCP_NODELAY
    keeps per-step latency flat.
    """

    host: str = "localhost"
    port: int = 8102
    config_task_path: str = "configs/task/pick.py"
    step_timing_threshold_ms: float = 30.0


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_task_config: Optional[Any] = None
# Monotonic suffix so each create_env yields a distinct env_id. Without this, two trainers that share
# one server (same env_name+env_usage) collide on a single env instance and corrupt each other's
# rollouts. Per-job unique ports (see train_robo_car_launch.sh) prevent sharing; this isolates envs
# even if a port is ever reused.
_env_create_counter: int = 0
_step_timing_threshold_ms: float = 30.0

# Human-in-the-loop: lazy spacemouse for droid envs
_spacemouse_policy: Optional[Any] = None
_HUMAN_OVERRIDE_NORM_THRESHOLD = 1e-4


def _get_human_override_action(task_config: Optional[Any] = None) -> tuple:
    """Return (action_7d or None, is_human). Assumes 7D action space."""
    global _spacemouse_policy
    try:
        if _spacemouse_policy is None:
            from client.real_utils.spacemouse import SpaceMousePolicy
            _spacemouse_policy = SpaceMousePolicy(
                max_lin_vel=task_config.collect_max_lin_vel,
                max_rot_vel=task_config.collect_max_rot_vel,
            )
        action_7d, _ = _spacemouse_policy.forward(None, include_info=True)
        is_active = np.linalg.norm(action_7d[:6]) > _HUMAN_OVERRIDE_NORM_THRESHOLD
        return (action_7d, True) if is_active else (None, False)
    except Exception as e:
        logging.getLogger(__name__).warning("Spacemouse unavailable (%s), using policy action.", e)
        return None, False


async def _handle_environment_request(websocket):
    """Serve environment operation requests over the dialed connection until it closes."""
    global _task_config, _env_create_counter
    logger = logging.getLogger(__name__)
    packer = msgpack_numpy.Packer()

    # Disable Nagle on this socket: tiny response frames (info/step) otherwise stall
    # ~40ms on the delayed-ACK timer. Effective on a direct app-owned socket.
    try:
        sock = websocket.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.warning("Could not set TCP_NODELAY on socket: %s", e)

    try:
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                operation = request.get("operation")
                
                if operation == "create_env":
                    task_config = load_task_config(_config_task_path)
                    _task_config = task_config
                    env_name = task_config.env_name
                    env_usage = request["env_usage"]
                    _env_create_counter += 1
                    env_id = f"{env_name}_{env_usage}_{_env_create_counter}"
                    
                    logger.info(f"Creating environment {env_id}...")
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env_kwargs["env_usage"] = env_usage
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    logger.info(f"Environment {env_id} created successfully")
                    
                    task_description = task_config.language_instruction
                    response = {"status": "success", "env_id": env_id, "task_description": task_description}
                    await websocket.send(packer.pack(response))
                    logger.info(f"Sent create_env response for {env_id}")
                    
                elif operation == "reset":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.reset()
                        response = {
                            "status": "success",
                            "observation": obs,
                            "done": False,
                        }
                    await websocket.send(packer.pack(response))
                    
                elif operation == "step":
                    step_rpc_start = time.perf_counter()
                    env_id = request["env_id"]
                    sent_action = np.array(request["action"])
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        sent_action = sent_action.astype(np.float64)
                        if not np.isfinite(sent_action).all():
                            logger.warning(
                                "Action contains NaN/Inf; replacing with zeros. "
                                "Check policy inputs (observations, encoder), training stability, or checkpoint."
                            )
                            sent_action = np.where(np.isfinite(sent_action), sent_action, 0.0)
                        real_action = sent_action.copy()
                        action_type = "policy"
                        is_human = False
                        t_human0 = time.perf_counter()
                        if _task_config is not None and _task_config.env_type == "droid":
                            sm_action, is_human = _get_human_override_action(_task_config)
                            if is_human and sm_action is not None:
                                real_action[:6] = sm_action[:6]
                                real_action[6] = sm_action[6]
                                action_type = "human"
                        human_ms = (time.perf_counter() - t_human0) * 1000.0
                        sent_is_invalid = np.allclose(sent_action, -1.0)
                        if is_human or not sent_is_invalid:
                            t_env_step0 = time.perf_counter()
                            step_result = env.step(real_action)
                            env_step_ms = (time.perf_counter() - t_env_step0) * 1000.0
                            executed_action = np.array(
                                step_result["executed_action"],
                                dtype=np.float64,
                            )
                        else:
                            env_step_ms = 0.0
                            executed_action = real_action

                        response = {
                            "status": "success",
                            "action": executed_action.tolist(),
                            "action_type": action_type,
                        }
                    t_send0 = time.perf_counter()
                    await websocket.send(packer.pack(response))
                    step_rpc_ms = (time.perf_counter() - step_rpc_start) * 1000.0
                    send_ms = (time.perf_counter() - t_send0) * 1000.0
                    if operation == "step" and step_rpc_ms >= _step_timing_threshold_ms:
                        logger.warning(
                            "[timing][server step] env_id=%s total=%.1fms human=%.1f env_step=%.1f send=%.1f "
                            "action_type=%s invalid=%s",
                            env_id,
                            step_rpc_ms,
                            human_ms if env is not None else 0.0,
                            env_step_ms if env is not None else 0.0,
                            send_ms,
                            action_type if env is not None else "missing_env",
                            sent_is_invalid if env is not None else False,
                        )

                elif operation == "get_observation":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.get_observation()
                        # Piggyback termination/detection on the (large) obs response so the
                        # client doesn't need a separate tiny get_info_for_step round-trip.
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "observation": obs,
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_info_for_step":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "render":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    elif not hasattr(env, "render"):
                        response = {"status": "error", "message": "Environment does not support render"}
                    else:
                        response = {
                            "status": "success",
                            "frame": np.asarray(env.render()),
                        }
                    await websocket.send(packer.pack(response))

                else:
                    response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    await websocket.send(packer.pack(response))
            
            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Connection closed by client {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                try:
                    response = {"status": "error", "message": str(e)}
                    await websocket.send(packer.pack(response))
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Connection closed while sending error response")
                    break
                
    except websockets.exceptions.ConnectionClosed:
        logger.debug(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Unexpected error in request handler: {e}", exc_info=True)


async def _run_client(
    host: str,
    port: int,
    config_task_path: Optional[str],
    step_timing_threshold_ms: float,
):
    """Dial the learner/eval host (which listens) and serve operations over that
    connection; redial on disconnect. Direct app-owned socket, so TCP_NODELAY is
    effective and per-step latency stays flat."""
    global _config_task_path, _step_timing_threshold_ms
    _config_task_path = config_task_path
    _step_timing_threshold_ms = step_timing_threshold_ms
    logger = logging.getLogger(__name__)
    uri = f"ws://{host}:{port}"
    while True:
        try:
            logger.info("Dialing learner/eval server at %s ...", uri)
            async with _client.connect(
                uri,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=100,
            ) as websocket:
                logger.info("Connected to %s; serving operations.", uri)
                await _handle_environment_request(websocket)
            logger.info("Connection closed; redialing in 5s ...")
        except (OSError, websockets.exceptions.WebSocketException) as e:
            logger.info("Server not reachable (%s); retrying in 5s ...", e)
        await asyncio.sleep(5)


async def main_async(args: Args) -> None:
    """Main async entry point."""
    await _run_client(
        args.host,
        args.port,
        args.config_task_path,
        args.step_timing_threshold_ms,
    )

def main(args: Args) -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
