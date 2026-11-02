"""A VectorEnv over N hosted BeamNG.tech games, stepped over ZeroMQ.

Every endpoint is a game host running ``host_game.py`` (which owns the game
and the tcom interaction loop, and serves it over ZeroMQ). The sub-envs are
:class:`RemoteBeamNGEnv` (connect-only): they send the action to the game
host and receive the observation it computed. The LiDAR point cloud never
crosses the network (it is read from shared memory on the game's host), and
only the small observation is sent back. The sub-environments are stepped in
parallel by :class:`gymnasium.vector.AsyncVectorEnv`.

Usage::

    ./host_game.py 192.168.1.50 5555    # game host 1
    ./host_game.py 192.168.1.51 5556    # game host 2
    venv = BeamNGTechHostedVectorEnv(
        [("192.168.1.50", 5555), ("192.168.1.51", 5556)])
    obs, info = venv.reset(seed=0)      # waits for all hosts, then steps them
"""
import functools
import socket
import time

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from beamng_tech_env.envs.remote import RemoteBeamNGEnv


def wait_for_port(host, port, timeout=300.0, poll=1.0):
    """Block until something accepts TCP connections on ``host:port``.

    The game host binds its ZeroMQ socket as soon as it starts (before the
    game is up); this waits for it so the sub-env connects on the first reset
    instead of failing fast.
    """
    deadline = time.monotonic() + timeout
    waiting = False
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=poll):
                return
        except OSError:
            if not waiting:
                print(f"waiting for {host}:{port} to open its ZeroMQ port...")
                waiting = True
            time.sleep(poll)
    raise TimeoutError(f"nothing listening on {host}:{port} after {timeout:.0f}s")


class EvalRoutingWrapper(gym.Wrapper):
    """Expose a single sub-env to eval, routed through the vector env.

    The hosted vector env is stepped as a whole, so eval cannot drive one
    sub-env directly. Instead every sub-env exposes ``_eval_reset`` /
    ``_eval_step``; the eval callback calls them on all sub-envs (via
    :meth:`AsyncVectorEnv.call`) passing the active index, and only the
    sub-env whose index matches acts - the rest return ``None``.

    The eval reset uses the env's special eval reset (``options={"eval":
    True}`` - a fresh random waypoint rather than a checkpoint resume), which
    the stack-based detection (``reset_is_eval``) cannot reach across the
    network (it runs on the game's host, where ``evaluate_policy`` is not on
    the call stack).
    """

    def __init__(self, env, index):
        super().__init__(env)
        self._eval_index = index

    def _eval_reset(self, active_index):
        if active_index != self._eval_index:
            return None
        return self.env.reset(seed=None, options={"eval": True})

    def _eval_step(self, active_index, action):
        if active_index != self._eval_index:
            return None
        return self.env.step(action)


def _make_sub_env(host, zmq_port, sub_env_wrapper=None, index=0):
    """Build one sub-env (a ZeroMQ client) and apply the wrappers.

    Runs in the :class:`AsyncVectorEnv` worker process, so it must be
    picklable (cloudpickle handles closures and ``functools.partial``). The
    :class:`EvalRoutingWrapper` is always outermost so a single sub-env can be
    driven for eval (see that class).
    """
    env = RemoteBeamNGEnv(host=host, zmq_port=zmq_port)
    if sub_env_wrapper is not None:
        env = sub_env_wrapper(env)
    return EvalRoutingWrapper(env, index)


class BeamNGTechHostedVectorEnv(AsyncVectorEnv):
    """One sub-env per game host (ZeroMQ client), stepped in parallel.

    Parameters
    ----------
    endpoints:
        Sequence of ``(host, zmq_port)`` pairs; each is a game host running
        ``host_game.py`` (which owns the game and serves it over ZeroMQ).
    sub_env_wrapper:
        Optional callable applied to every sub-env after construction (e.g.
        ``BeamsWrapper``); per-sub-env wrappers must be applied here because
        they cannot wrap the vector env itself.
    shared_memory:
        Pass-through to :class:`AsyncVectorEnv` (default True).
    host_wait_timeout:
        Seconds ``reset`` waits for each host's ZeroMQ port before giving up.
    """

    def __init__(self, endpoints, sub_env_wrapper=None, shared_memory=True,
                 host_wait_timeout=300.0):
        self.endpoints = list(endpoints)
        self.host_wait_timeout = host_wait_timeout
        super().__init__(
            [functools.partial(_make_sub_env, host=host, zmq_port=port,
                               sub_env_wrapper=sub_env_wrapper, index=i)
             for i, (host, port) in enumerate(self.endpoints)],
            shared_memory=shared_memory,
            # The workers run the cupy beam math, but importing this package
            # has already initialized CUDA in the parent. fork/forkserver
            # inherit that (fork-unsafe) CUDA context and die on the first cupy
            # op; spawn starts each worker fresh so it initializes CUDA itself.
            context="spawn",
        )

    def reset(self, *, seed=None, options=None):
        # the sub-envs only connect; wait for every host to open its ZeroMQ
        # port so the first reset does not fail fast
        for host, port in self.endpoints:
            wait_for_port(host, port, timeout=self.host_wait_timeout)
        return super().reset(seed=seed, options=options)
