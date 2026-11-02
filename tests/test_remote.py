"""Tests for the ZeroMQ remote env (envs/remote.py).

The serialization helpers are pure functions and run without the game. The
round-trip test (marked ``game``) launches a game via the server and steps a
remote client against it, verifying the observations match the env's spaces.
"""
import threading

import numpy as np
import pytest

from beamng_tech_env.core.constants import TCOM_PORT
from beamng_tech_env.core.obs import neutral_obs
from beamng_tech_env.envs.remote import (
    RemoteBeamNGEnv,
    RemoteGameServer,
    decode_value,
    encode_value,
)

import beamng_tech_env  # noqa: F401  (registers the env id)

# The RemoteGameServer's env uses tcom_port = zmq_port + 1, so pick the zmq
# port that puts the tcom port on the shared test game's port (the default
# TCOM_PORT): the server reuses the already-running game instead of launching
# a second one (a second instance on the same user path fails CefInitialize).
ZMQ_PORT = TCOM_PORT - 1


# -- serialization (no game) -------------------------------------------------

def _sample_obs():
    """A synthetic observation with the same layout as the env's."""
    obs = neutral_obs()
    obs["rpm"] = np.array([0.5], dtype=np.float32)
    obs["bev"] = (np.random.default_rng(0).random((1, 64, 64)) * 255).astype(np.uint8)
    return obs


def test_encode_decode_array():
    for dtype in (np.float32, np.float64, np.uint8, np.int32):
        for shape in ((1,), (3,), (1, 64, 64), (10,)):
            arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            out = decode_value(encode_value(arr))
            assert out.dtype == arr.dtype
            assert out.shape == arr.shape
            assert np.array_equal(out, arr)


def test_encode_decode_obs():
    obs = _sample_obs()
    out = decode_value(encode_value(obs))
    for key, value in obs.items():
        assert key in out, f"missing key {key}"
        assert out[key].dtype == value.dtype, f"{key}: dtype mismatch"
        assert out[key].shape == value.shape, f"{key}: shape mismatch"
        assert np.array_equal(out[key], value), f"{key}: value mismatch"


def test_encode_decode_info():
    info = {
        "reward_components": {"drift": 0.5, "dmg": -0.1},
        "game_fds": 300,
        "failure_reason": "damage",
        "applied_action": np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32),
        "spawn": "waypoint_1",
    }
    out = decode_value(encode_value(info))
    assert out["reward_components"] == info["reward_components"]
    assert out["game_fds"] == info["game_fds"]
    assert out["failure_reason"] == info["failure_reason"]
    assert np.array_equal(out["applied_action"], info["applied_action"])
    assert out["spawn"] == info["spawn"]


# -- server env wrapping (no game) -------------------------------------------

def test_server_debug_window_wraps_env():
    """debug_window=True wraps the env in a DebugWindowWrapper (no game)."""
    pytest.importorskip("cv2")
    from beamng_tech_env.debug import DebugWindowWrapper
    from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv

    server = RemoteGameServer(zmq_port=5590, debug_window=True)
    try:
        assert isinstance(server.env, DebugWindowWrapper)
        assert isinstance(server.env.env, BeamNGTechEnv)
    finally:
        server.close()


def test_server_default_no_debug_window():
    """By default the env is not wrapped (no game)."""
    from beamng_tech_env.debug import DebugWindowWrapper
    from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv

    server = RemoteGameServer(zmq_port=5591)
    try:
        assert not isinstance(server.env, DebugWindowWrapper)
        assert isinstance(server.env, BeamNGTechEnv)
    finally:
        server.close()


# -- round trip (needs the game) ---------------------------------------------

@pytest.mark.game
def test_remote_roundtrip(game_available):
    """The server (game host) and client (training host) agree end to end."""
    from beamng_tech_env.envs.base import make_observation_space

    server = RemoteGameServer(zmq_port=ZMQ_PORT)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    client = None
    try:
        client = RemoteBeamNGEnv(host="localhost", zmq_port=ZMQ_PORT)
        obs, info = client.reset(seed=0)
        _validate_obs(client.observation_space, obs)
        assert isinstance(info, dict)

        for _ in range(10):
            action = client.action_space.sample()
            obs, reward, terminated, truncated, info = client.step(action)
            _validate_obs(client.observation_space, obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                break
    finally:
        # close the client even if reset/step failed: an open ZeroMQ context
        # blocks pytest's final garbage collection (zmq_ctx_term)
        if client is not None:
            client.close()
        server.stop()
        thread.join(timeout=30)
        server.close()


@pytest.mark.game
def test_hosted_vector_env(game_available):
    """The hosted vector env (AsyncVectorEnv of remote clients) works."""
    from beamng_tech_env.envs.hosted_vector import BeamNGTechHostedVectorEnv

    server = RemoteGameServer(zmq_port=ZMQ_PORT)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    venv = None
    try:
        venv = BeamNGTechHostedVectorEnv([("localhost", ZMQ_PORT)])
        obs, info = venv.reset(seed=0)
        # batched observation: one env, so a leading batch dim of 1
        assert obs["bev"].shape == (1, 1, 64, 64)
        for _ in range(5):
            obs, rewards, terminated, truncated, info = venv.step(
                venv.action_space.sample())
            assert rewards.shape == (1,)
            assert terminated.shape == (1,)
            assert truncated.shape == (1,)
    finally:
        # close the venv even if reset/step failed: an open ZeroMQ context
        # blocks pytest's final garbage collection (zmq_ctx_term)
        if venv is not None:
            venv.close()
        server.stop()
        thread.join(timeout=30)
        server.close()


def _validate_obs(observation_space, obs):
    assert isinstance(obs, dict), f"observation must be a dict, got {type(obs)}"
    for key, space in observation_space.items():
        assert key in obs, f"missing observation key: {key}"
        val = obs[key]
        assert val.shape == space.shape, f"{key}: shape {val.shape} != {space.shape}"
        assert val.dtype == space.dtype, f"{key}: dtype {val.dtype} != {space.dtype}"
    extra = set(obs) - set(observation_space)
    assert not extra, f"unexpected observation keys: {extra}"
