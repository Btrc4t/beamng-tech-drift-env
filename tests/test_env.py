"""Gymnasium API tests for BeamNGTechEnv.

The env launches BeamNG.drive itself (``beamng.open()`` in ``_first_time_setup``),
so no manual game start is needed. These tests share one game instance for the
whole session (see the ``env`` fixture in ``conftest.py``) and each call
``env.reset()`` for a clean state.
"""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import beamng_tech_env  # noqa: F401  (registers the env id)
from beamng_tech_env.core.constants import BEV_N

pytestmark = pytest.mark.game


def validate_obs(env, obs):
    assert isinstance(obs, dict), f"observation must be a dict, got {type(obs)}"
    for key, space in env.observation_space.items():
        assert key in obs, f"missing observation key: {key}"
        val = obs[key]
        assert val.shape == space.shape, f"{key}: shape {val.shape} != expected {space.shape}"
        assert val.dtype == space.dtype, f"{key}: dtype {val.dtype} != expected {space.dtype}"
        assert space.contains(val), f"{key}: value outside declared space"
    extra = set(obs) - set(env.observation_space)
    assert not extra, f"unexpected observation keys: {extra}"


def test_conformance(env):
    """Official Gymnasium API conformance check."""
    check_env(env, skip_render_check=True)


def test_smoke(env, n_steps=30):
    """Reset + N steps, validating every observation against its space."""
    obs, info = env.reset(seed=0)
    validate_obs(env, obs)

    total_reward = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        validate_obs(env, obs)
        assert isinstance(reward, (float, np.floating)), f"reward not float: {type(reward)}"
        assert isinstance(terminated, (bool, np.bool_)), f"terminated not bool: {type(terminated)}"
        assert isinstance(truncated, (bool, np.bool_)), f"truncated not bool: {type(truncated)}"
        total_reward += float(reward)
        if terminated or truncated:
            break
    # reaching this point without raising is the pass condition


def test_debug_window(env, n_steps=10):
    """The live debug window wraps the env and streams frames without slowing
    the loop. Only the window is closed (the shared game env stays alive for
    the rest of the session)."""
    from beamng_tech_env.debug import DebugWindowWrapper
    wrapped = DebugWindowWrapper(env)
    try:
        obs, _ = wrapped.reset(seed=0)
        assert obs["bev"].shape == (1, BEV_N, BEV_N)
        for _ in range(n_steps):
            obs, _, term, trunc, _ = wrapped.step(env.action_space.sample())
            if term or trunc:
                break
        assert obs["bev"].shape == (1, BEV_N, BEV_N)
    finally:
        wrapped.window.close()


def test_diagnose(env):
    """Every observation key must be present and inside its declared space."""
    obs, info = env.reset(seed=0)
    validate_obs(env, obs)


def test_ai_mode(ai_env, n_steps=200):
    """Synthetic mode: the game's built-in AI drives the spawned vehicle.

    The env's ``ai_mode`` is set on the vehicle at reset (beamngpy's
    ``Vehicle.ai``), so the env sends no tcom Control: the game runs freely at
    real time and the AI steers/throttles/shifts on its own. We step with a
    dummy zero action only to advance and pace the loop, and the applied
    action is read back from the electrics sensor each step.
    """
    obs, info = ai_env.reset(seed=0)
    validate_obs(ai_env, obs)

    start = np.asarray(ai_env.unwrapped.ego.state["pos"], float)
    max_applied = 0.0
    for _ in range(n_steps):
        action = np.zeros(ai_env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = ai_env.step(action)
        validate_obs(ai_env, obs)
        assert "applied_action" in info, "AI mode must read back the applied action"
        max_applied = max(max_applied, float(np.max(np.abs(info["applied_action"]))))
        if terminated or truncated:
            break
    end = np.asarray(ai_env.unwrapped.ego.state["pos"], float)
    distance = float(np.linalg.norm(end - start))
    # the env sent no tcom Control, so a moving car proves the built-in AI
    # took over (and a non-zero applied action that it actually drove)
    assert distance > 5.0, f"the AI did not drive the car (moved {distance:.1f} m)"
    assert max_applied > 0.0, "the AI never applied any control input"


def test_human_resume_pose(human_env):
    """Human mode: the pose where an episode ends (crash or termination) is
    saved and restored on the next reset, so the driver resumes collecting
    from where they left off instead of a random waypoint."""
    env = human_env.unwrapped
    env._resume_pose = None  # start from a clean slate
    obs, info = env.reset(seed=0)
    # no saved end pose: the reset spawns at a random waypoint
    assert info["spawn"] != "resume"

    # save the current pose (as an episode end would), then reset: the car
    # must spawn at the saved pose, not a random waypoint (which is far away)
    env._save_resume_pose()
    saved_pos, _ = env._resume_pose
    obs, info = env.reset(seed=0)
    assert info["spawn"] == "resume"
    start = np.asarray(env.ego.state["pos"], float)
    assert np.linalg.norm(start - np.asarray(saved_pos, float)) < 5.0, \
        f"resumed at {start}, expected the saved pose {saved_pos}"
