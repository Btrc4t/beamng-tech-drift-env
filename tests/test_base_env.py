"""Tests for the shared env bookkeeping (no game needed).

The base env is constructed directly with a fake ego (sensors + state), so
the reward-state logic - in particular the spawn damage check that
invalidates the saved spawn point - runs without launching BeamNG.drive.
"""
from types import SimpleNamespace

import numpy as np

from beamng_tech_env.core.constants import (
    BEV_N,
    CHECKPOINT_DAMAGE,
    SPAWN_DAMAGE_STEPS,
)
from beamng_tech_env.envs.base import BeamNGBaseEnv


def _fake_ego(damage=0.0, pos=(0.0, 0.0, 5.0)):
    """A stand-in for the ego vehicle: a State sensor snapshot and the
    Damage sensor, the only two things _update_reward_state reads."""
    return SimpleNamespace(
        state={
            "pos": list(pos),
            "vel": [0.0, 0.0, 0.0],
            "dir": [1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "time": 0.0,
        },
        sensors={"damage": {"damage": damage}},
    )


def test_spawn_damage_invalidates_checkpoint():
    """Damage within the first SPAWN_DAMAGE_STEPS steps of an episode
    invalidates the saved spawn point (the car hit something right after
    spawning), so the next reset picks a waypoint instead of respawning
    at the bad pose."""
    env = BeamNGBaseEnv()
    try:
        env.ego = _fake_ego()
        env._checkpoint = ((1.0, 2.0, 5.0), (1.0, 0.0, 0.0, 0.0))
        # step 1: undamaged, the checkpoint is re-saved at the current pose
        env._update_reward_state()
        assert env._checkpoint is not None
        # step 2: damage (>= CHECKPOINT_DAMAGE, so the checkpoint would
        # otherwise persist at the spawn pose) -> invalidated
        env.ego.sensors["damage"]["damage"] = CHECKPOINT_DAMAGE + 0.5
        env._update_reward_state()
        assert env._checkpoint is None
    finally:
        env.close()


def test_spawn_damage_at_window_edge_invalidates():
    """The window is inclusive: damage at exactly step SPAWN_DAMAGE_STEPS
    still invalidates the saved spawn point."""
    env = BeamNGBaseEnv()
    try:
        env.ego = _fake_ego()
        env._checkpoint = ((1.0, 2.0, 5.0), (1.0, 0.0, 0.0, 0.0))
        for _ in range(SPAWN_DAMAGE_STEPS - 1):
            env._update_reward_state()
        assert env._checkpoint is not None
        env.ego.sensors["damage"]["damage"] = CHECKPOINT_DAMAGE + 0.5
        env._update_reward_state()
        assert env._checkpoint is None
    finally:
        env.close()


def test_damage_after_spawn_window_keeps_checkpoint():
    """Damage past the first SPAWN_DAMAGE_STEPS steps is a normal mid-episode
    hit, not a bad spawn: the saved spawn point (the last undamaged pose)
    is kept."""
    env = BeamNGBaseEnv()
    try:
        env.ego = _fake_ego()
        env._checkpoint = None
        for _ in range(SPAWN_DAMAGE_STEPS):
            env._update_reward_state()   # undamaged: re-saved every step
        assert env._checkpoint is not None
        saved = env._checkpoint
        env.ego.sensors["damage"]["damage"] = CHECKPOINT_DAMAGE + 0.5
        env._update_reward_state()       # step SPAWN_DAMAGE_STEPS + 1
        assert env._checkpoint == saved
    finally:
        env.close()


def test_spawn_damage_counter_resets_on_finish_reset():
    """_finish_reset anchors the new episode: the step counter starts at 0,
    so the spawn window applies to the fresh spawn."""
    env = BeamNGBaseEnv()
    try:
        env.ego = _fake_ego()
        for _ in range(SPAWN_DAMAGE_STEPS):
            env._update_reward_state()
        assert env._steps_since_reset == SPAWN_DAMAGE_STEPS
        # _finish_reset builds the reset observation via the rig; the bare
        # rig has no sensors, so stub it
        env.rig.get_obs = lambda: {"bev": np.zeros((1, BEV_N, BEV_N),
                                                   np.uint8)}
        env._finish_reset()
        assert env._steps_since_reset == 0
    finally:
        env.close()
