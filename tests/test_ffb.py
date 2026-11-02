"""Test that (virtual) steering-wheel force feedback can be read from the game.

The game computes the wheel FFB in ``lua/vehicle/hydros.lua`` and stores it in
``hydros.forceAtDriverNorm`` (normalised to [-1, 1]). It is not exposed as a
tcom/electrics sensor, so the env enables the virtual wheel
(``hydros.enableVirtualWheel(true)``) to make the FFB non-zero without any
physical wheel hardware, and reads it back via a lua command (the ``ffb``
observation channel).
"""
import pytest

from beamng_tech_env.core.constants import gear_to_action

pytestmark = pytest.mark.game


def test_ffb_channel_present(env):
    """The env must expose an ``ffb`` observation channel in its space."""
    obs, _ = env.reset(seed=0)
    assert "ffb" in obs, "missing 'ffb' observation channel"
    space = env.observation_space["ffb"]
    assert space.contains(obs["ffb"]), f"ffb value {obs['ffb']} outside space {space}"


def test_ffb_obtainable(env, n_steps=40):
    """Steering a moving car must produce a non-zero wheel force feedback.

    The env enables the virtual wheel on reset, so the ``ffb`` observation
    should become non-zero once the car is steered and moving (in a gear).
    """
    env.reset(seed=0)
    max_ffb = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()
        action[0] = 0.9                     # steer hard
        action[1] = 0.6                     # throttle
        action[2] = 0.0                     # release the parking brake
        action[3:] = gear_to_action(1)      # 1st gear (the action's encoding)
        obs, reward, terminated, truncated, _ = env.step(action)
        max_ffb = max(max_ffb, abs(float(obs["ffb"][0])))
        if terminated or truncated:
            break
    assert max_ffb > 0.01, (
        f"wheel force feedback never became non-zero: max |ffb| = {max_ffb:.4f}"
    )
