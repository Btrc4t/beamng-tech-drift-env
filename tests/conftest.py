"""Shared pytest configuration and fixtures for the BeamNG tech env tests.

All tests live in this folder. The pure-math tests (``test_drift_math.py``)
run without the game. The env and benchmark tests (marked ``game``) share a
single BeamNG.drive instance for the whole session: the game is launched (or
reused, if one is already running) on the first ``env.reset()`` and every test
calls ``env.reset()`` for a clean state.

Run only the fast tests (no game):
    pytest -m "not game"
Run everything (needs BNG_HOME, DISPLAY and XAUTHORITY set):
    pytest
"""
import os
import subprocess
import sys
import time

import pytest

# Make the package importable no matter where pytest is invoked from.
_GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _GYM_DIR not in sys.path:
    sys.path.insert(0, _GYM_DIR)

# Importing the package registers the env id used by gym.make below.
import beamng_tech_env  # noqa: F401


def _game_available():
    """Return (ok, reason): the game tests need the game home and an X display."""
    bng_home = os.environ.get("BNG_HOME", "")
    if not bng_home or not os.path.isdir(bng_home):
        return False, "BNG_HOME is not set or does not exist"
    if not os.environ.get("DISPLAY", ""):
        return False, "DISPLAY is not set"
    if not os.environ.get("XAUTHORITY", ""):
        return False, "XAUTHORITY is not set"
    return True, ""


def _kill_game():
    """Kill a leftover game process.

    Kept as a utility for a clean-slate run; the fixtures no longer call it
    (they reuse a game that is already running).
    """
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "BeamNG.tech.x64"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass
    time.sleep(1)


@pytest.fixture(scope="session")
def env():
    """One shared environment (and game) for the whole test session.

    The env reuses a game that is already running on the default tcom port
    (it connects to it on the first ``reset()``); if none is running it
    launches one. Each test is expected to call ``env.reset()`` for a clean
    state.
    """
    ok, reason = _game_available()
    if not ok:
        pytest.skip(f"game tests skipped: {reason}")
    import gymnasium as gym

    environment = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode=None)
    yield environment
    environment.close()


@pytest.fixture(scope="session")
def human_env():
    """A human-mode environment: a person drives with wheel/pedals (or keyboard).

    Used by the human-mode tests (the pose-resume test). It shares the same
    game as the other fixtures (``human_mode`` is an env-level setting, not a
    game-level one); the env recovers if another env on the same game
    despawns its ego.
    """
    ok, reason = _game_available()
    if not ok:
        pytest.skip(f"game tests skipped: {reason}")
    import gymnasium as gym

    environment = gym.make("beamng_tech_env/BeamNGTech-v0.39",
                           render_mode=None, human_mode=True)
    yield environment
    environment.close()


@pytest.fixture(scope="session")
def ai_env():
    """An AI (synthetic) mode environment: the game's built-in AI drives the
    spawned vehicle (its ``ai_mode`` is set on reset), so no human or agent
    input is needed.

    Used by the AI mode test, which steps the free-running game for a few
    seconds and checks that the car drives itself. It shares the same game as
    the other fixtures (``ai_mode`` is an env-level setting, not a game-level
    one); the env recovers if another env on the same game despawns its ego.
    """
    ok, reason = _game_available()
    if not ok:
        pytest.skip(f"game tests skipped: {reason}")
    import gymnasium as gym

    environment = gym.make("beamng_tech_env/BeamNGTech-v0.39",
                           render_mode=None, ai_mode=True)
    yield environment
    environment.close()


@pytest.fixture
def game_available():
    """Skip the test when the game is unavailable.

    For game tests that build their own env (the remote round-trip tests) and
    so do not go through the session-scoped ``env`` fixture above, which is
    what skips the other game tests. Without this they run (and fail) without
    the game, and a failed round trip leaks a ZeroMQ context whose ``term()``
    blocks pytest's final garbage collection, hanging the whole run."""
    ok, reason = _game_available()
    if not ok:
        pytest.skip(f"game test skipped: {reason}")
