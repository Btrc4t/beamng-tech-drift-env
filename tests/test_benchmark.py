"""End-to-end benchmark of the single-vehicle environment, as pytest tests.

Each test drives the real game (the shared session instance from
``conftest.py``) and asserts a meaningful property of one benchmark section.
Together they exercise the full pipeline: reset -> step -> sensors ->
observation -> reward -> drift parity. The standalone tool
``benchmark_single.py`` prints the same numbers without the assertions.
"""
import time

import pytest

from beamng_tech_env.core.constants import BEV_ROAD_LEVEL, DRIFT_CARS_6SPEED
from benchmark_single import (
    check_determinism,
    drift_parity,
    headline_fps,
    phase_breakdown,
)

pytestmark = pytest.mark.game


def _reset(env):
    # Pin the bx drift car (the scripted drift schedule is tuned for it) and
    # let the sensors warm up, mirroring benchmark_single.main().
    env.reset(seed=0, options={"car": DRIFT_CARS_6SPEED[0]})
    time.sleep(2)


def test_headline(env):
    """The pipelined step() must produce a positive, finite step rate."""
    _reset(env)
    fps, ms = headline_fps(env)
    assert fps > 0, f"non-positive FPS: {fps}"
    assert ms > 0, f"non-positive ms/step: {ms}"


def test_phase_breakdown(env):
    """Every phase must run, and the stale-read fallback must rarely trigger."""
    _reset(env)
    phase, total_ms, repolls = phase_breakdown(env)
    assert total_ms > 0, f"non-positive total ms/step: {total_ms}"
    assert all(t >= 0 for t in phase.values()), f"negative phase time: {phase}"
    # the batched step should not fall back to a re-poll on most steps
    assert repolls < 50, f"too many stale repolls: {repolls}"


def test_determinism(env):
    """step(1) must advance exactly 50 ms of sim time (deterministic physics)."""
    _reset(env)
    worst = check_determinism(env)
    assert worst < 1e-4, (
        f"sim time did not advance exactly 50 ms/step: worst |dt-50ms| = {worst * 1000:.3f} ms"
    )


def test_lidar(env):
    """The LiDAR must return points and the BEV grid must have filled cells."""
    _reset(env)
    raw = env.unwrapped
    pc = raw._poll_lidar_points()
    bev = raw._get_obs()["bev"]
    n_cells = int((bev != BEV_ROAD_LEVEL).sum())
    assert pc.shape[0] > 0, "LiDAR returned no points"
    assert n_cells > 0, "BEV grid has no filled cells"


def test_drift_parity(env):
    """drift_from_state must broadly agree with the game's Lua drift telemetry."""
    _reset(env)
    metrics = drift_parity(env)
    assert metrics["moving"] > 0, "car never moved during the scripted drift"
    assert metrics["agreement"] > 50.0, (
        f"drift telemetry diverges from the game: {metrics['agreement']:.1f}% agreement"
    )
