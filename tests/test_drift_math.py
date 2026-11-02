"""Pure-math tests for the drift telemetry computed from the State sensor.

These tests do not launch BeamNG.drive; they only exercise ``drift_from_state``,
which is the function that replaced the per-frame Lua round trip.
"""
import math

from beamng_tech_env.core.constants import (
    DRIFT_MAX_ANGLE,
    DRIFT_MIN_ANGLE,
    DRIFT_MIN_AIRSPEED,
)
from beamng_tech_env.core.obs import drift_from_state

KPH_TO_MS = 1.0 / 3.6


def state_at(angle_deg, speed_kph, heading=(0, 1, 0), up=(0, 0, 1)):
    """Build a State dict with the velocity rotated `angle_deg` from heading."""
    v = speed_kph * KPH_TO_MS
    a = math.radians(angle_deg)
    # rotate the heading toward +X by `angle` (a right-hand turn)
    vx, vy, vz = v * math.sin(a), v * math.cos(a), 0.0
    return {"vel": [vx, vy, vz], "dir": list(heading), "up": list(up)}


def test_stationary():
    r = drift_from_state({"vel": [0, 0, 0], "dir": [0, 1, 0], "up": [0, 0, 1]})
    assert r["angle"] == 0.0
    assert r["isDrifting"] is False
    assert r["scoreAdded"] == 0.0


def test_straight():
    r = drift_from_state(state_at(0.0, 100.0))
    assert abs(r["angle"]) < 1e-6
    assert r["isDrifting"] is False
    assert r["scoreAdded"] == 0.0


def test_drift_right():
    r = drift_from_state(state_at(45.0, 100.0))
    assert abs(abs(r["angle"]) - 45.0) < 1e-3
    assert r["isDrifting"] is True
    assert r["scoreAdded"] > 0.0
    assert r["angle"] < 0.0  # right drift is negative (game convention)


def test_drift_left():
    r = drift_from_state(state_at(-45.0, 100.0))
    assert abs(abs(r["angle"]) - 45.0) < 1e-3
    assert r["isDrifting"] is True
    assert r["angle"] > 0.0  # left drift is positive


def test_below_min_angle():
    r = drift_from_state(state_at(DRIFT_MIN_ANGLE - 4.0, 100.0))
    assert r["isDrifting"] is False
    assert r["scoreAdded"] == 0.0


def test_above_max_angle():
    r = drift_from_state(state_at(DRIFT_MAX_ANGLE + 5.0, 100.0))
    assert r["isDrifting"] is False
    assert r["scoreAdded"] == 0.0


def test_below_min_speed():
    r = drift_from_state(state_at(45.0, DRIFT_MIN_AIRSPEED - 1.0))
    assert r["isDrifting"] is False
    assert r["scoreAdded"] == 0.0


def test_angle_bounded():
    # the signed angle must always fit the observation space [-1, 1] after /180
    for angle in (-179, -90, 0, 90, 179):
        r = drift_from_state(state_at(angle, 120.0))
        assert -180.0 <= r["angle"] <= 180.0


def test_score_monotonic_in_angle():
    # a wider drift at the same speed should add more score
    low = drift_from_state(state_at(20.0, 100.0))["scoreAdded"]
    high = drift_from_state(state_at(60.0, 100.0))["scoreAdded"]
    assert high > low > 0.0
