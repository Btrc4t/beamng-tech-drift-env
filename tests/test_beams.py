"""Unit tests for the ultrasonic beams (no game needed)."""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from beamng_tech_env.core.array import to_cpu
from beamng_tech_env.core.constants import (
    BEV_AGGRESSIVE_LEVEL,
    BEV_N,
    BEV_ROAD_LEVEL,
    LIDAR_MAX_DIST,
)
from beamng_tech_env.core.obs import correct_lidar_bev
from beamng_tech_env.wrappers import (
    BEAM_ANGLES,
    BeamsWrapper,
    bev_to_beams,
    obs_to_beams,
)


def _bev(value=128):
    return np.full((1, BEV_N, BEV_N), value, dtype=np.uint8)


def test_all_road_reads_full_range():
    beams = bev_to_beams(_bev())
    assert beams.shape == (20,)
    assert np.all(beams == 1.0)


def test_forward_wall_distance():
    bev = _bev()
    # beam 10 is the closest to forward (+7.1 deg, 20 beams leave no beam at
    # exactly 0); its ray crosses 10 m ahead at cell (row 24, col 31)
    bev[0, 24, 31] = 255
    beams = bev_to_beams(bev)
    # the ray enters the cell (center 9.375 m ahead) at some radius inside
    # the cell's extent
    assert abs(beams[10] - 9.375 / LIDAR_MAX_DIST) < (2 * LIDAR_MAX_DIST / BEV_N) / LIDAR_MAX_DIST
    # neighbouring beams miss the single cell
    assert beams[9] == 1.0 and beams[0] == 1.0 and beams[-1] == 1.0


def test_return_in_car_region_is_an_obstacle():
    # the car is no longer painted or masked: a bright return in the car's
    # region (a wall right at the car's edge, its points just outside the
    # bounding box but in a footprint cell) reads as an obstacle
    bev = _bev()
    bev[0, 30, 31:33] = 255   # the front-edge wall's cells
    assert bev_to_beams(bev)[10] < 1.0


def test_center_cell_return_is_not_an_obstacle():
    # the beam ignores the cell it lands on (the grid center, where the car
    # is): a return in the car's own cell is not an obstacle, so an
    # obstacle-free frame with only that return reads full range
    bev = _bev()
    bev[0, 32, 32] = 200   # the grid-center cell (beam sample 0)
    assert np.all(bev_to_beams(bev) == 1.0)


def test_beam_hits_wall_at_car_edge():
    # the reported bug: a wall right at the car's front edge used to be
    # erased by the car paint, so the beams read full range; the
    # point-cloud filter keeps the wall's low points (below the body-height
    # z band) even though the wall is now inside the enlarged x/y box
    from beamng_tech_env.core.obs import bev_to_obs, lidar_bev
    state = {"pos": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0],
             "up": [0.0, 0.0, 1.0]}
    drift = {"angle": 0.0, "airSpeed": 50.0}
    # the wall face is 2.3 m ahead (inside the enlarged body box)
    wall = np.array([[2.3, y, z] for y in np.linspace(-6.0, 6.0, 49)
                     for z in (0.0, 0.5, 1.0)], dtype=np.float32)
    bev = bev_to_obs(lidar_bev(wall, state, drift))
    beams = bev_to_beams(bev)
    # the forward beams (10 at +7.1 deg, 9 at -7.1 deg) hit the wall near
    # its face, not full range
    assert beams[10] < 1.0 and beams[9] < 1.0
    assert abs(beams[10] - 2.3 / LIDAR_MAX_DIST) \
        < (2 * LIDAR_MAX_DIST / BEV_N) / LIDAR_MAX_DIST


def test_banked_road_is_not_an_obstacle():
    # the road tilts in the car frame (the vehicle banks while drifting): a
    # smooth gradient of road returns in the central band must not trigger
    bev = _bev()
    for r in range(27, 37):
        bev[0, r, 29:36] = 128 + (r - 27)             # 128..139
    assert np.all(bev_to_beams(bev) == 1.0)
    # a curb ~20 cm above the road, 9.4 m ahead, is an obstacle
    bev[0, 24, 31:34] = 160
    assert bev_to_beams(bev)[10] < 1.0


def test_batched_input():
    bev = _bev()
    bev[0, 24, 31] = 255
    batch = np.stack([bev, _bev()])
    beams = bev_to_beams(batch)
    assert beams.shape == (2, 20)
    assert abs(beams[0][10] - 9.375 / LIDAR_MAX_DIST) < (2 * LIDAR_MAX_DIST / BEV_N) / LIDAR_MAX_DIST
    assert np.all(beams[1] == 1.0)


def test_angle_coverage():
    # 20 beams evenly spaced over -135..+135, + = left; an even count leaves
    # no beam at exactly 0 (the paper's layout), the closest are +/-7.1 deg
    angles = to_cpu(BEAM_ANGLES)
    assert len(angles) == 20
    assert angles[0] == -135.0 and angles[-1] == 135.0
    assert abs(angles[10]) == pytest.approx(7.105, abs=0.01)
    assert np.allclose(np.diff(angles), 270.0 / 19)


# -- correct_lidar_bev (the road-corrected LiDAR) --------------------------

def _road_mask(rows=(20, 30), cols=(27, 37)):
    road = np.zeros((BEV_N, BEV_N), float)
    road[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
    return road


def test_correct_lidar_bev_fills_road_surface():
    # road cells that read below the aggressive level become the road level
    bev = _bev(100)
    out = correct_lidar_bev(bev, _road_mask())
    assert (out[0, 20:30, 27:37] == BEV_ROAD_LEVEL).all()
    # off-road cells are untouched
    assert out[0, 0, 0] == 100
    assert out[0, -1, -1] == 100


def test_correct_lidar_bev_keeps_aggressive_obstacles():
    # a return above the aggressive level survives on the road
    bev = _bev(100)
    bev[0, 25, 32] = BEV_AGGRESSIVE_LEVEL + 1
    out = correct_lidar_bev(bev, _road_mask())
    assert out[0, 25, 32] == BEV_AGGRESSIVE_LEVEL + 1
    # a return at the aggressive level is still road surface
    bev = _bev(100)
    bev[0, 25, 32] = BEV_AGGRESSIVE_LEVEL
    out = correct_lidar_bev(bev, _road_mask())
    assert out[0, 25, 32] == BEV_ROAD_LEVEL


def test_correct_lidar_bev_no_road_mask():
    bev = _bev(100)
    assert np.array_equal(correct_lidar_bev(bev, None), bev)


def test_beams_on_corrected_lidar():
    road = np.ones((BEV_N, BEV_N), float)
    # a ~20 cm curb on the road (160) stops beam 10 uncorrected...
    bev = _bev(128)
    bev[0, 24, 31:34] = 160
    assert bev_to_beams(bev)[10] < 1.0
    # ...but the correction clears it (160 <= aggressive level): full range
    assert np.all(bev_to_beams(correct_lidar_bev(bev, road)) == 1.0)
    # an aggressive return (250) survives the correction and still stops it
    bev = _bev(128)
    bev[0, 24, 31:34] = 250
    assert bev_to_beams(correct_lidar_bev(bev, road))[10] < 1.0


# -- obs_to_beams (corrected LiDAR BEV by default, road BEV fallback) ------

def _road_bev(value=1.0):
    return np.full((1, BEV_N, BEV_N), value, dtype=np.float32)


def test_obs_to_beams_uses_lidar_bev():
    # a road-level LiDAR BEV reads full range; the road BEV (no drivable
    # cells) is ignored when the LiDAR BEV is present
    obs = {"bev": _bev(128), "road_bev": _road_bev(0.0)}
    assert np.all(obs_to_beams(obs) == 1.0)
    # an all-obstacle LiDAR BEV hits everywhere even with a drivable road BEV
    obs = {"bev": _bev(255), "road_bev": _road_bev(1.0)}
    assert np.all(obs_to_beams(obs) < 1.0)


def test_obs_to_beams_falls_back_to_road_bev():
    # no LiDAR BEV: the road BEV is used (drivable = full range, off-road = 0)
    obs = {"road_bev": _road_bev(1.0)}
    assert np.all(obs_to_beams(obs) == 1.0)
    obs = {"road_bev": _road_bev(0.0)}
    assert np.all(obs_to_beams(obs) == 0.0)


def test_obs_to_beams_batched():
    # a batch of obs (each an array) gives (T, 20)
    obs = {"bev": np.stack([_bev(128), _bev(255)])}
    beams = obs_to_beams(obs)
    assert beams.shape == (2, 20)
    assert np.all(beams[0] == 1.0) and np.all(beams[1] < 1.0)


class _FakeEnv(gym.Env):
    observation_space = spaces.Dict({
        "rpm": spaces.Box(0.0, 1.0, (1,), np.float32),
        "drift_angle": spaces.Box(-1.0, 1.0, (1,), np.float32),
        "bev": spaces.Box(0, 255, (1, BEV_N, BEV_N), np.uint8),
    })
    action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def test_wrapper_swaps_bev_for_beams():
    env = BeamsWrapper(_FakeEnv())
    assert "bev" not in env.observation_space.spaces
    assert env.observation_space.spaces["beams"].shape == (20,)
    obs, _ = env.reset()
    assert "bev" not in obs and obs["beams"].shape == (20,)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert "bev" not in obs and obs["beams"].shape == (20,)
