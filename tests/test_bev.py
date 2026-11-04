"""Pure-math tests for the LiDAR bird's-eye view (``lidar_bev``).

These tests do not launch BeamNG.drive; they feed a synthetic point cloud to
``lidar_bev`` and check the height filtering: tunnel/overpass ceilings (returns
well above the car) must be dropped so they don't read as obstacles, while real
low obstacles (walls, curbs) and the road must be kept.
"""
import numpy as np

from beamng_tech_env.core.constants import (
    BEV_MAX_HEIGHT,
    BEV_N,
    BEV_ROAD_LEVEL,
    LIDAR_MAX_DIST,
    TRAVEL_ALIGN_MIN_KPH,
)
from beamng_tech_env.core.obs import bev_rotation_angle, bev_to_obs, lidar_bev

N = BEV_N
R = LIDAR_MAX_DIST

# car at the origin, facing +X, up +Z, no drift: the point->car rotation is the
# identity and the grid rotation is th = pi/2, so dx = -y and dy = x
STATE = {"pos": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}
DRIFT = {"angle": 0.0}


def obs_rc(x, y):
    """Map a world (x, y) at the car's location to the BEV obs (row, col).

    Mirrors ``lidar_bev``: the grid row is flipped (row = N-1-iy), the column is
    ix. Only valid for the fixed STATE/DRIFT above (slip = 0).
    """
    ix = int((-y + R) * N / (2 * R))
    iy = int((x + R) * N / (2 * R))
    return N - 1 - iy, ix


def render(points, drift=None):
    pc = np.asarray(points, dtype=np.float32)
    return bev_to_obs(lidar_bev(pc, STATE, drift if drift is not None else DRIFT))[0]


def test_road_is_road_level():
    obs = render([[10.0, 0.0, 0.0]])
    r, c = obs_rc(10.0, 0.0)
    assert obs[r, c] == BEV_ROAD_LEVEL


def test_ceiling_only_cell_is_not_an_obstacle():
    # a return well above BEV_MAX_HEIGHT (a tunnel/overpass ceiling) is dropped,
    # leaving the cell at the road level rather than a bright obstacle
    obs = render([[20.0, 5.0, 5.0]])
    r, c = obs_rc(20.0, 5.0)
    assert obs[r, c] == BEV_ROAD_LEVEL


def test_ceiling_does_not_override_road():
    # a cell with both a road return and a ceiling return must show the road,
    # not the ceiling (which would clamp to the max and read as an obstacle)
    obs = render([[10.0, 0.0, 0.0], [10.0, 0.0, 5.0]])
    r, c = obs_rc(10.0, 0.0)
    assert obs[r, c] == BEV_ROAD_LEVEL


def test_low_obstacle_is_kept():
    # a return at or below BEV_MAX_HEIGHT (a wall/curb) is a real obstacle and
    # must read brighter than the road
    obs = render([[-10.0, 0.0, 1.0]])
    r, c = obs_rc(-10.0, 0.0)
    assert obs[r, c] > BEV_ROAD_LEVEL


def test_tall_wall_is_kept():
    # a wall taller than BEV_MAX_HEIGHT still reads as an obstacle: it has
    # returns from the ground up, so the per-cell max keeps the low ones and the
    # height cap only trims the top, not the detection
    wall = [[10.0, 0.0, z] for z in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    obs = render(wall)
    r, c = obs_rc(10.0, 0.0)
    assert obs[r, c] > BEV_ROAD_LEVEL


def test_ceiling_filter_threshold_is_respected():
    # a return just below the limit is kept; just above it is dropped
    kept = render([[10.0, 0.0, BEV_MAX_HEIGHT - 0.1]])
    dropped = render([[10.0, 0.0, BEV_MAX_HEIGHT + 0.1]])
    r, c = obs_rc(10.0, 0.0)
    assert kept[r, c] > BEV_ROAD_LEVEL
    assert dropped[r, c] == BEV_ROAD_LEVEL


def test_car_body_returns_are_dropped():
    # the car's own body returns (the roof LiDAR catches the car, mostly the
    # rear) are dropped at the point-cloud stage: a return inside the body
    # box (x/y footprint AND the body-height z band) leaves its cell at the
    # road level, not a bright obstacle
    obs = render([[0.0, 0.0, 1.2]])            # on the roof, in the body box
    r, c = obs_rc(0.0, 0.0)
    assert obs[r, c] == BEV_ROAD_LEVEL
    r, c = obs_rc(-2.0, 0.0)                   # rear body (spoiler), in the box
    assert render([[-2.0, 0.0, 1.2]])[r, c] == BEV_ROAD_LEVEL


def test_return_below_body_band_is_kept():
    # a return in the x/y footprint but below the body-height z band (the
    # road, a low curb) is not the car's body: it survives the filter and
    # reads as road/obstacle, not dropped
    obs = render([[0.0, 0.0, 0.5]])            # in the x/y box, z below the band
    r, c = obs_rc(0.0, 0.0)
    assert obs[r, c] > BEV_ROAD_LEVEL


def test_wall_at_car_edge_is_kept():
    # a wall inside the enlarged x/y box but below the body-height z band is
    # not the car's body: its points survive the filter (the z band, not the
    # x/y box, keeps them) and read as an obstacle
    wall = [[2.5, y, 0.5] for y in np.linspace(-2.0, 2.0, 21)]
    obs = render(wall)
    r, c = obs_rc(2.5, 0.0)
    assert obs[r, c] > BEV_ROAD_LEVEL
    # a wall just outside the x/y box (at body height) is kept too
    wall = [[3.5, y, 1.2] for y in np.linspace(-2.0, 2.0, 21)]
    obs = render(wall)
    r, c = obs_rc(3.5, 0.0)
    assert obs[r, c] > BEV_ROAD_LEVEL


# -- BEV rotation (damped below TRAVEL_ALIGN_MIN_KPH) ------------------------

def test_rotation_angle_damps_below_limit():
    # at rest the grid stays unrotated; below the limit only a fraction of
    # the drift angle is applied (airSpeed / TRAVEL_ALIGN_MIN_KPH)
    assert bev_rotation_angle({"angle": 30.0, "airSpeed": 0.0}) == 0.0
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": TRAVEL_ALIGN_MIN_KPH / 2.0}) == 15.0
    assert bev_rotation_angle({"angle": 30.0}) == 0.0  # no airSpeed -> 0
    # at and above the limit the full drift angle is used
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": TRAVEL_ALIGN_MIN_KPH}) == 30.0
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": TRAVEL_ALIGN_MIN_KPH + 1.0}) == 30.0
    assert bev_rotation_angle({"angle": -45.0, "airSpeed": 100.0}) == -45.0


def test_rotation_angle_reversing_in_reverse_gear_is_0():
    # moving backwards in gear -1: unrotated, the grid stays aligned with
    # the car's heading (independent of the drift angle)
    assert bev_rotation_angle(
        {"angle": 0.0, "airSpeed": 50.0, "reversing": True}, -1) == 0.0
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": 50.0, "reversing": True}, -1) == 0.0
    # moving backwards in any other gear: the alignment is kept (damped
    # below the speed limit, full above it)
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": 50.0, "reversing": True}, 1) == 30.0
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": 5.0, "reversing": True}, 1) \
        == 30.0 * (5.0 / TRAVEL_ALIGN_MIN_KPH)
    # not reversing: the drift angle is used (any gear)
    assert bev_rotation_angle(
        {"angle": 30.0, "airSpeed": 50.0, "reversing": False}, -1) == 30.0


def test_bev_rotation_damps_below_limit():
    # a drift at a crawl: the grid is rotated by a fraction of the drift
    # angle (airSpeed / TRAVEL_ALIGN_MIN_KPH) - the same render as the scaled
    # angle at full speed
    pts = [[10.0, 5.0, 0.0], [-8.0, -3.0, 0.5], [3.0, -12.0, 1.0]]
    speed = 5.0
    low = render(pts, {"angle": 30.0, "airSpeed": speed})
    scaled = render(pts, {"angle": 30.0 * speed / TRAVEL_ALIGN_MIN_KPH,
                          "airSpeed": 100.0})
    assert np.array_equal(low, scaled)


def test_bev_rotated_above_limit():
    # the same drift at speed: the grid is rotated by the drift angle, so it
    # differs from the unrotated render
    pts = [[10.0, 5.0, 0.0], [-8.0, -3.0, 0.5], [3.0, -12.0, 1.0]]
    fast = render(pts, {"angle": 30.0, "airSpeed": 50.0})
    none = render(pts, {"angle": 0.0, "airSpeed": 50.0})
    assert not np.array_equal(fast, none)


def test_bev_reversing_stays_aligned_with_heading():
    # STATE faces +X: a point at +X is in front of the car, -X behind it.
    # reversing is unrotated (the grid stays aligned with the car's heading,
    # like the slow case), so the front point renders above center and the
    # rear point below center. The point is at a low obstacle height so it
    # reads brighter than the road but not the car.
    center = N // 2

    def obstacle_row(world_x):
        obs = render([[world_x, 0.0, 0.5]],
                     {"angle": 0.0, "airSpeed": 50.0, "reversing": True})
        rows, _ = np.where((obs > BEV_ROAD_LEVEL) & (obs < 255))
        assert len(rows) == 1
        return int(rows[0])

    assert obstacle_row(10.0) < center   # front of the car -> top
    assert obstacle_row(-10.0) > center  # rear of the car -> bottom
