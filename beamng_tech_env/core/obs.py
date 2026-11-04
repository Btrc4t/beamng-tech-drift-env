"""Pure observation math for the BeamNG.tech drift environments.

Everything in this module is a function of plain data (a State snapshot, a
point cloud, a drift reading) with no dependency on the game or on any
environment instance, so it can be unit-tested without launching BeamNG.tech
and reused by any environment that builds the same observation.
"""
import inspect

from beamng_tech_env.core.array import np
from beamng_tech_env.core.constants import (
    AIRSPEED_MAX_KPH,
    ANGLE_MAX,
    BEV_AGGRESSIVE_LEVEL,
    BEV_MAX_HEIGHT,
    BEV_N,
    BEV_ROAD_LEVEL,
    CAR_LEN,
    CAR_WID,
    CAR_Z_HI,
    CAR_Z_LO,
    DRIFT_MAX_ANGLE,
    DRIFT_MIN_AIRSPEED,
    DRIFT_MIN_ANGLE,
    IMU_ACC_MAX,
    IMU_ANGVEL_MAX,
    LIDAR_MAX_DIST,
    MAX_GEAR,
    TRAVEL_ALIGN_MIN_KPH,
    Z_HI,
    Z_LO,
)


def drift_from_state(state):
    """Per-vehicle drift telemetry from the State sensor, no game round trip.

    Mirrors the game's formulas (drift.lua:230-231): the drift angle is the
    absolute yaw difference between the velocity direction and the heading,
    signed by vel . (up x dir). The game's drift score is player-centric, so
    for a group of vehicles we compute a per-frame score proxy here instead.

    The angle is reported at any speed (0 at rest, where the velocity
    direction is undefined); the BEV alignment damps it toward 0 below
    TRAVEL_ALIGN_MIN_KPH (see bev_rotation_angle).
    """
    vel = np.asarray(state["vel"], float)
    dirv = np.asarray(state["dir"], float)
    up = np.asarray(state["up"], float)
    speed_kph = float(np.linalg.norm(vel) * 3.6)
    if speed_kph > 0.0:
        d = dirv / np.linalg.norm(dirv)
        v = vel / np.linalg.norm(vel)
        # the game measures slip as a horizontal-plane (x, y) yaw difference
        # via atan2, not the full 3D angle (drift.lua:201-230); mirror that
        # exactly so the parity holds on slopes, where the 3D arccos would
        # pick up pitch
        yaw_diff = np.arctan2(d[1], d[0]) - np.arctan2(v[1], v[0])
        yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
        angle = float(abs(np.degrees(yaw_diff)))
        sign = float(np.sign(np.dot(vel, np.cross(up, dirv))))
        # moving in reverse: the velocity has a backward component (the signed
        # angle is 0 for a straight reverse, so it can't be used for this)
        reversing = bool(np.dot(v, d) < 0.0)
    else:
        # at rest the velocity direction is undefined
        angle, sign, reversing = 0.0, 0.0, False
    is_drifting = (DRIFT_MIN_ANGLE < angle < DRIFT_MAX_ANGLE
                   and speed_kph > DRIFT_MIN_AIRSPEED)
    # per-frame score proxy, scaled so the reward stays in a similar range
    score_added = ((angle / DRIFT_MAX_ANGLE)
                   * (speed_kph / AIRSPEED_MAX_KPH) * 100.0
                   if is_drifting else 0.0)
    return {"angle": angle * sign, "airSpeed": speed_kph,
            "isDrifting": is_drifting, "scoreAdded": score_added,
            "reversing": reversing}


def car_rotation(state):
    """Rotation matrix whose COLUMNS are the car's [forward, left, up] axes in
    world space, built from the (normalised) dir and up vectors."""
    d = np.asarray(state["dir"], float); d /= np.linalg.norm(d)
    u = np.asarray(state["up"], float);  u /= np.linalg.norm(u)
    l = np.cross(u, d)                     # left
    l /= np.linalg.norm(l)
    return np.stack([d, l, u], axis=1)     # columns [fwd, left, up]


def drift_rotation(drift_angle_deg):
    """(cos, sin) of the grid->car-frame rotation at a drift angle: the BEV
    grid is aligned with the travel direction, so the car frame is rotated by
    pi/2 - slip relative to it."""
    slip = float(np.deg2rad(drift_angle_deg))
    th = 0.5 * np.pi - slip
    return np.cos(th), np.sin(th)


def bev_rotation_angle(drift, gear=0):
    """The BEV grid rotation (deg) for a drift reading: the drift angle,
    damped toward 0 below TRAVEL_ALIGN_MIN_KPH, or 0 while reversing in
    reverse gear.

    Below that speed the travel direction is unreliable, so only a fraction
    of the drift angle is applied (airSpeed / TRAVEL_ALIGN_MIN_KPH), ramping
    the grid from the car's heading to the travel direction as the speed
    builds up. While moving backwards in gear -1 the grid stays aligned with
    the car's heading (angle 0); moving backwards in any other gear (e.g.
    rolling back in 1st) keeps the alignment.
    """
    if drift.get("reversing", False) and int(gear) == -1:
        return 0.0
    angle = drift.get("angle", 0.0)
    airspeed = drift.get("airSpeed", 0.0)
    if airspeed <= TRAVEL_ALIGN_MIN_KPH:
        return angle * (airspeed / TRAVEL_ALIGN_MIN_KPH)
    return angle


def lidar_bev(pc, state, drift, n=BEV_N, range_m=LIDAR_MAX_DIST, rot=None,
              gear=0):
    """Bird's eye view height grid: the car is in the center, the grid spans
    [-range_m, +range_m] around it, values are the highest LiDAR return per
    cell clamped to [0, 1] (0.5 = road level). Returns higher than
    BEV_MAX_HEIGHT (tunnel/overpass ceilings) are dropped so they don't read
    as obstacles. The grid is aligned with the travel direction (rotated by
    the drift angle, damped toward the car's heading below
    TRAVEL_ALIGN_MIN_KPH - see bev_rotation_angle). The car is not painted:
    its own body returns (the roof LiDAR catches the car, mostly the rear)
    are dropped at the point-cloud stage by the body box, so a wall right at
    the car's edge keeps its points (outside the box) and the beams
    (wrappers.bev_to_beams) read the grid as-is, with no car mask. The box is
    wide in x/y but narrow in z (CAR_Z_LO..CAR_Z_HI, the body height): only
    returns at the car's own height are cut, so the road and low obstacles
    (a wall's base, a curb) survive even just outside the x/y box."""
    angle = bev_rotation_angle(drift, gear)
    c, s = drift_rotation(angle)

    pc = np.asarray(pc, float)
    out = np.full((n, n), np.nan, np.float32)
    if pc.size == 0:
        return out
    R = rot if rot is not None else car_rotation(state)
    p = (pc - np.asarray(state["pos"], float)) @ R   # [fwd, left, up]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    # drop the car's own body returns in the car frame: wide in x/y (covers
    # the full LiDAR-visible body) but narrow in z (the body height), so the
    # road and low obstacles just outside the box keep their points
    in_car = ((np.abs(x) <= CAR_LEN / 2) & (np.abs(y) <= CAR_WID / 2)
              & (z >= CAR_Z_LO) & (z <= CAR_Z_HI))
    x, y, z = x[~in_car], y[~in_car], z[~in_car]
    if x.size == 0:
        return out
    dx, dy = x * c - y * s, x * s + y * c
    ix = ((dx + range_m) * n / (2 * range_m)).astype(int)
    iy = ((dy + range_m) * n / (2 * range_m)).astype(int)
    m = (ix >= 0) & (iy >= 0) & (ix < n) & (iy < n) & (z <= BEV_MAX_HEIGHT)
    cell = (n - 1 - iy[m]) * n + ix[m]
    h = np.clip((z[m] - Z_LO) / (Z_HI - Z_LO), 0.0, 1.0)
    # max height per cell in one O(N) pass via an unbuffered ufunc;
    # ~3-4x faster than argsort + unique + reduceat on a full point cloud
    out_flat = np.full(n * n, -np.inf, np.float32)
    np.maximum.at(out_flat, cell, h)
    out_flat[out_flat == -np.inf] = np.nan
    return out_flat.reshape(n, n)


def bev_to_obs(grid):
    """[0, 1] float BEV grid (NaN = no LiDAR return) -> the uint8 observation.

    Empty cells become the road level (0.5 -> BEV_ROAD_LEVEL); a leading
    channel dim is added so SB3's CombinedExtractor sees a 3-D image space
    and runs a CNN on the BEV instead of flattening it.
    """
    grid = np.nan_to_num(grid, nan=0.5)
    return (np.clip(grid, 0.0, 1.0) * 255).round().astype(np.uint8)[None, ...]


def correct_lidar_bev(bev, road):
    """The road-corrected LiDAR BEV: the road network fills the road surface
    with the road level, everything else stays as the LiDAR saw it.

    Every cell the road mask marks drivable is set to exactly
    BEV_ROAD_LEVEL, except the most aggressive obstacles (returns above
    BEV_AGGRESSIVE_LEVEL), which are kept; off-road cells are untouched.
    ``bev`` is a uint8 LiDAR BEV ((1, n, n) or (n, n)); ``road`` a road mask
    (1.0 = drivable, same grid convention as the LiDAR BEV) or None (no
    correction, ``bev`` returned as-is). The beams are computed on the
    result (wrappers.bev_to_beams).
    """
    if road is None:
        return bev
    bev = np.asarray(bev)
    road = np.asarray(road, float)
    if road.ndim == 3 and road.shape[0] == 1:
        road = road[0]
    fill = (road > 0.5) & (bev <= BEV_AGGRESSIVE_LEVEL)
    return np.where(fill, BEV_ROAD_LEVEL, bev).astype(bev.dtype, copy=False)


def reset_is_eval():
    """True when reset() runs inside SB3's evaluate_policy: the eval callback
    reuses this env, so those resets interrupt the training episode and are
    not episode terminations."""
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_code.co_name == "evaluate_policy":
            return True
        frame = frame.f_back
    return False


def derive_imu(prev, state):
    """Accel and angular velocity from two consecutive State snapshots.

    Stateless form of the per-step IMU derivation (the caller owns the
    previous snapshot). Channels are in the car frame with the [fwd, left, up]
    axis order: acc = [longitudinal, lateral, vertical] (kinematic d(vel)/dt,
    so no gravity offset in the vertical channel, unlike the raw IMU),
    angvel = [roll, pitch, yaw] rates in rad/s.

    ``prev`` is a dict with ``vel``/``dir``/``up``/``time`` (or ``None`` for
    the first step). Returns (acc, angvel) as float32 arrays, clipped to the
    sensor limits.
    """
    if prev is None:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    dt = float(state["time"]) - float(prev["time"])
    if dt < 0.04:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    R = car_rotation(state)
    v_car = R.T @ np.asarray(state["vel"], float)
    v_car_prev = car_rotation(prev).T @ np.asarray(prev["vel"], float)
    acc = (v_car - v_car_prev) / dt

    dR = R.T @ car_rotation(prev)
    skew = 0.5 * np.array([
        dR[2, 1] - dR[1, 2],
        dR[0, 2] - dR[2, 0],
        dR[1, 0] - dR[0, 1],
    ])
    theta = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    if theta > 1e-4:
        angvel = skew * (theta / np.sin(theta)) / dt
    else:
        angvel = skew / dt

    acc = np.clip(acc / IMU_ACC_MAX, -1.0, 1.0)
    angvel = np.clip(angvel / IMU_ANGVEL_MAX, -1.0, 1.0)
    return acc.astype(np.float32), angvel.astype(np.float32)


def state_snapshot(state):
    """Copy the fields of a State snapshot that the derived IMU needs.

    ``vehicle.state`` is mutated in place by the next sensor poll, so a
    reference would make the previous snapshot identical to the current one
    (dt = 0); copy the fields instead.
    """
    return {
        "vel": np.array(state["vel"], float),
        "dir": np.array(state["dir"], float),
        "up": np.array(state["up"], float),
        "time": float(state["time"]),
    }


def neutral_obs():
    """A zeroed observation with the BEV at the road level, used to end an
    episode when a recovery leaves no usable sensor data."""
    return {
        "rpm":         np.zeros(1, dtype=np.float32),
        "imu_acc":     np.zeros(3, dtype=np.float32),
        "imu_angvel":  np.zeros(3, dtype=np.float32),
        "drift_angle": np.zeros(1, dtype=np.float32),
        "airspeed":    np.zeros(1, dtype=np.float32),
        "is_shifting": np.zeros(1, dtype=np.float32),
        "gear":        np.zeros(1, dtype=np.float32),
        "ffb":         np.zeros(1, dtype=np.float32),
        "bev":         bev_to_obs(np.full((BEV_N, BEV_N), 0.5, np.float32)),
        "road_bev":    np.zeros((1, BEV_N, BEV_N), dtype=np.float32),
    }


def gear_to_obs(gear):
    """Absolute gear -> the normalised [0, 1] gear observation channel."""
    return np.array([(float(gear) + 1.0) / (MAX_GEAR + 1.0)], np.float32)
