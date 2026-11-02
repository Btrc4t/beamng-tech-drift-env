"""Shared constants for the BeamNG.tech drift environments.

Moved out of the (former) monolithic env module so the drive and replay
environments, and any future ones, share a single source of truth for the
tuning values, sensor limits and tcom ports.

The array-valued data (BEAM_ANGLES, the gear one-hot) is plain numpy: it is
small, shared with the numpy-based callers (the env's action encoding, the
dataset, the debug window), and the cupy hot path converts it where it needs
it (core.roads / wrappers beam tables).
"""
import numpy as _numpy

from beamng_tech_env.core.array import np

# general limits
DAMAGE_LIMIT        = 1000
EPISODE_TIMEOUT     = 60.0
FPS                 = 20   # physics tick rate (fixed by the engine, step(1) = 50 ms of sim time)
TICK_DT             = 1.0 / FPS  # one physics tick in seconds (0.05 s)
RENDER_FPS          = 120  # render loop rate: how often the game's main loop wakes up to process socket requests
REALTIME_FPS_LIMIT  = 40   # render cap used for real-time (1x) pacing, see GameDriver.set_realtime
FPS_FACTOR          = 1
TCOM_PORT           = 25252  # tcom port: connect to a game here, or launch one on it
# progress reward: a single reward from the per-step distance, positive when
# the car travels more than the break-even distance, held at 0 when the car
# just jiggles (travels less), halved when traveling backwards
PROGRESS_RATE        = 1.0   # reward per metre relative to the break-even distance
PROGRESS_BREAK_SPEED = 3.0   # kph; zero reward at this speed
PROGRESS_BREAK_DIST  = PROGRESS_BREAK_SPEED / 3.6 / FPS  # break-even metres/step

# checkpoint: respawn at the last undamaged pose so a crash doesn't force the
# agent to repeat the whole episode from the start
CHECKPOINT_DAMAGE   = 1.0   # damage below this counts as "undamaged"
# spawn verification: damage within this many steps of a spawn means the
# saved spawn point is bad (the car hit something right after spawning), so
# it is invalidated and the next reset picks a waypoint
SPAWN_DAMAGE_STEPS  = 5
# progress timeout: if the car covers less than this in a rolling window, the
# episode is truncated (it has stopped making forward progress)
PROGRESS_WINDOW_SECONDS = 15.0  # rolling window length (s)
PROGRESS_MIN_DISTANCE   = 15.0  # min metres in the window before truncation
# centerline reward: the distance from the car to the road centerline of the
# surface it is closest to is clamped to [0, CENTERLINE_DIST_MAX] and inverted
# (on-centerline = 1.0, far = 0.0) to scale the progress reward - the closer the
# car is to the centerline, the more of the distance reward is kept
CENTERLINE_DIST_MAX = 40.0   # (m); the max centerline distance for the reward
# off-road / reversing: the total reward is scaled down while the car is off
# the road (the env's on_road determination is False) or moving backwards
# (the drift data's reversing flag); when both apply, the lowest multiplier
# wins (the two never stack)
OFFROAD_REWARD_MULT   = 0.2
REVERSING_REWARD_MULT = 0.5

# tcom has no protocol-level timeout: if the game wedges in an internal
# blocking state (scenario restart, vehicle spawn, vehicle connection),
# every request waits forever. These limits detect that and restart the game.
STEP_TIMEOUT        = 10.0   # s; a normal step round trip is well under 1 s
RESET_TIMEOUT       = 90.0   # s; a normal reset (restart + respawn) is 5-15 s
SETUP_TIMEOUT       = 180.0  # s; recovery includes a game launch + map load, which can take minutes
MAX_HANG_RECOVERIES = 5      # consecutive unresponsive games before giving up
# the game leaks 2 shared-memory fds per LiDAR lifecycle (its C++ removeSensor
# never closes them); once the total fd count approaches the 1024 limit of
# select()'s fd_set, the tcom server's checkMessages() dies with "descriptor
# too large for set size" and the vehicle hangs. A fresh game starts at ~230
# fds, so this restarts the game after ~330 resets. For hosted games the
# budget is applied by host_game.py on the game's host (the training host
# can't read a remote /proc).
MAX_GAME_FDS = 900

# human mode: the driver's wheel (electrics.steering, in degrees) is normalised
# by the car's steering wheel lock back into the [-1, 1] action space; this is
# the fallback if the lock can't be read from the game
DEFAULT_STEERING_WHEEL_LOCK = 450.0  # deg

# drift band, mirrored from the game's driftOptions (drift.lua:86-98)
DRIFT_MIN_ANGLE    = 7.0    # deg, below this is not a drift
DRIFT_MAX_ANGLE    = 165.0  # deg, above this is a spin
DRIFT_MIN_AIRSPEED = 25.0    # kph, minimum speed to count as drifting

# sensor value limits
IMU_ACC_MAX      = 20.0   # m/s^2  (~2 g per axis, covers hard drift loads)
IMU_ANGVEL_MAX   = 10.0   # rad/s
ANGLE_MAX        = 180.0  # drift angle, deg -> [-1, 1]
AIRSPEED_MAX_KPH = 300.0  # kph -> [0, 1]
BEV_N            = 64     # LiDAR BEV grid size
# ego body box: drops the car's own LiDAR returns at the point-cloud stage.
# x/y are enlarged to cover the full LiDAR-visible body (the roof LiDAR
# catches the spoiler when stopped and the banked front while drifting); the
# narrow z band (CAR_Z_LO..CAR_Z_HI, the body height) is what makes the large
# x/y safe - only returns at the car's own height are cut, so the road and
# low obstacles (a wall's base, a curb) just outside the box keep their
# points and read as obstacles.
CAR_LEN, CAR_WID = 6.0, 3.0   # (m) ego footprint: |x| <= CAR_LEN/2, |y| <= CAR_WID/2
CAR_Z_LO, CAR_Z_HI = 1.0, 1.4  # (m) body-height band: only returns here are dropped
Z_LO, Z_HI       = -1.0, 1.0   # (m) road band: squeeze the scale so the raised divider pops
BEV_MAX_HEIGHT   = 2.0    # (m) drop LiDAR returns above this (tunnel/overpass ceilings)
BEV_ROAD_LEVEL   = 128    # uint8 value of the 0.5 road level (0.5 * 255, rounded)
# the road correction fills the road surface with the road level, but keeps
# returns above this level: the most aggressive obstacles (walls, other
# vehicles, overpass structures)
BEV_AGGRESSIVE_LEVEL = 240  # uint8
# the BEV grid is aligned with the travel direction only above this speed
# and while the car moves forward: below it (or in reverse) the travel
# direction is unreliable (a crawl, a stop), so the rotation angle is forced
# to 0 and the grid stays aligned with the car's heading (see
# obs.bev_rotation_angle).
TRAVEL_ALIGN_MIN_KPH = 7.0  # kph
# LiDAR: full 360-degree scan, attached to the ego car's roof
# We rotate the display / observation to the travel direction.
LIDAR_RES_V      = 64                   # vertical resolution (number of horizontal slices)
LIDAR_V_ANGLE    = 26.9                 # vertical aperture (deg)
LIDAR_HZ         = FPS * FPS_FACTOR     # lidar rotation frequency (Hz)
LIDAR_MAX_DIST   = 40.0                 # max range (m)
LIDAR_DENSITY    = 50                   # point density (1 = dense, 100 = sparse)

# ultrasonic-style beams: N_BEAMS radial distances, evenly spaced over
# -135..+135 degrees (0 = travel direction, positive = left). Shared by the
# LiDAR-derived beams (wrappers) and the road-network beams (core.roads).
N_BEAMS = 20
BEAM_ANGLES = _numpy.linspace(-135.0, 135.0, N_BEAMS)
# road beams: a beam that starts on a non-drivable cell (the car on a road
# divider) looks past it, up to this distance, to the edge of the road it
# enters; a beam with no road within it reads zero.
BEAM_SKIP_MAX = 5.0    # (m)
# road-network BEV: only segments at the car's height are rasterized - an
# overpass or a road under a bridge is a different level and would otherwise
# mark its top-down projection drivable. The car's state pos sits ~0.5 m
# above the road surface, so 3 m covers road undulation within the BEV range
# while excluding overpass decks.
ROAD_BEV_Z_TOL = 3.0   # (m)
# road-surface removal: LiDAR returns within this distance of the road
# surface (the nearby navigraph segments' ribbons) are dropped from the
# point cloud, so the BEV reads the road as empty (road level) and only the
# obstacles (returns above the road) as returns. The ribbon follows the
# road's longitudinal slope and lateral bank (the navigraph's per-node
# surface normals; the flat unbanked approximation when the navigraph lacks
# them). 0.5 m covers the low structures (a wall's base, a curb, a tree
# branch) that sit 0.25-0.5 m above the road (measured live: the majority of
# the on-road false obstacles are such structures, not road surface), while
# real obstacles sit well above the road.
ROAD_SURFACE_TOL = 0.5   # (m)
# on-road detection (debug window; a future observation, see
# Navigraph.is_on_road): the car is on-road when its position is within this
# distance of the road surface (horizontal distance to the nearest segment
# centerline minus the local half-width). The navigraph represents the road
# as individual lanes whose road surface is continuous, so the tolerance
# covers the car sitting at a lane divider (between two lane ribbons) or
# straddling the road edge (the state pos is the car's center, ~half a car
# width from the wheels).
ON_ROAD_TOL = 1.0   # (m)
# on-road fast-track: within this lateral distance of the road centerline the
# car is on the road (a real road's half-width plus shoulders fits inside it),
# so the determination defaults to True without the strict edge check (see
# VehicleRig.on_road, fast_track).
ON_ROAD_FAST_TRACK_DIST = 7.0   # (m)
# road-surface merge: each surface's half-width is grown by this margin when
# the road BEV is built, so two surfaces less than 2*margin apart merge into
# one (the lane dividers fill in) and wider gaps stay open. A dilation, not a
# closing: the outer edge grows by the margin too (negligible for the BEV
# fill, and it leaves the centerline unchanged since it is symmetric).
ROAD_MERGE_MARGIN = 0.25   # (m); merges surfaces < 0.5 m apart
# road centerline (debug window, a future reward): the merged road mask is
# Gaussian-blurred and the centerline is the center of mass of each row of the
# blurred mask (the middle of the merged road, not a lane centerline). The
# sigma is wide enough that a typical road is narrower than the blur, so the
# blurred mask peaks at the road's middle.
ROAD_CENTERLINE_SIGMA = 4.0   # (m)

DRIFT_CARS_6SPEED = [
    {"model": "bx", "part_config": "pro_drift_M"},
    {"model": "bastion", "part_config": "drift"},
    {"model": "etk800", "part_config": "858_drift_M"},
    {"model": "etkc", "part_config": "kc8_drift_M"},
    {"model": "etki", "part_config": "drift"},
    {"model": "fullsize", "part_config": "drift"},
    {"model": "sbr", "part_config": "drift"},
]

# The car is fixed for reproducibility (the drift setup is tuned for it); the
# benchmark can still override it per-reset via options={"car": ...}
FIXED_CAR = DRIFT_CARS_6SPEED[4]

# AI (synthetic) mode: the built-in AI routes on the road network and does not
# drive dirt tracks, so waypoints with these name prefixes are excluded from
# AI spawns (the AI would sit still, unable to route from a dirt track)
AI_SPAWN_EXCLUDE = ("dirttrack",)

# Gearbox: the action's gear is one [0, 1] value per allowed gear (see
# action_to_gear / gear_to_action); the gear with the max value is selected,
# and none above 0.5 means neutral. Neutral, 5th and 6th are removed from the
# action space, so the agent picks from reverse and 1st-4th. The env sends the
# absolute gear (tcom convention: -1 reverse, 0 neutral, 1..N forward); the
# game clamps it to the car's range.
MIN_GEAR = -1
MAX_GEAR = 6
# The gears the action can select: reverse and 1st-4th (neutral, 5th and 6th
# are removed from the action space).
ALLOWED_GEARS = [-1, 1, 2, 3, 4]


def action_to_gear(gear_actions):
    """Map the action's gear vector (one [0, 1] value per allowed gear) to an
    absolute gear. The gear with the max value is selected; if none is above
    0.5 the car is in neutral (gear 0).
    """
    gear_actions = np.asarray(gear_actions, float)
    if gear_actions.max() > 0.5:
        return ALLOWED_GEARS[int(np.argmax(gear_actions))]
    return 0


def gear_to_action(gear):
    """Map an absolute gear to the action's gear vector (one [0, 1] value per
    allowed gear). The closest allowed gear gets 1.0, the rest 0.0; neutral
    (gear 0) is all zeros (none above 0.5). Encodes a selected or recorded gear
    (human mode, offline data) into the action space.
    """
    vec = _numpy.zeros(len(ALLOWED_GEARS), dtype=_numpy.float32)
    if gear != 0:
        idx = min(range(len(ALLOWED_GEARS)),
                  key=lambda i: abs(ALLOWED_GEARS[i] - gear))
        vec[idx] = 1.0
    return vec
# The game's shift state machine restarts from "clutchIn" on every gear
# request, so re-sending the same gear every tick restarts the shift before
# it completes (the car revs in neutral forever). The gear is therefore sent
# only when the requested gear changes, plus a timed retry so a failed shift
# self-heals.
GEAR_RESEND_DT = 1.0
# After an agent action that results in a shift between engaged gears, the
# gear stays engaged for at least this many steps: new gear requests are
# ignored in the meantime so the agent can't chatter between gears and
# restart the shift before it settles. Not applied when the car is in
# neutral (gear 0): a shift out of neutral is always honoured, so the agent
# can freely pick its first gear.
MIN_STEPS_ENGAGED = 10
# stalled-engine recovery: below this RPM the engine is considered stalled
# (off or dead) and the starter is engaged to restart it - the code
# equivalent of holding the starter key (V)
ENGINE_STALL_RPM = 100.0
