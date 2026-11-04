"""Shared gymnasium scaffolding for the BeamNG.tech drift environments.

``BeamNGBaseEnv`` holds everything the drive and replay environments have in
common: the action/observation spaces, the reward/failure bookkeeping, the
human-action reader, the game-hang recovery, and the thin delegates that expose
the underlying :class:`~beamng_tech_env.core.game.GameDriver` and
:class:`~beamng_tech_env.core.rig.VehicleRig` as env attributes (so existing
callers that poke at ``env.ego`` / ``env.beamng`` / ``env._poll_sensors`` keep
working).

Subclasses provide the parts that differ: how a step advances the simulation
(drive sends a tcom Control + explicit step; replay runs free and paces to the
physics tick) and how a vehicle is obtained (drive spawns one; replay attaches
to the vehicle a replay spawns).
"""
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from beamngpy.logging import BNGDisconnectedError, BNGError, BNGValueError
from beamngpy.misc.quat import flip_y_axis

from beamng_tech_env.core.constants import (
    ALLOWED_GEARS,
    BEV_N,
    CHECKPOINT_DAMAGE,
    DEFAULT_STEERING_WHEEL_LOCK,
    DAMAGE_LIMIT,
    FPS,
    FPS_FACTOR,
    MAX_GAME_FDS,
    PROGRESS_WINDOW_SECONDS,
    RESET_TIMEOUT,
    SETUP_TIMEOUT,
    SPAWN_DAMAGE_STEPS,
    TCOM_PORT,
    gear_to_action,
)
from beamng_tech_env.core.game import GameDriver, _dbg
from beamng_tech_env.core.obs import neutral_obs
from beamng_tech_env.core.reward import compute_reward
from beamng_tech_env.core.rig import VehicleRig


def make_action_space():
    """The drive action space: [steering, throttle/brake, parkingbrake, gears].

    The gears are one [0, 1] value per allowed gear (reverse, 1st-4th); the
    gear with the max value is selected, and none above 0.5 means neutral (see
    constants.action_to_gear). Throttle and parkingbrake are exclusive: the
    env applies only the larger of the two. Shared by the live env and the
    remote client (envs/remote.py) so the two always agree on the action
    layout.
    """
    n_gears = len(ALLOWED_GEARS)
    low = np.concatenate([np.array([-1.0, -1.0, -1.0]),
                          np.zeros(n_gears, dtype=np.float32)])
    high = np.concatenate([np.array([1.0, 1.0, 1.0]),
                           np.ones(n_gears, dtype=np.float32)])
    return spaces.Box(low=low, high=high, dtype=np.float32)


def make_observation_space():
    """The drive observation space: one Dict, one key per sensor channel.

    Every value is a float already normalized into a bounded range (the BEV
    grids are the exception: uint8 / float32 images). Shared by the live env
    and the remote client (envs/remote.py) so the two always agree.
    """
    return spaces.Dict({
        "rpm":         spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
        "imu_acc":     spaces.Box(low=-1.0,            high=1.0,            shape=(3,),           dtype=np.float32),
        "imu_angvel":  spaces.Box(low=-1.0,            high=1.0,            shape=(3,),           dtype=np.float32),
        "drift_angle": spaces.Box(low=-1.0,            high=1.0,            shape=(1,),           dtype=np.float32),
        "airspeed":    spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
        "is_shifting": spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
        "gear":        spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
        "ffb":         spaces.Box(low=-1.0,            high=1.0,            shape=(1,),           dtype=np.float32),
        "bev":         spaces.Box(low=0,               high=255,            shape=(1, BEV_N, BEV_N), dtype=np.uint8),
        "road_bev":    spaces.Box(low=0.0,             high=1.0,            shape=(1, BEV_N, BEV_N), dtype=np.float32),
    })


class BeamNGBaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS * FPS_FACTOR}

    def __init__(self, render_mode=None, human_mode=False, ai_mode=False,
                 host="localhost", tcom_port=TCOM_PORT, launch=True,
                 transport="ros2"):
        self.human_mode = human_mode
        # AI (synthetic) mode: the game's built-in AI drives the car instead
        # of a human or an agent (see the drive env's _setup_ai_mode)
        self.ai_mode = ai_mode
        # the sensor/control transport: "ros2" (default) reads the sensor
        # stream from the C++ bridge's ROS2 topics (control published to the
        # bridge); "shmem" uses the BeamNGpy socket + shared memory
        self.transport = transport
        # steering wheel lock (deg) of the current car, used to normalise the
        # wheel angle (electrics.steering) into the [-1, 1] action space
        self._steering_wheel_lock = DEFAULT_STEERING_WHEEL_LOCK
        # the reusable building blocks: the game driver and the vehicle rig
        self.driver = GameDriver(host=host, tcom_port=tcom_port, launch=launch,
                                 ros2=(transport == "ros2"))
        self.rig = VehicleRig()
        # the vehicle this env tracks (spawned for drive, discovered for
        # replay); None until the first reset
        self.ego = None
        # the spawn descriptor for the last reset (waypoint name / replay file)
        self.key = ""

        self.action_space = make_action_space()
        self.observation_space = make_observation_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # relevant information for a previous frame, for reward calculation
        self.previous = dict(
            speed = 0.0,
            score = 0.0,
            damage = 0.0,
            distance = 0.0,
            centerline_dist = float("nan"),
            on_road = None,
            reversing = False,
        )

        # relevant information for a current frame, for reward calculation
        self.current = dict(
            speed = 0.0,
            score = 0.0,
            damage = 0.0,
            distance = 0.0,
            centerline_dist = float("nan"),
            on_road = None,
            reversing = False,
        )

        # running total of the self-computed drift score for the episode
        self._score_total = 0.0
        # simulation time at the last reset, for the episode timeout
        self._t0 = 0.0
        # sim time of the last completed step, for the stale-read check
        self._last_step_time = 0.0
        # info dict from the last completed step (read by the reward-components callback)
        self._last_info = {}
        # bird's-eye view at the last reset (read by the callback to log the
        # episode start state)
        self._reset_bev = None
        # BEV of the last step (read by the callback for the crash image when
        # the env is wrapped and the observation no longer carries the BEV)
        self._last_bev = None
        # spawn point of the last reset (pos + where it came from), read by the
        # callback to log where each episode started
        self._reset_pos = None
        self._reset_spawn = ""
        # previous position, for the per-step distance reward
        self._prev_pos = None
        # rolling window of per-step distances, for the progress timeout
        self._dist_window = deque(maxlen=int(PROGRESS_WINDOW_SECONDS * FPS))
        self._dist_window_sum = 0.0
        # last undamaged (pos, quat) checkpoint; persists across episodes
        self._checkpoint = None
        # human mode: the pose to resume from on the next reset, saved when an
        # episode ends (crash or termination); None until the first end
        self._resume_pose = None
        # steps since the last reset (spawn), for the spawn damage check
        self._steps_since_reset = 0
        # last step's is_shifting flag and the last gear we sent; while the
        # gearbox is mid-shift the game needs a quiet period (no throttle, no
        # new gear request) to complete the shift, so step() holds these
        self._last_is_shifting = False
        self._last_requested_gear = 0
        # edge-triggered gear sending: the last gear actually sent and when
        self._last_sent_gear = 0
        self._last_sent_time = 0.0
        # steps left to keep the gear engaged after an agent action shifted
        # the gearbox (see MIN_STEPS_ENGAGED); new requests are ignored while
        # this is above zero
        self._gear_engaged_steps = 0
        # stalled-engine recovery: True while the starter is engaged (see
        # _recover_stalled_engine in the drive env)
        self._engine_cranking = False

    # -- driver / rig access (compat + convenience) -------------------------
    @property
    def free_driving(self):
        """True when the car is driven by an external driver (a human or the
        game's built-in AI): no tcom Control is sent, the game runs free at
        real time, and the applied action is read back from the electrics
        sensor each step."""
        return self.human_mode or self.ai_mode

    @property
    def beamng(self):
        return self.driver.beamng

    @property
    def scenario(self):
        return self.driver.scenario

    @property
    def waypoints(self):
        return self.driver.waypoints

    @property
    def window(self):
        return self.driver.window

    @property
    def _imu_acc(self):
        return self.rig._imu_acc

    @_imu_acc.setter
    def _imu_acc(self, value):
        self.rig._imu_acc = value

    @property
    def _imu_angvel(self):
        return self.rig._imu_angvel

    @_imu_angvel.setter
    def _imu_angvel(self, value):
        self.rig._imu_angvel = value

    def _call_with_timeout(self, func, timeout, *args, **kwargs):
        return self.driver.call_with_timeout(func, timeout, *args, **kwargs)

    def _poll_sensors(self):
        return self.rig.poll_sensors()

    def _get_obs(self):
        return self.rig.get_obs()

    def _poll_lidar_points(self):
        return self.rig.poll_lidar_points()

    def _read_ffb(self):
        return self.rig.read_ffb()

    def _send_sensor_request(self):
        return self.rig.send_sensor_request()

    def _recv_sensor_data(self):
        return self.rig.recv_sensor_data()

    def _derive_imu(self, state):
        return self.rig.update_imu(state)

    # -- reward / failure (shared by drive and replay) ----------------------
    def _update_reward_state(self):
        """Update the reward bookkeeping from fresh sensor data. Called in
        step() before _compute_reward, so the reward reflects the tick that
        just ran."""
        from beamng_tech_env.core.obs import drift_from_state
        drift = drift_from_state(self.ego.state)
        self.current["speed"] = drift["airSpeed"]
        # moving backwards (the velocity has a backward component), scales the
        # reward down (see compute_reward)
        self.current["reversing"] = drift["reversing"]
        self._score_total += drift["scoreAdded"]
        self.current["score"] = self._score_total
        dmg = self.ego.sensors["damage"]
        if isinstance(dmg, dict):
            self.current["damage"] = dmg.get("damage", 0.0)
        else:
            self.current["damage"] = 0.0
        # distance traveled this step (m), rewarded instead of raw speed
        pos = np.asarray(self.ego.state["pos"], float)
        self.current["distance"] = (float(np.linalg.norm(pos - self._prev_pos))
                                    if self._prev_pos is not None else 0.0)
        self._prev_pos = pos
        # distance to the road centerline (m), scales the progress reward; NaN
        # when there is no road nearby (off-road) or the navigraph is missing
        self.current["centerline_dist"] = self.rig.road_centerline_dist()
        # on-road determination: the fast-track reuses the centerline distance
        # just computed for the reward (within ON_ROAD_FAST_TRACK_DIST of the
        # centerline -> on road, no strict edge check); otherwise the strict
        # navigraph check decides
        self.current["on_road"] = self.rig.on_road(
            centerline_dist=self.current["centerline_dist"])
        # rolling distance window for the progress timeout: evict the oldest
        # sample once the window is full, then add the current step's distance
        if len(self._dist_window) == self._dist_window.maxlen:
            self._dist_window_sum -= self._dist_window[0]
        self._dist_window.append(self.current["distance"])
        self._dist_window_sum += self.current["distance"]
        # spawn verification: damage in the first steps after a spawn means
        # the saved spawn point is bad (the car hit something right after
        # spawning), so invalidate it - the next reset picks a waypoint
        self._steps_since_reset += 1
        if (self._steps_since_reset <= SPAWN_DAMAGE_STEPS
                and self.current["damage"] > 0.0):
            self._checkpoint = None
            self._resume_pose = None
        # checkpoint: remember the last undamaged pose so reset can respawn there
        if self.current["damage"] < CHECKPOINT_DAMAGE:
            if pos[2] < 1.0:
                # the car fell into the water: don't respawn here, reset the
                # saved spawn to a waypoint (None makes reset pick one)
                self._checkpoint = None
            else:
                # rotate the saved pose 180 deg about the vertical (z) axis so the
                # car respawns facing the opposite direction
                self._checkpoint = (tuple(self.ego.state["pos"]),
                                    flip_y_axis(tuple(self.ego.state["rotation"])))

    def _check_failure(self):
        """Return (failed: bool, reason: str) from the Damage sensor poll."""
        dmg = self.ego.sensors["damage"]
        if not isinstance(dmg, dict):
            return False, ""
        overall = dmg.get("damage", 0.0)
        tire_flat = dmg.get("lowpressure")

        if overall > DAMAGE_LIMIT:
            return True, f"damage={overall:.3f}"
        if tire_flat:
            return True, f"tire flat (lowpressure={tire_flat})"
        return False, ""

    def _compute_reward(self):
        """Return (total, components_dict). All values are floats."""
        # off-road: the on_road determination is False (None = unknown, e.g.
        # no navigraph, counts as on-road); reversing: the drift data
        total, comps = compute_reward(
            self.current["distance"],
            self.current["score"] - self.previous["score"],
            self.current["damage"] - self.previous["damage"],
            self.current["centerline_dist"],
            offroad=self.current["on_road"] is False,
            reversing=self.current["reversing"])
        # copy, not a reference: _update_reward_state mutates self.current in
        # place, so a reference would make every delta zero after the first step
        self.previous = dict(self.current)
        return total, comps

    def _read_human_action(self):
        """Reconstruct the applied action from the electrics sensor.

        The wheel angle (electrics.steering, in degrees) is normalised by the
        car's steering wheel lock into the [-1, 1] steering axis. Throttle and
        brake are read from the ``*_input`` fields (the actual pedal input, each
        0..1) rather than the applied values, which carry a non-zero baseline
        (e.g. ``brake`` idles at ~0.3); they are combined into the single
        [-1, 1] throttle/brake axis (positive = throttle, negative = brake),
        matching the action the env would have sent via tcom Control. The gear
        is one [0, 1] value per allowed gear, encoding the absolute gear the
        driver selected via ``gear_to_action`` (the nearest allowed gear).
        """
        e = self.ego.sensors["electrics"]
        steering = float(e.get("steering", 0.0)) / self._steering_wheel_lock
        combined = float(e.get("throttle_input", 0.0)) - float(e.get("brake_input", 0.0))
        parkingbrake = float(e.get("parkingbrake_input", 0.0))
        gear = int(round(float(e.get("gear", 0))))
        return np.concatenate([
            np.array([
                float(np.clip(steering, -1.0, 1.0)),
                float(np.clip(combined, -1.0, 1.0)),
                float(np.clip(parkingbrake, -1.0, 1.0)),
            ], dtype=np.float32),
            gear_to_action(gear),
        ])

    def _get_info(self):
        return {}

    # -- recovery (shared) --------------------------------------------------
    def _recover_from_hang(self, err):
        """The game stopped answering tcom (or the ego was lost): clear the
        env's vehicle/rig and force a full re-setup on the next reset, so the
        env reconnects (if needed) and re-spawns its ego.

        Resetting ``driver.window`` to None is what makes the next reset re-run
        ``_first_time_setup`` (reconnect + re-spawn). Without it, a lost ego
        (e.g. one despawned by another env sharing the same game) would leave
        the env stuck: it would skip ``_first_time_setup`` and then fail in
        ``_reset_running_game`` with a None/despawned ego.
        """
        self.driver.recover_from_hang(err)
        self.ego = None
        self.rig.reset()
        self.driver.window = None

    def _recover_and_end_episode(self, err):
        """A step wedged the game: kill and relaunch it, then end the episode.

        If the relaunch itself wedges, return a neutral observation and let
        the next reset() try again; each attempt counts toward
        MAX_HANG_RECOVERIES, so a systemic problem still crashes training.
        """
        self._recover_from_hang(err)
        try:
            # rebuild the game and get a vehicle + rig the same way a normal
            # reset would (drive re-spawns the ego; replay plays the replay and
            # attaches to the recorded vehicle)
            self._call_with_timeout(self._first_time_setup, SETUP_TIMEOUT)
            self._call_with_timeout(self._reset_running_game, RESET_TIMEOUT, None)
            self.rig.reset_imu_state()
            self._last_step_time = self.ego.state["time"]
            self._score_total = 0.0
            self._t0 = self.ego.state["time"]
            _dbg("recover: fresh game ready, ending episode")
            return self.rig.get_obs(), 0.0, False, True, self._get_info()
        except (OSError, BNGError, BNGDisconnectedError, BNGValueError,
                TimeoutError) as err2:
            self._recover_from_hang(err2)
            _dbg("recover: relaunch also failed, ending episode with neutral obs")
            return self._neutral_obs(), 0.0, False, True, self._get_info()

    def _check_game_fd_budget(self):
        """Restart the game if its open-fd count is past MAX_GAME_FDS.

        The game leaks 2 shm fds per LiDAR lifecycle (see
        GameDriver.game_fd_count); past ~1024 its tcom select() dies. Returns
        True if the game was restarted (the caller must retry the reset).
        Only applies to a game we launched (we own the process and can
        restart it); a game we only connected to has its fd budget owned by
        whoever launched it, so we leave it alone.
        """
        if not self.driver.owns_process:
            return False
        fds = self.driver.game_fd_count()
        if fds is not None and fds > MAX_GAME_FDS:
            _dbg(f"reset: game fd count {fds} > {MAX_GAME_FDS}; restarting game")
            self._recover_from_hang(RuntimeError(f"game fd count {fds}"))
            return True
        return False

    # -- reset helpers (shared) ---------------------------------------------
    def _finish_reset(self):
        self.rig.reset_imu_state()
        self._last_is_shifting = False
        self._last_requested_gear = 0
        self._last_sent_gear = 0
        self._last_sent_time = self.ego.state["time"]
        self._gear_engaged_steps = 0
        self._last_step_time = self.ego.state["time"]
        self._engine_cranking = False
        # anchor the distance baseline to the spawn pose so the first step's
        # distance is measured from the respawn point, not the prior episode
        self._prev_pos = np.asarray(self.ego.state["pos"], float)
        # reset the progress window for the new episode
        self._dist_window.clear()
        self._dist_window_sum = 0.0
        # reset the spawn damage counter for the new episode
        self._steps_since_reset = 0

        obs = self.rig.get_obs()
        # keep the reset bird's-eye view so the callback can log the start state
        self._reset_bev = obs["bev"]
        # keep the full reset observation so the callback can also log the
        # numeric spawn channels (rpm/gear/airspeed/...) next to the BEV image
        self._reset_obs = obs
        # keep the spawn point (fixed for the episode; _prev_pos moves each step)
        self._reset_pos = np.asarray(self.ego.state["pos"], float)
        # anchor the episode clock and score to the freshly spawned car
        self._score_total = 0.0
        self._t0 = self.ego.state["time"]
        info = self._get_info()
        # the spawn descriptor (waypoint name for drive, replay file for replay)
        info["spawn"] = self.key
        _dbg("reset: done")
        return obs, info

    def _neutral_obs(self):
        return neutral_obs()

    def close(self):
        self.driver.close()
