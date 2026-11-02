"""The replay environment: play back a recorded .rpl and regenerate data.

A BeamNG replay (``.rpl``) is the recorded simulation state, so playing it
re-creates and animates the recorded vehicle in the scene. This environment
loads the map the replay was recorded on, plays the replay, attaches its sensor
rig to the (single) driving vehicle the replay spawns, and steps in lock-step
with the playback - regenerating the same observation/reward structure as the
drive environment, but driven by the human's recorded driving.

Differences from the drive env (all the shared logic lives in
:class:`~beamng_tech_env.envs.base.BeamNGBaseEnv`):

- No ego spawn: it attaches to the vehicle the replay creates.
- Real-time pacing (``GameDriver.set_realtime``) so the physics is not faster
  than real time, and it paces each step to the physics tick.
- The applied action is read back from the electrics sensor (the recorded
  driving), like human mode.
- The episode truncates when the replay ends (``core_replay`` leaves playback,
  which is also when the game pauses).
"""
import time

from beamngpy.logging import BNGDisconnectedError, BNGError, BNGValueError
from beamngpy.sensors import Electrics

from beamng_tech_env.core.constants import (
    RESET_TIMEOUT,
    SETUP_TIMEOUT,
    STEP_TIMEOUT,
    TICK_DT,
)
from beamng_tech_env.core.game import _dbg
from beamng_tech_env.core.obs import reset_is_eval
from beamng_tech_env.core.replay import ReplayController
from beamng_tech_env.core.rpl import RplTrajectory, game_save_dir
from beamng_tech_env.envs.base import BeamNGBaseEnv


class BeamNGReplayEnv(BeamNGBaseEnv):
    """Gymnasium env that plays a .rpl and regenerates data from its vehicle."""

    def __init__(self, render_mode=None, replay_path=None, level="west_coast_usa"):
        super().__init__(render_mode=render_mode, human_mode=False)
        # game-relative path of the .rpl to play by default (e.g.
        # "replays/drift_....rpl"); reset() can override it per-episode with
        # options={"replay": path}
        self.replay_path = replay_path
        self.level = level
        # the .rpl this reset will play (set in reset)
        self._active_replay = None
        # the parsed .rpl trajectory for the active replay (set in reset)
        self._rpl = None
        # the active replay's total length in seconds (set in reset)
        self._replay_total = None
        # the playback position (s) the previous step was taken at; step()
        # waits for the position to advance one dataset tick past this (set
        # in reset)
        self._last_replay_pos = 0.0
        # created in _first_time_setup once the connection exists
        self.replay = None

    # -- reset --------------------------------------------------------------
    def _first_time_setup(self):
        self.driver.connect()
        self.driver.load_scenario(self.level, "Replay")
        # no ego spawn: the replay re-creates the recorded vehicle
        self.driver.start_scenario()

        self.driver.set_window([1280, 720])
        self.driver.disable_driving_assists()
        self.driver.set_deterministic()
        # real-time pacing: the replay drives the sim, and we want it to run at
        # wall-clock speed, not several times faster. set_realtime uses the
        # wall-clock speed factor (0), so the sim - and the replay playback -
        # is 1x regardless of the render speed (see GameDriver.set_realtime).
        self.driver.set_realtime()

        # settle
        time.sleep(1)
        self.replay = ReplayController(self.beamng)
        _dbg("setup: done (real time, no ego)")

    def _find_driving_vehicle(self, tries=25):
        """Find the single driving vehicle the replay spawned.

        The replay re-creates the recorded vehicle in the scene; we attach our
        rig to it. It may take a couple of frames for the vehicle to appear
        after playback starts, so retry. If several vehicles are present (e.g.
        a stale one from an interrupted replay), prefer the player's, else the
        first.
        """
        vehicles = {}
        for _ in range(tries):
            vehicles = self.beamng.vehicles.get_current(include_config=False)
            if vehicles:
                break
            time.sleep(0.2)
        if not vehicles:
            raise BNGError("no vehicles found in the scene after replay start")
        # log everything in the scene so a wrong-vehicle attach is diagnosable
        for vid, veh in vehicles.items():
            _dbg(f"reset: vehicle in scene: vid={vid!r}")
        if len(vehicles) == 1:
            return next(iter(vehicles.values()))
        try:
            player_vid = self.beamng.vehicles.get_player_vehicle_id()["vid"]
            if player_vid in vehicles:
                return vehicles[player_vid]
        except (BNGError, KeyError, TypeError):
            pass
        return next(iter(vehicles.values()))

    def _reset_running_game(self, options):
        # the game pauses when a replay ends, so resume before (re)playing
        self.driver.resume()
        # remove the old LiDAR while the old vehicle still exists (after the
        # replay stops the vehicle is gone and CloseLidar would be a no-op,
        # leaking a simulated LiDAR in the game)
        self.rig.remove_lidar()
        # clear any stale playback / vehicle from a previous (interrupted) run
        if self.replay is not None:
            self.replay.stop()
        # let the game finish tearing down the old playback (and its vehicle)
        # before spawning the new one, so we don't attach to a stale vehicle
        time.sleep(1.0)

        # The State sensor (pos/dir/up/vel) and the electrics are stale during
        # playback, so source them from the .rpl (the recorded ground truth),
        # indexed by the playback time. The LiDAR (BEV) and the FFB are read
        # live from the game, which is why the replay is played at all.
        self._rpl = RplTrajectory(game_save_dir() / self._active_replay)
        self._replay_total = float(self._rpl.t[-1])
        _dbg(f"reset: loaded .rpl trajectory ({len(self._rpl)} frames)")

        if not self.replay.play(self._active_replay, autoplay=True):
            raise BNGError(f"replay failed to start: {self._active_replay!r}")
        _dbg(f"reset: replay playing ({self._active_replay})")

        self.ego = self._find_driving_vehicle()
        _dbg(f"reset: attached to vehicle {self.ego.vid!r}")
        self.rig.attach(self.ego, self.beamng, human_mode=False)
        _dbg("reset: sensors set up")

        self.rig.poll_sensors()
        # correct the (stale) reset observation from the .rpl too
        self._last_replay_pos = self.replay.position_seconds()
        self._apply_rpl_state(self._last_replay_pos)
        self.key = self._active_replay
        time.sleep(0.5)

    def _apply_rpl_state(self, secs):
        """Override the (stale) State sensor and electrics with the .rpl's
        recorded values at playback time ``secs``, so the observation (drift,
        IMU, BEV, rpm, gear, ...) is built from the true pose/telemetry."""
        st = self.ego.sensors["state"]
        rpl = self._rpl.state_at(secs)
        st["pos"] = rpl["pos"]
        st["vel"] = rpl["vel"]
        st["dir"] = rpl["dir"]
        st["up"] = rpl["up"]
        e = self._rpl.electrics_at(secs)
        if e:
            cur = self.ego.sensors["electrics"]
            e = cur._reassign_values(Electrics._rename_values(dict(e)))
            cur.clear()
            cur.update(e)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        eval = reset_is_eval()
        if options is None:
            options = dict(eval=eval)
        # which replay to play: a per-reset override, else the constructor default
        self._active_replay = options.get("replay", self.replay_path)
        if not self._active_replay:
            raise ValueError(
                "no replay to play: pass replay_path= to BeamNGReplayEnv or "
                "options={'replay': path} to reset()")
        _dbg(f"reset requested{" [eval]" if eval else ""}; replay: {self._active_replay}")

        while True:
            try:
                # start BeamNG.drive on the first reset (works with reset-first ordering)
                if self.window is None:
                    self._call_with_timeout(self._first_time_setup, SETUP_TIMEOUT)
                    _dbg("reset: first_time_setup done")
                if self._check_game_fd_budget():
                    continue
                self._call_with_timeout(self._reset_running_game, RESET_TIMEOUT, options)
                self.driver.reset_hang_count()
                return self._finish_reset()
            except (OSError, BNGError, BNGDisconnectedError, BNGValueError,
                    TimeoutError) as err:
                self._recover_from_hang(err)

    # -- step ---------------------------------------------------------------
    def _step_tick(self):
        """Seek the playback forward by one dataset tick and check if it plays.
        Returns ``(playing, target)``: True and the target position if the
        replay is still in playback (and the seek was performed), False and the
        target if it has ended.

        The seek-forward pacing consumes exactly one tick (TICK_DT) of replay
        time per step, regardless of the (variable) per-step processing time or
        the round-trip latency. Waiting for the playback position to advance
        naturally (the old approach) overshoots the one-tick target because the
        tcom round trips that poll it advance the 1x playback, and the poll
        period (~31 ms) does not divide the 50 ms tick evenly. Seeking to an
        absolute target (``anchor + TICK_DT``) in a single round trip avoids the
        overshoot: the seek lands exactly on the target (at the moment of
        execution), so each step consumes exactly one .rpl tick and a
        regenerated episode always spans the full .rpl length.

        The state check and the seek happen in a single tcom round trip
        (``replay.seek_and_check``), so the check costs no extra round trip.
        """
        target = self._last_replay_pos + TICK_DT
        playing = self.replay.seek_and_check(target)
        return playing, target

    def step(self, action):
        t_step = time.perf_counter()
        # The action is ignored: the replay drives the car. The recorded action
        # is read back from the electrics sensor below so the regenerated data
        # has the same structure as the drive env.
        try:
            playing, secs = self._call_with_timeout(self._step_tick, STEP_TIMEOUT)
        except (OSError, BNGError, BNGDisconnectedError) as err:
            return self._recover_and_end_episode(err)

        # The State sensor / electrics are stale during playback: replace them
        # with the .rpl's recorded pose + telemetry at the current playback
        # time so every downstream computation (reward, IMU, drift, BEV, obs)
        # uses the true values. The LiDAR and FFB are read live in get_obs.
        # Poll the sensors once here; the poll's own ~25 ms of playback advance
        # is irrelevant because the pacing is seek-based (each step seeks to an
        # absolute target, so the position is re-anchored every step).
        self._last_replay_pos = secs
        self.rig.poll_sensors()
        self._apply_rpl_state(secs)

        self._update_reward_state()
        self._derive_imu(self.ego.state)
        terminated, failure_reason = self._check_failure()
        reward, comps = self._compute_reward()
        observation = self.rig.get_obs()
        self._last_step_time = self.ego.state["time"]

        # the replay ended (core_replay left playback, or the playback position
        # reached the end while the state still reports 'playback'): either way
        # the episode is over. (The clone's State-sensor time never advances,
        # so it can't be used as the episode clock.)
        replay_ended = (not playing or secs >= self._replay_total - 0.1)
        truncated = replay_ended
        info = self._get_info()
        info["reward_components"] = comps
        info["replay_ended"] = replay_ended
        if terminated:
            info["failure_reason"] = failure_reason
        # read the recorded action back from the electrics sensor (fresh after
        # the tick), matching the drive env's human-mode applied_action
        info["applied_action"] = self._read_human_action()
        self._last_info = info
        self.driver.reset_hang_count()

        # pace to real time (one 50 ms sim tick per 50 ms of wall clock)
        time.sleep(max(0.0, TICK_DT - (time.perf_counter() - t_step)))

        return observation, reward, terminated, truncated, info

    def render(self):
        # the game takes care of this without any work.
        pass
