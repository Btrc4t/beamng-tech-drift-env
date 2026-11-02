"""The drive environment: spawn a car and let an agent (or a human) drive it.

This is the gymnasium layer on top of the reusable core blocks. It composes a
:class:`~beamng_tech_env.core.game.GameDriver` (game + scenario) and a
:class:`~beamng_tech_env.core.rig.VehicleRig` (sensors + observation) and adds
the drive-specific behaviour: spawning a fresh ego vehicle each reset, sending
a tcom ``Control`` + explicit step each tick, the reward/failure rules, and
optional human mode (a person drives) or AI mode (the game's built-in AI
drives, for synthetic data collection).

The pure observation math lives in :mod:`beamng_tech_env.core.obs` and the
shared constants in :mod:`beamng_tech_env.core.constants`; both are re-exported
here so existing imports (``from ...beamng_tech import FPS``) keep working.
"""
import random
import time

import numpy as np
from beamngpy import Vehicle
from beamngpy.logging import BNGDisconnectedError, BNGError, BNGValueError

from beamng_tech_env.core.constants import (
    AI_SPAWN_EXCLUDE,
    DEFAULT_STEERING_WHEEL_LOCK,
    ENGINE_STALL_RPM,
    FIXED_CAR,
    FPS,
    GEAR_RESEND_DT,
    MIN_STEPS_ENGAGED,
    PROGRESS_MIN_DISTANCE,
    RENDER_FPS,
    RESET_TIMEOUT,
    SETUP_TIMEOUT,
    STEP_TIMEOUT,
    TCOM_PORT,
    EPISODE_TIMEOUT,
    action_to_gear,
)
from beamng_tech_env.core.game import _dbg
from beamng_tech_env.core.obs import reset_is_eval
from beamng_tech_env.envs.base import BeamNGBaseEnv


class BeamNGTechEnv(BeamNGBaseEnv):
    """Gymnasium env that spawns a car and steps it with tcom Control + Step.

    Parameters
    ----------
    host:
        The tcom host to connect to (or launch the game on).
    tcom_port:
        The tcom port to connect to (or launch the game on).
    launch:
        Whether the env may launch a local game when nothing is listening.
        Set to False for a hosted instance (launched by ``host_game.py``).
    """

    def __init__(self, render_mode=None, human_mode=False, ai_mode=False,
                 ai_speed=None, ai_aggression=0.3, host="localhost",
                 tcom_port=TCOM_PORT, launch=True, transport="ros2"):
        super().__init__(render_mode=render_mode, human_mode=human_mode,
                         ai_mode=ai_mode, host=host, tcom_port=tcom_port,
                         launch=launch, transport=transport)
        # AI mode settings (beamngpy's Vehicle.ai): True means the default
        # "span" mode (drive the whole road network); a string selects a
        # specific mode ("random", "traffic", ...)
        self._ai_mode_name = "span" if ai_mode is True else ai_mode
        self._ai_speed = ai_speed
        self._ai_aggression = ai_aggression

    # -- reset --------------------------------------------------------------
    def _first_time_setup(self):
        self.driver.connect()
        # Reuse the scenario if the game already has it loaded (saves the
        # ~35 s map load); otherwise load it fresh.
        if not self.driver.reuse_scenario("west_coast_usa", "Gymnasium"):
            self.driver.load_scenario("west_coast_usa", "Gymnasium")

        # done in reset (the actual spawn); pick a waypoint and a vehicle
        car = FIXED_CAR
        self.key = random.choice(list(self.driver.waypoints.keys()))
        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.driver.scenario.add_vehicle(
            self.ego,
            pos=self.driver.waypoints[self.key].pos,
            rot_quat=self.driver.waypoints[self.key].rot,
        )
        self.driver.start_scenario()

        self.driver.set_window([1280, 720] if self.human_mode else [640, 480])
        self.driver.disable_driving_assists()
        self.driver.set_deterministic()
        self.driver.set_render_fps(RENDER_FPS)

        # settle
        time.sleep(1)
        # pause physics / time
        self.driver.pause()
        _dbg("setup: done (paused)")

    def _reset_running_game(self, options):
        options = options or {}
        r_control = self.ego._send(dict(
            type="Control",
            steering=0.0,
            throttle=0.0,
            brake=0.0,
            parkingbrake=1.0
        ))
        r_control.recv("Controlled")
        # unpause physics / time
        self.driver.resume()
        time.sleep(0.5)
        _dbg("reset: resumed")
        # vehicle switch: remove the old LiDAR first, while its vehicle still
        # exists (after removal CloseLidar is a no-op and the LiDAR leaks in
        # the game, slowing the main loop on every subsequent reset)
        self.rig.remove_lidar()
        self.driver.scenario.remove_vehicle(self.ego)
        self.driver.beamng.scenario.restart()
        _dbg("reset: scenario restarted")
        car = FIXED_CAR
        waypoint_keys = list(self.driver.waypoints.keys())
        if self.ai_mode:
            # the built-in AI routes on the road network and does not drive
            # dirt tracks, so keep only road waypoints for its spawns
            waypoint_keys = [k for k in waypoint_keys
                             if not k.startswith(AI_SPAWN_EXCLUDE)]
        self.key = waypoint_keys[self.np_random.integers(len(waypoint_keys))]
        if "car" in options:
            car = options["car"]
        if "eval" in options and options["eval"] is True:
            self.key = waypoint_keys[self.np_random.integers(len(waypoint_keys))]
        # respawn at the last undamaged checkpoint if we have one, otherwise
        # fall back to the fixed spawn waypoint
        # on eval switch the spawn point
        # human mode resumes from the saved end pose (crash/termination); AI
        # mode and a human with no saved pose spawn at a random waypoint
        if self.free_driving:
            if self.human_mode and self._resume_pose is not None:
                spawn_pos, spawn_quat = self._resume_pose
                spawn_desc = "resume"
                self.key = "resume"
            else:
                spawn_pos = self.driver.waypoints[self.key].pos
                spawn_quat = self.driver.waypoints[self.key].rot
                spawn_desc = f"waypoint {self.key}"
        elif ("eval" not in options or options["eval"] is False) and self._checkpoint is not None:
            spawn_pos, spawn_quat = self._checkpoint
            spawn_desc = "checkpoint"
        else:
            spawn_pos = self.driver.waypoints[self.key].pos
            spawn_quat = self.driver.waypoints[self.key].rot
            spawn_desc = f"waypoint {self.key}"
        # remember where this episode spawned from, for the TB callback
        self._reset_spawn = spawn_desc
        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.driver.scenario.add_vehicle(
            self.ego, pos=spawn_pos, rot_quat=spawn_quat
        )
        _dbg(f"setup: vehicle added {car} at {spawn_desc} {tuple(round(c, 1) for c in spawn_pos)}")

        self.rig.attach(self.ego, self.beamng, self.human_mode,
                        transport=self.transport)
        _dbg("reset: sensors set up")

        # Human and AI mode let the game run freely (not paused): the FFB
        # force is computed in lua either way, but it only reaches the wheel
        # hardware while the game is running, so a paused/stepped car has no
        # force feedback. The parked car (parking brake engaged) won't move on
        # its own, and step() paces to real time by waiting for the physics
        # tick.
        if self.free_driving:
            _dbg(f"reset: running free ({'human' if self.human_mode else 'AI'} mode), polling sensors")
        else:
            self.driver.pause()
            _dbg("reset: paused, polling sensors")

        self.rig.poll_sensors()

        if self.human_mode:
            self._setup_human_mode()
        elif self.ai_mode:
            self._setup_ai_mode()
        else:
            # tcom vehicles spawn in neutral; release the parking brake and
            # hand the car over in neutral (the policy shifts for itself)
            r = self.ego._send(dict(type="Control", steering=0.0, throttle=0.0,
                                    brake=0.0, parkingbrake=0.0, gear=0))
            r.recv("Controlled")
        time.sleep(1)

    def _save_resume_pose(self):
        """Human mode: remember the end pose so the next reset resumes here.

        Called when an episode ends (crash or termination); the saved pose is
        restored on the next reset so the driver keeps collecting from where
        they left off. A pose in the water is skipped (the previously saved
        pose, if any, is kept).
        """
        pos = self.ego.state["pos"]
        if pos[2] < 1.0:
            return
        self._resume_pose = (tuple(pos), tuple(self.ego.state["rotation"]))

    def _setup_human_mode(self):
        """Human mode: read the wheel lock and hand the car to the driver.

        No tcom Control is sent during the episode, so the game's own input
        (wheel/pedals/keyboard) drives the car. We read the steering wheel
        lock to normalise the wheel angle, then zero all inputs (including
        releasing the parking brake a fresh spawn engages) so the car is
        drivable the moment the driver takes over.
        """
        lock = self.ego.queue_lua_command(
            "return v.data.input.steeringWheelLock", response=True)
        try:
            self._steering_wheel_lock = float(lock)
        except (TypeError, ValueError):
            self._steering_wheel_lock = DEFAULT_STEERING_WHEEL_LOCK
            _dbg(f"reset: could not read steering wheel lock ({lock!r}); "
                 f"using default {DEFAULT_STEERING_WHEEL_LOCK}")
        # hand the car over in neutral: tcom vehicles spawn in neutral, and
        # the driver shifts for themselves
        r = self.ego._send(dict(type="Control", steering=0.0, throttle=0.0,
                                brake=0.0, parkingbrake=0.0, gear=0))
        r.recv("Controlled")
        # Run the physics at real time (1x) - see GameDriver.set_realtime
        self.driver.set_realtime()
        _dbg(f"reset: human mode ready (steering wheel lock="
             f"{self._steering_wheel_lock:g} deg, physics at real time)")

    def _setup_ai_mode(self):
        """AI (synthetic) mode: hand the car to the game's built-in AI driver.

        Sets the AI mode on the spawned vehicle (beamngpy's ``Vehicle.ai``)
        and switches the gearbox to arcade (the AI shifts for itself), then
        runs the physics at real time, exactly like human mode. No tcom
        Control is sent at all (a Control can reset the arcade gearbox, and
        the AI sends its own inputs once it starts driving), so step() reads
        the applied action back from the electrics sensor and the collector
        records it as the episode's actions.
        """
        ai = self.ego.ai
        ai.set_mode(self._ai_mode_name)
        if self._ai_speed is not None:
            ai.set_speed(self._ai_speed)
        ai.set_aggression(self._ai_aggression)
        # the AI shifts for itself: arcade gearbox (the rig attached in
        # realistic_manual, which is for a human or agent shifting). The
        # vehicle's lua controller initializes after spawn and re-applies its
        # default gearbox behavior, so a shift-mode set right after spawn is
        # reverted (notebook 01 hit this); wait for the init to settle, then
        # set arcade and verify it stuck
        behavior = None
        for _ in range(10):
            time.sleep(0.5)
            self.ego.set_shift_mode("arcade")
            behavior = self.ego.queue_lua_command(
                "return controller.mainController and "
                "controller.mainController.gearboxBehavior or 'none'",
                response=True)
            if behavior == "arcade":
                break
        _dbg(f"reset: arcade gearbox {'set' if behavior == 'arcade' else 'NOT set (reverted?)'}")
        # No tcom Control at all in AI mode: a Control request can reset the
        # arcade gearbox (a gear request in arcade behavior switches it back
        # to realistic). The AI sends its own inputs (including releasing the
        # parking brake a fresh spawn engages) once it starts driving.
        # Run the physics at real time (1x) - see GameDriver.set_realtime
        self.driver.set_realtime()
        _dbg(f"reset: AI mode ready (mode={self._ai_mode_name}, "
             f"speed={self._ai_speed}, aggression={self._ai_aggression}, "
             f"physics at real time)")

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        time = self.ego.state['time'] - self._t0 if self.ego is not None else 0
        dmg = self.ego.sensors["damage"] if self.ego is not None else None
        lowpressure = dmg.get("lowpressure", 0.0) if isinstance(dmg, dict) else 0.0
        eval = reset_is_eval()
        _dbg(f"reset requested{" [eval]" if eval else ""}; prev_damage: {self.current['damage']}; prev_timer: {time}; lowpressure: {lowpressure}")
        if options is None:
            options = dict(
              eval = eval
            )

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
    def _recover_stalled_engine(self):
        """Restart a stalled engine (RPM < ENGINE_STALL_RPM) by engaging the
        starter - the code equivalent of holding the starter key (V). Setting
        the ignition to the cranking level (3) engages the starter; the game's
        vehicle controller drops it back to the running level (2) once the
        engine catches, so a single crank request is enough. The
        ``_engine_cranking`` flag avoids re-issuing the (blocking) lua request
        on every tick while the starter is already engaged."""
        rpm = float(self.ego.sensors["electrics"].get("rpm", 0.0))
        if rpm >= ENGINE_STALL_RPM:
            self._engine_cranking = False
            return
        if self._engine_cranking:
            return
        self.ego.queue_lua_command("electrics.setIgnitionLevel(3)")
        self._engine_cranking = True

    def _step_tick(self, steering, throttle, brake, parkingbrake, gear):
        # Human / AI mode: the external driver (wheel/pedals or the game's
        # built-in AI) drives the car, so no tcom Control is sent; the applied
        # action is read back from the electrics sensor after the tick (see
        # step).
        if not self.free_driving:
            if self.rig.ros2 is not None:
                # ROS2 transport: publish control (fire-and-forget; the bridge
                # applies it in onPhysicsStep when a new frame arrives and only
                # shifts when the gear actually changed), fire the Step, wait
                # for the tick to complete (tick-pacing off the bridge's Time
                # topic - the ROS2 equivalent of the socket path's blocking
                # sensor recv), then read the fresh sensors from the topics (no
                # socket round trip).
                self.rig.ros2.publish_control(
                    steering, throttle, brake,
                    0.0 if parkingbrake < 0.0 else parkingbrake,
                    gear if gear is not None else self._last_sent_gear)
                self.beamng._send(dict(type="Step", count=1, ack=False))
                # queue the FFB read (socket; the bridge has no FFB topic) so
                # it is processed while we wait for the tick (read in get_obs)
                self.rig.send_ffb_request()
                self.rig.ros2.wait_for_tick(self._last_step_time)
                self.rig.recv_sensor_data()  # populates from the topics
                return
            # Store the control input and wait for the game to accept
            # it. This must complete before the step so the tick runs with the
            # new input regardless of the order the game drains its sockets.
            # gear is None when the requested gear did not change (see step):
            # omitting the field keeps the game's shift state machine from
            # restarting a shift that is already in progress.
            control = dict(
                type="Control",
                steering=steering,
                throttle=throttle,
                brake=brake,
                parkingbrake=0.0 if parkingbrake < 0.0 else parkingbrake,
            )
            if gear is not None:
                control["gear"] = gear
            r_control = self.ego._send(control)
            r_control.recv("Controlled")
            # One tick plus the sensor read, sent as one batch. No ack
            # on the Step: the game processes requests in order and the tick runs
            # inside the Step handler, so by the time the sensor data arrives the
            # tick is done. The state["time"] check below catches a stale read
            # (vehicle socket drained before the engine socket in the same frame)
            # and re-polls once.
            self.beamng._send(dict(type="Step", count=1, ack=False))
            self.rig.send_sensor_request()
            # queue the FFB read in the same batch: the game runs the tick
            # first, then the lua command, so the force is the post-tick
            # value and the read costs no extra round trip
            self.rig.send_ffb_request()
            self.rig.recv_sensor_data()
            if self.ego.state["time"] - self._last_step_time < 0.04:
                self.rig.poll_sensors()
        else:
            # Free driving: the game runs freely, so the physics ticks on its
            # own. Wait for the sim time to advance by a full tick (this also
            # paces the loop to real time), then read the fresh state.
            # Queue the FFB read first: the vehicle socket only drains at the
            # vehicle's own update rate, so a blocking FFB round trip costs a
            # full vehicle-lua period (~30 ms); queued now, it is processed
            # while we wait for the tick and get_obs' read_ffb only receives
            # the already-sent response (the force is a few ms stale, which
            # is fine for a 50 ms tick).
            self.rig.send_ffb_request()
            for _ in range(20):
                self.rig.poll_sensors()
                if self.ego.state["time"] - self._last_step_time >= 0.04:
                    break
                time.sleep(0.01)

    def step(self, action):
        t_step = time.perf_counter()
        action = np.asarray(action, dtype=np.float32)
        steering, combined, parkingbrake = (float(value) for value in action[:3])
        brake = 0.0
        throttle = 0.0
        if combined > 0.0:
            throttle = combined
        else:
            brake = -combined
        # Throttle and parking brake are exclusive: only the larger of the two
        # contradicting actions is applied. Simple max on the raw throttle
        # (not abs): a negative combined means braking, which does not
        # contradict the parking brake.
        if throttle >= parkingbrake:
            parkingbrake = 0.0
        else:
            throttle = 0.0
        # Absolute gear: the gear vector is one [0, 1] value per allowed gear
        # (reverse and 1st-4th); the max picks the gear, none above 0.5 means
        # neutral; the game clamps the result to the car's range.
        gear = action_to_gear(action[3:])

        if not self.free_driving and self._last_is_shifting:
            # The gearbox is mid-shift: the game's shift state machine needs a
            # quiet period to complete. Lifting the throttle and holding the
            # last requested gear (not re-requesting a new one) lets it finish;
            # the model resumes full control on the next step.
            throttle = 0.0
            gear = self._last_requested_gear
        elif not self.free_driving and self._gear_engaged_steps > 0 and self._last_sent_gear != 0:
            # The previous agent action shifted the gearbox: keep the gear
            # engaged for MIN_STEPS_ENGAGED steps so the agent can't chatter
            # between gears; ignore new requests in the meantime. Skipped in
            # neutral: a shift out of gear 0 is always honoured.
            gear = self._last_requested_gear
            self._gear_engaged_steps -= 1
        else:
            self._last_requested_gear = gear
            if not self.free_driving:
                if self._last_sent_gear == 0:
                    # Shifting out of neutral: no minimum engagement, so the
                    # agent can freely pick (and re-pick) its first gear.
                    self._gear_engaged_steps = 0
                elif gear != self._last_sent_gear:
                    # The action results in a shift: keep the gear engaged for
                    # at least MIN_STEPS_ENGAGED steps.
                    self._gear_engaged_steps = MIN_STEPS_ENGAGED

        # Send the gear only when it changes (or on a timed retry): the game
        # restarts the shift state machine on every gear request, so sending
        # the same gear every tick keeps restarting the shift before it can
        # complete and the car sits in neutral.
        send_gear = (gear != self._last_sent_gear
                     or self.ego.state["time"] - self._last_sent_time >= GEAR_RESEND_DT)
        if send_gear:
            self._last_sent_gear = gear
            self._last_sent_time = self.ego.state["time"]

        try:
            self._call_with_timeout(
                self._step_tick, STEP_TIMEOUT,
                steering, throttle, brake, parkingbrake,
                gear if send_gear else None,
            )
        except (OSError, BNGError, BNGDisconnectedError) as err:
            return self._recover_and_end_episode(err)

        # reward and failure from fresh data, then build the observation
        self._update_reward_state()
        self._derive_imu(self.ego.state)
        # the agent can stall the engine (e.g. a high gear at low speed);
        # restart it the way a driver would (hold the starter) so the episode
        # can continue - human mode lets the driver do it themselves
        if not self.human_mode:
            self._recover_stalled_engine()
        terminated, failure_reason = self._check_failure()
        reward, comps = self._compute_reward()
        observation = self.rig.get_obs()
        self._last_is_shifting = bool(observation["is_shifting"][0])
        self._last_step_time = self.ego.state["time"]
        # truncate if the car covered less than PROGRESS_MIN_DISTANCE over the
        # full rolling window (it has stopped making forward progress)
        no_progress = (len(self._dist_window) >= self._dist_window.maxlen
                       and self._dist_window_sum < PROGRESS_MIN_DISTANCE)
        truncated = ((self.ego.state["time"] - self._t0) > EPISODE_TIMEOUT
                     or no_progress)
        if self.human_mode and (terminated or truncated):
            self._save_resume_pose()
        info = self._get_info()
        info["reward_components"] = comps
        # the game's open-fd count (it leaks 2 per reset; the fd-budget check
        # in reset restarts the game past MAX_GAME_FDS)
        info["game_fds"] = self.driver.game_fd_count()
        if terminated:
            info["failure_reason"] = failure_reason
        if self.free_driving:
            # the external driver (wheel/pedals or the built-in AI) drove the
            # car this tick: read the applied action back from the electrics
            # sensor (fresh after the tick)
            info["applied_action"] = self._read_human_action()
        self._last_info = info
        self.driver.reset_hang_count()

        if self.free_driving:
            # the game runs free (not stepped), so pace the loop to real time
            # (one 50 ms sim tick per 50 ms of wall clock); otherwise the car
            # would outrun the driver's reactions
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t_step)))

        return observation, reward, terminated, truncated, info

    def render(self):
        # the game takes care of this without any work.
        pass
