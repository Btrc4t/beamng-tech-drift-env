"""A sensor suite attached to a single vehicle, by handle.

``VehicleRig`` is the reusable "attach sensors to a vehicle and read its
observation" building block. It does not care *how* the vehicle got there: the
drive environment spawns it, the replay environment discovers an already
running one. Given a ``beamngpy.Vehicle`` handle it attaches the electrics,
damage and LiDAR sensors, polls them (blocking or pipelined), derives the IMU
from State snapshots, and builds the shared observation dict.
"""
import numpy as _numpy
from contextlib import nullcontext

from beamngpy.sensors import Damage, Electrics, Lidar

from beamng_tech_env.core.array import (
    make_stream,
    np,
    to_cpu,
    to_gpu,
    use_device,
)
from beamng_tech_env.core.constants import (
    AIRSPEED_MAX_KPH,
    ANGLE_MAX,
    BEV_N,
    CAR_LEN,
    CAR_WID,
    CAR_Z_HI,
    CAR_Z_LO,
    LIDAR_DENSITY,
    LIDAR_HZ,
    LIDAR_MAX_DIST,
    LIDAR_RES_V,
    LIDAR_V_ANGLE,
    MAX_GEAR,
    ON_ROAD_FAST_TRACK_DIST,
    ROAD_MERGE_MARGIN,
)
from beamng_tech_env.core.obs import (
    bev_to_obs,
    car_rotation,
    derive_imu,
    drift_from_state,
    lidar_bev,
    state_snapshot,
)
from beamng_tech_env.core.roads import (
    car_grid_axes,
    get_navigraph,
    road_centerline,
    road_centerline_segments,
)
from beamng_tech_env.core import ros2_bridge


class VehicleRig:
    """Sensors + observation for one vehicle.

    Call :meth:`attach` with a vehicle handle (and the BeamNGpy connection) to
    set up the sensor suite; the vehicle is connected to the game if it is not
    already. Then use :meth:`poll_sensors` (or the pipelined
    :meth:`send_sensor_request` / :meth:`recv_sensor_data`) to refresh the
    readings and :meth:`get_obs` to build the observation.
    """

    def __init__(self):
        self.vehicle = None
        self.beamng = None
        self.electrics = None
        self.damage = None
        self.lidar = None
        # the ROS2 bridge (transport="ros2"); None on the shmem path. Owns the
        # in-game LiDAR when active (the bridge creates it), so remove_lidar /
        # reset also tear it down (see below).
        self.ros2 = None
        # State snapshot from the previous step, for the derived IMU
        self._prev_state = None
        # derived IMU channels for the current step (see update_imu)
        self._imu_acc = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        # BEV of the last get_obs (read by callbacks)
        self._last_bev = None
        # road BEV of the last get_obs (1.0 = drivable; read by callbacks)
        self._last_road_bev = None
        # the full navigraph, fetched once and cached (core.roads.get_navigraph)
        self._navigraph = None
        # in-flight FFB lua response (see send_ffb_request)
        self._ffb_resp = None
        # in-flight sensor request responses (see send_sensor_request)
        self._sensor_engine_resp = None
        self._sensor_vehicle_resp = None
        # non-blocking CUDA streams for the two independent get_obs sub-paths
        # (the LiDAR point-cloud path and the road raster path share no written
        # memory, so they overlap on the GPU; None when cupy is not used)
        self._pc_stream = make_stream()
        self._road_stream = make_stream()

    # -- vehicle access -----------------------------------------------------
    @property
    def state(self):
        """The vehicle's latest State snapshot (refreshed by a poll)."""
        return self.vehicle.state

    @property
    def sensors(self):
        return self.vehicle.sensors

    # -- setup --------------------------------------------------------------
    def remove_lidar(self):
        """Remove the LiDAR from the game and release its shared memory.

        The game's LiDAR registry is keyed by name and only consulted by
        CloseLidar; when the owning vehicle is destroyed the sensor is
        dropped from the game's sensor manager but the LiDAR keeps running
        in the game's per-frame sensor pass (and its shared memory stays
        open). Call this while the vehicle still exists, before it is
        removed, so CloseLidar finds the live sensor: otherwise every reset
        leaks one simulated LiDAR and the game's main loop slows down in
        proportion to the number of resets.
        """
        # ROS2 transport: stop the bridge (stopRos) to remove its ROS2 node
        # and sensor stream. The LiDAR is the beamngpy one (shared memory),
        # removed below - the bridge does not create its own LiDAR.
        if self.ros2 is not None:
            self.ros2.stop()
            self.ros2 = None
        lidar = self.lidar
        if lidar is None:
            return
        self.lidar = None
        try:
            lidar.remove()
        except Exception:
            pass
        for shmem in (lidar.point_cloud_shmem, lidar.colour_shmem):
            if shmem is not None:
                try:
                    shmem.unlink()
                except Exception:
                    pass

    def attach(self, vehicle, beamng, human_mode=False, connect=True,
               transport="ros2"):
        """Attach the sensor suite to ``vehicle`` and (optionally) connect it.

        The vehicle must already exist in the running game (spawned for the
        drive env, created by the replay playback for the replay env).

        ``transport`` selects the sensor/control path: "ros2" (default) reads
        the sensor stream from the C++ bridge's ROS2 topics (control is
        published to the bridge and the LiDAR is created by it); "shmem" uses
        the BeamNGpy socket + shared memory (the legacy path). If ROS2 is not
        available or the bridge fails to start, "ros2" falls back to the shmem
        path (with a warning), so the env still runs.
        """
        # defensive: if the caller did not remove the previous LiDAR while
        # its vehicle still existed, do it now (a no-op once the vehicle is
        # gone, but still better than leaking a simulated LiDAR)
        self.remove_lidar()

        self.vehicle = vehicle
        self.beamng = beamng

        # connect the vehicle first (opens its socket); a vehicle added through
        # the scenario API is already connected, so this is a no-op for the
        # drive env and does the real work for a replay-discovered vehicle. The
        # calls below (set_shift_mode, queue_lua_command) send over this socket,
        # so the connection must exist before them.
        if connect and not vehicle.is_connected():
            vehicle.connect(beamng)

        # realistic manual gearbox: the agent shifts gears (sequentially) and
        # the game's autoClutch assist handles the clutch
        vehicle.set_shift_mode("realistic_manual")

        # Enable the virtual steering wheel so the game computes the wheel
        # force feedback (read back in read_ffb) with no physical wheel
        # hardware. In human mode a real wheel is connected and already drives
        # the FFB (FFBID >= 0), so the virtual wheel is not enabled: it would
        # divert the FFB away from the real wheel and change the steering input
        # source. The FFB is still readable via hydros.forceAtDriver either way.
        if not human_mode:
            vehicle.queue_lua_command("hydros.enableVirtualWheel(true)")

        # Sensors (State is auto-attached by beamngpy; the ROS2 bridge
        # repopulates it from its topic, so it is attached either way)
        self.electrics = Electrics()
        self.damage = Damage()
        vehicle.sensors.attach("electrics", self.electrics)
        vehicle.sensors.attach("damage", self.damage)

        # no AdvancedIMU: acc/angvel are derived from the State sensor in
        # update_imu, which saves one blocking round trip per step

        if transport == "ros2" and ros2_bridge._HAS_RCLPY:
            # start the C++ bridge (the ROS2 node + the State/Electrics/Time
            # sensor stream + the control topic). The bridge does NOT create
            # the LiDAR: the env reads it from shared memory (the direct,
            # zero-hop path), created below. If the bridge fails, fall back to
            # the shmem transport (socket for the sensor stream).
            try:
                self.ros2 = ros2_bridge.Ros2Bridge(vehicle, beamng)
                self.ros2.start(create_lidar=False)
            except Exception as err:
                print(f"[rig] ROS2 bridge failed to start ({err!r}); "
                      f"falling back to the shmem transport", flush=True)
                if self.ros2 is not None:
                    try:
                        self.ros2.stop()
                    except Exception:
                        pass
                    self.ros2 = None

        # Both transports read the LiDAR from shared memory (the game writes
        # the point cloud to /dev/shm, read in poll_lidar_points). The ROS2
        # bridge does not create its own LiDAR, so this is the only one.
        self.lidar = Lidar(
            "roof",
            beamng,
            vehicle,                             # attached to the car -> rides with it
            requested_update_time= 1.0/LIDAR_HZ, # 20 Hz
            pos=(0.0, 0.0, 1.5),                 # on the roof, in vehicle space
            vertical_resolution=LIDAR_RES_V,
            vertical_angle=LIDAR_V_ANGLE,
            frequency=LIDAR_HZ,
            max_distance=LIDAR_MAX_DIST,
            density=LIDAR_DENSITY,
            is_360_mode=True,                    # full 360 degrees
            is_rotate_mode=False,
            is_using_shared_memory=True,
            is_visualised=False,
            is_streaming=True,
        )

    def reset(self):
        """Drop the current vehicle and clear all per-vehicle state."""
        self.vehicle = None
        self.beamng = None
        self.electrics = None
        self.damage = None
        self.lidar = None
        self.ros2 = None
        self._prev_state = None
        self._imu_acc = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        self._last_bev = None
        self._last_road_bev = None
        self._navigraph = None
        self._ffb_resp = None
        self._sensor_engine_resp = None
        self._sensor_vehicle_resp = None

    def reset_imu_state(self):
        self._prev_state = None
        self._imu_acc = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        if self._ffb_resp is not None:
            # drain a response that was queued but never read (e.g. the step
            # was interrupted), so it does not sit in the socket buffer
            try:
                self._ffb_resp.recv("ExecutedLuaChunkVE")
            except Exception:
                pass
            self._ffb_resp = None

    # -- sensor polling -----------------------------------------------------
    def send_sensor_request(self):
        """Send the bundled sensor request without waiting (send half of poll)."""
        engine_reqs, vehicle_reqs = self.vehicle.sensors._encode_requests(
            tuple(self.vehicle.sensors._sensors.keys()))
        self._sensor_engine_resp = (self.beamng._send(engine_reqs)
                                    if engine_reqs["sensors"] else None)
        self._sensor_vehicle_resp = (self.vehicle._send(vehicle_reqs)
                                     if vehicle_reqs["sensors"] else None)

    def recv_sensor_data(self):
        """Receive and decode a previously sent sensor request (recv half of poll)."""
        # ROS2 transport: the sensor data comes from the bridge's topics (no
        # socket round trip); the tick-pacing (wait_for_tick) is done in the
        # step loop before this is called
        if self.ros2 is not None:
            self.ros2.populate_sensors()
            return
        sensor_data = dict()
        if self._sensor_engine_resp is not None:
            sensor_data.update(self._sensor_engine_resp.recv("SensorData")["data"])
        if self._sensor_vehicle_resp is not None:
            sensor_data.update(self._sensor_vehicle_resp.recv("SensorData")["data"])
        result = self.vehicle.sensors._decode_response(sensor_data)
        for sensor, data in result.items():
            self.vehicle.sensors._sensors[sensor].replace(data)
        self._sensor_engine_resp = None
        self._sensor_vehicle_resp = None

    def poll_sensors(self):
        """Blocking sensor poll; off the hot path (reset, stale-read fallback)."""
        # ROS2 transport: the sensor data comes from the bridge's topics (no
        # socket request/response)
        if self.ros2 is not None:
            self.ros2.populate_sensors()
            return
        self.send_sensor_request()
        self.recv_sensor_data()

    def poll_lidar_points(self):
        """Point cloud from shared memory, skipping the colour buffer (unused).

        Mirrors Lidar._convert_binary_to_array: in streaming mode the game
        writes the point count as the last float of the buffer. The buffer is
        read into a numpy array (the shared memory is a CPU buffer) and then
        moved to the active backend (the GPU when cupy is used) with to_gpu.
        """
        # The LiDAR is read from shared memory in both transports (the direct,
        # zero-hop path); the ROS2 bridge does not create its own LiDAR.
        if self.lidar is None:
            # Defensive: no shmem LiDAR. Fall back to the bridge's topic if it
            # is active and created one.
            if self.ros2 is not None:
                return self.ros2.read_lidar()
            return to_gpu(_numpy.empty((0, 3), dtype=_numpy.float32))
        shmem = self.lidar.point_cloud_shmem
        buf = _numpy.frombuffer(shmem.read(self.lidar.point_cloud_shmem_size),
                                dtype=_numpy.float32)
        n = int(buf[-1])
        if n <= 0 or n * 3 > len(buf) - 1:
            return to_gpu(_numpy.empty((0, 3), dtype=_numpy.float32))
        return to_gpu(buf[: 3 * n].reshape(-1, 3).copy())

    def send_ffb_request(self):
        """Queue the FFB read without waiting (send half of read_ffb).

        Called in the same batch as the Step so the game processes the tick
        first and the lua command right after: the returned force is the
        post-tick value and the read costs no extra round trip.
        """
        self._ffb_resp = self.vehicle._send(dict(
            type="QueueLuaCommandVE",
            chunk="return hydros.forceAtDriverNorm",
            resp=True))

    def read_ffb(self):
        """Read the (virtual) steering-wheel force feedback from the game's
        hydros lua module, normalised to [-1, 1] (0 on a failed read).

        The virtual wheel (enabled in attach) makes the game compute the wheel
        FFB without any physical wheel hardware; the value is the torque the
        wheel would apply to the driver (hydros.forceAtDriverNorm).

        If a request was queued via send_ffb_request (the step batch), this
        only receives the already-sent response; otherwise it does a full
        blocking round trip.
        """
        try:
            if self._ffb_resp is not None:
                resp = self._ffb_resp.recv("ExecutedLuaChunkVE")["resp"]
                self._ffb_resp = None
            else:
                resp = self.vehicle.queue_lua_command(
                    "return hydros.forceAtDriverNorm", response=True)
            return max(-1.0, min(1.0, float(resp)))
        except (TypeError, ValueError, KeyError):
            self._ffb_resp = None
            return 0.0

    def update_imu(self, state):
        """Refresh the derived IMU channels from a fresh State snapshot.

        Returns (acc, angvel); also caches them on the rig for get_obs.
        """
        acc, angvel = derive_imu(self._prev_state, state)
        # copy the fields: vehicle.state is mutated in place by the next poll,
        # so a reference would make prev identical to state (dt = 0)
        self._prev_state = state_snapshot(state)
        self._imu_acc, self._imu_angvel = acc, angvel
        return acc, angvel

    def _gear(self):
        """The engaged gear from the Electrics sensor (0 when unavailable).

        Feeds the BEV alignment's reverse-gear gate (obs.bev_rotation_angle).
        """
        try:
            return int(self.vehicle.sensors["electrics"]["gear"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0

    # -- observation --------------------------------------------------------
    def _road_bev(self):
        """The merged road BEV (1.0 = drivable) around the car, rasterized from
        the full navigraph (fetched once and cached - core.roads.get_navigraph).
        The surfaces are merged (each half-width grown by ROAD_MERGE_MARGIN) so
        lane dividers less than 2*margin apart fill in. Pure computation: the
        navigraph is in memory and the pose is the cached State, so no game
        round trip per step. Returns (BEV_N, BEV_N) or None on any failure (no
        vehicle / no connection yet).

        The mask is no longer used to correct the LiDAR BEV (the point cloud
        does the road-surface masking now, see _remove_road_surface); it is
        only the "road_bev" observation channel.
        TODO REFACTOR into secondary BEV observation channel"""
        try:
            if self.vehicle is None or self.beamng is None:
                return None
            if self._navigraph is None:
                self._navigraph = get_navigraph(self.beamng)
            state = self.vehicle.state
            pos = np.asarray(state["pos"], float)
            fwd, left = car_grid_axes(state, self._gear())
            mask = self._navigraph.to_bev(pos, fwd=fwd, left=left,
                                          merge_margin=ROAD_MERGE_MARGIN)
            self._last_road_mask = mask
            return mask
        except Exception:
            return None

    def _remove_road_surface(self, pc, rot=None):
        """The point cloud with the car body and the road surface removed, in
        one pass: the returns in the car's body box (the car frame, the same
        box obs.lidar_bev uses) and the returns within ROAD_SURFACE_TOL of the
        road surface (the nearby navigraph segments' ribbons,
        core.roads.Navigraph.road_surface_mask) are dropped, so the BEV reads
        the road as empty (road level) and only the obstacles (returns above
        the road) as returns. Pure computation: the navigraph is cached (like
        _road_bev) and the pose is the cached State, so no game round trip.
        The point cloud is returned unchanged when the navigraph is
        unavailable (no vehicle / no connection yet) or on any failure."""
        if pc.shape[0] == 0:
            return pc
        try:
            if self.vehicle is None or self.beamng is None:
                return pc
            state = self.vehicle.state
            pos = np.asarray(state["pos"], float)
            # the car's own body returns (the car frame, the same box
            # obs.lidar_bev uses) - grouped with the road-surface removal
            R = rot if rot is not None else car_rotation(state)
            p_car = (pc - pos) @ R
            x, y, z = p_car[:, 0], p_car[:, 1], p_car[:, 2]
            in_car = ((np.abs(x) <= CAR_LEN / 2) & (np.abs(y) <= CAR_WID / 2)
                      & (z >= CAR_Z_LO) & (z <= CAR_Z_HI))
            if self._navigraph is None:
                self._navigraph = get_navigraph(self.beamng)
            road = self._navigraph.road_surface_mask(pc, pos)
            keep = ~(in_car | road)
            return pc[keep]
        except Exception:
            return pc

    def road_centerline(self):
        """The centerline of the road surface the car is closest to for the
        current pose (see roads.road_centerline_segments: the closest surface's
        navigraph segment centerlines, tracked one point per row from the car
        outward - pinned to the surface the car is on, not the middle of a
        merged multi-surface road), in BEV grid (row, col) units, and the
        distance from the car to it (metres). Falls back to the merged-mask
        center of mass (roads.road_centerline) when the segment geometry is
        unavailable (no navigraph / vehicle / no nearby segments); ([], NaN)
        when the rig has no road mask yet. Called by the debug window, so it
        only runs while a window is open."""
        try:
            if self._navigraph is not None and self.vehicle is not None:
                state = self.vehicle.state
                pos = np.asarray(state["pos"], float)
                fwd, left = car_grid_axes(state, self._gear())
                centerline, dist = road_centerline_segments(
                    self._navigraph, pos, fwd=fwd, left=left)
                if centerline.size:
                    return centerline, dist
        except Exception:
            pass
        mask = getattr(self, "_last_road_mask", None)
        if mask is None:
            return [], float("nan")
        return road_centerline(mask)

    def road_centerline_dist(self):
        """The distance (metres) from the car to the road centerline of the
        surface it is closest to, for the current pose (see
        roads.road_centerline_segments). The navigraph is fetched once and
        cached (like _road_bev), so this is a pure in-memory computation. NaN
        when the vehicle / navigraph is unavailable or there is no road
        nearby (off-road). Used by the reward to scale the progress term."""
        try:
            if self.vehicle is None or self.beamng is None:
                return float("nan")
            if self._navigraph is None:
                self._navigraph = get_navigraph(self.beamng)
            state = self.vehicle.state
            pos = np.asarray(state["pos"], float)
            fwd, left = car_grid_axes(state, self._gear())
            _, dist = road_centerline_segments(self._navigraph, pos,
                                               fwd=fwd, left=left)
            return float(dist)
        except Exception:
            return float("nan")

    def on_road(self, fast_track=True, centerline_dist=None):
        """Whether the car is on the road (a bool) for the current pose.

        Fast-track (default): within ON_ROAD_FAST_TRACK_DIST of the road
        centerline the car is on the road, so the determination defaults to
        True without the strict edge check. Pass ``centerline_dist`` (metres,
        NaN when off-road) to reuse an already-computed distance (the env
        computes one for the reward every step). Otherwise (or with
        fast_track=False) the strict navigraph check applies (within
        ON_ROAD_TOL of the road surface of any nearby segment, see
        roads.Navigraph.is_on_road). The navigraph is fetched once and cached
        (like _road_bev), so this is a pure in-memory computation. None when
        the vehicle / navigraph is unavailable or the read fails. Shown in the
        debug window.
        TODO: expose as an observation channel ("on_road") - not part of the
        observation yet."""
        try:
            if self.vehicle is None or self.beamng is None:
                return None
            if self._navigraph is None:
                self._navigraph = get_navigraph(self.beamng)
            if fast_track:
                if centerline_dist is None:
                    centerline_dist = self.road_centerline_dist()
                # NaN (no road nearby / no navigraph) compares False and falls
                # through to the strict check
                if centerline_dist < ON_ROAD_FAST_TRACK_DIST:
                    return True
            state = self.vehicle.state
            pos = np.asarray(state["pos"], float)
            return bool(self._navigraph.is_on_road(pos))
        except Exception:
            return None

    def road_surface_outlines(self):
        """The edge outlines of the road surface that alters the BEV (the
        navigraph segments the road BEV rasterizes for the current pose), as
        (outline, active) tuples in BEV grid (row, col) units (see
        roads.Navigraph.surface_outlines): ``outline`` is the edge polyline,
        ``active`` is True for the road surface the car is closest to. Pure
        computation (the navigraph is cached and the pose is the cached
        State); [] when the rig is not attached or the read fails. Called by
        the debug window, so it only runs while a window is open."""
        try:
            if self.vehicle is None or self.beamng is None:
                return []
            if self._navigraph is None:
                self._navigraph = get_navigraph(self.beamng)
            state = self.vehicle.state
            pos = np.asarray(state["pos"], float)
            fwd, left = car_grid_axes(state, self._gear())
            return self._navigraph.surface_outlines(pos, fwd=fwd, left=left)
        except Exception:
            return []

    def get_obs(self):
        """Build the observation dict from the cached sensor data.

        Pure computation: no round trips here, the caller (step/reset) polls
        before calling this.
        """
        e = self.vehicle.sensors["electrics"]
        drift = drift_from_state(self.vehicle.state)

        acc, angvel = self._imu_acc, self._imu_angvel
        if drift["airSpeed"] <= 1.0:
            acc = np.zeros(3, dtype=np.float32)
            angvel = np.zeros(3, dtype=np.float32)

        # the two sub-paths share no written memory (both only read the cached
        # navigraph and the State dict), so with cupy each runs on its own
        # non-blocking CUDA stream and they overlap on the GPU; without cupy
        # the contexts are null and they run sequentially as before. Both
        # streams are synced before the results are read back to the CPU.
        use_device()
        with (self._pc_stream or nullcontext()):
            # the car-frame rotation, computed once and shared by the road-
            # surface removal and the LiDAR BEV (both transform the point cloud
            # into the car frame; car_rotation does H2D copies of the State
            # vectors, so computing it twice per step is wasteful)
            rot = car_rotation(self.vehicle.state)
            # the LiDAR point cloud with the road surface removed (the returns
            # within ROAD_SURFACE_TOL of the road surface are dropped), so the
            # BEV reads the road as empty (road level) and only the obstacles
            # (returns above the road) as returns
            pc = self._remove_road_surface(self.poll_lidar_points(), rot)
            bev = bev_to_obs(lidar_bev(pc, self.vehicle.state, drift, n=BEV_N,
                                       rot=rot, gear=self._gear()))
        with (self._road_stream or nullcontext()):
            road = self._road_bev()
        for stream in (self._pc_stream, self._road_stream):
            if stream is not None:
                stream.synchronize()

        # the road BEV (1.0 = drivable); an all-zero grid when the navigraph
        # is unavailable (off-road / no game)
        road_bev = (road if road is not None
                    else np.zeros((BEV_N, BEV_N), np.float32))
        road_bev = road_bev[None, ...].astype(np.float32)

        # the observation is computed on the active backend (the GPU when
        # cupy is used); move it back to the CPU so the callbacks and the
        # rest of the stack keep seeing plain numpy arrays
        bev = to_cpu(bev)
        road_bev = to_cpu(road_bev)
        self._last_road_bev = road_bev
        self._last_bev = bev

        obs = {
            "rpm":         np.clip(np.array([e["rpm"] / e["maxrpm"]], np.float32), 0.0, 1.0),
            "imu_acc":     acc,
            "imu_angvel":  angvel,
            "drift_angle": np.clip(np.array([drift["angle"] / ANGLE_MAX], np.float32), -1.0, 1.0),
            "airspeed":    np.array([min(AIRSPEED_MAX_KPH, drift["airSpeed"]) / AIRSPEED_MAX_KPH], np.float32),
            "is_shifting": np.array([1.0 if e["is_shifting"] else 0.0], np.float32),
            "gear":        np.array([(float(e["gear"]) + 1.0) / (MAX_GEAR + 1.0)], np.float32),
            "ffb":         np.array([self.read_ffb()], np.float32),
            "bev":         bev,
            "road_bev":    road_bev,
        }
        return {k: to_cpu(v) for k, v in obs.items()}
