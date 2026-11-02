"""The ROS2 (C++ bridge) transport for a single vehicle.

``Ros2Bridge`` drives the in-game C++ bridge (``libros2_beamng.so``) for one
vehicle and reads its sensor stream over ROS2 topics instead of the BeamNGpy
tcom socket. It is the "ROS2 transport" behind :class:`VehicleRig`: the env
loop publishes control to ``/bng<id>/control``, fires the tcom ``Step``, waits
for the tick to complete (tick-pacing, off the bridge's ``Time`` topic), and
reads State/Electrics/Damage from the bridge's topics.

The bridge is a **sensor stream + control** path: the C++ node is started and
stopped programmatically through the game's lua hooks (``tech_ros_GECore``
``loadRosLib`` / ``startRos`` / ``stopRos``), so Python owns its lifecycle. It
creates the in-game scalar sensors (State/Electrics/Time) itself. The LiDAR is
optional (``start(create_lidar=...)``): the env reads the LiDAR from shared
memory by default (the direct, zero-hop path), so the bridge does not create
its own LiDAR and the beamngpy ``Lidar`` is created by ``VehicleRig``.

ROS2 is optional: the module imports (and the shmem transport runs) on a
machine without ROS2; ``_HAS_RCLPY`` is False there and ``start`` raises.

Topics (node ``bng<numeric_id>``): sensors ``/bng<id>/<sensor_id>`` (State ->
``StateSensor``, Electrics -> ``ElectricsSensor``, Time -> ``TimeSensor``,
Damage -> ``DamageSensor``, and, when the bridge creates the LiDAR, LiDAR ->
``sensor_msgs/PointCloud2``), control ``/bng<id>/control``
(``VehicleControl``). The LiDAR PointCloud2 carries the raw shared-memory
buffer (a flat float32 xyz array with the valid point count as the last
float), so it is parsed exactly like the shmem read.
"""
import os
import threading
import time

import numpy as _numpy

from beamng_tech_env.core.array import to_gpu
from beamng_tech_env.core.constants import (
    LIDAR_HZ,
    LIDAR_MAX_DIST,
    LIDAR_RES_V,
    LIDAR_V_ANGLE,
)

# ROS2 is optional: the bridge is only used when rclpy is importable AND the
# caller asks for it (attach(transport="ros2")). Guarded so this file imports
# (and the shmem path runs) on a machine without ROS2.
try:
    import rclpy
    from beamng_msgs.msg import (DamageSensor, ElectricsSensor, StateSensor,
                                 TimeSensor, VehicleControl)
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import (DurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import PointCloud2
    _HAS_RCLPY = True
except ImportError:  # ROS2 not installed -> shmem fallback only
    rclpy = None
    _HAS_RCLPY = False

# game-home-relative path of the packaged default sensor config (the same one
# the in-game ROS2 Editor reads); jsonReadFile wants a game-home-relative path
_ROS_DEFAULTS_LUA = "/lua/ge/extensions/tech/ros/rosDefaults.json"


def ros_lib_path():
    """The built C++ bridge ``.so`` (``BinLinux/libros2_beamng.so``).

    Overridable via ``BNG_ROS_LIB``; otherwise derived from ``BNG_HOME``.
    """
    env = os.environ.get("BNG_ROS_LIB")
    if env:
        return env
    bng_home = os.environ.get("BNG_HOME")
    if bng_home:
        return os.path.join(bng_home, "BinLinux", "libros2_beamng.so")
    return "libros2_beamng.so"


def _start_ros_lua(vehicle_name, ros_lib, lidar, create_lidar):
    """Lua: find the vehicle by name, load the bridge, start its node with
    State/Electrics/Time (and optionally the Lidar), and return the vehicle's
    numeric id (or an error string).

    The sensor entries come from the game's packaged ``rosDefaults.json``
    (read with ``jsonReadFile``). When ``create_lidar`` is set, the LiDAR's
    ``create`` config is overridden to match the env's socket LiDAR (roof
    position, range, 360 deg, not visualised) and its buffer is capped (the
    default 24 MB PointCloud2 intermittently crashes the FastDDS SHM
    transport). The env reads the LiDAR from shared memory by default, so the
    bridge does not create its own LiDAR (``create_lidar=False``).
    """
    wanted = "State = 'state', Electrics = 'electrics', Time = 'time'"
    if create_lidar:
        wanted += ", Lidar = 'lidar'"
    return f"""
local target = {vehicle_name!r}
local vid = nil
for id, obj in activeVehiclesIterator() do
  if obj:getName() == target then vid = id break end
end
if vid == nil then
  for id, obj in activeVehiclesIterator() do vid = id break end
end
if vid == nil then return 'NOVEHICLE' end
if not extensions.isExtensionLoaded('tech_ros_GECore') then
  extensions.load('tech_ros_GECore')
end
local rosCore = extensions.tech_ros_GECore
local ok, err = rosCore.loadRosLib({ros_lib!r})
if not ok then return 'LOADFAIL: ' .. tostring(err) end
if not rosCore.isRosLibLoaded() then return 'NOTLOADED after loadRosLib' end
if be:getObjectByID(vid) == nil then return 'NOVEHICBEOBJ vid=' .. tostring(vid) end
local doc = jsonReadFile('{_ROS_DEFAULTS_LUA}')
if doc == nil then return 'NODOC' end
local wanted = {{ {wanted} }}
local sensors = {{}}
for _, entry in ipairs(doc.sensors) do
  local id = wanted[entry.type]
  if id then
    local s = {{}}
    for k, v in pairs(entry) do s[k] = v end
    s.id = id
    if entry.type == 'Lidar' then
      s.config = s.config or {{}}
      s.config.blobLayout = s.config.blobLayout or {{}}
      s.config.blobLayout.maxPoints = {lidar['max_points']}
      s.config.create = s.config.create or {{}}
      s.config.create.horizontalResolution = {lidar['horizontal_resolution']}
      s.config.create.pos = {{ {lidar['pos'][0]}, {lidar['pos'][1]}, {lidar['pos'][2]} }}
      s.config.create.maxDistance = {lidar['max_distance']}
      s.config.create.frequency = {lidar['frequency']}
      s.config.create.verticalResolution = {lidar['vertical_resolution']}
      s.config.create.verticalAngle = {lidar['vertical_angle']}
      s.config.create.is360 = {str(bool(lidar['is_360'])).lower()}
      s.config.create.isRotate = {str(bool(lidar['is_rotate'])).lower()}
      s.config.create.isVisualised = {str(bool(lidar['is_visualised'])).lower()}
    end
    if entry.type == 'Electrics' then
      -- The packaged config points two fields at camelCase paths the game
      -- never sets (the real electrics.values keys are lowercase); repoint.
      for _, f in ipairs(s.fields or {{}}) do
        if f.name == 'maxrpm' then f.path = 'electrics.values.maxrpm' end
        if f.name == 'lowpressure' then f.path = 'electrics.values.lowpressure' end
      end
    end
    sensors[#sensors + 1] = s
  end
end
local ok2, err2 = rosCore.startRos(vid, {{ sensors = sensors }})
if not ok2 then return 'STARTFAIL: ' .. tostring(err2) .. ' vid=' .. tostring(vid) end
return vid
"""


def _quat_from_dir_up(direction, up):
    """A (x, y, z, w) quaternion for the rotation whose world columns are the
    car's [forward, left, up] axes, built from the (normalised) dir and up
    vectors. The ROS2 StateSensor has no rotation field, so the pose the env
    saves/restores (checkpoint, human resume) is reconstructed this way - the
    same axes ``obs.car_rotation`` uses."""
    d = _numpy.asarray(direction, float)
    d /= _numpy.linalg.norm(d)
    u = _numpy.asarray(up, float)
    u /= _numpy.linalg.norm(u)
    l = _numpy.cross(u, d)
    l /= _numpy.linalg.norm(l)
    R = _numpy.stack([d, l, u], axis=1)  # columns [fwd, left, up]
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / _numpy.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * _numpy.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * _numpy.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * _numpy.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def _scalar_msg_to_dict(msg):
    """A ROS2 scalar sensor message -> a plain dict of its scalar fields.

    The ``header`` and any nested sub-message fields (the Electrics
    wheel-thermal blocks) are skipped; the rest are read by name, so the
    result matches the beamngpy sensor's field naming (the bridge publishes
    the beamngpy-renamed names).
    """
    out = {}
    for name in msg._fields_and_field_types:
        if name == "header":
            continue
        value = getattr(msg, name)
        if hasattr(value, "_fields_and_field_types"):
            continue
        out[name] = value
    return out


class Ros2Bridge:
    """The ROS2 bridge for one vehicle: control publisher + sensor readers.

    Subscribes to the bridge's State/Electrics/Time/Damage/LiDAR topics and
    publishes ``VehicleControl`` to its control topic. A background thread
    spins the rclpy executor so the subscription callbacks (which store the
    latest message in lock-guarded slots) fire without the env loop having to
    pump ROS2. The env loop reads the slots (``read_state`` / ``read_electrics``
    / ``read_damage`` / ``read_lidar`` / ``sim_time``), waits for a tick
    (``wait_for_tick``), and publishes control (``publish_control``).
    """

    def __init__(self, vehicle, beamng):
        self.vehicle = vehicle
        self.beamng = beamng
        self._vehicle_id = None
        self._node = None
        self._executor = None
        self._thread = None
        self._pub = None
        self._lock = threading.Lock()
        self._state = None       # latest StateSensor
        self._sim_time = None    # latest sim time (s), from the Time topic
        self._electrics = None   # latest ElectricsSensor
        self._damage = None      # latest DamageSensor
        self._lidar = None       # latest PointCloud2
        self._started = False
        # the LiDAR create config (matches the env's socket LiDAR) + the buffer
        # cap (the default 24 MB PointCloud2 crashes the FastDDS SHM transport)
        self._lidar = dict(
            pos=(0.0, 0.0, 1.5),
            max_distance=LIDAR_MAX_DIST,
            frequency=LIDAR_HZ,
            vertical_resolution=LIDAR_RES_V,
            vertical_angle=LIDAR_V_ANGLE,
            is_360=True,
            is_rotate=False,
            is_visualised=False,
            max_points=100000,
            horizontal_resolution=5000,
        )

    # -- lifecycle ----------------------------------------------------------
    def start(self, create_lidar=False):
        """Start the in-game bridge node and the rclpy subscriptions.

        ``create_lidar`` controls whether the bridge creates the in-game
        LiDAR. The env reads the LiDAR from shared memory (the direct,
        zero-hop path), so the bridge does not create its own LiDAR by
        default (avoiding two LiDARs).
        """
        if self._started:
            return
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy is not installed; the ROS2 transport "
                               "needs ROS2 (use transport='shmem' to fall "
                               "back to the socket path)")
        lua = _start_ros_lua(self.vehicle.vid, ros_lib_path(), self._lidar,
                             create_lidar)
        result = self.beamng.queue_lua_command(lua, response=True)
        if isinstance(result, str) and not result.strip().lstrip("-").isdigit():
            raise RuntimeError(f"startRos failed: {result}")
        self._vehicle_id = int(float(result))

        rclpy.init()  # idempotent: re-attach in the same process is a no-op
        self._node = rclpy.create_node(f"drift_ros2_{self._vehicle_id}")
        # sensor streams: best-effort, keep only the newest message
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        base = f"/bng{self._vehicle_id}"
        self._node.create_subscription(StateSensor, f"{base}/state",
                                       self._on_state, qos)
        self._node.create_subscription(TimeSensor, f"{base}/time",
                                       self._on_time, qos)
        self._node.create_subscription(ElectricsSensor, f"{base}/electrics",
                                       self._on_electrics, qos)
        self._node.create_subscription(DamageSensor, f"{base}/damage",
                                       self._on_damage, qos)
        self._node.create_subscription(PointCloud2, f"{base}/lidar",
                                       self._on_lidar, qos)
        self._pub = self._node.create_publisher(VehicleControl,
                                                f"{base}/control", 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._wait_ready()
        self._started = True

    def _wait_ready(self, timeout=10.0):
        """Block until the first State and Time messages arrive (topic
        discovery + the first 2 kHz publish), so the caller can read fresh
        sensor data immediately after ``start``. Raises on timeout."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._state is not None and self._sim_time is not None:
                    return
            time.sleep(0.005)
        raise RuntimeError("ROS2 bridge timed out waiting for the first "
                           "State/Time messages (topic discovery?)")

    def stop(self):
        """Tear down the rclpy node and stop the in-game bridge (idempotent)."""
        if self._thread is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._executor = None
        self._pub = None
        with self._lock:
            self._state = None
            self._sim_time = None
            self._electrics = None
            self._damage = None
            self._lidar = None
        # stop the in-game node (removes the LiDAR + sensors). Best-effort: the
        # connection may be gone if the game is quitting.
        if self.beamng is not None and self._vehicle_id is not None:
            try:
                self.beamng.queue_lua_command(
                    f"extensions.tech_ros_GECore.stopRos({self._vehicle_id})")
            except Exception:
                pass
        self._started = False

    @property
    def vehicle_id(self):
        return self._vehicle_id

    # -- subscription callbacks (run in the spin thread) ---------------------
    def _on_state(self, msg):
        with self._lock:
            self._state = msg

    def _on_time(self, msg):
        t = msg.beamng_simulation_time
        with self._lock:
            self._sim_time = float(t.sec) + float(t.nanosec) * 1e-9

    def _on_electrics(self, msg):
        with self._lock:
            self._electrics = msg

    def _on_damage(self, msg):
        with self._lock:
            self._damage = msg

    def _on_lidar(self, msg):
        with self._lock:
            self._lidar = msg

    # -- control -------------------------------------------------------------
    def publish_control(self, steering, throttle, brake, parkingbrake, gear):
        """Publish a ``VehicleControl`` frame (fire-and-forget; the bridge
        applies it in ``onPhysicsStep`` when a new frame arrives). Units match
        the BeamNGpy tcom Control (steering -1..1, pedals 0..1, gear index)."""
        if self._pub is None:
            return
        m = VehicleControl()
        m.steering = float(steering)
        m.throttle = float(throttle)
        m.brake = float(brake)
        m.parkingbrake = float(parkingbrake)
        m.clutch = 0.0
        m.gear = int(gear)
        self._pub.publish(m)

    # -- sensor reads --------------------------------------------------------
    def sim_time(self):
        """The latest sim time (s) from the Time topic, or None."""
        with self._lock:
            return self._sim_time

    def wait_for_tick(self, last_time, timeout=0.25):
        """Block until the sim time advances a full tick (>= 40 ms) past
        ``last_time`` - the ROS2 equivalent of the socket path's blocking
        sensor recv (which waits for the tick). Returns True on a completed
        tick, False on timeout (the caller uses the latest state)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            t = self.sim_time()
            if t is not None and t - last_time >= 0.04:
                return True
            time.sleep(0.001)
        return False

    def read_state(self):
        """The latest State + Time as the State-sensor dict the env reads:
        ``pos``/``vel``/``dir``/``up``/``front`` (from the State topic),
        ``time`` (from the Time topic), and ``rotation`` (reconstructed from
        dir/up - the StateSensor has no rotation field). None when no State
        message has arrived yet."""
        with self._lock:
            msg = self._state
            sim_time = self._sim_time
        if msg is None:
            return None
        state = {
            "pos": (msg.position.x, msg.position.y, msg.position.z),
            "vel": (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            "dir": (msg.dir.x, msg.dir.y, msg.dir.z),
            "up": (msg.up.x, msg.up.y, msg.up.z),
            "front": (msg.front.x, msg.front.y, msg.front.z),
        }
        if sim_time is not None:
            state["time"] = sim_time
        state["rotation"] = _quat_from_dir_up(state["dir"], state["up"])
        return state

    def read_electrics(self):
        """The latest Electrics as a dict (beamngpy field naming), or None."""
        with self._lock:
            msg = self._electrics
        if msg is None:
            return None
        return _scalar_msg_to_dict(msg)

    def read_damage(self):
        """The latest Damage as the dict the env reads: ``damage`` (the sum of
        the per-part damage) and ``lowpressure`` (the tire-flat flag, read from
        the Electrics topic - the DamageSensor has no such field). None when
        nothing has arrived yet."""
        with self._lock:
            msg = self._damage
            electrics = self._electrics
        if msg is None:
            return None
        damage = sum(float(v) for v in msg.part_damage)
        lowpressure = (float(electrics.lowpressure)
                       if electrics is not None else 0.0)
        return {"damage": damage, "lowpressure": lowpressure}

    def read_lidar(self):
        """The latest LiDAR point cloud as (N, 3) float32 on the active
        backend, matching ``VehicleRig.poll_lidar_points``. The PointCloud2
        carries the raw shared-memory buffer (a flat float32 xyz array with the
        valid point count as the last float), so it is parsed like the shmem
        read. Empty when nothing has arrived yet."""
        with self._lock:
            msg = self._lidar
        if msg is None:
            return to_gpu(_numpy.empty((0, 3), dtype=_numpy.float32))
        buf = _numpy.frombuffer(msg.data, dtype=_numpy.float32)
        n = int(buf[-1])
        if n <= 0 or n * 3 > len(buf) - 1:
            return to_gpu(_numpy.empty((0, 3), dtype=_numpy.float32))
        return to_gpu(buf[: 3 * n].reshape(-1, 3).copy())

    def populate_sensors(self):
        """Refresh the beamngpy sensor dicts from the latest topics, so the
        env loop (which reads ``ego.state`` / ``ego.sensors[...]``) sees fresh
        data with no socket round trip. Returns True when a State message was
        applied, False when nothing has arrived yet."""
        state = self.read_state()
        if state is None:
            return False
        self.vehicle.state.replace(state)
        electrics = self.read_electrics()
        if electrics is not None:
            self.vehicle.sensors["electrics"].replace(electrics)
        damage = self.read_damage()
        if damage is not None:
            self.vehicle.sensors["damage"].replace(damage)
        return True
