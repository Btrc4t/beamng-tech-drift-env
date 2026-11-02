"""Owns the BeamNG.tech process/connection and the loaded scenario.

``GameDriver`` is the reusable "talk to the game" building block: it launches
(or connects to) the game, loads a map scenario, applies the global settings a
drift environment needs, and recovers when the game wedges. It knows nothing
about gymnasium, vehicles or sensors - those live in the environment and in
:class:`~beamng_tech_env.core.rig.VehicleRig`.
"""
import os
import socket
import subprocess
import threading
import time

from beamngpy import BeamNGpy, Scenario
from beamngpy.logging import BNGDisconnectedError, BNGError, BNGValueError

from beamng_tech_env.core.constants import MAX_HANG_RECOVERIES, RENDER_FPS, REALTIME_FPS_LIMIT, TCOM_PORT

_T0 = time.perf_counter()
def _dbg(msg):
    print(f"[{time.perf_counter() - _T0:8.2f}s] {msg}", flush=True)


class GameDriver:
    """Connect/launch the game, load a scenario, and keep it alive.

    Parameters
    ----------
    host:
        The host of the tcom server to connect to (or launch the game on).
    tcom_port:
        The tcom port to connect to (or launch the game on).
    launch:
        Whether ``connect`` may launch a local game when nothing is listening
        on ``host:tcom_port``. Set to False for a hosted instance that is
        launched out-of-band (e.g. by ``host_game.py``): the env then only
        connects and fails fast if the game is not up.
    """

    def __init__(self, host="localhost", tcom_port=TCOM_PORT, launch=True,
                 ros2=False):
        self.host = host
        self.tcom_port = tcom_port
        self.launch = launch
        # the ROS2 transport: the game must be launched with -disable-sandbox
        # (the C++ bridge .so's constructor calls rclcpp::init) and inherit the
        # ROS2 env of the process that launches it
        self.ros2 = ros2
        self.beamng = None
        self.scenario = None
        self.waypoints = {}
        self.window = None
        self._hang_count = 0

    # -- connection / launch ------------------------------------------------
    def connect(self):
        """Create the BeamNGpy handle and open (launching the game if needed)."""
        _dbg(f"setup: connecting to game at {self.host}:{self.tcom_port}")
        self.beamng = BeamNGpy(self.host, self.tcom_port)
        # a launched game inherits stderr; a game relaunched by recovery
        # outlives the training process (reparented to init) and would keep
        # the write end of the output pipe open, so the job that launched
        # the training never sees EOF. Launch with stderr on /dev/null.
        saved_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            # the ROS2 transport needs -disable-sandbox (the C++ bridge .so's
            # constructor calls rclcpp::init, which needs the ROS2 env + an
            # unsandboxed process); the game inherits this process's ROS2 env.
            # Only relevant when we launch (a connected game keeps its flags).
            if self.ros2:
                self.beamng.open(None, "-disable-sandbox",
                                 launch=self.launch)
            else:
                self.beamng.open(launch=self.launch)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        _dbg("setup: connected, loading scenario")

    # -- scenario -----------------------------------------------------------
    def load_scenario(self, level, name):
        """Make a scenario for ``level``, load it, and cache its waypoints."""
        self.scenario = Scenario(level, name)
        self.scenario.make(self.beamng)
        self.beamng.scenario.load(self.scenario)
        _dbg("setup: scenario loaded")
        self.waypoints = {w.name: w for w in self.scenario.find_waypoints()}
        _dbg(f"setup: {len(self.waypoints)} waypoints")

    def reuse_scenario(self, level, name):
        """Adopt the scenario the game has already loaded, if it matches
        ``level``/``name``.

        Caches its waypoints and, if there are leftover vehicles (e.g. an ego
        from a previous env), restarts the scenario to a clean, running state
        so the env can spawn a fresh ego without reloading the map. If there
        are no leftover vehicles, the scenario is left as-is (it is already in
        a clean state, e.g. just restarted by another env sharing the game).

        Returns True if the scenario was reused, False if the game has no
        matching scenario loaded (the caller should load one).
        """
        try:
            current = self.beamng.scenario.get_current(connect=True)
        except (BNGError, BNGValueError, BNGDisconnectedError, OSError):
            # no scenario loaded (BNGValueError) or the game is not
            # answering: fall back to loading the scenario fresh
            return False
        current_level = (current.level.name
                         if hasattr(current.level, "name") else str(current.level))
        if current_level != level or current.name != name:
            return False
        self.scenario = current
        self.beamng._scenario = current
        self.waypoints = {w.name: w for w in current.find_waypoints()}
        _dbg(f"setup: reusing already-loaded scenario '{name}' on {current_level} "
             f"({len(self.waypoints)} waypoints)")
        # clear any leftover vehicles (e.g. an ego from a previous env) and
        # reset the scenario to a clean, running state. Skip the restart when
        # there are no leftover vehicles: the scenario is already clean (e.g.
        # just restarted by another env sharing the game), and restarting
        # while another env is mid-operation can wedge the game.
        if current.vehicles:
            _dbg(f"setup: clearing {len(current.vehicles)} leftover vehicle(s)")
            self.beamng.scenario.restart()
        return True

    def start_scenario(self):
        """Start the simulation with the (already added) vehicles."""
        self.beamng.scenario.start()
        _dbg("setup: scenario started")

    # -- global settings ----------------------------------------------------
    def disable_driving_assists(self):
        """Turn off the driving assists and enable the manual-gear assists."""
        self.beamng.queue_lua_command('''
            settings.setValue("steeringAutocenterEnabled", false)
            settings.setValue("steeringSlowdownEnabled", false)
            settings.setValue("steeringStabilizationEnabled", false)
            settings.setValue("steeringUndersteerReductionEnabled", false)
            settings.setValue("steeringLimitEnabled", false)
            settings.setValue("autoClutch", true)
            settings.setValue("autoThrottle", true)
            settings.setValue("absBehavior", "off")
        ''')

    def set_window(self, window):
        """Fixed windowed resolution for the game."""
        self.window = list(window)
        self.beamng.settings.change("GraphicDisplayMode", "Windowed")
        self.beamng.settings.change("GraphicDisplayResolutions", " ".join(map(str, self.window)))
        self.beamng.settings.apply_graphics()

    def set_deterministic(self):
        self.beamng.queue_lua_command('be:setPhysicsDeterministic(true)')

    def set_render_fps(self, fps=RENDER_FPS):
        """Run the render loop faster than the physics tick so the game's main
        loop wakes up more often to process socket requests (each blocking
        round trip otherwise waits up to a full frame period). The physics
        stays on its fixed 20 Hz tick - step(1) still advances exactly 50 ms of
        sim time - but decoupling the render loop from the tick means the
        simulation can run FASTER than real time when the game is resumed. That
        is fine for the stepped (drive) env: it steps explicitly and measures
        wall-clock FPS, not sim speed."""
        self.beamng.queue_lua_command(f"settings.setValue('fpsLimit', {fps})")
        self.beamng.queue_lua_command('settings.setValue("fpsLimitEnabled", true)')
        self.beamng.queue_lua_command('be:setPhysicsSpeedFactor(1)')

    def set_realtime(self, fps=REALTIME_FPS_LIMIT):
        """Run the physics at real time (1x), independent of the render speed.

        Speed factor 0 advances the physics by the wall-clock delta since the
        last graphics frame, so the sim runs at exactly 1x real time no matter
        how fast the render runs. The older factor -1 (1/fpslimit seconds of
        physics per rendered frame) is only real time while the render
        constantly reaches the fps cap; the render routinely exceeds the cap
        (unfocused window, cap not enforced), which made the sim - and any
        replay playback - run several times faster than real time and truncate
        episodes early. The fps cap is kept (harmless to the sim speed, and it
        limits the render's CPU use when the window is focused).
        """
        self.beamng.queue_lua_command(
            f"settings.setValue('fpsLimit', {fps})\n"
            "settings.setValue('fpsLimitEnabled', true)\n"
            "be:setPhysicsSpeedFactor(0)")

    def pause(self):
        self.beamng.control.pause()

    def resume(self):
        self.beamng.control.resume()

    # -- timeout / recovery -------------------------------------------------
    def call_with_timeout(self, func, timeout, *args, **kwargs):
        """Run a blocking tcom call in a worker thread.

        tcom has no protocol-level timeout, so a wedged game blocks the call
        forever. If the call takes longer than `timeout` seconds, raise
        TimeoutError. The abandoned worker unblocks itself once recovery
        kills the game (the socket then gets a RST).
        """
        box = {}
        def worker():
            try:
                box["result"] = func(*args, **kwargs)
            except BaseException as exc:
                box["error"] = exc
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"tcom call timed out after {timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def kill_game(self):
        proc = self.beamng.process if self.beamng is not None else None
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
            return
        # no process handle: the game was launched outside of us (we only
        # connect to it), so kill it by command line; the main process and
        # all of its helper children carry the same binary name
        _dbg("kill: no process handle, killing game by command line")
        subprocess.run(["pkill", "-f", "BeamNG.tech.x64"], check=False)
        time.sleep(2)
        subprocess.run(["pkill", "-9", "-f", "BeamNG.tech.x64"], check=False)

    def wait_for_port_free(self, port=None, timeout=30.0):
        """Wait until nothing accepts connections on the tcom port.

        A dying game (e.g. wedged in the GPU driver) keeps its tcom socket
        open for a while after we kill it; if open() connects to that corpse
        it assumes an instance already exists and never launches a new one.
        """
        if port is None:
            port = self.tcom_port
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.2)
                if probe.connect_ex((self.host, port)) != 0:
                    _dbg("recover: tcom port free")
                    return
            time.sleep(0.2)
        _dbg(f"recover: tcom port still occupied after {timeout:.0f}s; "
             f"open() may connect to a dying game")

    def game_fd_count(self):
        """The game process's open-fd count, or None if the pid is unknown.

        The game leaks 2 shared-memory fds per LiDAR lifecycle (its C++
        removeSensor never closes them); when the count approaches the 1024
        limit of select()'s fd_set, the tcom server's checkMessages() dies
        with "descriptor too large for set size" and the vehicle hangs.
        """
        if self.beamng is not None and self.beamng.process is not None:
            pid = self.beamng.process.pid
        else:
            # no process handle: the game was launched outside of us, find it
            # by command line (the same match kill_game uses)
            out = subprocess.run(["pgrep", "-f", "BeamNG.tech.x64"],
                                 capture_output=True, text=True)
            pids = out.stdout.split()
            if not pids:
                return None
            pid = int(pids[0])
        try:
            return len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            return None

    def recover_from_hang(self, err):
        """The game stopped answering tcom.

        BeamNG's tcom server skips all request handling while it waits for an
        internal blocking condition (scenario restart, vehicle spawn, vehicle
        connection). If that condition is never met, every client hangs
        forever. For a game we launched (we own the process) the only reliable
        fix is to kill it and relaunch it. For a game we only connected to
        (launched out-of-band) we own no process handle and must not kill it
        (``kill_game`` would fall back to a global pkill that also hits any
        other instance), so we just disconnect and let the caller retry the
        connection.
        """
        self._hang_count += 1
        if self._hang_count > MAX_HANG_RECOVERIES:
            raise RuntimeError(
                f"BeamNG unresponsive {self._hang_count} times in a row, giving up"
            ) from err
        if self.owns_process:
            _dbg(f"recover: game unresponsive ({type(err).__name__}: {err}); "
                 f"killing and relaunching (hang #{self._hang_count})")
            self.kill_game()
            self.wait_for_port_free()
        else:
            _dbg(f"recover: connected game unresponsive ({type(err).__name__}: {err}); "
                 f"retrying connection (hang #{self._hang_count})")
        if self.beamng is not None:
            try:
                self.beamng.disconnect()
            except Exception:
                pass
        self.beamng = None
        self.scenario = None
        self.waypoints = {}
        self.window = None

    @property
    def owns_process(self):
        """True if this driver launched the game (owns the process handle).

        A game we only connected to (launched out-of-band) has no process
        handle; we must not kill or quit it, only drop the connection.
        """
        return self.beamng is not None and self.beamng.process is not None

    @property
    def hang_count(self):
        return self._hang_count

    def reset_hang_count(self):
        self._hang_count = 0

    # -- teardown -----------------------------------------------------------
    def close(self, timeout=30.0):
        if self.window is not None:
            if self.owns_process:
                # game we launched: close() quits it (kill on failure)
                try:
                    self.call_with_timeout(self.beamng.close, timeout)
                except (OSError, BNGError, BNGDisconnectedError, TimeoutError):
                    self.kill_game()
            else:
                # game we only connected to (launched out-of-band): just drop
                # the connection; close() would send a quit command and kill it
                try:
                    self.beamng.disconnect()
                except Exception:
                    pass
            self.window = None
