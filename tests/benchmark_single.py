"""Benchmark the single-vehicle environment.

Measures headline FPS, a fine-grained phase breakdown of the pipelined
step(), determinism (state["time"] advances exactly 50 ms per step), the
LiDAR point count, and drift parity against the game's Lua telemetry.

Usage:
    BNG_HOME=/path/to/BeamNG.tech venv/bin/python src/env/gymnasium/tests/benchmark_single.py
"""
import os
import sys
import time

import numpy as np

# the env package lives in the parent folder (this script sits in tests/)
_GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _GYM_DIR)
from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv
from beamng_tech_env.core.constants import (
    BEV_N,
    BEV_ROAD_LEVEL,
    DRIFT_CARS_6SPEED,
    action_to_gear,
)
from beamng_tech_env.core.obs import drift_from_state


def _raw(env):
    """Unwrap a gymnasium env wrapper to reach the underlying BeamNGTechEnv.

    The standalone script passes the raw env; the pytest suite passes the
    gym.make-wrapped env, whose internal attributes (ego, beamng, ...) live on
    the unwrapped instance.
    """
    return env.unwrapped if hasattr(env, "unwrapped") else env


def read_drift(bng):
    """Drift telemetry from the game's Lua side, as a dict. The lua side packs
    the fields into a pipe-delimited string (booleans as 1/0); we split and
    cast here, so no json round-trip is needed."""
    chunk = (
        "local s = gameplay_drift_scoring.getScore() "
        "local b2s = function(x) return x and 1 or 0 end "
        "return string.format('%s|%s|%s|%s|%s|%s', "
        "s.potentialScore, s.score, "
        "b2s(gameplay_drift_drift.getIsDrifting()), "
        "b2s(gameplay_drift_general.getFrozen()), "
        "gameplay_drift_drift.getCurrDegAngleSigned(), "
        "gameplay_drift_drift.getAirSpeed())"
    )
    raw = bng.queue_lua_command(chunk, response=True)
    try:
        p = str(raw).split("|")
        return {"potentialScore": float(p[0]), "score": float(p[1]),
                "isDrifting": p[2] == "1", "frozen": p[3] == "1",
                "angle": float(p[4]), "airSpeed": float(p[5])}
    except (TypeError, ValueError, IndexError):
        print("read_drift: unexpected lua response:", repr(raw))
        return {"potentialScore": 0.0, "score": 0.0, "isDrifting": False,
                "frozen": False, "angle": 0.0, "airSpeed": 0.0}


def headline_fps(env, n_steps=100):
    env = _raw(env)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(env.action_space.sample())
    dt = time.perf_counter() - t0
    return n_steps / dt, dt / n_steps * 1000


def phase_breakdown(env, n_steps=100):
    """Mirror step() with per-phase timing (same phases, same order)."""
    env = _raw(env)
    phase = {"control": 0.0, "batch send": 0.0, "batch recv": 0.0,
             "repoll": 0.0, "reward+imu": 0.0, "obs+bev": 0.0}
    repolls = 0
    for _ in range(n_steps):
        action = np.asarray(env.action_space.sample(), dtype=np.float32)
        steering, combined, parkingbrake = (float(v) for v in action[:3])
        throttle = combined if combined > 0.0 else 0.0
        brake = -combined if combined < 0.0 else 0.0
        gear = action_to_gear(action[3:])
        t0 = time.perf_counter()
        r_control = env.ego._send(dict(
            type="Control", steering=steering, throttle=throttle, brake=brake,
            parkingbrake=0.0 if parkingbrake < 0.0 else parkingbrake, gear=gear))
        r_control.recv("Controlled")
        t1 = time.perf_counter()
        env.beamng._send(dict(type="Step", count=1, ack=False))
        env._send_sensor_request()
        t2 = time.perf_counter()
        env._recv_sensor_data()
        t3 = time.perf_counter()
        if env.ego.state["time"] - env._last_step_time < 0.04:
            env._poll_sensors()
            repolls += 1
        t4 = time.perf_counter()
        env._update_reward_state()
        env._imu_acc, env._imu_angvel = env._derive_imu(env.ego.state)
        env._check_failure()
        env._compute_reward()
        t5 = time.perf_counter()
        env._get_obs()
        env._last_step_time = env.ego.state["time"]
        t6 = time.perf_counter()
        phase["control"] += t1 - t0
        phase["batch send"] += t2 - t1
        phase["batch recv"] += t3 - t2
        phase["repoll"] += t4 - t3
        phase["reward+imu"] += t5 - t4
        phase["obs+bev"] += t6 - t5
    total = sum(phase.values())
    return phase, total / n_steps * 1000, repolls


def check_determinism(env, n_steps=50):
    """state["time"] must advance exactly 50 ms per step."""
    env = _raw(env)
    t_prev = env.ego.state["time"]
    worst = 0.0
    for _ in range(n_steps):
        env.step(env.action_space.sample())
        dt = env.ego.state["time"] - t_prev
        worst = max(worst, abs(dt - 0.05))
        t_prev = env.ego.state["time"]
    return worst


def drift_parity(env, iterations=160):
    """Scripted drift; compare drift_from_state against the game's Lua.

    The Lua probe is queued after the state poll, so it runs one frame later:
    a large angle diff is expected only at the wrap point of the spin phase,
    and the airspeed diff grows with the car's acceleration.
    """
    env = _raw(env)
    env.beamng.queue_lua_command('settings.setValue("enableDriftFreeroamCruising", true)')
    env.beamng.queue_lua_command('gameplay_drift_general.setContext("inFreeroam")')
    env.ego.ai.set_mode("none")

    # the car has been driven with random actions by the earlier sections;
    # teleport it to a known waypoint (teleport also resets its velocity)
    key = "Bridge15_C" if "Bridge15_C" in env.waypoints else list(env.waypoints.keys())[0]
    env.ego.teleport(env.waypoints[key].pos, rot_quat=env.waypoints[key].rot)
    time.sleep(1.05)
    env.beamng.control.pause()

    # clean control state: the earlier sections leave throttle/brake at
    # random values, and the drift schedule below never touches the brake -
    # a residual brake would make the car slide backwards through the drift.
    # The realistic gearbox starts in neutral, so select 1st gear (the
    # autoClutch assist handles the launch).
    env.ego.control(throttle=0.0, steering=0.0, brake=0.0, parkingbrake=0.0, gear=1)

    hist_game, hist_ours = [], []
    def hook(step):
        env.ego.sensors.poll()
        hist_game.append(read_drift(env.beamng))
        hist_ours.append(drift_from_state(env.ego.state))

    # scripted drift schedule (same as notebook 01/04)
    for step in range(1, iterations):
        t, s, p = 0.0, 0.0, 0.0
        if step < 40:
            t, s = 0.4, 0.3
        elif step < 55:
            t, s = 1.0, -0.8
        elif step < 70:
            t, s = 1.0, -0.5
        elif step < 80:
            t, s = 0.5, -0.8
        elif step < 95:
            t, s = 0.55, -0.5
        elif step < 110:
            p, s = 0.8, -0.8
        elif step < 120:
            s = 0.4
        elif step < 125:
            t, s = 0.8, -0.99
        elif step >= 145:
            p = 0.9
        env.ego.control(throttle=t, steering=s, parkingbrake=p)
        hook(step)
        env.beamng.control.step(1)

    env.ego.control(throttle=0.0, steering=0.0)
    ang_g = np.array([g["angle"] for g in hist_game])
    ang_o = np.array([o["angle"] for o in hist_ours])
    sp_g = np.array([g["airSpeed"] for g in hist_game])
    sp_o = np.array([o["airSpeed"] for o in hist_ours])
    drift_g = np.array([g["isDrifting"] for g in hist_game])
    drift_o = np.array([o["isDrifting"] for o in hist_ours])
    mask = sp_g > 3.0
    if mask.any():
        max_angle_diff = float(np.max(np.abs(ang_g[mask] - ang_o[mask])))
        max_airspeed_diff = float(np.max(np.abs(sp_g[mask] - sp_o[mask])))
    else:
        max_angle_diff = 0.0
        max_airspeed_diff = 0.0
    metrics = {
        "frames": len(hist_game),
        "moving": int(mask.sum()),
        "max_angle_diff": max_angle_diff,
        "max_airspeed_diff": max_airspeed_diff,
        "agreement": float(100 * (drift_g == drift_o).mean()),
    }
    print(f"frames: {metrics['frames']}, moving: {metrics['moving']}")
    print(f"max |angle diff|:    {metrics['max_angle_diff']:.2f} deg (wrap at spin expected)")
    print(f"max |airspeed diff|: {metrics['max_airspeed_diff']:.2f} kph (lua lags one frame)")
    print(f"isDrifting agreement: {metrics['agreement']:.1f} %")
    return metrics


def main():
    env = BeamNGTechEnv()
    try:
        # the scripted drift schedule in drift_parity was tuned for the bx drift
        # setup (the notebook's scenario used it), so pin the car for the run
        obs, info = env.reset(seed=0, options={"car": DRIFT_CARS_6SPEED[0]})
        time.sleep(2)  # let the sensors warm up

        print("=== headline ===")
        fps, ms = headline_fps(env)
        print(f"headline: {fps:.1f} FPS ({ms:.0f} ms/step)")

        print("\n=== phase breakdown ===")
        phase, total_ms, repolls = phase_breakdown(env)
        print(f"total {total_ms:.0f} ms/step, stale repolls: {repolls}")
        for name, t in phase.items():
            print(f"  {name:10s} {t / 100 * 1000:7.1f} ms")

        print("\n=== determinism ===")
        worst = check_determinism(env)
        print(f"max |dt - 50 ms| over 50 steps: {worst * 1000:.3f} ms "
              f"({'OK' if worst < 1e-4 else 'FAIL'})")

        print("\n=== lidar ===")
        pc = env._poll_lidar_points()
        bev = env._get_obs()["bev"]
        n_cells = int((bev != BEV_ROAD_LEVEL).sum())
        print(f"points per scan: {pc.shape[0]}, filled BEV cells: {n_cells}/{BEV_N * BEV_N}")

        print("\n=== drift parity ===")
        drift_parity(env)
    finally:
        # always close the game, so a crash never leaves a paused window behind
        env.close()


if __name__ == "__main__":
    main()
