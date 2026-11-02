"""Benchmark the remote (host_game.py) env path against the direct env.

The direct env (``benchmark_single.py``) steps the game in-process. The
hosted path (``host_game.py``) splits the env across ZeroMQ: the
``RemoteGameServer`` (game host) owns the game and the training host steps a
``RemoteBeamNGEnv`` client, so every step pays for a msgpack round trip of
the observation. This script measures that overhead in two parts:

1. pure remote overhead (no game): an echo REP server answers every step
   with a synthetic observation of the real layout, isolating the msgpack +
   ZeroMQ + client work of the wrapper path.
2. remote path with the game: the same setup ``host_game.py`` runs
   (``RemoteGameServer`` + ``RemoteBeamNGEnv``, localhost), measured against
   the direct env on the same game instance.

The server's tcom port is ``zmq_port + 1``, so the ZeroMQ port is chosen as
``TCOM_PORT - 1``: the hosted env reuses the game already running on
``TCOM_PORT`` (a second instance on the same user path fails to launch).

Usage:
    BNG_HOME=/path/to/BeamNG.tech venv/bin/python src/env/gymnasium/tests/benchmark_remote.py
"""
import os
import sys
import threading
import time

import msgpack
import numpy as np

# the env package lives in the parent folder (this script sits in tests/)
_GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _GYM_DIR)
from beamng_tech_env.core.array import to_cpu, use_device
from beamng_tech_env.core.constants import DRIFT_CARS_6SPEED, TCOM_PORT
from beamng_tech_env.core.obs import neutral_obs
from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv
from beamng_tech_env.envs.remote import (
    RemoteBeamNGEnv,
    RemoteGameServer,
    decode_value,
    encode_value,
)
from benchmark_single import headline_fps

N_STEPS = 100
ROUNDS = 2
ECHO_PORT = 5599
ZMQ_PORT = TCOM_PORT - 1  # the server's tcom port is then TCOM_PORT
# the GPU also serves inference, which skews step timing: settle it before
# every measured section (and after each, so the next reset starts clean)
SETTLE_SLEEP = 5.0


def _sample_obs():
    """A synthetic observation with the real layout (random BEV grids).

    The live env returns plain numpy (rig.get_obs moves every channel to the
    CPU); do the same so the payload matches what the server encodes.
    """
    obs = neutral_obs()
    obs["bev"] = (np.random.default_rng(0).random((1, 64, 64)) * 255
                  ).astype(np.uint8)
    obs["road_bev"] = np.random.default_rng(1).random((1, 64, 64)) \
        .astype(np.float32)
    return {k: to_cpu(v) for k, v in obs.items()}


def _sample_info():
    return {
        "reward_components": {"drift": 0.5, "dmg": -0.1},
        "game_fds": 300,
        "applied_action": np.zeros(4, dtype=np.float32),
        "spawn": "waypoint_1",
    }


def _echo_server(port, obs, info, stop):
    """A REP server that answers every message with a fixed step response."""
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    packed = msgpack.packb(
        {"ok": True, "obs": encode_value(obs), "reward": 1.0,
         "terminated": False, "truncated": False,
         "info": encode_value(info)}, use_bin_type=True)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    while not stop.is_set():
        try:
            socket.recv()
        except zmq.Again:
            continue
        socket.send(packed)
    socket.close()
    context.term()


def pure_overhead(n_steps=200):
    """The wrapper path's fixed cost with no game in the loop."""
    obs, info = _sample_obs(), _sample_info()
    stop = threading.Event()
    thread = threading.Thread(target=_echo_server,
                              args=(ECHO_PORT, obs, info, stop), daemon=True)
    thread.start()
    time.sleep(0.2)  # let the server bind
    client = RemoteBeamNGEnv(host="localhost", zmq_port=ECHO_PORT)
    action = np.zeros(4, dtype=np.float32)
    try:
        client.step(action)  # warm up (the zmq handshake)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            client.step(action)
        dt = time.perf_counter() - t0
    finally:
        client.close()
        stop.set()
        thread.join(timeout=5)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        [float(v) for v in np.asarray(action)]
    t_action = (time.perf_counter() - t0) / n_steps * 1000
    t0 = time.perf_counter()
    for _ in range(n_steps):
        packed = msgpack.packb(
            {"ok": True, "obs": encode_value(obs), "reward": 1.0,
             "terminated": False, "truncated": False,
             "info": encode_value(info)}, use_bin_type=True)
    t_encode = (time.perf_counter() - t0) / n_steps * 1000
    t0 = time.perf_counter()
    for _ in range(n_steps):
        decode_value(msgpack.unpackb(packed, raw=False))
    t_decode = (time.perf_counter() - t0) / n_steps * 1000

    total_ms = dt / n_steps * 1000
    print(f"echo round trip: {total_ms:.2f} ms/step ({n_steps / dt:.0f} steps/s)")
    print(f"  wire size:           {len(packed) / 1024:.1f} KB")
    print(f"  action -> floats:    {t_action:.3f} ms")
    print(f"  encode+pack:         {t_encode:.3f} ms")
    print(f"  unpack+decode:       {t_decode:.3f} ms")
    print(f"  zmq + scheduling:    {total_ms - t_action - t_encode - t_decode:.3f} ms")
    return total_ms


def _run_server(server):
    """The server's run loop on the right CUDA device.

    host_game.py runs it on the main thread (which core.array put on
    CUPY_DEVICE at import); a new thread defaults to device 0, so select the
    device first or to_gpu allocates on the wrong one.
    """
    use_device()
    server.run()


def _measure(env, n_steps=N_STEPS):
    """headline_fps as (fps, ms/step), skipping terminated episodes."""
    fps, ms = headline_fps(env, n_steps)
    return fps, ms


def game_benchmark(configs=("remote", "remote-debug", "direct")):
    """The selected configurations, measured on the shared game.

    All configurations share the one game running on TCOM_PORT; each gets a
    fresh reset before its measured rounds. Rounds alternate so game load
    drifts equally across the configurations.
    """
    results = {}
    for round_no in range(1, ROUNDS + 1):
        print(f"=== round {round_no}/{ROUNDS} ===")
        # rotate the order so each config gets one fresh-state slot and one
        # post-steps slot (the game degrades across a round, so a fixed order
        # would bias the configs that run first)
        order = configs[(round_no - 1) % len(configs):] + \
            configs[: (round_no - 1) % len(configs)]
        for config in order:
            time.sleep(SETTLE_SLEEP)  # let the GPU settle (inference load)
            if config in ("remote", "remote-debug"):
                server = RemoteGameServer(zmq_port=ZMQ_PORT,
                                          debug_window=(config == "remote-debug"))
                thread = threading.Thread(target=_run_server,
                                          args=(server,), daemon=True)
                thread.start()
                client = RemoteBeamNGEnv(host="localhost", zmq_port=ZMQ_PORT)
            else:
                server = thread = None
                client = BeamNGTechEnv()
            try:
                client.reset(seed=0, options={"car": DRIFT_CARS_6SPEED[0]})
                time.sleep(2)  # sensor warm-up (mirrors benchmark_single)
                time.sleep(SETTLE_SLEEP)  # and let the GPU settle
                fps, ms = _measure(client)
                results.setdefault(config, []).append((fps, ms))
                print(f"{config:18s} {fps:.1f} FPS ({ms:.0f} ms/step)")
            finally:
                client.close()
                if server is not None:
                    server.stop()
                    thread.join(timeout=30)
                    server.close()

    print("\n=== summary (mean over rounds) ===")
    for name in configs:
        ms = np.mean([ms for _, ms in results[name]])
        fps = np.mean([fps for fps, _ in results[name]])
        print(f"{name:18s} {fps:5.1f} FPS ({ms:5.0f} ms/step)")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",
                        choices=("echo", "direct", "remote", "remote-debug"),
                        help="run a single section (default: all)")
    parser.add_argument("--configs",
                        help="comma-separated game configs to run "
                             "(direct,remote,remote-debug); overrides --only")
    args = parser.parse_args()
    if args.configs:
        game_benchmark(configs=tuple(
            c.strip() for c in args.configs.split(",") if c.strip()))
        return
    if args.only in (None, "echo"):
        print("=== pure remote overhead (no game) ===")
        pure_overhead()
        print()
    if args.only in (None, "remote"):
        game_benchmark(configs=("remote",))
    elif args.only == "remote-debug":
        game_benchmark(configs=("remote-debug",))
    elif args.only == "direct":
        game_benchmark(configs=("direct",))


if __name__ == "__main__":
    main()
