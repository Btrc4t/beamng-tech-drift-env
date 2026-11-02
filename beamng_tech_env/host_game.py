#!/usr/bin/env python3
"""Host a BeamNG.tech instance and serve its env over ZeroMQ.

Runs on the game's host. It owns the game (launches and recovers it, via the
env) and serves the drive env over a ZeroMQ REP socket, so the training host
can step it from another machine. The LiDAR reads from shared memory, which
only exists on this host, so the tcom interaction loop (and the observation
build) must run here; only the small observation crosses the network. The
tcom port stays internal to this host - the ZeroMQ port is the only external
interface.

    ./host_game.py <ip> [zmq_port] [--debug-window]

The game runs until Ctrl+C (which stops it for good). The ZeroMQ port
defaults to 5555. ``--debug-window`` shows a live cv2 debug window (BEV +
beams + telemetry) on this host. Needs DISPLAY and XAUTHORITY set in your
shell (see env-drift-tools); machine-specific paths (BNG_HOME, VENV_PATH)
are read from env-drift-tools next to this script.
"""

import os
import re
import sys
import time


def log(msg):
    print(f"[host_game] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_tools():
    """Read the ``export KEY=VALUE`` pairs from env-drift-tools (next to this
    script): machine-specific paths (BNG_HOME, VENV_PATH)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env-drift-tools")
    tools = {}
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", line)
                if m:
                    tools[m.group(1)] = m.group(2)
    except OSError:
        pass
    return tools


def use_venv_python():
    """Re-exec under the venv's python (the env needs beamngpy and pyzmq)."""
    tools = load_tools()
    venv = tools.get("VENV_PATH") or os.environ.get("VENV_PATH")
    if venv and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        for name in ("python", "python3"):
            venv_python = os.path.join(venv, "bin", name)
            if os.path.isfile(venv_python):
                os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])
                break
    try:
        import beamng_tech_env  # noqa: F401  (fail early if the env is missing)
        import zmq  # noqa: F401
    except ImportError as err:
        sys.exit(f"error: {err.name} not available under {sys.executable}; "
                 f"run with the venv's python (VENV_PATH={venv})")
    return tools


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Host a BeamNG.tech instance and serve its env over ZeroMQ.")
    parser.add_argument("ip", help="the IP to bind the ZeroMQ socket to")
    parser.add_argument("zmq_port", nargs="?", type=int, default=None,
                        help="the ZeroMQ port (default: the env's ZMQ_PORT)")
    parser.add_argument("--debug-window", action="store_true",
                        help="show a live cv2 debug window (BEV + beams + "
                             "telemetry) on this host")
    args = parser.parse_args()
    ip = args.ip

    tools = use_venv_python()
    bng_home = os.environ.get("BNG_HOME") or tools.get("BNG_HOME")
    if not bng_home:
        sys.exit("error: BNG_HOME not set (is env-drift-tools next to this script?)")

    # the server owns the game (launches and recovers it) and serves the env
    # over ZeroMQ; the tcom port stays internal to this host
    from beamng_tech_env.envs.remote import RemoteGameServer, ZMQ_PORT
    zmq_port = args.zmq_port if args.zmq_port is not None else ZMQ_PORT
    server = RemoteGameServer(zmq_port=zmq_port, debug_window=args.debug_window)
    log(f"serving the env on {ip}:{zmq_port}"
        + (" (debug window on)" if args.debug_window else "")
        + " (Ctrl+C to stop)")
    try:
        server.run()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        server.close()


if __name__ == "__main__":
    main()
