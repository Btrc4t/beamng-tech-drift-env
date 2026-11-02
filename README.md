# beamng-tech-drift-env

A Gymnasium environment for drifting in BeamNG.tech (v0.39). It drives a
spawned car through the `west_coast_usa` Gymnasium track, streams the
sensors (LiDAR, IMU, electrics, force feedback) at the game's 120 Hz
physics rate, and exposes a normalized action space (throttle, steering,
brake, gear selection).

This package is the environment used by the
[drift-ai](https://github.com/Btrc4t/beamng-tech-drift-ai) project, which
checks it out as a git submodule. See the
[drift-ai README](https://github.com/Btrc4t/beamng-tech-drift-ai/blob/dev/README.md)
for the full workflow around it: data collection, curation, online and
offline training, and live driving.

## Installation

You need Python 3.10 or newer and a copy of BeamNG.tech v0.39 on disk (the
env reads its install path from the `BNG_HOME` environment variable).

```sh
pip install -e .
# optional extras:
pip install -e ".[debug]"   # live cv2 debug window (opencv-python, scipy)
pip install -e ".[dev]"     # pytest, pre-commit
```

## Usage

```python
import gymnasium as gym

env = gym.make("beamng_tech_env/BeamNGTech-v0.39")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The constructor takes:

- `human_mode` - a person drives (wheel/pedals/keyboard); the env sends no
  control and reads the applied action back from the game
- `ai_mode` - the game's built-in AI drives the car (synthetic mode;
  `ai_speed` and `ai_aggression` tune it)
- `host`, `tcom_port` - where the game's tcom server listens
  (default: `localhost`, the package's `TCOM_PORT`)
- `launch` - launch the game if it is not running (default: true)
- `transport` - `"ros2"` (default) reads the sensor stream from the C++
  bridge's ROS2 topics; `"shmem"` uses the BeamNGpy socket + shared memory

The action is a flat vector: `[throttle, steering, brake]` in `[-1, 1]`
plus a one-hot gear selection over reverse and gears 1-4. The observation
is a `Dict` with one key per sensor channel: `rpm`, `imu_acc`,
`imu_angvel`, `drift_angle`, `airspeed`, `is_shifting`, `gear`, `ffb`
(force feedback), `bev` (the LiDAR bird's-eye view, a 64x64 `uint8`
image), and `road_bev` (the road network rasterized into the same grid).

A second env id, `beamng_tech_env/BeamNGReplay-v0.39`, replays a recorded
`.rpl` file through the same observation pipeline.

## GPU processing (cupy)

The observation and point-cloud processing runs on the GPU via cupy when
it is installed; otherwise it falls back to the CPU. Set the `USE_CUPY`
environment variable to `0` (or `false`/`no`/`off`) to force the CPU
backend even when cupy is installed. `CUPY_DEVICE` selects the CUDA device
by numeric id (default: the last one, so the game's renderer and the env
can share a machine without contending for the same GPU).

## Remote env (ZeroMQ)

The tcom interaction loop must run on the machine that hosts the game
(the LiDAR reads from shared memory there). `host_game.py` hosts a game
instance and serves the env over a ZeroMQ REP socket, so a training host
can step it from another machine:

```sh
./host_game.py <ip> [zmq_port] [--debug-window]
```

On the training host, `RemoteBeamNGEnv` (in `beamng_tech_env.envs.remote`)
is a drop-in gymnasium env with the same action/observation spaces, and
`BeamNGTechHostedVectorEnv` wraps several game hosts in a vector env.

## Tests

The test suite lives in `tests/`. The pure-math tests run without the
game; the env and benchmark tests (marked `game`) launch BeamNG.drive once
per session and need `BNG_HOME` plus a display.

```sh
pytest -m "not game"   # fast tests only
pytest                 # everything, including the game tests
```
