"""Parse BeamNG ``.rpl`` replay files offline.

A ``.rpl`` is an Ogg container of msgpack packets at 40 Hz (the render rate);
each packet holds ~38 objects, the ones that matter being:

  19: vehicle position (x, y, z) in the world frame (z up)
  20: vehicle orientation quaternion (x, y, z, w)
  37: dict of JSON telemetry blobs (electrics, engineInfo, wheelThermalData, ...)

The quaternion's rotation matrix maps world -> local (its rows are the vehicle
local axes in world space); the vehicle local frame is X=left, Y=back, Z=up.
Verified live against the State sensor's dir/up: forward = -R[1], left = R[0],
up = R[2].

This module is the shared offline parser used by both the data-regeneration
converter and the replay environment (which sources position/orientation from
the .rpl because the State sensor is stale during playback).
"""
import json
import struct
from pathlib import Path

import msgpack
import numpy as np

# The game's save directory (where ``replays/`` lives). Overridable via env.
_GAME_SAVE_ENV = "BNG_SAVE_DIR"


def game_save_dir() -> Path:
    """The game's save directory (``.../BeamNG.tech/current``)."""
    import os
    env = os.environ.get(_GAME_SAVE_ENV)
    if env:
        return Path(env)
    return Path.home() / ".local/share/BeamNG/BeamNG.tech/current"


def read_packets(path):
    """Yield (granule, packet_bytes) for every Ogg packet in the file."""
    data = open(path, "rb").read()
    n = len(data)
    cur = b""
    off = 0
    prev_seq = None
    while True:
        if off + 27 > n or data[off:off + 4] != b"OggS" or data[off + 4] != 0:
            break
        granule = struct.unpack("<q", data[off + 6:off + 14])[0]
        seq = struct.unpack("<I", data[off + 18:off + 22])[0]
        if prev_seq is not None and seq != prev_seq + 1:
            break
        prev_seq = seq
        n_seg = data[off + 26]
        lacing = data[off + 27:off + 27 + n_seg]
        total = sum(255 if v == 255 else v for v in lacing)
        seg = data[off + 27 + n_seg:off + 27 + n_seg + total]
        sp = 0
        for v in lacing:
            take = 255 if v == 255 else v
            cur += seg[sp:sp + take]
            sp += take
            if v < 255:
                yield granule, cur
                cur = b""
        off += 27 + n_seg + total


def quat_to_axes(q):
    """(forward, left, up) unit vectors from a (x, y, z, w) quaternion.

    The quaternion's rotation matrix maps world -> local (its rows are the
    vehicle's local axes in world space); the vehicle local frame is
    X=left, Y=back, Z=up. So forward = -R[1], left = R[0], up = R[2].
    """
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - z * w); r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w); r21 = 2 * (y * z + x * w); r22 = 1 - 2 * (x * x + y * y)
    forward = np.array([-r10, -r11, -r12])
    left = np.array([r00, r01, r02])
    up = np.array([r20, r21, r22])
    return forward, left, up


def frames(path):
    """Yield (t, pos, quat, electrics) for every replay frame.

    ``t`` is seconds from the start of the file; ``pos`` and ``quat`` are lists
    of floats; ``electrics`` is the decoded electrics JSON dict (or None).
    """
    for granule, packet in read_packets(path):
        u = msgpack.Unpacker(raw=False, strict_map_key=False)
        u.feed(packet)
        objs = list(u)
        if len(objs) < 38:
            continue
        pos, quat, tele = objs[19], objs[20], objs[37]
        if not isinstance(pos, list) or not isinstance(quat, list):
            continue
        electrics = None
        if isinstance(tele, dict) and "electrics" in tele:
            try:
                electrics = json.loads(tele["electrics"])
            except Exception:
                electrics = None
        yield (granule / 1e6, [float(v) for v in pos],
               [float(v) for v in quat], electrics)


class RplTrajectory:
    """A loaded .rpl: the vehicle (t, pos, quat, electrics) trajectory.

    Provides ``state_at(secs)`` returning a State-sensor-shaped dict
    (pos/vel/dir/up/time) built from the .rpl, and ``electrics_at(secs)``.
    The State sensor is stale during playback, so the replay environment uses
    this to source the true position/orientation/telemetry.
    """

    def __init__(self, path):
        rows = list(frames(path))
        if not rows:
            raise ValueError(f"no frames in {path}")
        self.t = np.array([r[0] for r in rows])
        self.pos = np.array([r[1] for r in rows])
        self.quat = np.array([r[2] for r in rows])
        self.electrics = [r[3] for r in rows]
        # velocity by central difference (40 Hz)
        self.vel = np.empty_like(self.pos)
        self.vel[1:-1] = (self.pos[2:] - self.pos[:-2]) / (self.t[2:] - self.t[:-2])[:, None]
        self.vel[0] = (self.pos[1] - self.pos[0]) / (self.t[1] - self.t[0])
        self.vel[-1] = (self.pos[-1] - self.pos[-2]) / (self.t[-1] - self.t[-2])

    def _idx(self, secs):
        return int(np.clip(np.searchsorted(self.t, secs), 0, len(self.t) - 1))

    def state_at(self, secs):
        """A State-sensor-shaped dict (pos/vel/dir/up/time) at ``secs``."""
        i = self._idx(secs)
        f, l, u = quat_to_axes(self.quat[i])
        return {"pos": self.pos[i].copy(), "vel": self.vel[i].copy(),
                "dir": f, "up": u, "time": float(self.t[i])}

    def electrics_at(self, secs):
        """The electrics dict at ``secs`` (may be None if not recorded)."""
        return self.electrics[self._idx(secs)]

    def __len__(self):
        return len(self.t)
