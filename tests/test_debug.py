"""Fast tests for the live debug window (no game, no display required).

The render path is exercised directly (frame shape + content), and the
agent-loop contract is checked: ``enqueue_latest`` never blocks and drops the
oldest frame, ``update`` stays cheap even when the display thread is gone,
and ``close`` always joins the thread.
"""
import queue
import threading
import time

import gymnasium as gym
import numpy as np
import pytest

from beamng_tech_env.core.array import to_cpu
from beamng_tech_env.core.constants import ALLOWED_GEARS, BEV_N, gear_to_action
from beamng_tech_env.debug import (DebugWindow, DebugWindowWrapper,
                                   TRACE_LEN, PANEL_MIN_W, _TRACE_SPECS,
                                   enqueue_latest,
                                   make_layout,
                                   make_traces, _payload_from,
                                   _render_frame)

cv2 = pytest.importorskip("cv2")


def _payload(bev=None, **overrides):
    p = {
        "bev": bev,
        "speed": 30.0,
        "angle": 20.0,
        "rpm": 0.4,
        "gear": 2.0 / 7.0,
        "shifting": 0.0,
        "ffb": 0.1,
        "reward": 0.5,
        "on_road": None,
        "action": np.concatenate([[0.1, 0.5, 0.0],
                                  to_cpu(gear_to_action(1))]).astype(np.float32),
    }
    p.update(overrides)
    return p


def _obs(bev=None):
    return {
        "rpm": np.array([0.4], np.float32),
        "imu_acc": np.zeros(3, np.float32),
        "imu_angvel": np.zeros(3, np.float32),
        "drift_angle": np.array([20.0 / 180.0], np.float32),
        "airspeed": np.array([30.0 / 300.0], np.float32),
        "is_shifting": np.array([0.0], np.float32),
        "gear": np.array([2.0 / 7.0], np.float32),
        "ffb": np.array([0.1], np.float32),
        "bev": bev,
    }


def test_enqueue_latest_drop_oldest():
    q = queue.Queue(maxsize=1)
    q.put_nowait("old")
    enqueue_latest(q, "new")
    assert q.get_nowait() == "new"
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_render_frame_shape():
    L = make_layout(960)
    rng = np.random.default_rng(0)
    bev = rng.integers(0, 255, (1, BEV_N, BEV_N), dtype=np.uint8)
    frame = _render_frame(cv2, _payload(bev=bev), make_traces(), L)
    assert frame.shape == (L.height, L.width, 3)
    assert frame.dtype == np.uint8


def test_render_frame_smaller_width():
    L = make_layout(640)
    bev = np.full((BEV_N, BEV_N), 128, np.uint8)
    frame = _render_frame(cv2, _payload(bev=bev), make_traces(), L)
    assert frame.shape == (L.height, L.width, 3)


def test_make_layout_panels_do_not_overlap():
    for width in (640, 960, 1440):
        L = make_layout(width)
        assert L.height == width * 9 // 16  # 16:9
        assert 0 < L.bev_px <= L.height     # square BEV, at most full height
        assert L.panel_x0 >= L.bev_px
        assert L.panel_x1 > L.panel_x0
        assert L.panel_x1 <= L.width
        # the panel is wide enough for the text list + the action widgets
        assert L.panel_x1 - L.panel_x0 >= PANEL_MIN_W


def test_render_frame_without_bev():
    L = make_layout(960)
    frame = _render_frame(cv2, _payload(bev=None), make_traces(), L)
    assert frame.shape == (L.height, L.width, 3)


def test_render_frame_no_action():
    L = make_layout(960)
    bev = np.full((BEV_N, BEV_N), 128, np.uint8)
    frame = _render_frame(cv2, _payload(bev=bev, action=None), make_traces(), L)
    assert frame.shape == (L.height, L.width, 3)


def test_render_frame_on_road():
    # the on-road flag renders in the telemetry panel (yes / no / ---)
    L = make_layout(960)
    bev = np.full((BEV_N, BEV_N), 128, np.uint8)
    for on_road in (True, False, None):
        frame = _render_frame(cv2, _payload(bev=bev, on_road=on_road),
                              make_traces(), L)
        assert frame.shape == (L.height, L.width, 3)


def _trace_band(L, index):
    """The (y0, y1) of the index-th trace panel (mirrors _draw_panel)."""
    s = L.s

    def px(v):
        return max(1, int(round(v * s)))

    trace_top, trace_bottom = px(232), L.height - px(16)
    gap = px(8)
    trace_h = (trace_bottom - trace_top - 2 * gap) // 3
    y0 = trace_top + index * (trace_h + gap)
    return y0, y0 + trace_h


def _trace_xspan(frame, L, index, color, tol=60):
    """The (xmin, xmax) of the trace (its color) in one trace panel."""
    y0, y1 = _trace_band(L, index)
    seg = frame[y0:y1, L.panel_x0:L.panel_x1].astype(int)
    mask = (np.abs(seg - np.array(color)).max(axis=2) <= tol)
    xs = np.flatnonzero(mask.any(axis=0))
    assert len(xs), "trace not found in the panel"
    return L.panel_x0 + xs.min(), L.panel_x0 + xs.max()


def test_trace_slides_at_fixed_scale():
    """The trace scrolls at a fixed scale (the newest sample at the right
    edge, the older ones sliding out the left), not rescaled to fit the
    panel: shorter than a full window it occupies only the right part of
    the panel, and at a full window it spans the panel exactly once."""
    L = make_layout(960)
    x0, x1 = L.panel_x0, L.panel_x1
    step = (x1 - x0) / (TRACE_LEN - 1)
    traces = make_traces()
    # 50 samples + the one the render appends = 51: the trace must end at
    # the right edge and start 50 fixed steps left of it, NOT at x0
    for label, *_ in _TRACE_SPECS:
        traces[label].extend([0.5] * 50)
    frame = _render_frame(cv2, _payload(bev=None), traces, L)
    xmin, xmax = _trace_xspan(frame, L, 0, (0, 0, 255))
    assert xmax >= x1 - 3
    assert xmin >= x1 - 50 * step - 3
    assert xmin > x0 + (x1 - x0) // 2  # not stretched across the panel
    # a full window (200 samples + the render's) spans the panel exactly
    for label, *_ in _TRACE_SPECS:
        traces[label].extend([0.5] * (TRACE_LEN - 50))
    frame = _render_frame(cv2, _payload(bev=None), traces, L)
    xmin, xmax = _trace_xspan(frame, L, 0, (0, 0, 255))
    assert xmax >= x1 - 3
    assert xmin <= x0 + 3


def test_payload_from_on_road():
    # the on-road flag passes through the payload as a plain bool (or None)
    p = _payload_from(_obs(bev=None), None, 0.0, None, on_road=True)
    assert p["on_road"] is True
    p = _payload_from(_obs(bev=None), None, 0.0, None, on_road=False)
    assert p["on_road"] is False
    p = _payload_from(_obs(bev=None), None, 0.0, None)
    assert p["on_road"] is None


def test_update_never_blocks():
    win = DebugWindow()
    try:
        obs = _obs(bev=np.full((1, BEV_N, BEV_N), 128, np.uint8))
        action = np.zeros(3 + len(ALLOWED_GEARS), np.float32)
        t0 = time.perf_counter()
        for _ in range(200):
            win.update(obs, action=action, reward=0.1)
        dt = time.perf_counter() - t0
        # 200 snapshots + non-blocking puts: well under a second (typically ms)
        assert dt < 1.0
    finally:
        win.close()


def test_close_joins_thread():
    win = DebugWindow()
    win.close()
    assert win._thread.is_alive() is False
    win.close()  # idempotent


def test_payload_from_cupy_bev():
    """A cupy BEV must end up as a plain numpy array in the payload (cupy 14
    forbids the implicit np.asarray conversion, and the display thread
    renders with cv2/numpy)."""
    from beamng_tech_env.core.array import USE_CUPY, to_gpu
    if not USE_CUPY:
        pytest.skip("cupy not installed")
    bev = to_gpu(np.full((1, BEV_N, BEV_N), 128, np.uint8))
    p = _payload_from(_obs(bev=None), None, 0.0, bev)
    assert type(p["bev"]) is np.ndarray
    assert p["bev"].shape == (1, BEV_N, BEV_N)


def test_render_frame_in_new_thread():
    """The display thread is a new thread whose current CUDA device defaults
    to 0; the render must select CUPY_DEVICE or bev_to_beams raises a device
    mismatch (its beam tables live on CUPY_DEVICE)."""
    from beamng_tech_env.core.array import USE_CUPY
    if not USE_CUPY:
        pytest.skip("cupy not installed")
    L = make_layout(960)
    bev = np.full((1, BEV_N, BEV_N), 128, np.uint8)
    out = []

    def work():
        try:
            out.append(_render_frame(cv2, _payload(bev=bev),
                                     make_traces(), L).shape)
        except Exception as err:
            out.append(err)

    t = threading.Thread(target=work)
    t.start()
    t.join(timeout=30)
    assert isinstance(out[0], tuple)


class _FakeEnv(gym.Env):
    """A minimal env with the env's Dict observation (no game)."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.n = 0
        self.observation_space = gym.spaces.Dict({
            "rpm": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "imu_acc": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            "imu_angvel": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            "drift_angle": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "airspeed": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "is_shifting": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "gear": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "ffb": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "bev": gym.spaces.Box(0, 255, (1, BEV_N, BEV_N), np.uint8),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    def _obs(self):
        return _obs(bev=np.full((1, BEV_N, BEV_N), 128, np.uint8))

    def reset(self, seed=None, options=None):
        self.n = 0
        return self._obs(), {}

    def step(self, action):
        self.n += 1
        return self._obs(), 0.1, False, False, {}

    def close(self):
        pass


def test_wrapper_step_and_close():
    env = DebugWindowWrapper(_FakeEnv())
    try:
        obs, info = env.reset()
        obs, r, term, trunc, info = env.step(np.zeros(4, np.float32))
        assert obs["bev"].shape == (1, BEV_N, BEV_N)
        assert r == 0.1
    finally:
        env.close()


def test_render_frame_road_outlines():
    L = make_layout(960)
    bev = np.full((1, BEV_N, BEV_N), 128, np.uint8)
    # a horizontal outline at grid row 20, spanning cols 10..50
    row, c0, c1 = 20.0, 10.0, 50.0
    poly = np.stack([np.full(5, row), np.linspace(c0, c1, 5)], axis=1)
    # an active outline (the surface the car is closest to) and a dim one
    # (another surface), the dim one at row 30
    dim_poly = np.stack([np.full(5, 30.0), np.linspace(c0, c1, 5)], axis=1)
    outlines = [(poly, True), (dim_poly, False)]
    frame = _render_frame(cv2, _payload(bev=bev, road_outlines=outlines),
                          make_traces(), L)
    scale = L.bev_px / BEV_N
    y = int(row * scale)
    x0, x1 = int(c0 * scale), int(c1 * scale)
    target = np.array((255, 255, 0))
    # the active outline is drawn bright (a +/-2 px band, antialiasing tol)
    band = frame[max(0, y - 2):y + 3, x0:x1 + 1]
    close = np.abs(band.astype(int) - target).max(axis=2) <= 60
    assert close.sum() > (x1 - x0) // 2
    # the other surface's outline is drawn in the dim color
    yd = int(30.0 * scale)
    dim_target = np.array((120, 120, 0))
    dim_band = frame[max(0, yd - 2):yd + 3, x0:x1 + 1]
    dim_close = np.abs(dim_band.astype(int) - dim_target).max(axis=2) <= 60
    assert dim_close.sum() > (x1 - x0) // 2
    # and nothing is drawn outside the BEV panel
    assert not (np.abs(frame[:, L.panel_x0:].astype(int) - target)
                .max(axis=2) <= 60).any()


def test_payload_from_outlines():
    bev = np.full((1, BEV_N, BEV_N), 128, np.uint8)
    # (outline, active) tuples; a bare outline array is treated as active
    outlines = [(np.array([[10.0, 20.0], [30.0, 40.0]]), True),
                (np.array([[0.0, 0.0], [1.0, 1.0]]), False)]
    p = _payload_from(_obs(bev=bev), None, 0.5, bev, outlines)
    assert len(p["road_outlines"]) == 2
    np.testing.assert_allclose(p["road_outlines"][0][0], outlines[0][0])
    assert p["road_outlines"][0][1] is True
    assert p["road_outlines"][1][1] is False
    p2 = _payload_from(_obs(bev=bev), None, 0.5, bev,
                       [np.array([[10.0, 20.0], [30.0, 40.0]])])
    assert p2["road_outlines"][0][1] is True
    # no outlines -> None
    assert _payload_from(_obs(bev=bev), None, 0.5, bev)["road_outlines"] is None


def test_wrapper_extras_use_rig():
    class _Rig:
        _last_bev = np.full((1, BEV_N, BEV_N), 100, np.uint8)

        def road_surface_outlines(self):
            return [(np.array([[10.0, 20.0], [30.0, 40.0]]), True)]

        def road_centerline(self):
            return np.array([[10.0, 20.0], [30.0, 40.0]]), 1.5

        def on_road(self, fast_track=True, centerline_dist=None):
            return True

    env = _FakeEnv()
    env.rig = _Rig()
    wrapped = DebugWindowWrapper(env)
    try:
        # an obs without a BEV: the BEV, outlines, centerline and on-road
        # flag come from the rig
        obs = {k: v for k, v in env._obs().items() if k != "bev"}
        bev, outlines, centerline, cl_dist, on_road = wrapped._extras(obs)
        assert bev is env.rig._last_bev
        assert outlines is not None and len(outlines) == 1
        assert outlines[0][1] is True
        assert centerline is not None and centerline.shape == (2, 2)
        assert cl_dist == 1.5
        assert on_road is True
        # an obs that carries the BEV: the BEV from the obs, the rest from the rig
        bev2, outlines2, centerline2, cl_dist2, on_road2 = \
            wrapped._extras(env._obs())
        assert bev2 is not None
        assert len(outlines2) == len(outlines)
        np.testing.assert_allclose(outlines2[0][0], outlines[0][0])
        assert outlines2[0][1] == outlines[0][1]
        np.testing.assert_allclose(centerline2, centerline)
        assert cl_dist2 == cl_dist
        assert on_road2 is True
    finally:
        wrapped.close()
