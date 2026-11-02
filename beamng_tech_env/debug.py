"""Live debug window: a fast, async cv2 view of the BEVs and telemetry.

Like ``data.curate video`` (the LiDAR BEV with the 20 ultrasonic beams
overlaid, plus the action and the telemetry), but live and async: the agent
loop never renders. The window shows the (road-corrected) LiDAR BEV with the
road surface edge outlines (the navigraph segments that fill the road cells)
and its beams, and the telemetry panel on the right. ``DebugWindow.update`` snapshots the small
frame payload (one 64x64 BEV plus a handful of scalars) into a single-slot
queue and returns without blocking; a daemon thread renders with plain
cv2/numpy (no matplotlib) and shows the window. If the thread falls behind,
frames are dropped, so the loop is paced by the game, not by the display.

Needs a display (DISPLAY/XAUTHORITY), like the game itself. If there is none
(or it goes away), the display thread stops and ``update`` stays a cheap
no-op, so the agent loop keeps running.

Usage:
    from beamng_tech_env.debug import DebugWindow, DebugWindowWrapper
    env = DebugWindowWrapper(env)            # one window per env
    # or standalone:
    win = DebugWindow()
    win.update(obs, action=action, reward=reward)
    win.close()
"""
import queue
import threading
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from beamng_tech_env.core.array import to_cpu, use_device
from beamng_tech_env.core.constants import (
    AIRSPEED_MAX_KPH,
    BEV_N,
    LIDAR_MAX_DIST,
    MAX_GEAR,
)
from beamng_tech_env.wrappers import BEAM_ANGLES, bev_to_beams

__all__ = ["DebugWindow", "DebugWindowWrapper", "make_layout"]

# rolling telemetry traces: ~10 s at 20 FPS
TRACE_LEN = 200
# beam ray angles (rad), precomputed once (0 = travel direction, + = left)
_TH = np.deg2rad(BEAM_ANGLES)
# fixed trace ranges (no per-frame rescaling): (label, lo, hi, BGR color)
_TRACE_SPECS = (
    ("drift angle (deg)", -180.0, 180.0, (0, 0, 255)),
    ("airspeed (kph)", 0.0, 120.0, (255, 160, 0)),
    ("force feedback", -1.0, 1.0, (0, 255, 0)),
)
# the telemetry panel must be at least this wide: the observation text list and
# the action bars + steering gauge sit side by side in it
PANEL_MIN_W = 260
# the road surface edge outlines on the BEV (cyan: distinct from the grayscale
# BEV and the red->green beam rays): bright for the road surface the car is
# closest to, dim for the other nearby surfaces
_ROAD_OUTLINE_COLOR = (255, 255, 0)
_ROAD_OUTLINE_DIM_COLOR = (120, 120, 0)
# the merged road centerline on the BEV (magenta: distinct from the cyan
# outlines, the grayscale BEV and the red->green beam rays)
_CENTERLINE_COLOR = (255, 0, 255)


def make_layout(width=960):
    """Pixel layout for a 16:9 window: the square LiDAR BEV on the left (as
    large as it fits, at x=0), the telemetry panel on the right, and the beam
    ray unit vectors in pixels per metre. Grid convention (see
    wrappers.bev_to_beams): row 0 = forward, col 0 = left, 1.25 m per cell."""
    height = width * 9 // 16
    bev_px = min(height, width - PANEL_MIN_W - 32)
    gap = 16
    k = bev_px / (2.0 * LIDAR_MAX_DIST)
    return SimpleNamespace(
        width=width,
        height=height,
        bev_px=bev_px,
        panel_x0=bev_px + gap,
        panel_x1=width - 16,
        beam_ux=-np.sin(_TH) * k,
        beam_uy=-np.cos(_TH) * k,
        center=(bev_px // 2, bev_px // 2),
        s=height / 480.0,  # panel scale factor (the layout is drawn at 480)
    )


def make_traces():
    """The rolling telemetry windows, one per trace (see _TRACE_SPECS)."""
    return {label: deque(maxlen=TRACE_LEN) for label, *_ in _TRACE_SPECS}


def enqueue_latest(q, payload):
    """Put payload into a maxsize=1 queue without blocking: if the slot is
    full (the display thread is behind), drop the older frame."""
    try:
        q.put_nowait(payload)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(payload)


def _map_window(cv2, title):
    """Make the cv2 window visible so it actually shows.

    cv2/Qt creates the window from the display thread, but in some contexts
    (notably once the game has been launched) it is left in the X11 Withdrawn
    state - created but never mapped - so it never shows. Ask Qt to show it
    (WND_PROP_VISIBLE) and keep it on top (WND_PROP_TOPMOST, best-effort: some
    builds ignore it). Pure cv2, no external tools.
    """
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_VISIBLE, 1)
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except Exception:  # noqa: BLE001 - the window is a best-effort extra
        pass


def _scalar(value):
    """One float from an observation channel ((1,) array) or 0.0."""
    if value is None:
        return 0.0
    return float(np.asarray(value).reshape(-1)[0])


def _draw_bev(cv2, frame, grid, dists, L, x0, outlines=None, centerline=None):
    """One BEV panel at column offset ``x0``: the grid resized to the panel
    (uint8 as-is, float scaled 0..1 -> 0..255) with the road surface edge
    outlines (``outlines``: (outline, active) tuples, outline a (row, col)
    polyline in grid units from the rig, active = the road surface the car is
    closest to - drawn bright, the other surfaces dim), the merged road
    centerline (``centerline``: (row, col) points in grid units) and the beam
    rays (``dists`` in metres, one per beam) drawn from the panel center, red
    (near) to green (far)."""
    if grid.ndim == 3:
        grid = grid[0]
    if grid.dtype != np.uint8:
        grid = (np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = cv2.resize(grid, (L.bev_px, L.bev_px), interpolation=cv2.INTER_NEAREST)
    frame[:L.bev_px, x0:x0 + L.bev_px] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = L.bev_px / BEV_N
    if outlines:
        # grid (row, col) units -> panel pixels (row 0 = top = forward,
        # col 0 = left = the panel's x0)
        for item in outlines:
            poly, active = (item if isinstance(item, (tuple, list))
                            else (item, True))
            color = _ROAD_OUTLINE_COLOR if active else _ROAD_OUTLINE_DIM_COLOR
            pts = np.asarray(poly, float)
            if pts.shape[0] < 2:
                continue
            xy = np.stack([x0 + pts[:, 1] * scale, pts[:, 0] * scale], axis=1)
            cv2.polylines(frame, [xy.astype(np.int32).reshape(-1, 1, 2)], False,
                          color, 1, cv2.LINE_AA)
    if centerline is not None:
        pts = np.asarray(centerline, float)
        if pts.shape[0] >= 2:
            xy = np.stack([x0 + pts[:, 1] * scale, pts[:, 0] * scale], axis=1)
            cv2.polylines(frame, [xy.astype(np.int32).reshape(-1, 1, 2)], False,
                          _CENTERLINE_COLOR, 2, cv2.LINE_AA)
    cx, cy = x0 + L.center[0], L.center[1]
    for ux, uy, dist in zip(L.beam_ux, L.beam_uy, dists):
        ex, ey = int(cx + ux * dist), int(cy + uy * dist)
        t = min(1.0, max(0.0, dist / LIDAR_MAX_DIST))
        color = (0, int(255 * t), int(255 * (1.0 - t)))
        cv2.line(frame, (cx, cy), (ex, ey), color, 1)
        cv2.circle(frame, (ex, ey), 2, color, -1)


def _label(cv2, frame, text, x0, L):
    cv2.putText(frame, text, (x0 + 4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * L.s, (255, 255, 255), max(1, int(round(L.s))),
                cv2.LINE_AA)


def _draw_steering(cv2, frame, steering, cx, cy, r, s):
    """A semi-circle steering gauge (the upper half of a circle) with an
    indicator needle sweeping -90 (left) .. +90 (right), 0 = straight ahead.
    ``steering`` is the normalised action in [-1, 1]."""
    lw = max(1, int(round(s)))
    # the gauge arc: cv2 angles 0 (right) .. 180 (left) = the upper half
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, 180, (120, 120, 120), lw)
    # tick marks at -90, -45, 0, 45, 90
    for tick in (-90, -45, 0, 45, 90):
        th = np.deg2rad(90.0 - tick)
        cv2.line(frame,
                 (int(cx + (r - 8 * s) * np.cos(th)),
                  int(cy - (r - 8 * s) * np.sin(th))),
                 (int(cx + r * np.cos(th)), int(cy - r * np.sin(th))),
                 (120, 120, 120), lw)
    # the needle: steering 0 = up, +1 = right, -1 = left
    th = np.deg2rad(90.0 - steering * 90.0)
    ex, ey = int(cx + r * np.cos(th)), int(cy - r * np.sin(th))
    cv2.line(frame, (cx, cy), (ex, ey), (0, 200, 255),
             max(2, int(round(2 * s))), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(2, int(round(3 * s))), (0, 200, 255), -1)
    cv2.putText(frame, f"steer {steering:+.2f}",
                (cx - int(35 * s), cy + int(16 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, (255, 255, 255), lw,
                cv2.LINE_AA)


def _draw_panel(cv2, frame, p, traces, L):
    x0, x1 = L.panel_x0, L.panel_x1
    s = L.s

    def px(v):
        return max(1, int(round(v * s)))

    gear = int(round(p["gear"] * (MAX_GEAR + 1) - 1.0))
    shift = "yes" if p["shifting"] > 0.5 else "no"
    # the gear line is green in a forward gear, red in reverse, white in neutral
    gear_color = ((0, 255, 0) if gear >= 1 else
                  (0, 0, 255) if gear < 0 else (255, 255, 255))
    cl_dist = p.get("centerline_dist")
    cl_text = f"{cl_dist:5.2f} m" if cl_dist is not None else "   ---"
    onroad = p.get("on_road")
    or_text = ("yes" if onroad else "no") if onroad is not None else "---"
    or_color = ((0, 255, 0) if onroad else (0, 0, 255)) \
        if onroad is not None else (255, 255, 255)
    lines = (
        (f"SPEED   {p['speed']:6.1f} kph", (255, 255, 255)),
        (f"ANGLE   {p['angle']:6.1f} deg", (255, 255, 255)),
        (f"RPM     {p['rpm']:6.2f}", (255, 255, 255)),
        (f"GEAR    {gear:6d}", gear_color),
        (f"FFB     {p['ffb']:6.2f}", (255, 255, 255)),
        (f"REWARD  {p['reward']:6.3f}",
         (0, 255, 0) if p["reward"] >= 0 else (0, 0, 255)),
        (f"SHIFT   {shift:>6s}", (0, 200, 255) if shift == "yes" else (255, 255, 255)),
        (f"CENTER  {cl_text:>7s}", _CENTERLINE_COLOR),
        (f"ONROAD  {or_text:>4s}", or_color),
    )
    y = px(24)
    for text, color in lines:
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s,
                    color, max(1, int(round(s))), cv2.LINE_AA)
        y += px(24)

    # actions: vertical bars (throttle / brake / parking brake, 0..1) and a
    # semi-circle steering gauge, on the right of the text list (the traces
    # fill the space below)
    a = p["action"]
    if a is not None:
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        ax0 = x0 + px(132)  # right of the text list
        bar_top = px(24)
        bar_h = px(80)
        bar_w = px(24)
        bar_gap = px(8)
        bars = (
            ("thr", max(0.0, float(a[1])), (0, 165, 255)),
            ("brake", max(0.0, -float(a[1])), (0, 0, 255)),
            ("pb", max(0.0, float(a[2])), (160, 32, 240)),
        )
        bx = ax0
        for label, value, color in bars:
            by0, by1 = bar_top, bar_top + bar_h
            cv2.rectangle(frame, (bx, by0), (bx + bar_w, by1), (60, 60, 60), 1)
            fill = int(bar_h * value)
            if fill > 0:
                cv2.rectangle(frame, (bx, by1 - fill), (bx + bar_w, by1), color, -1)
            cv2.putText(frame, label, (bx, by1 + px(14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * s, (255, 255, 255), max(1, int(round(s))), cv2.LINE_AA)
            bx += bar_w + bar_gap
        # steering gauge to the right of the bars
        r = px(45)
        cx = bx + px(12) + r
        cy = bar_top + bar_h
        _draw_steering(cv2, frame, float(a[0]), cx, cy, r, s)

    # telemetry traces: fill the remaining space below the text list / actions
    # (the text list's last line, ONROAD, has its baseline at px(216), so the
    # traces start below it to avoid overlapping the text)
    trace_top = px(232)
    trace_bottom = L.height - px(16)
    n_traces = len(_TRACE_SPECS)
    trace_gap = px(8)
    trace_h = max(1,
                  (trace_bottom - trace_top - (n_traces - 1) * trace_gap) // n_traces)
    ty = trace_top
    for (label, lo, hi, color), value in zip(
            _TRACE_SPECS, (p["angle"], p["speed"], p["ffb"])):
        trace = traces[label]
        trace.append(value)
        ty0, ty1 = ty, ty + trace_h
        cv2.rectangle(frame, (x0, ty0), (x1, ty1), (60, 60, 60), 1)
        cv2.putText(frame, label, (x0 + 4, ty0 + px(14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * s, (180, 180, 180), max(1, int(round(s))), cv2.LINE_AA)
        if lo < 0.0 < hi:
            zy = int(ty1 - (0.0 - lo) / (hi - lo) * (ty1 - ty0))
            cv2.line(frame, (x0, zy), (x1, zy), (90, 90, 90), 1)
        m = len(trace)
        if m > 1:
            # fixed scale (one full window of TRACE_LEN samples spans the
            # panel): the newest sample sits at the right edge and the older
            # ones slide out the left, instead of the trace being rescaled to
            # fit the panel (which compressed it while it was shorter than a
            # full window)
            step = (x1 - x0) / (TRACE_LEN - 1)
            xs = x1 - (m - 1 - np.arange(m)) * step
            ys = ty1 - (np.asarray(trace) - lo) / (hi - lo) * (ty1 - ty0)
            pts = np.stack([xs, ys], axis=-1).astype(np.int32)
            pts = np.clip(pts, 0, (L.width, L.height)).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
        ty = ty1 + trace_gap


def _render_frame(cv2, p, traces, L):
    """Draw one frame (the (road-corrected) LiDAR BEV with its road surface
    outlines and beams on the left, telemetry on the right) and return it as
    a uint8 BGR array. Runs on the display thread, a new thread whose current
    CUDA device defaults to 0: select CUPY_DEVICE first, or the cupy ops in
    bev_to_beams (whose beam tables live on CUPY_DEVICE) raise a device
    mismatch."""
    use_device()
    frame = np.zeros((L.height, L.width, 3), np.uint8)
    if p["bev"] is not None:
        # ultrasonic beam overlay: one ray per beam to the first obstacle
        dists = bev_to_beams(p["bev"]) * LIDAR_MAX_DIST
        _draw_bev(cv2, frame, p["bev"], dists, L, 0, p.get("road_outlines"),
                  p.get("centerline"))
        _label(cv2, frame, "LiDAR BEV", 0, L)
    _draw_panel(cv2, frame, p, traces, L)
    return frame


def _payload_from(obs, action, reward, bev, outlines=None, centerline=None,
                  centerline_dist=None, on_road=None):
    """The frame payload dict for one update (the display thread renders it).
    ``outlines``: the road surface edge outlines in BEV grid (row, col) units,
    as (outline, active) tuples (active = the road surface the car is closest
    to); a bare outline array is treated as active (None when the env has no
    rig / navigraph). ``centerline``: the merged road centerline in BEV grid
    (row, col) units (None when there is no road); ``centerline_dist``: the
    distance from the car to it, in metres. ``on_road``: whether the car is on
    the road (a bool, see rig.on_road; None when unknown)."""
    if bev is None:
        bev = obs.get("bev")
    return {
        "bev": to_cpu(bev).copy() if bev is not None else None,
        "road_outlines": (
            [((np.asarray(to_cpu(o[0]), float), bool(o[1])) if isinstance(o, (tuple, list))
              else (np.asarray(to_cpu(o), float), True)) for o in outlines]
            if outlines else None),
        "centerline": (np.asarray(to_cpu(centerline), float)
                       if centerline is not None and len(centerline) else None),
        "centerline_dist": (float(centerline_dist)
                            if centerline_dist is not None
                            and np.isfinite(centerline_dist) else None),
        "on_road": (bool(on_road) if on_road is not None else None),
        "speed": _scalar(obs.get("airspeed")) * AIRSPEED_MAX_KPH,
        "angle": _scalar(obs.get("drift_angle")) * 180.0,
        "rpm": _scalar(obs.get("rpm")),
        "gear": _scalar(obs.get("gear")),
        "shifting": _scalar(obs.get("is_shifting")),
        "ffb": _scalar(obs.get("ffb")),
        "reward": float(reward),
        "action": (np.asarray(action, dtype=np.float32).copy()
                   if action is not None else None),
    }


class DebugWindow:
    """A live cv2 window fed by a background thread (see the module docstring).

    ``update`` is the agent-loop side: it snapshots the frame payload and
    enqueues it without blocking (drop-oldest), so a step pays only for the
    snapshot and the put. ``close`` stops the thread and destroys the window.
    """

    def __init__(self, title="drift debug", width=960):
        import cv2  # lazy: the window is optional, so don't import it otherwise
        self._cv2 = cv2
        self._title = title
        self._layout = make_layout(width)
        self._traces = make_traces()
        self._queue = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, obs, action=None, reward=0.0, bev=None, outlines=None,
               centerline=None, centerline_dist=None, on_road=None):
        """Queue one frame for the display thread (non-blocking, drop-oldest).

        obs: the env's Dict observation (missing channels show as 0).
        action: the 4-axis action [steering, throttle/brake, parking brake,
        gear] (optional; shown as vertical throttle/brake/park-brake bars and
        a semi-circle steering gauge).
        bev: the (road-corrected) LiDAR BEV image when the observation
        carries none (e.g. the obs went through BeamsWrapper).
        outlines: the road surface edge outlines that alter the BEV, as
        (outline, active) tuples in BEV grid (row, col) units (active = the
        road surface the car is closest to, drawn bright; the others dim).
        centerline: the merged road centerline in BEV grid (row, col) units
        (drawn on the BEV panel); centerline_dist: the distance from the car
        to it, in metres (shown in the telemetry panel).
        on_road: whether the car is on the road (a bool, see rig.on_road;
        shown in the telemetry panel; None when unknown).
        """
        if self._closed.is_set():
            return
        enqueue_latest(self._queue,
                       _payload_from(obs, action, reward, bev, outlines,
                                     centerline, centerline_dist, on_road))

    def close(self):
        """Stop the display thread and destroy the window (idempotent)."""
        if self._closed.is_set():
            return
        self._closed.set()
        enqueue_latest(self._queue, None)
        self._thread.join(timeout=2.0)

    def reopen(self):
        """Restart the display thread if it has stopped.

        The thread stops when the window is closed (e.g. the user presses Esc)
        or when the env is closed. A hosted env (host_game.py) is reused across
        training runs, so the window must come back on the next reset rather
        than staying dead for the rest of the host's life.
        """
        if self._thread.is_alive():
            return
        self._closed.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        cv2 = self._cv2
        shown = False
        mapped = False
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            try:
                frame = _render_frame(cv2, payload, self._traces, self._layout)
                cv2.imshow(self._title, frame)
                shown = True
                if cv2.waitKey(1) & 0xFF == 27:  # Esc closes the window
                    break
                if not mapped:
                    # cv2/Qt can leave the window in the X11 Withdrawn state
                    # (created but never mapped), so it never shows; make it
                    # visible (a no-op when it is already mapped)
                    _map_window(cv2, self._title)
                    mapped = True
            except Exception as err:
                print(f"[debug] display thread stopped: {err!r}")
                break
        if shown:
            cv2.destroyAllWindows()


class DebugWindowWrapper(gym.Wrapper):
    """Show the env in a live debug window (see DebugWindow).

    Wrap the env anywhere in the chain; the BEV and the road surface
    outlines are found either way (the observation, or the raw env's rig when
    the obs went through BeamsWrapper). step() and reset() hand the frame to
    the display thread through a single-slot queue, so the agent loop only
    pays for a non-blocking put.

        env = DebugWindowWrapper(env)
    """

    def __init__(self, env, **window_kwargs):
        super().__init__(env)
        self.window = DebugWindow(**window_kwargs)

    def _extras(self, obs):
        """The BEV, road surface outlines, merged road centerline, and on-road
        flag for the window: from the observation when it carries the BEV,
        else from the raw env's rig (the obs went through BeamsWrapper; the
        rig keeps the last BEV it rendered and derives the outlines/
        centerline/on-road flag from its cached navigraph). All extras are
        None when the env has no rig (e.g. a replay-free dummy env)."""
        bev = obs.get("bev")
        outlines = None
        centerline = None
        centerline_dist = None
        on_road = None
        rig = getattr(self.env.unwrapped, "rig", None)
        if rig is not None:
            if bev is None:
                bev = getattr(rig, "_last_bev", None)
            fn = getattr(rig, "road_surface_outlines", None)
            if fn is not None:
                outlines = fn() or None
            fn = getattr(rig, "road_centerline", None)
            if fn is not None:
                centerline, centerline_dist = fn()
            fn = getattr(rig, "on_road", None)
            if fn is not None:
                # reuse the centerline distance just computed above (the
                # fast-track needs it; NaN/None falls back to computing it)
                on_road = fn(centerline_dist=centerline_dist)
        return bev, outlines, centerline, centerline_dist, on_road

    def reset(self, seed=None, options=None):
        # bring the window back if it was closed (e.g. Esc) - a hosted env is
        # reused across runs, so the window must not stay dead
        self.window.reopen()
        obs, info = super().reset(seed=seed, options=options)
        bev, outlines, centerline, centerline_dist, on_road = self._extras(obs)
        self.window.update(obs, bev=bev, outlines=outlines,
                           centerline=centerline, centerline_dist=centerline_dist,
                           on_road=on_road)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        bev, outlines, centerline, centerline_dist, on_road = self._extras(obs)
        self.window.update(obs, action=action, reward=reward,
                           bev=bev, outlines=outlines,
                           centerline=centerline, centerline_dist=centerline_dist,
                           on_road=on_road)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.window.close()
        super().close()
