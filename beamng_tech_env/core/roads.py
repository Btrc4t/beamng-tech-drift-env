"""Road-geometry math for the road sensor: derive a road BEV / road beams from
the game's road data.

The primary source is the **full navigraph**, fetched once from the game
(``GetRoadGraph``) and cached in memory (:class:`Navigraph`): it is the whole
road network (every segment, with its half-width), so ``Navigraph.to_bev``
rasterizes the complete road around the car - including the road *behind* it
and intersecting roads - with no per-step game round trip. It covers tunnels.
Only segments at the car's height (within ``ROAD_BEV_Z_TOL``) are rasterized,
so overpasses and roads under bridges do not appear in the BEV.
``get_navigraph`` fetches it once per process and caches it.

Two secondary sources are kept. A per-vehicle ``beamngpy.sensors.RoadsSensor``
gives the road the vehicle is on as parametric cubic curves (centerline +
left/right edges); ``roadsensor_polylines`` / ``roadsensor_to_bev`` consume a
reading. The **DecalRoad** network (``scenario.get_road_network``) is a dict of
roads with sampled ``edges`` (left/middle/right cross-sections);
``RoadNetwork`` / ``roads_to_bev`` consume it; it does not include tunnels.
The world frame is X/Y horizontal, Z up (the State sensor's spawn up is
``[0, 0, 1]``).

All BEVs share one grid convention (matching the LiDAR BEV): row 0 = forward,
col 0 = the car's left, 1.0 = drivable. Everything here is a function of plain
data, so it can be unit-tested without launching BeamNG.tech. This is the
exploration of road geometry as an alternative source for the ultrasonic-style
beams: instead of the distance to the nearest LiDAR obstacle, the road beams
give the distance to the road edge (how far the car can go in a direction
before leaving the drivable road). A beam that starts on a non-drivable cell
(the car on a road divider) looks past it, up to ``BEAM_SKIP_MAX``, to the
edge of the road it enters.
"""
import numpy as _numpy

from beamng_tech_env.core.array import USE_CUPY, np, to_cpu, to_gpu
from beamng_tech_env.core.constants import (
    BEAM_ANGLES, BEAM_SKIP_MAX, BEV_N, LIDAR_MAX_DIST,
    ON_ROAD_TOL, ROAD_BEV_Z_TOL, ROAD_CENTERLINE_SIGMA, ROAD_SURFACE_TOL,
)
from beamng_tech_env.core.obs import bev_rotation_angle, drift_from_state

# A fused CUDA kernel for the road_surface_mask (N, M) pass (cupy only): one
# thread per (point, segment) pair computes the ribbon distance directly, so
# the pass writes only the (N, M) output (the point cloud and the per-segment
# data are read from L2) instead of the several (N, M) temporaries the
# vectorized form needs. The per-segment data is packed into one (M, 17) row:
# ab(3), l2(1), lat(3), normal(3), ha(1), hb(1), a.ab(1), a.lat(1),
# ab.lat(1), a.normal(1), ab.normal(1). ``normal`` is the road surface normal
# (the navigraph's, banked; the flat one when the navigraph lacks it) and
# ``lat`` the in-surface lateral direction; the flat ribbon is the special
# case lat = horizontal left, normal = cross(ab_dir, lat). None when cupy is
# not used (the vectorized form runs instead).
if USE_CUPY:
    _RSM_SRC = r"""
    extern "C" __global__
    void road_surface_dist(const float* __restrict__ pc,
                           const float* __restrict__ seg,
                           float* __restrict__ out, int N, int M) {
        int n = blockIdx.y * blockDim.y + threadIdx.y;
        int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N || m >= M) return;
        const float* p = pc + 3 * n;
        const float* s = seg + 17 * m;
        float abx = s[0], aby = s[1], abz = s[2], l2 = s[3];
        float lnx = s[4], lny = s[5], lnz = s[6];
        float nx = s[7], ny = s[8], nz = s[9];
        float ha = s[10], hb = s[11];
        float a_ab = s[12], a_lat = s[13], ab_lat = s[14];
        float a_normal = s[15], ab_normal = s[16];
        float t = (p[0] * abx + p[1] * aby + p[2] * abz - a_ab) / l2;
        t = fminf(fmaxf(t, 0.f), 1.f);
        float d_lat = (p[0] * lnx + p[1] * lny + p[2] * lnz)
                      - a_lat - t * ab_lat;
        float d_perp = fabsf(p[0] * nx + p[1] * ny + p[2] * nz
                             - a_normal - t * ab_normal);
        float hw = ha + t * (hb - ha);
        float lat = fabsf(d_lat) - hw;
        if (lat < 0.f) lat = 0.f;
        out[n * M + m] = sqrtf(lat * lat + d_perp * d_perp);
    }
    """
    _RSM_KERNEL = np.RawKernel(_RSM_SRC, "road_surface_dist")
else:
    _RSM_KERNEL = None


def car_grid_axes(state):
    """The world vectors for the road BEV's grid axes (fwd = grid "up",
    left = grid "right"), aligned the same way the LiDAR BEV is (travel
    direction when moving fast forward, the heading when slow or reversing -
    see obs.bev_rotation_angle)."""
    dirv = np.asarray(state["dir"], float); dirv /= np.linalg.norm(dirv)
    up = np.asarray(state["up"], float); up /= np.linalg.norm(up)
    left = np.cross(up, dirv); left /= np.linalg.norm(left)
    theta = np.deg2rad(bev_rotation_angle(drift_from_state(state)))
    fwd = np.cos(theta) * dirv + np.sin(theta) * left
    left_grid = np.cross(up, fwd); left_grid /= np.linalg.norm(left_grid)
    return fwd, left_grid


class RoadNetwork:
    """A cached, vectorized view of a ``get_road_network`` result.

    The fetch itself (``beamng.scenario.get_road_network``) takes ~1.5 s on
    west_coast_usa, so it is done once and this object is reused across
    steps; per-step ``nearby`` is a single vectorized distance op over all
    edge points, not a per-road Python loop.
    """

    def __init__(self, road_spec):
        self._roads = {}
        self._ids = []
        pts, idx = [], []
        for road_id, road in road_spec.items():
            left, middle, right = road_edge_arrays(road)
            if left.shape[0] == 0:
                continue
            i = len(self._ids)
            self._ids.append(road_id)
            self._roads[i] = {
                "id": road_id,
                "left": left,
                "middle": middle,
                "right": right,
                "half_width": 0.5 * np.linalg.norm(left - right, axis=1),
            }
            pts.append(np.vstack((left, middle, right)))
            idx.append(np.full(3 * left.shape[0], i, np.int32))
        if pts:
            self._pts = np.concatenate(pts, axis=0)
            self._road_idx = np.concatenate(idx)
        else:
            self._pts = np.empty((0, 3), float)
            self._road_idx = np.empty((0,), np.int32)

    def nearby(self, pos, limit_m=LIDAR_MAX_DIST, z_tol=ROAD_BEV_Z_TOL):
        """The roads that come within ``limit_m`` of ``pos`` (horizontal X/Y
        distance to any edge node, within ``z_tol`` of its height) as a list
        of per-road dicts in the ``nearby_road_edges`` format. Intersecting
        roads are included: a road counts if *any* of its nodes is within the
        limit. The height filter keeps other levels of the network (overpasses
        and roads under bridges) out of the BEV."""
        pos = np.asarray(pos, float)
        if self._pts.shape[0] == 0:
            return []
        d2 = ((self._pts[:, 0] - pos[0]) ** 2
              + (self._pts[:, 1] - pos[1]) ** 2)
        dz = np.abs(self._pts[:, 2] - pos[2])
        near = np.unique(self._road_idx[(d2 <= limit_m * limit_m) & (dz <= z_tol)])
        return [self._roads[int(i)] for i in near]


def nearby_road_edges(road_spec, pos, limit_m=LIDAR_MAX_DIST,
                      z_tol=ROAD_BEV_Z_TOL):
    """The roads of a ``get_road_network`` result that come within ``limit_m``
    of ``pos`` (horizontal X/Y distance to any edge node, within ``z_tol`` of
    its height).

    This is not just the road the car is on: a road counts if *any* of its
    edge nodes is within the limit, so intersecting roads (whose nodes reach
    into the limit) are included too. Returns a list of dicts, one per nearby
    road: ``{"id", "left", "middle", "right", "half_width"}`` where the
    arrays are (N, 3) and ``half_width`` is (N,) = half the left-right edge
    distance per cross-section (the local road half-width).

    Convenience wrapper over ``RoadNetwork`` (which caches the per-road
    arrays when the same network is queried repeatedly).
    """
    return RoadNetwork(road_spec).nearby(pos, limit_m, z_tol)


def road_edge_arrays(road):
    """One road's ``edges`` ({"left","middle","right"} triplet list) -> three
    (N, 3) float arrays (left, middle, right), aligned per cross-section.

    A road without edges gives three empty (0, 3) arrays.
    """
    edges = road.get("edges") or []
    if not edges:
        empty = np.empty((0, 3), float)
        return empty, empty, empty
    left = np.array([e["left"] for e in edges], float)
    middle = np.array([e["middle"] for e in edges], float)
    right = np.array([e["right"] for e in edges], float)
    return left, middle, right


def _rasterize_centerline(mids, hws, pos, n, range_m, fwd, left):
    """Rasterize centerline points (``mids``: (N, 2), ``hws``: (N,) local
    half-widths) onto an (n, n) grid centered on the car. A cell is drivable
    (1.0) if its center is within the local half-width of any centerline
    point, else 0.0.

    Grid convention (matches the LiDAR BEV): row 0 = forward (``fwd``), col 0
    = the car's left (``left``).

    The union test is one (n, n, N) pass: the squared gap from every cell to
    every sample, minus the sample's half-width squared, min over the samples
    (a cell is painted when that min is <= 0). A few large async kernels with
    no per-sample Python loop and no boolean-index syncs. The result is
    accurate to the sampling spacing (well below the cell size).
    """
    mids = np.asarray(mids, float)
    hws = np.asarray(hws, float)
    pos = np.asarray(pos, float)
    fwd = np.asarray(fwd, float)
    left = np.asarray(left, float)
    if mids.shape[0] == 0:
        return np.zeros((n, n), float)
    cell = 2.0 * range_m / n
    # sample centers in grid units: col 0 = far left (+left), row 0 = forward
    d = mids - pos[:2]
    col0 = (range_m - d @ left[:2]) / cell - 0.5
    row0 = (range_m - d @ fwd[:2]) / cell - 0.5
    # integer stamp radius per sample; drop samples whose disk is fully
    # outside the grid (they can paint no cell)
    R = np.ceil(hws / cell).astype(np.int64)
    keep = ((row0 + R >= 0) & (row0 - R < n)
            & (col0 + R >= 0) & (col0 - R < n))
    if not keep.any():
        return np.zeros((n, n), float)
    row0, col0, hws = row0[keep], col0[keep], hws[keep]
    # the exact union test as one (n, n, N) pass: a cell is drivable when its
    # center is within the half-width of any sample (the min over samples of
    # the squared gap to the sample's disk is <= 0)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dr = rows.astype(float)[:, :, None] - row0[None, None, :]   # (n, n, N)
    dc = cols.astype(float)[:, :, None] - col0[None, None, :]   # (n, n, N)
    d2 = (dr * dr + dc * dc) * (cell * cell)                    # (n, n, N)
    ok = (d2 - (hws * hws)[None, None, :]).min(axis=2) <= 0.0   # (n, n)
    return ok.astype(float)


def roads_to_bev(roads, pos, n=BEV_N, range_m=LIDAR_MAX_DIST,
                 fwd=None, left=None):
    """Rasterize the nearby roads onto an (n, n) grid centered on the car.

    The grid spans [-range_m, +range_m] around the car in the frame given by
    the world vectors ``fwd`` (grid "up", row 0) and ``left`` (grid "left",
    column 0 - matching the LiDAR BEV) - pass the car's heading axes for a
    car-aligned grid, or rotated axes to match the LiDAR BEV's travel-direction
    alignment. Defaults to the world X (fwd) and Y (left) axes.

    A cell is drivable (1.0) if it is within the local half-width of any
    nearby road's centerline, else 0.0.
    """
    pos = np.asarray(pos, float)
    fwd = np.ones(3) if fwd is None else np.asarray(fwd, float)
    left = np.array([0.0, 1.0, 0.0]) if left is None else np.asarray(left, float)
    fwd = fwd / np.linalg.norm(fwd)
    left = left / np.linalg.norm(left)
    if not roads or all(r["middle"].shape[0] == 0 for r in roads):
        return np.zeros((n, n), float)
    mids = np.concatenate([r["middle"][:, :2] for r in roads])
    hws = np.concatenate([r["half_width"] for r in roads])
    return _rasterize_centerline(mids, hws, pos, n, range_m, fwd, left)


def _resample_polyline(pts, granularity):
    """Densely resample a 2D polyline (``pts``: (N, 2)) by arc length.

    Returns ``(points, s)``: (granularity+1, 2) points evenly spaced along the
    polyline, and their (granularity+1,) arc-length positions."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = float(seglen.sum())
    if total < 1e-9:
        return np.tile(pts[0], (granularity + 1, 1)), np.zeros(granularity + 1)
    cum = np.concatenate([np.array([0.0]), np.cumsum(seglen)])
    s = np.linspace(0.0, total, granularity + 1)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    local = np.clip((s - cum[idx]) / np.maximum(seglen[idx], 1e-9), 0.0, 1.0)
    return pts[idx] + local[:, None] * seg[idx], s


def _edge_half_width(reading, granularity):
    """The mean centerline-to-edge distance of the reading's left/right edge
    cubics (the segment ahead of the vehicle), or NaN if the edges are not
    usable (not sampled yet, or the car is off the navigraph)."""
    def curve2d(uA, uB, uC, uD, vA, vB, vC, vD, x0, y0):
        p = np.linspace(0.0, 1.0, granularity + 1)
        p2 = p * p
        p3 = p2 * p
        u = uA + p * uB + p2 * uC + p3 * uD
        v = vA + p * vB + p2 * vC + p3 * vD
        return np.stack([u + x0, v + y0], axis=1)

    r = reading
    le = curve2d(r["uAofLeftRE"], r["uBofLeftRE"], r["uCofLeftRE"], r["uDofLeftRE"],
                 r["vAofLeftRE"], r["vBofLeftRE"], r["vCofLeftRE"], r["vDofLeftRE"],
                 r["xStartL"], r["yStartL"])
    re = curve2d(r["uAofRightRE"], r["uBofRightRE"], r["uCofRightRE"], r["uDofRightRE"],
                 r["vAofRightRE"], r["vBofRightRE"], r["vCofRightRE"], r["vDofRightRE"],
                 r["xStartR"], r["yStartR"])
    if not (np.isfinite(le).all() and np.isfinite(re).all()):
        return float("nan")
    return float(0.5 * np.mean(np.linalg.norm(le - re, axis=1)))


def roadsensor_polylines(reading, granularity=200):
    """Discretize a RoadsSensor reading into world polylines for the road
    ribbon.

    The reading (``beamngpy.sensors.RoadsSensor.poll``) carries the four
    navigraph nodes bracketing the vehicle (``xP0onCL..zP3onCL``; P0 is the
    node nearest the car, P3 the farthest ahead - the sensor sets trailing
    nodes to NaN when fewer than four exist) plus parametric cubics for the
    left/right road edges. The centerline is the straight polyline through the
    nodes (the true navigraph geometry - unlike the single-segment centerline
    cubic it also covers the segment the vehicle is currently on, which is
    where the car sits), and the left/right edges are that polyline offset by
    the local half-width (the mean centerline-to-edge distance of the edge
    cubics). This is the navigraph road the vehicle is on - unlike the
    DecalRoad network it covers tunnels. Returns ``(centerline, left_edge,
    right_edge)``, each a ``(granularity+1, 3)`` array; three empty ``(0, 3)``
    arrays when the reading has no valid geometry (the sensor has not sampled
    yet, or the car is off the navigraph).
    """
    r = reading
    nodes = np.array([
        [r["xP0onCL"], r["yP0onCL"], r["zP0onCL"]],
        [r["xP1onCL"], r["yP1onCL"], r["zP1onCL"]],
        [r["xP2onCL"], r["yP2onCL"], r["zP2onCL"]],
        [r["xP3onCL"], r["yP3onCL"], r["zP3onCL"]],
    ], float)
    # the sensor sets trailing nodes to NaN when fewer than four exist: keep
    # the longest leading run of finite nodes
    keep = 0
    for finite in np.isfinite(nodes).all(axis=1):
        if not finite:
            break
        keep += 1
    if keep < 2:
        empty = np.empty((0, 3), float)
        return empty, empty, empty
    nodes = nodes[:keep]

    half_width = _edge_half_width(r, granularity)
    if not np.isfinite(half_width) or half_width <= 0:
        empty = np.empty((0, 3), float)
        return empty, empty, empty

    center, s = _resample_polyline(nodes[:, :2], granularity)
    node_cum = np.concatenate([np.array([0.0]), np.cumsum(
        np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1))])
    z = np.interp(s, node_cum, nodes[:, 2])

    # local unit direction and its left normal (left of travel)
    d = np.gradient(center, axis=0)
    dlen = np.linalg.norm(d, axis=1, keepdims=True)
    dlen[dlen < 1e-9] = 1.0
    d = d / dlen
    leftn = np.stack([-d[:, 1], d[:, 0]], axis=1)

    left = center + leftn * half_width
    right = center - leftn * half_width
    return (np.column_stack([center, z]), np.column_stack([left, z]),
            np.column_stack([right, z]))


def roadsensor_to_bev(reading, pos, n=BEV_N, range_m=LIDAR_MAX_DIST,
                      fwd=None, left=None, granularity=200):
    """Rasterize a RoadsSensor reading's road ribbon onto an (n, n) grid
    centered on the car.

    Same grid convention as ``roads_to_bev`` (row 0 = forward, col 0 = the
    car's left, defaulting to world X/Y). A cell is drivable (1.0) if it is
    within the local half-width of the road centerline
    (``roadsensor_polylines``), else 0.0. The source is the navigraph (via the
    RoadsSensor), so tunnels - absent from the DecalRoad network - are
    covered.
    """
    pos = np.asarray(pos, float)
    fwd = np.ones(3) if fwd is None else np.asarray(fwd, float)
    left = np.array([0.0, 1.0, 0.0]) if left is None else np.asarray(left, float)
    fwd = fwd / np.linalg.norm(fwd)
    left = left / np.linalg.norm(left)

    centerline, left_edge, right_edge = roadsensor_polylines(reading, granularity)
    if centerline.shape[0] == 0:
        return np.zeros((n, n), float)
    mids = centerline[:, :2]
    hws = 0.5 * np.linalg.norm(left_edge[:, :2] - right_edge[:, :2], axis=1)
    return _rasterize_centerline(mids, hws, pos, n, range_m, fwd, left)


def _sample_segments(a, b, ha, hb, spacing):
    """Sample each segment (``a[i]`` -> ``b[i]``, half-width ``ha[i]`` ->
    ``hb[i]``) into a dense centerline + local half-width, ~``spacing`` arc
    length between points (the half-width varies linearly along the segment).

    Every segment is sampled at the densest rate needed (the ``max_k`` of the
    longest segment), so short segments are over-sampled (harmless - the
    rasterizer is accurate to the sampling spacing, well below the cell size)
    and no per-segment Python loop or scalar sync is needed.

    Returns ``(mids (N, 2), hws (N,))`` over all segments."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ha = np.asarray(ha, float)
    hb = np.asarray(hb, float)
    if a.shape[0] == 0:
        return np.empty((0, 2), float), np.empty(0, float)
    seglen = np.linalg.norm(b - a, axis=1)                       # (N,)
    max_k = max(2, int(np.ceil(seglen.max() / spacing)) + 1)
    t = np.linspace(0.0, 1.0, max_k)                             # (max_k,)
    mids = a[:, None, :2] + t[None, :, None] \
        * (b[:, None, :2] - a[:, None, :2])                    # (N, max_k, 2)
    hws = ha[:, None] + t[None, :] * (hb[:, None] - ha[:, None])  # (N, max_k)
    return mids.reshape(-1, 2), hws.reshape(-1)


def _point_seg_dist_2d(p, a, b):
    """The distance from point ``p`` (2-D) to each segment ``a[i] -> b[i]``
    (``a`` and ``b`` are (N, 2) arrays): the shortest point-to-segment
    distance (clamped to the segment, not the infinite line through it). Runs
    on the CPU (numpy): N is small (the nearby segments), so the GPU transfer
    is not worth it."""
    p = _numpy.asarray(to_cpu(p), float)
    a = _numpy.asarray(to_cpu(a), float)
    b = _numpy.asarray(to_cpu(b), float)
    ab = b - a                                              # (N, 2)
    l2 = (ab * ab).sum(axis=1)                              # (N,)
    t = _numpy.clip(((p[None, :] - a) * ab).sum(axis=1)
                    / _numpy.maximum(l2, 1e-12), 0.0, 1.0)  # (N,)
    closest = a + t[:, None] * ab                           # (N, 2)
    return _numpy.hypot(*(p[None, :] - closest).T)          # (N,)


def _flat_normals(a, b):
    """The flat (unbanked) surface normal per segment: the cross product of
    the segment direction and the horizontal left (unit, since both are unit
    and perpendicular). It follows the longitudinal slope but not the lateral
    bank - the fallback when the navigraph carries no node normals. Runs on
    the active backend (the inputs are converted)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ab_dir = ab / np.sqrt(l2)[:, None]
    d2 = ab[:, :2] / np.maximum(np.hypot(ab[:, 0], ab[:, 1]), 1e-9)[:, None]
    leftn3d = np.stack([-d2[:, 1], d2[:, 0],
                        np.zeros(a.shape[0], dtype=ab.dtype)], axis=1)
    return np.cross(ab_dir, leftn3d)


class Navigraph:
    """A cached, vectorized view of the full navigraph road network.

    Fetched once from the game (a single ``GetRoadGraph`` tcom round trip,
    ~0.3 s on west_coast_usa) and reused across steps; per-step ``to_bev`` is
    a vectorized proximity filter + rasterization with no game round trip.
    Unlike the per-vehicle RoadsSensor it covers the whole network - the road
    behind the car, intersecting roads, and tunnels - so the road BEV is
    complete, not just the segment ahead.
    """

    def __init__(self, data):
        self._build(data)

    @classmethod
    def from_beamng(cls, bng):
        """Fetch the full navigraph from a running game (one tcom round trip).

        The scenario must be loaded (the navigraph is built when the level
        loads). ``bng`` is a connected ``beamngpy.BeamNGpy``."""
        from beamngpy.connection.comm_base import CommBase
        data = CommBase._send_recv(bng.connection, "GetRoadGraph")["data"]
        return cls(data)

    def _build(self, data):
        graph = data["graph"]
        coords = data["coords"]
        widths = data["widths"]
        node_normals = data.get("normals") or {}

        def _pt(p, node):
            if p is not None:
                return (p["x"], p["y"], p["z"])
            c = coords[node]
            return (c[0], c[1], c[2])

        def _nrm(node):
            n = node_normals.get(node)
            if n is None:
                return None
            n = _numpy.asarray(n, float)
            nrm = float(_numpy.linalg.norm(n))
            if nrm < 1e-9:
                return None
            return (float(n[0] / nrm), float(n[1] / nrm), float(n[2] / nrm))

        # the graph is directed (each undirected edge appears twice, once per
        # direction); dedupe on eId (falling back to the node pair)
        seen = set()
        a_list, b_list, ha_list, hb_list, n_list = [], [], [], [], []
        for a, nbrs in graph.items():
            for b, e in nbrs.items():
                key = e.get("eId")
                if key is None:
                    key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                in_node = e.get("inNode", a)
                out_node = e.get("outNode", b)
                a_list.append(_pt(e.get("inPos"), in_node))
                b_list.append(_pt(e.get("outPos"), out_node))
                ha_list.append(e.get("inRadius", widths.get(in_node, 0.0)))
                hb_list.append(e.get("outRadius", widths.get(out_node, 0.0)))
                n_list.append((_nrm(in_node), _nrm(out_node)))
        self.a = np.array(a_list, float)
        self.b = np.array(b_list, float)
        self.hw_a = np.array(ha_list, float)
        self.hw_b = np.array(hb_list, float)
        self.max_hw = float(max(self.hw_a.max(), self.hw_b.max())) \
            if self.a.shape[0] else 0.0
        # the road surface normal per segment: the navigraph carries a
        # per-node surface normal (data["normals"]), which captures the
        # lateral bank the flat ribbon cannot; the segment normal is the
        # average of its endpoint node normals (re-normalised), the flat
        # normal when either node lacks one
        self.normal = (self._segment_normals(self.a, self.b, n_list)
                       if self.a.shape[0] else np.empty((0, 3), float))
        # cached packed per-segment data for the road_surface_mask kernel
        # (see _seg_data_all); None until first use
        self._seg_cache = None

    @staticmethod
    def _segment_normals(a, b, node_normals):
        """The per-segment road surface normal: the average of the two
        endpoint node normals, re-normalised; the flat (unbanked) normal
        (see _flat_normals) when either node lacks one. The per-node math is
        plain numpy (the normals are small python tuples from the game), so
        no per-segment GPU sync; the result is moved to the active backend
        once."""
        flat = to_cpu(_flat_normals(a, b))
        out = _numpy.empty_like(flat)
        for i, (na, nb) in enumerate(node_normals):
            if na is None or nb is None:
                out[i] = flat[i]
                continue
            n = _numpy.asarray(na, float) + _numpy.asarray(nb, float)
            nrm = float(_numpy.linalg.norm(n))
            out[i] = n / nrm if nrm > 1e-9 else flat[i]
        return to_gpu(out)

    @property
    def num_segments(self):
        """The number of undirected road segments in the network."""
        return int(self.a.shape[0])

    def _arrays(self):
        """The segment endpoint/half-width arrays on the active backend (the
        network may hold numpy data, e.g. when built outside :meth:`__init__`)."""
        return (np.asarray(self.a, float), np.asarray(self.b, float),
                np.asarray(self.hw_a, float), np.asarray(self.hw_b, float))

    def nearby(self, pos, limit_m=LIDAR_MAX_DIST, z_tol=ROAD_BEV_Z_TOL):
        """A boolean mask of the segments whose point nearest the car is within
        ``limit_m + max_hw`` (horizontal X/Y) of ``pos`` and within ``z_tol`` of
        its height - the same filter ``to_bev`` uses. The height is evaluated at
        the near point (not the endpoints or midpoint), so a sloped road that
        passes through the car's height is kept everywhere along it: sampling
        the z at the endpoints/midpoint opens dead zones on a long grade, where
        all the samples sit above or below the car even though the road under it
        is at the car's height. The height filter keeps other levels of the
        network (overpasses, roads under bridges) out of the BEV."""
        pos = np.asarray(pos, float)
        if self.a.shape[0] == 0:
            return np.zeros(0, bool)
        a, b, _, _ = self._arrays()
        # the point of each segment nearest the car (horizontally): the clamped
        # parameter t of the closest point, then the 3-D point at that t
        ab = b - a
        l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((pos[None, :2] - a[:, :2]) * ab[:, :2]).sum(axis=1)
                    / l2, 0.0, 1.0)
        near_pt = a + t[:, None] * ab                 # (N, 3)
        lim = limit_m + self.max_hw
        near = ((near_pt[:, 0] - pos[0]) ** 2
                + (near_pt[:, 1] - pos[1]) ** 2) <= lim * lim
        zok = np.abs(near_pt[:, 2] - pos[2]) <= z_tol
        return near & zok

    def _seg_data_all(self):
        """The packed per-segment data for the road_surface_mask kernel
        (_RSM_KERNEL), for ALL segments, computed once and cached: it depends
        only on the segment geometry (endpoints + half-widths + surface
        normals), never on the car or the point cloud, so a step only gathers
        the nearby rows. Layout per row (17 floats): ab(3), l2(1), lat(3),
        normal(3), ha(1), hb(1), a.ab(1), a.lat(1), ab.lat(1), a.normal(1),
        ab.normal(1) - the same quantities the vectorized pass derives,
        precomputed here."""
        cache = getattr(self, "_seg_cache", None)
        if cache is not None:
            return cache
        a, b, ha, hb = (x.astype(np.float32) for x in self._arrays())
        normal = np.asarray(self.normal, np.float32)
        ab = b - a                                                    # (S, 3)
        l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)                 # (S,)
        ab_dir = ab / np.sqrt(l2)[:, None]                            # (S, 3)
        # the in-surface lateral direction: perpendicular to the segment
        # direction and to the surface normal (the flat ribbon's horizontal
        # left when the normal is the flat one)
        lat = np.cross(ab_dir, normal)
        lat = lat / np.maximum(np.linalg.norm(lat, axis=1)[:, None], 1e-9)
        seg = np.empty((a.shape[0], 17), np.float32)
        seg[:, 0:3] = ab
        seg[:, 3] = l2
        seg[:, 4:7] = lat
        seg[:, 7:10] = normal
        seg[:, 10] = ha
        seg[:, 11] = hb
        seg[:, 12] = (a * ab).sum(axis=1)
        seg[:, 13] = (a * lat).sum(axis=1)
        seg[:, 14] = (ab * lat).sum(axis=1)
        seg[:, 15] = (a * normal).sum(axis=1)
        seg[:, 16] = (ab * normal).sum(axis=1)
        self._seg_cache = seg
        return seg

    def to_bev(self, pos, n=BEV_N, range_m=LIDAR_MAX_DIST, fwd=None, left=None,
               z_tol=ROAD_BEV_Z_TOL, merge_margin=0.0):
        """Rasterize the navigraph segments near the car onto an (n, n) grid.

        Same grid convention as ``roads_to_bev`` (row 0 = forward, col 0 = the
        car's left). The nearby segments (see :meth:`nearby` - within the
        horizontal limit *and* ``z_tol`` of the car's height, so overpasses
        and roads under bridges are excluded) are sampled into a dense
        centerline with a linearly-varying half-width and rasterized (a cell
        is drivable within the local half-width).

        ``merge_margin`` (metres) grows each surface's half-width before
        rasterizing, so surfaces less than ``2 * merge_margin`` apart merge
        into one (the lane dividers fill in); 0.0 (the default) rasterizes the
        unmerged surfaces."""
        pos = np.asarray(pos, float)
        fwd = np.ones(3) if fwd is None else np.asarray(fwd, float)
        left = np.array([0.0, 1.0, 0.0]) if left is None else np.asarray(left, float)
        fwd = fwd / np.linalg.norm(fwd)
        left = left / np.linalg.norm(left)
        sel = self.nearby(pos, range_m, z_tol)
        if not sel.any():
            return np.zeros((n, n), float)
        spacing = range_m / n   # ~one cell between samples
        a, b, ha, hb = self._arrays()
        mids, hws = _sample_segments(a[sel], b[sel], ha[sel], hb[sel], spacing)
        if merge_margin:
            hws = hws + merge_margin
        return _rasterize_centerline(mids, hws, pos, n, range_m, fwd, left)

    def to_bev_height(self, pos, n=BEV_N, range_m=LIDAR_MAX_DIST, fwd=None,
                      left=None, span_m=20.0, z_tol=None):
        """The road network as a 0-255 BEV channel that also encodes the
        height of each road segment, on the same grid as :meth:`to_bev`
        (row 0 = forward, col 0 = the car's left).

        A cell is filled when it is drivable (within the local half-width of
        a nearby segment, the same rasterization as :meth:`to_bev`). Its
        value is the height of the nearest painting sample, encoded relative
        to the car: the road under the car (at the car's height) is the
        middle of the range, and ``-span_m`` / ``+span_m`` metres (default
        20) are 0 / 255, i.e. ``(dz / span_m + 1) * 127.5`` clamped to
        [0, 255] (``dz`` the sample's height minus the car's). Non-drivable
        cells are 0.0. The result is a float (n, n) grid in [0, 255] (cast
        to uint8 for an observation channel).

        ``z_tol`` (metres) is the vertical tolerance of the segment filter;
        None (the default) uses ``span_m``: a segment further away than the
        encoding span would clamp to 0/255 anyway, so it is dropped.

        TODO: using this channel in the observation is planned (it is not
        part of the observation space yet).
        """
        pos = np.asarray(pos, float)
        fwd = np.ones(3) if fwd is None else np.asarray(fwd, float)
        left = np.array([0.0, 1.0, 0.0]) if left is None else np.asarray(left, float)
        fwd = fwd / np.linalg.norm(fwd)
        left = left / np.linalg.norm(left)
        if z_tol is None:
            z_tol = span_m
        sel = self.nearby(pos, range_m, z_tol)
        if not sel.any():
            return np.zeros((n, n), float)
        spacing = range_m / n   # ~one cell between samples
        a, b, ha, hb = self._arrays()
        aa, bb = a[sel], b[sel]
        # the same dense centerline as to_bev, plus each sample's height
        # (linear along the segment)
        seglen = np.linalg.norm(bb - aa, axis=1)
        max_k = max(2, int(np.ceil(seglen.max() / spacing)) + 1)
        t = np.linspace(0.0, 1.0, max_k)
        mids = (aa[:, None, :2] + t[None, :, None]
                * (bb[:, None, :2] - aa[:, None, :2])).reshape(-1, 2)
        hws = (ha[sel][:, None] + t[None, :]
               * (hb[sel][:, None] - ha[sel][:, None])).reshape(-1)
        zs = (aa[:, None, 2:3] + t[None, :, None]
              * (bb[:, None, 2:3] - aa[:, None, 2:3])).reshape(-1)
        cell = 2.0 * range_m / n
        # sample centers in grid units (col 0 = far left, row 0 = forward)
        d = mids - pos[:2]
        col0 = (range_m - d @ left[:2]) / cell - 0.5
        row0 = (range_m - d @ fwd[:2]) / cell - 0.5
        R = np.ceil(hws / cell).astype(np.int64)
        keep = ((row0 + R >= 0) & (row0 - R < n)
                & (col0 + R >= 0) & (col0 - R < n))
        if not keep.any():
            return np.zeros((n, n), float)
        row0, col0, hws, zs = row0[keep], col0[keep], hws[keep], zs[keep]
        # one (n, n, N) pass: the gap of every cell to every sample's disk;
        # a cell is drivable when the min gap is <= 0, and its height is the
        # nearest painting sample's (the argmin of the gap)
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dr = rows.astype(float)[:, :, None] - row0[None, None, :]
        dc = cols.astype(float)[:, :, None] - col0[None, None, :]
        gap = ((dr * dr + dc * dc) * (cell * cell)
               - (hws * hws)[None, None, :])
        ok = gap.min(axis=2) <= 0.0
        dz = zs[gap.argmin(axis=2)] - pos[2]
        val = np.clip((dz / span_m + 1.0) * 127.5, 0.0, 255.0)
        return np.where(ok, val, 0.0)

    def road_surface_mask(self, pc, pos, tol=ROAD_SURFACE_TOL):
        """A boolean mask of the points of ``pc`` (N, 3, world coords) that are
        within ``tol`` of the road surface (the nearby segments' ribbons) - the
        LiDAR returns to drop from the point cloud so the BEV reads the road as
        empty (road level) and only the obstacles (returns above the road) as
        returns.

        The distance from a point to a segment's ribbon: the point is projected
        onto the segment's centerline (the clamped parameter t), and the
        distance is the gap along the road surface normal when the point is
        within the local half-width (directly above/below the road), else the
        3-D distance to the nearest edge (the point is off the road). The gap
        is measured along the surface normal (the surface "up"), not world Z:
        the car and its roof LiDAR are inclined relative to the road (the
        suspension, acceleration, braking and drifting tilt the car, and the
        road itself is sloped or banked), so the car's Z axis is aligned to the
        road's for the measurement. The ribbon is banked: the surface normal is
        the navigraph's per-segment normal (the average of its endpoint node
        normals - it captures the lateral bank on curves), and the lateral
        direction is the in-surface direction perpendicular to the segment
        (the flat unbanked ribbon when the navigraph lacks node normals). The
        mask is True for the points whose minimum distance over the nearby
        segments (see :meth:`nearby`) is <= ``tol``. All-False when the point
        cloud is empty or no segment is nearby."""
        pc = np.ascontiguousarray(np.asarray(pc, dtype=np.float32))
        pos = np.asarray(pos, float)
        n = pc.shape[0]
        if n == 0 or self.a.shape[0] == 0:
            return np.zeros(n, bool)
        sel = self.nearby(pos, LIDAR_MAX_DIST, ROAD_BEV_Z_TOL)
        if not sel.any():
            return np.zeros(n, bool)
        if _RSM_KERNEL is not None:
            # fused kernel: the per-segment data is cached (it depends only on
            # the segment geometry), so a step is one boolean gather of the
            # nearby rows + one kernel (one thread per point-segment pair
            # computes the ribbon distance, writing only the (N, M) output) +
            # one min reduction
            seg = self._seg_data_all()[sel]                    # (M, 17)
            m = seg.shape[0]
            out = np.empty((n, m), np.float32)
            _RSM_KERNEL(((m + 7) // 8, (n + 7) // 8), (8, 8),
                        (pc, seg, out, n, m))
            return out.min(axis=1) <= tol
        # vectorized (N, M) pass (the numpy backend, or cupy without the kernel)
        a, b, ha, hb = (x[sel].astype(np.float32) for x in self._arrays())
        normal = np.asarray(self.normal[sel], np.float32)       # (M, 3)
        ab = b - a                                                    # (M, 3)
        l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)                 # (M,)
        # the surface "up": the road surface normal (the navigraph's, banked;
        # the flat cross-product normal when the navigraph lacks it). The car
        # and its roof LiDAR are inclined relative to the road (suspension,
        # acceleration, braking, drifting, and the road's own slope/bank), so
        # the gap is measured along this normal, not world Z: the car's Z
        # axis is aligned to the road's for the measurement.
        ab_dir = ab / np.sqrt(l2)[:, None]                            # (M, 3)
        # the in-surface lateral direction: perpendicular to the segment
        # direction and to the surface normal
        lat = np.cross(ab_dir, normal)
        lat = lat / np.maximum(np.linalg.norm(lat, axis=1)[:, None], 1e-9)
        # per-segment dot products: every per-(point, segment) quantity is the
        # point's dot product with a per-segment vector minus one of these, so
        # the (N, M) pass is (N, K) @ (K, M) matmuls that touch only (N, M)
        # arrays (the explicit-broadcast form materialises (N, M, 3) temps)
        a_ab = (a * ab).sum(axis=1)                                   # (M,)
        a_lat = (a * lat).sum(axis=1)                                 # (M,)
        ab_lat = (ab * lat).sum(axis=1)                               # (M,)
        a_normal = (a * normal).sum(axis=1)                           # (M,)
        ab_normal = (ab * normal).sum(axis=1)                         # (M,)
        t = np.clip((pc @ ab.T - a_ab[None, :]) / l2[None, :],
                    0.0, 1.0)                                         # (N, M)
        d_lat = (pc @ lat.T - a_lat[None, :]
                 - t * ab_lat[None, :])                               # (N, M)
        d_perp = np.abs(pc @ normal.T - a_normal[None, :]
                        - t * ab_normal[None, :])                     # (N, M)
        hw = ha[None, :] + t * (hb[None, :] - ha[None, :])            # (N, M)
        # distance to the ribbon: the lateral excess is 0 inside the
        # half-width, so hypot(max(|d_lat|-hw, 0), d_perp) covers both the
        # on-ribbon case (d_perp) and the off-ribbon edge case in one form -
        # no where(), which would evaluate both branches for every pair
        dist = np.hypot(np.maximum(np.abs(d_lat) - hw, 0.0), d_perp)  # (N, M)
        return dist.min(axis=1) <= tol

    def is_on_road(self, pos, tol=ON_ROAD_TOL):
        """Whether ``pos`` is on the road (a plain bool): True when the car is
        on a street - the horizontal distance from ``pos`` to the nearest
        nearby segment's centerline minus that segment's local half-width is
        <= ``tol`` (i.e. within ``tol`` of the road surface).

        The tolerance is needed because the navigraph represents the road as
        individual lanes whose road surface is continuous: a car on the
        surface at a lane divider (between two lane ribbons) or straddling the
        road edge (the state pos is the car's center, ~half a car width from
        the wheels) is still on-road. The height filter (see :meth:`nearby`)
        keeps other levels of the network out: a car under an overpass is on
        its own road, not on the overpass.

        False when the network is empty or no segment is nearby (off-road).
        Pure in-memory computation (the navigraph is cached), one small sync.
        TODO: expose as an observation channel ("on_road") - for now it is a
        debug window information point only (see rig.on_road).
        """
        pos = np.asarray(pos, float)
        if self.a.shape[0] == 0:
            return False
        sel = self.nearby(pos, LIDAR_MAX_DIST, ROAD_BEV_Z_TOL)
        if not sel.any():
            return False
        a, b, ha, hb = self._arrays()
        a, b, ha, hb = a[sel], b[sel], ha[sel], hb[sel]
        ab = b - a
        l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((pos[None, :2] - a[:, :2]) * ab[:, :2]).sum(axis=1)
                    / l2, 0.0, 1.0)                                   # (M,)
        closest = a[:, :2] + t[:, None] * ab[:, :2]                   # (M, 2)
        d = np.hypot(pos[0] - closest[:, 0], pos[1] - closest[:, 1])  # (M,)
        hw = ha + t * (hb - ha)                                       # (M,)
        return bool((d - hw).min() <= tol)

    def surface_outlines(self, pos, fwd, left, n=BEV_N, range_m=LIDAR_MAX_DIST,
                         z_tol=ROAD_BEV_Z_TOL, node_tol=0.5):
        """The edge outlines of the road surface that ``to_bev`` rasterizes
        for ``pos`` (the same nearby segments), in grid (row, col) units:
        row 0 = forward, col 0 = left, continuous cell coordinates with the
        car at (n/2, n/2) - the same axes and grid as ``to_bev``.

        Returns one ``(outline, active)`` tuple per segment edge (the left
        edge, then the right edge, per segment). ``outline`` is a (K, 2) float
        array clipped to the [0, n] grid (a road leaving the BEV range is
        drawn along the panel edge); ``active`` is True when the edge belongs
        to the road surface the car is closest to (see
        :meth:`_closest_surface_mask`) and False for the other nearby
        surfaces. Empty when no segment is nearby (nothing alters the BEV)."""
        # The per-segment loop syncs on every iteration, so it runs on the CPU
        # (numpy): the nearby data is small, so the GPU transfer is not worth it.
        pos_c = _numpy.asarray(to_cpu(pos), float)
        fwd_c = (_numpy.asarray(to_cpu(fwd), float) if fwd is not None
                 else _numpy.ones(3))
        left_c = (_numpy.asarray(to_cpu(left), float) if left is not None
                  else _numpy.array([0.0, 1.0, 0.0]))
        fwd_c = fwd_c / _numpy.linalg.norm(fwd_c)
        left_c = left_c / _numpy.linalg.norm(left_c)
        sel = self.nearby(pos, range_m, z_tol)
        if not sel.any():
            return []
        idx = np.flatnonzero(sel)
        active = self._closest_surface_mask(pos, idx, node_tol)
        idx_c = to_cpu(idx)
        active_c = to_cpu(active)
        cell = 2.0 * range_m / n
        spacing = range_m / n   # ~one cell between samples, as in to_bev
        a_all, b_all, ha_all, hb_all = (to_cpu(x) for x in self._arrays())
        outlines = []
        for j, i in enumerate(idx_c):
            i = int(i)
            a, b = a_all[i], b_all[i]
            d = b - a
            seglen = float(_numpy.linalg.norm(d))
            if seglen < 1e-9:
                continue
            k = max(2, int(_numpy.ceil(seglen / spacing)) + 1)
            t = _numpy.linspace(0.0, 1.0, k)
            mids = a[:2] + t[:, None] * (b[:2] - a[:2])
            hw = ha_all[i] + t * (hb_all[i] - ha_all[i])
            d2 = d[:2] / seglen
            leftn = _numpy.array([-float(d2[1]), float(d2[0])])
            for edge in (mids + leftn * hw[:, None], mids - leftn * hw[:, None]):
                dp = edge - pos_c[:2]
                row = (range_m - dp @ fwd_c[:2]) / cell
                col = (range_m - dp @ left_c[:2]) / cell
                outlines.append((_numpy.stack([row, col], axis=1).clip(0.0, float(n)),
                                 bool(active_c[j])))
        return outlines

    def _closest_surface_mask(self, pos, idx, node_tol=0.5):
        """A boolean mask (one per segment in ``idx``) that is True for the
        segments of the road surface the car (``pos``) is closest to.

        The nearby segments are grouped into road surfaces: two segments are
        on the same surface when they share an endpoint (within ``node_tol`` -
        a navigraph node), and membership is transitive (a road is a chain of
        segments through its nodes). The closest surface is the one with the
        segment nearest the car (the min point-to-segment distance). On a
        single road every segment is on the closest surface (all True); with
        several nearby roads (a parallel road, an intersection) only the road
        the car is on is True."""
        n = len(idx)
        if n == 0:
            return np.zeros(0, bool)
        # the union-find is a Python loop with per-step scalar syncs, which is
        # far faster on the CPU: move the (small) segment endpoints to numpy
        a, b, _, _ = self._arrays()
        a2 = to_cpu(a[idx])[:, :2]
        b2 = to_cpu(b[idx])[:, :2]
        # the segment endpoints, a/b interleaved (ends2[2s] = seg s start,
        # ends2[2s + 1] = seg s end)
        ends2 = _numpy.stack((a2, b2), axis=1).reshape(-1, 2)  # (2n, 2)
        parent = _numpy.arange(2 * n)

        def find(x):
            x = int(x)
            while int(parent[x]) != x:
                parent[x] = int(parent[int(parent[x])])
                x = int(parent[x])
            return x

        # merge endpoints that are the same navigraph node (within node_tol)
        d2 = ((ends2[:, None, :] - ends2[None, :, :]) ** 2).sum(axis=2)
        nz = _numpy.nonzero(_numpy.triu(d2 <= node_tol * node_tol, 1))
        for i, j in zip(nz[0].tolist(), nz[1].tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        # a segment joins its two endpoints into one surface
        for s in range(n):
            ra, rb = find(2 * s), find(2 * s + 1)
            if ra != rb:
                parent[ra] = rb
        comp = _numpy.array([find(2 * s) for s in range(n)])
        dists = _point_seg_dist_2d(to_cpu(pos)[:2], a2, b2)
        mask = comp == comp[int(_numpy.argmin(dists))]
        return to_gpu(mask)

    def save(self, path):
        """Persist the network to a compressed .npz (so it is fetched from the
        game only once, even across processes)."""
        np.savez_compressed(path, a=self.a, b=self.b,
                            hw_a=self.hw_a, hw_b=self.hw_b,
                            normal=self.normal)

    @classmethod
    def load(cls, path):
        """Load a network saved by :meth:`save` (no game needed). A file saved
        before the surface normals were added (no "normal" key) loads with the
        flat (unbanked) normals as the fallback."""
        with np.load(path) as z:
            nav = cls.__new__(cls)
            nav.a = z["a"]
            nav.b = z["b"]
            nav.hw_a = z["hw_a"]
            nav.hw_b = z["hw_b"]
            nav.max_hw = float(max(nav.hw_a.max(), nav.hw_b.max())) \
                if nav.a.shape[0] else 0.0
            try:
                nav.normal = z["normal"]
            except KeyError:  # a file saved before the normals were added
                nav.normal = _flat_normals(nav.a, nav.b)
            nav._seg_cache = None
            return nav


# the full navigraph, fetched once per (key) and kept in memory: the fetch is
# one tcom round trip, but per-step use is a vectorized in-memory op, so the
# network is fetched once and reused for every step and every wrapper
_NAVIGRAPH_CACHE = {}


def get_navigraph(bng, key="default"):
    """The full navigraph (see :class:`Navigraph`), fetched once per ``key``
    and cached in memory. ``key`` distinguishes different networks (e.g. the
    level name); the same key reuses the cached fetch. A disk cache can be
    layered on with ``Navigraph.save`` / ``Navigraph.load``."""
    nav = _NAVIGRAPH_CACHE.get(key)
    if nav is None:
        nav = Navigraph.from_beamng(bng)
        _NAVIGRAPH_CACHE[key] = nav
    return nav


def _beam_table(n, range_m, angles):
    """Sample points (rows, cols, radii, valid) for every beam angle.

    Half-cell steps from the grid center out to the edge (mirrors
    wrappers._beam_table, kept local so roads.py has no wrappers dependency
    beyond the shared beam angles).
    """
    cell = 2.0 * range_m / n
    half = 0.5 * cell
    nrad = int(np.ceil(range_m / half))
    radii = half * np.arange(nrad)
    th = np.deg2rad(np.asarray(angles))
    x = -np.outer(np.sin(th), radii)              # (nbeams, nrad) lateral
    y = np.outer(np.cos(th), radii)               # (nbeams, nrad) forward
    cols = np.clip(((x + range_m) / cell).astype(int), 0, n - 1)
    rows = np.clip(((range_m - y) / cell).astype(int), 0, n - 1)
    valid = (np.abs(x) < range_m) & (np.abs(y) < range_m)
    return rows, cols, radii, valid


def road_bev_to_beams(grid, angles=BEAM_ANGLES, n=BEV_N, range_m=LIDAR_MAX_DIST,
                      skip_m=BEAM_SKIP_MAX):
    """Distance to the road edge per beam, normalized to [0, 1] (1 = still on
    the road at full range).

    ``grid`` is a roads_to_bev output (1.0 = drivable, 0.0 = not), (n, n), with
    a channel dim (1, n, n) / (..., 1, n, n), or batched (..., n, n). Each beam
    raycasts from the grid center outward and stops at the first non-drivable
    cell (the road edge); a beam that stays on the road to the edge reads full
    range. A beam that starts on a non-drivable cell (the car on a road
    divider) looks past it - up to ``skip_m`` - and reads the edge of the road
    it enters; a beam with no road within ``skip_m`` reads zero. This is the
    road-network analogue of wrappers.bev_to_beams (distance to the nearest
    obstacle).
    """
    grid = np.asarray(grid, float)
    if grid.shape[-2:] != (n, n):
        raise ValueError(f"expected an {n}x{n} grid, got {grid.shape}")
    if grid.ndim >= 3 and grid.shape[-3] == 1:
        grid = grid[..., 0, :, :]
    rows, cols, radii, valid = _beam_table(n, range_m, angles)
    vals = grid[..., rows, cols]                       # (..., nbeams, nrad)
    on_road = (vals >= 0.5) & valid
    off = ~on_road
    first_on = on_road.argmax(axis=-1)                 # 0 if the ray never enters a road
    any_on = on_road.any(axis=-1)
    # the ray may start on an obstacle the car is on (a road divider): skip
    # the leading non-drivable cells (up to skip_m) and read the edge of the
    # road the ray enters
    idx = np.arange(radii.size)
    after = off & (idx > first_on[..., None])
    any_after = after.any(axis=-1)
    first_after = after.argmax(axis=-1)
    dist = np.where(any_on & (radii[first_on] <= skip_m),
                    np.where(any_after, radii[first_after], range_m),
                    0.0)
    return to_cpu((dist / range_m).astype(np.float32))


def road_centerline(mask, n=BEV_N, range_m=LIDAR_MAX_DIST,
                    sigma_m=ROAD_CENTERLINE_SIGMA):
    """The centerline of a merged road mask, via a Gaussian blur.

    The mask is blurred and the centerline is tracked as a continuous path from
    the car (the grid center): at each row it is the center of mass of the road
    blob closest to the previous point. On a single road this is the road's
    middle (center of mass, not an edge - the blurred top is flat on wide
    roads, where an argmax would pick an edge). Where the road diverges into
    several branches (a Y fork) a row has several blobs; the path follows the
    branch that continues it - the surface the car is closest to - instead of
    the gap between the branches. The merge margin (ROAD_MERGE_MARGIN) keeps
    close branches connected through the split, so the path stays continuous
    there and only branches once the surfaces actually separate.

    ``mask`` is a (n, n) road mask (1.0 = drivable, the BEV grid convention:
    row 0 = forward, col 0 = left). Returns ``(centerline, dist)``:
    - ``centerline``: a (K, 2) float array of grid (row, col) points, one per
      row where the road is present (for drawing on the BEV panel); empty when
      the mask has no road.
    - ``dist``: the distance from the grid center (the car) to the nearest
      centerline point, in metres (NaN when the mask has no road).
    """
    from scipy.ndimage import gaussian_filter
    # The per-row tracking loop syncs on every iteration, so it runs on the CPU
    # (numpy): the (n, n) mask is small, so the GPU transfer is not worth it.
    mask_c = _numpy.asarray(to_cpu(mask), float)
    if not (mask_c > 0.5).any():
        return np.empty((0, 2), float), float("nan")
    cell = 2.0 * range_m / n
    blurred = gaussian_filter(mask_c, sigma=sigma_m / cell)
    on = mask_c > 0.5

    def row_centers(r):
        """The column center of mass (weighted by the blurred mask) of each
        road blob - a contiguous run of drivable cells - in row r."""
        row_on = on[r]
        if not row_on.any():
            return []
        centers = []
        lo = 0
        while lo < n:
            if not row_on[lo]:
                lo += 1
                continue
            hi = lo
            while hi + 1 < n and row_on[hi + 1]:
                hi += 1
            w = blurred[r, lo:hi + 1]
            cols = _numpy.arange(lo, hi + 1)
            centers.append(float((w * cols).sum() / _numpy.maximum(w.sum(), 1e-9)))
            lo = hi + 1
        return centers

    # track the path from the car (row n//2, the grid center) outward, both
    # directions; at each row pick the blob center closest to the current point
    # (continuity), so a fork follows the branch the car is on, not the gap
    c0 = n // 2
    path = _numpy.full(n, _numpy.nan)
    start = row_centers(c0)
    path[c0] = min(start, key=lambda c: abs(c - c0)) if start else float(c0)
    for r in range(c0 - 1, -1, -1):
        centers = row_centers(r)
        if centers:
            path[r] = min(centers, key=lambda c: abs(c - path[r + 1]))
    for r in range(c0 + 1, n):
        centers = row_centers(r)
        if centers:
            path[r] = min(centers, key=lambda c: abs(c - path[r - 1]))

    rows = _numpy.flatnonzero(on.any(axis=1) & ~_numpy.isnan(path))
    if rows.size == 0:
        return np.empty((0, 2), float), float("nan")
    centerline = to_gpu(_numpy.stack([rows.astype(float), path[rows]], axis=1))
    c = 0.5 * n
    dist = float(_numpy.hypot(rows - c, path[rows] - c).min() * cell)
    return centerline, dist


def road_centerline_segments(nav, pos, fwd, left, n=BEV_N,
                             range_m=LIDAR_MAX_DIST, z_tol=ROAD_BEV_Z_TOL,
                             node_tol=0.5, smooth=5):
    """The centerline from the segment centerlines of the road surface the car
    is closest to.

    Unlike :func:`road_centerline` (the blurred merged-mask center of mass),
    this is derived from the *actual navigraph geometry*: the closest surface's
    segments (:meth:`Navigraph._closest_surface_mask`) are sampled into a dense
    centerline, and one point per row is tracked from the car outward (the
    point closest to the previous one, so a fork follows the branch the car is
    on). It is far smoother than the mask center of mass - which wobbles when
    the road is two close parallel surfaces that intermittently merge/split in
    the rasterized mask - and it is pinned to the surface the car is on rather
    than the middle of a merged multi-surface road.

    ``smooth`` (a moving-average window in rows; 0 disables) attenuates the
    residual sampling jitter (the per-row point is the nearest dense sample,
    up to half a cell off the true centerline).

    ``nav`` is a :class:`Navigraph`, ``pos`` the car position, ``fwd``/``left``
    the grid axes (see :func:`car_grid_axes`). Returns ``(centerline, dist)``
    like :func:`road_centerline` (empty / NaN when no segment is nearby)."""
    # The per-row tracking is sequential (each row depends on the previous
    # reference), so it runs on the CPU (numpy): a per-row GPU pass would sync
    # on every iteration. The (possibly cupy) inputs and the nearby segment
    # rows are moved to the CPU once at the boundary.
    pos_c = _numpy.asarray(to_cpu(pos), float)
    fwd_c = _numpy.asarray(to_cpu(fwd), float)
    left_c = _numpy.asarray(to_cpu(left), float)
    sel = nav.nearby(pos, range_m, z_tol)
    if not sel.any():
        return _numpy.empty((0, 2), float), float("nan")
    idx = np.flatnonzero(sel)
    active = nav._closest_surface_mask(pos, idx, node_tol)
    aidx = to_cpu(idx[active])
    # dense centerline samples of the closest surface's segments (the
    # sampling runs on the active backend; only the small result is pulled
    # to the CPU for the sequential tracking below)
    spacing = range_m / n
    a, b, ha, hb = nav._arrays()
    mids, _ = _sample_segments(a[aidx], b[aidx], ha[aidx], hb[aidx], spacing)
    mids = to_cpu(mids)
    cell = 2.0 * range_m / n
    d = mids - pos_c[:2]
    row0 = (range_m - d @ fwd_c[:2]) / cell
    col0 = (range_m - d @ left_c[:2]) / cell
    ok = (row0 >= 0) & (row0 < n) & (col0 >= 0) & (col0 < n)
    row0, col0 = row0[ok], col0[ok]
    if row0.size == 0:
        return _numpy.empty((0, 2), float), float("nan")

    # track one point per row from the car (row n//2) outward; a running
    # reference survives rows the centerline does not cross
    c0 = n // 2
    path = _numpy.full(n, _numpy.nan)
    ref = float(col0[int(_numpy.argmin(_numpy.hypot(row0 - c0, col0 - c0)))])
    path[c0] = ref
    for r in range(c0 - 1, -1, -1):
        m = _numpy.abs(row0 - r) <= 0.5
        if m.any():
            ref = float(min(col0[m], key=lambda c: abs(c - ref)))
            path[r] = ref
    ref = float(path[c0])
    for r in range(c0 + 1, n):
        m = _numpy.abs(row0 - r) <= 0.5
        if m.any():
            ref = float(min(col0[m], key=lambda c: abs(c - ref)))
            path[r] = ref
    rows = _numpy.flatnonzero(~_numpy.isnan(path))
    if rows.size == 0:
        return _numpy.empty((0, 2), float), float("nan")

    if smooth and smooth > 1:
        # fill the rows the centerline does not cross (so the average does not
        # propagate NaN) and attenuate the residual sampling jitter
        valid = ~_numpy.isnan(path)
        gaps = _numpy.flatnonzero(~valid)
        if gaps.size:
            v = _numpy.flatnonzero(valid)
            path = path.copy()
            path[gaps] = _numpy.interp(gaps, v, path[v])
        k = _numpy.ones(smooth) / smooth
        path = _numpy.convolve(_numpy.pad(path, smooth // 2, mode="edge"), k,
                               mode="valid")[: n]

    centerline = _numpy.stack([rows.astype(float), path[rows]], axis=1)
    c = 0.5 * n
    dist = float(_numpy.hypot(rows - c, path[rows] - c).min() * cell)
    return centerline, dist