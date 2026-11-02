"""Pure-math tests for the road-network helpers (no game needed).

``RoadNetwork`` / ``nearby_road_edges`` filter a ``get_road_network`` result
to the roads within a horizontal limit of the car (including intersecting
roads, not just the one the car is on); ``roads_to_bev`` rasterizes them onto
a grid; ``road_bev_to_beams`` raycasts the distance to the road edge per
beam; ``car_grid_axes`` gives the world vectors for the grid axes.
"""
import numpy as np

from beamng_tech_env.core.array import to_cpu
from beamng_tech_env.core.constants import BEV_N, LIDAR_MAX_DIST
from beamng_tech_env.core.roads import (
    Navigraph,
    RoadNetwork,
    _rasterize_centerline,
    car_grid_axes,
    nearby_road_edges,
    road_bev_to_beams,
    road_centerline,
    road_centerline_segments,
    road_edge_arrays,
    roads_to_bev,
    roadsensor_polylines,
    roadsensor_to_bev,
)


def straight_road(axis, width=6.0, lo=-50.0, hi=50.0, step=2.0, z=0.0):
    """A straight road along world ``axis`` ('x' or 'y'), centered on the
    other axis, at height ``z``, sampled every ``step`` metres."""
    edges = []
    v = lo
    while v <= hi:
        if axis == "x":
            left, mid, right = [v, width / 2, z], [v, 0, z], [v, -width / 2, z]
        else:
            left, mid, right = [width / 2, v, z], [0, v, z], [-width / 2, v, z]
        edges.append({"left": left, "middle": mid, "right": right})
        v += step
    return edges


def crossing_spec():
    """Two roads crossing at the origin (a plus intersection) plus a road far
    away that must be filtered out."""
    return {
        "road_x": {"drivability": "1", "edges": straight_road("x")},
        "road_y": {"drivability": "1", "edges": straight_road("y")},
        "far_road": {"drivability": "1",
                     "edges": straight_road("x", lo=100.0, hi=120.0)},
    }


def test_road_edge_arrays():
    left, mid, right = road_edge_arrays({"edges": straight_road("x", lo=0, hi=4)})
    assert left.shape == (3, 3) and mid.shape == (3, 3) and right.shape == (3, 3)
    # a road without edges gives empty arrays
    e, m, r = road_edge_arrays({"edges": []})
    assert e.shape == (0, 3) and m.shape == (0, 3) and r.shape == (0, 3)


def test_nearby_includes_intersecting_excludes_far():
    # the car sits on the intersection: both crossing roads are nearby (their
    # nodes reach into the limit), the far road is not
    near = nearby_road_edges(crossing_spec(), [0, 0, 0], limit_m=40.0)
    assert sorted(r["id"] for r in near) == ["road_x", "road_y"]
    # each nearby road carries its edge arrays and a per-section half-width
    for r in near:
        assert r["left"].shape == r["middle"].shape == r["right"].shape
        assert r["half_width"].shape == (r["middle"].shape[0],)
        assert np.allclose(r["half_width"], 3.0)  # width 6 -> half 3


def test_nearby_limit_is_horizontal():
    # a road 30 m away horizontally is within 40 m; the same road 50 m away
    # is not (the vertical offset is ignored)
    spec = {"r": {"edges": straight_road("x", lo=30.0, hi=40.0, width=4.0)}}
    assert len(nearby_road_edges(spec, [0, 0, 0], limit_m=40.0)) == 1
    spec = {"r": {"edges": straight_road("x", lo=50.0, hi=60.0, width=4.0)}}
    assert len(nearby_road_edges(spec, [0, 0, 0], limit_m=40.0)) == 0


def test_nearby_empty_when_no_road_close():
    spec = {"far": {"edges": straight_road("x", lo=100.0, hi=120.0)}}
    assert nearby_road_edges(spec, [0, 0, 0], limit_m=40.0) == []


def test_nearby_excludes_other_heights():
    # a road at the car's height is nearby; the same road 8 m up (an
    # overpass) is not
    spec = {"r": {"edges": straight_road("x", lo=10.0, hi=30.0, width=4.0)}}
    assert len(nearby_road_edges(spec, [0, 0, 0], limit_m=40.0)) == 1
    spec = {"r": {"edges": straight_road("x", lo=10.0, hi=30.0, width=4.0, z=8.0)}}
    assert len(nearby_road_edges(spec, [0, 0, 0], limit_m=40.0)) == 0


def _bev_cell(bev, wx, wy, range_m=LIDAR_MAX_DIST, n=BEV_N):
    """The BEV cell at world (wx, wy) for the default fwd=[1,0,0], left=[0,1,0]
    grid (wx = forward/row, wy = lateral; col 0 = the car's left = +wy,
    matching the LiDAR BEV)."""
    cell = 2.0 * range_m / n
    row = int((range_m - wx) / cell)
    col = int((range_m - wy) / cell)
    return bev[row, col]


def test_roads_to_bev_drivable_and_not():
    near = nearby_road_edges(crossing_spec(), [0, 0, 0], limit_m=40.0)
    bev = roads_to_bev(near, [0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                       fwd=[1, 0, 0], left=[0, 1, 0])
    assert bev.shape == (BEV_N, BEV_N)
    # on the road (the intersection center and along each arm) -> drivable
    assert _bev_cell(bev, 0, 0) == 1.0
    assert _bev_cell(bev, 10, 0) == 1.0    # along road_x
    assert _bev_cell(bev, 0, 10) == 1.0    # along road_y
    # off both roads -> not drivable
    assert _bev_cell(bev, 10, 10) == 0.0
    assert _bev_cell(bev, 30, 30) == 0.0


def test_roads_to_bev_no_roads():
    bev = roads_to_bev([], [0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST)
    assert bev.shape == (BEV_N, BEV_N)
    assert bev.sum() == 0.0


def test_rasterize_centerline_matches_bruteforce():
    # the stamp rasterizer paints exactly the cells whose center is within
    # the half-width of any centerline sample (the union of half-width disks)
    rng = np.random.default_rng(0)
    mids = rng.uniform(-45.0, 45.0, (200, 2))
    hws = rng.uniform(1.0, 6.0, 200)
    pos = np.array([0.0, 0.0, 0.0])
    fwd = np.array([1.0, 0.0, 0.0])
    left = np.array([0.0, 1.0, 0.0])
    bev = _rasterize_centerline(mids, hws, pos, BEV_N, LIDAR_MAX_DIST,
                                fwd, left)
    # brute-force reference: cell centers in world XY (grid convention:
    # row 0 = forward, col 0 = left)
    cell = 2.0 * LIDAR_MAX_DIST / BEV_N
    dx = -LIDAR_MAX_DIST + (np.arange(BEV_N) + 0.5) * cell
    dy = LIDAR_MAX_DIST - (np.arange(BEV_N) + 0.5) * cell
    DX, DY = np.meshgrid(dx, dy)
    wx = pos[0] + DY * fwd[0] - DX * left[0]
    wy = pos[1] + DY * fwd[1] - DX * left[1]
    ref = np.zeros((BEV_N, BEV_N), bool)
    for i in range(mids.shape[0]):
        d2 = (wx - mids[i, 0]) ** 2 + (wy - mids[i, 1]) ** 2
        ref |= d2 <= hws[i] ** 2
    assert np.array_equal(to_cpu(bev) > 0.5, ref)


def test_rasterize_centerline_empty():
    bev = _rasterize_centerline(np.empty((0, 2)), np.empty(0),
                                [0, 0, 0], BEV_N, LIDAR_MAX_DIST,
                                [1, 0, 0], [0, 1, 0])
    assert bev.shape == (BEV_N, BEV_N)
    assert bev.sum() == 0.0


def test_road_bev_to_beams():
    near = nearby_road_edges(crossing_spec(), [0, 0, 0], limit_m=40.0)
    bev = roads_to_bev(near, [0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                       fwd=[1, 0, 0], left=[0, 1, 0])
    beams = road_bev_to_beams(bev)
    assert beams.shape == (20,)
    assert np.all((beams >= 0.0) & (beams <= 1.0))
    # the lateral beams (near +/-90 deg) run along road_y to full range; the
    # near-forward beams (no beam at exactly 0) leave the 6 m road_x early
    assert beams[3] == 1.0 and beams[16] == 1.0
    assert beams[10] < 1.0


def test_road_bev_to_beams_all_road():
    # a fully drivable grid: every beam reads full range
    bev = np.ones((BEV_N, BEV_N), float)
    assert np.all(road_bev_to_beams(bev) == 1.0)


def test_road_bev_to_beams_no_road():
    # a fully non-drivable grid: every beam reads zero (off the road at once)
    bev = np.zeros((BEV_N, BEV_N), float)
    assert np.all(road_bev_to_beams(bev) == 0.0)


def test_road_bev_to_beams_car_on_divider():
    # the car sits on a non-drivable patch (a road divider): the beams look
    # past it to the edge of the road they enter, not zero
    bev = np.ones((BEV_N, BEV_N), float)
    bev[30:34, 30:34] = 0.0
    assert np.all(road_bev_to_beams(bev) == 1.0)
    # with no skip, the beams stop on the divider itself
    assert np.all(road_bev_to_beams(bev, skip_m=0.0) == 0.0)


def test_road_bev_to_beams_no_road_within_skip():
    # the car is off-road and the nearest road is beyond the skip distance:
    # the beams read zero (the car has left the road)
    bev = np.zeros((BEV_N, BEV_N), float)
    bev[0:8, :] = 1.0   # a road 30+ m ahead
    assert np.all(road_bev_to_beams(bev) == 0.0)


# -- RoadsSensor (navigraph) functions --------------------------------------

def straight_reading(x0=-50.0, x1=50.0, width=8.0):
    """A synthetic RoadsSensor reading: a straight road along world X from
    x0 to x1, centered on Y=0, ``width`` metres wide (two navigraph nodes,
    the edge cubics carry the half-width)."""
    w = width / 2.0
    du = x1 - x0
    nan = float("nan")
    return {
        # the 4 navigraph nodes (only the first two are valid here; the
        # sensor sets trailing nodes to NaN when fewer than four exist)
        "xP0onCL": x0, "yP0onCL": 0.0, "zP0onCL": 0.0,
        "xP1onCL": x1, "yP1onCL": 0.0, "zP1onCL": 0.0,
        "xP2onCL": nan, "yP2onCL": nan, "zP2onCL": nan,
        "xP3onCL": nan, "yP3onCL": nan, "zP3onCL": nan,
        "uAofCL": 0.0, "uBofCL": du, "uCofCL": 0.0, "uDofCL": 0.0,
        "vAofCL": 0.0, "vBofCL": 0.0, "vCofCL": 0.0, "vDofCL": 0.0,
        "xStartCL": x0, "yStartCL": 0.0, "zStartCL": 0.0,
        "uAofLeftRE": 0.0, "uBofLeftRE": du, "uCofLeftRE": 0.0, "uDofLeftRE": 0.0,
        "vAofLeftRE": 0.0, "vBofLeftRE": 0.0, "vCofLeftRE": 0.0, "vDofLeftRE": 0.0,
        "xStartL": x0, "yStartL": w, "zStartL": 0.0,
        "uAofRightRE": 0.0, "uBofRightRE": du, "uCofRightRE": 0.0, "uDofRightRE": 0.0,
        "vAofRightRE": 0.0, "vBofRightRE": 0.0, "vCofRightRE": 0.0, "vDofRightRE": 0.0,
        "xStartR": x0, "yStartR": -w, "zStartR": 0.0,
    }


def test_roadsensor_polylines_straight():
    cl, le, re = roadsensor_polylines(straight_reading(), granularity=10)
    assert cl.shape == (11, 3) and le.shape == (11, 3) and re.shape == (11, 3)
    assert np.allclose(cl[0, :2], [-50.0, 0.0])
    assert np.allclose(cl[-1, :2], [50.0, 0.0])
    assert np.allclose(le[:, 1], 4.0)   # left edge at Y = +4
    assert np.allclose(re[:, 1], -4.0)  # right edge at Y = -4


def test_roadsensor_polylines_invalid():
    r = straight_reading()
    r["xP0onCL"] = float("nan")
    cl, le, re = roadsensor_polylines(r)
    assert cl.shape == (0, 3) and le.shape == (0, 3) and re.shape == (0, 3)


def test_roadsensor_to_bev_drivable_and_not():
    bev = roadsensor_to_bev(straight_reading(), [0, 0, 0],
                            n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0])
    assert bev.shape == (BEV_N, BEV_N)
    # on the road (center, along it, and within the 4 m half-width) -> drivable
    assert _bev_cell(bev, 0, 0) == 1.0
    assert _bev_cell(bev, 10, 0) == 1.0
    assert _bev_cell(bev, 0, 2) == 1.0
    # off the road (beyond the half-width, behind it, or far away) -> not
    assert _bev_cell(bev, 0, 10) == 0.0
    assert _bev_cell(bev, 0, -10) == 0.0
    assert _bev_cell(bev, 30, 30) == 0.0


def test_roadsensor_to_bev_no_road():
    r = straight_reading()
    r["xP0onCL"] = float("nan")
    bev = roadsensor_to_bev(r, [0, 0, 0])
    assert bev.sum() == 0.0


def test_roadsensor_to_bev_car_on_segment_behind_cubic():
    # regression: the centerline cubic only spans P1->P2 (ahead of the car);
    # the car sits on the P0->P1 segment. The node-polyline centerline must
    # cover it, so the car's own cell is drivable.
    r = {
        "xP0onCL": -5.0, "yP0onCL": 0.0, "zP0onCL": 0.0,
        "xP1onCL": 15.0, "yP1onCL": 0.0, "zP1onCL": 0.0,
        "xP2onCL": 35.0, "yP2onCL": 0.0, "zP2onCL": 0.0,
        "xP3onCL": 55.0, "yP3onCL": 0.0, "zP3onCL": 0.0,
        # edge cubics over P1->P2 giving a 4 m half-width
        "uAofLeftRE": 0.0, "uBofLeftRE": 20.0, "uCofLeftRE": 0.0, "uDofLeftRE": 0.0,
        "vAofLeftRE": 0.0, "vBofLeftRE": 0.0, "vCofLeftRE": 0.0, "vDofLeftRE": 0.0,
        "xStartL": 15.0, "yStartL": 4.0, "zStartL": 0.0,
        "uAofRightRE": 0.0, "uBofRightRE": 20.0, "uCofRightRE": 0.0, "uDofRightRE": 0.0,
        "vAofRightRE": 0.0, "vBofRightRE": 0.0, "vCofRightRE": 0.0, "vDofRightRE": 0.0,
        "xStartR": 15.0, "yStartR": -4.0, "zStartR": 0.0,
    }
    bev = roadsensor_to_bev(r, [0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0])
    # the car (at the origin, on the P0->P1 segment) is on the road
    assert _bev_cell(bev, 0, 0) == 1.0
    # and the road continues ahead
    assert _bev_cell(bev, 10, 0) == 1.0
    assert _bev_cell(bev, 30, 0) == 1.0
    # off the road laterally
    assert _bev_cell(bev, 0, 10) == 0.0


# -- RoadNetwork (the cached, vectorized view) -----------------------------

def test_road_network_matches_function():
    # RoadNetwork.nearby is the vectorized nearby_road_edges: same roads,
    # same arrays
    spec = crossing_spec()
    fn = nearby_road_edges(spec, [0, 0, 0], limit_m=40.0)
    near = RoadNetwork(spec).nearby([0, 0, 0], limit_m=40.0)
    assert sorted(r["id"] for r in near) == sorted(r["id"] for r in fn)
    for a, b in zip(sorted(near, key=lambda r: r["id"]),
                    sorted(fn, key=lambda r: r["id"])):
        assert np.array_equal(a["middle"], b["middle"])
        assert np.allclose(a["half_width"], b["half_width"])


def test_road_network_nearby_limit():
    spec = {"r": {"edges": straight_road("x", lo=30.0, hi=40.0, width=4.0)}}
    assert len(RoadNetwork(spec).nearby([0, 0, 0], limit_m=40.0)) == 1
    spec = {"r": {"edges": straight_road("x", lo=50.0, hi=60.0, width=4.0)}}
    assert len(RoadNetwork(spec).nearby([0, 0, 0], limit_m=40.0)) == 0


def test_road_network_empty():
    rn = RoadNetwork({})
    assert rn.nearby([0, 0, 0]) == []
    assert roads_to_bev(rn.nearby([0, 0, 0]), [0, 0, 0]).sum() == 0.0


# -- car_grid_axes (the road BEV's world axes) ------------------------------

def _state(vel=(10.0, 0.0, 0.0), dirv=(1.0, 0.0, 0.0)):
    return {"pos": [0.0, 0.0, 0.0], "vel": list(vel), "dir": list(dirv),
            "up": [0.0, 0.0, 1.0]}


def test_car_grid_axes_slow_keeps_heading():
    # below TRAVEL_ALIGN_MIN_KPH: the grid stays aligned with the car's heading
    fwd, left = car_grid_axes(_state(vel=(1.0, 0.0, 0.0)))  # 3.6 kph
    assert np.allclose(fwd, [1.0, 0.0, 0.0])
    assert np.allclose(left, [0.0, 1.0, 0.0])


def test_car_grid_axes_reversing_keeps_heading():
    # reversing at speed: the rotation is 0, so the grid stays aligned with
    # the car's heading (not the backward direction)
    fwd, left = car_grid_axes(_state(vel=(-10.0, 0.0, 0.0)))
    assert np.allclose(fwd, [1.0, 0.0, 0.0])
    assert np.allclose(left, [0.0, 1.0, 0.0])


def test_car_grid_axes_drift_aligns_with_travel():
    # drifting at speed: the grid "up" is the horizontal travel direction
    th = np.deg2rad(-30.0)
    vel = tuple(10.0 * c for c in (np.cos(th), np.sin(th), 0.0))
    fwd, left = car_grid_axes(_state(vel=vel))
    assert np.allclose(fwd, np.array(vel) / np.linalg.norm(vel))
    assert np.allclose(left, np.cross([0.0, 0.0, 1.0], fwd))


# -- Navigraph (the full cached network) ------------------------------------

def _navigraph_data(segments, normals=None):
    """Synthetic full-navigraph data from a list of (a, b, hw) segments (a, b
    are (x, y, z) endpoints, hw the half-width). Each segment is one
    undirected edge (two directed entries sharing an eId, as the real
    navigraph does). ``normals``: an optional dict node -> (x, y, z) surface
    normal (the real navigraph carries one per node; absent here unless
    given, so the flat normal is the fallback)."""
    graph, coords, widths = {}, {}, {}
    for i, (a, b, hw) in enumerate(segments):
        na, nb = f"n{i}a", f"n{i}b"
        coords[na] = list(a); coords[nb] = list(b)
        widths[na] = hw; widths[nb] = hw
        ea = {"eId": i, "inNode": na, "outNode": nb,
              "inPos": {"x": a[0], "y": a[1], "z": a[2]},
              "outPos": {"x": b[0], "y": b[1], "z": b[2]},
              "inRadius": hw, "outRadius": hw}
        eb = dict(ea, inNode=nb, outNode=na,
                  inPos={"x": b[0], "y": b[1], "z": b[2]},
                  outPos={"x": a[0], "y": a[1], "z": a[2]})
        graph[na] = {nb: ea}
        graph[nb] = {na: eb}
    data = {"graph": graph, "coords": coords, "widths": widths}
    if normals is not None:
        data["normals"] = normals
    return data


def test_navigraph_build_dedupes():
    # each segment appears in both directions (same eId): it dedupes to one
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    assert nav.num_segments == 1
    assert np.allclose(nav.a[0][:2], (-50.0, 0.0))
    assert np.allclose(nav.b[0][:2], (50.0, 0.0))
    assert nav.max_hw == 4.0


def test_navigraph_to_bev():
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    bev = nav.to_bev([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                     fwd=[1, 0, 0], left=[0, 1, 0])
    assert bev.shape == (BEV_N, BEV_N)
    # the car (at the origin, on the road) is drivable; off the road is not
    assert _bev_cell(bev, 0, 0) == 1.0
    assert _bev_cell(bev, 10, 0) == 1.0
    assert _bev_cell(bev, 0, 2) == 1.0
    assert _bev_cell(bev, 0, 10) == 0.0
    assert _bev_cell(bev, 0, -10) == 0.0


def test_navigraph_to_bev_height():
    # a flat road under the car: the road cells encode the middle of the
    # range (the road is at the car's height), off-road cells are 0
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    bev = nav.to_bev_height([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0])
    assert bev.shape == (BEV_N, BEV_N)
    assert np.all((bev >= 0.0) & (bev <= 255.0))
    assert _bev_cell(bev, 0, 0) == 127.5
    assert _bev_cell(bev, 10, 0) == 127.5
    assert _bev_cell(bev, 0, 10) == 0.0
    # a road 10 m above the car encodes 127.5 + 10/20 * 127.5 = 191.25 (the
    # default z_tol is the 20 m span, so other levels within it appear)
    nav = Navigraph(_navigraph_data([((-50, 0, 10), (50, 0, 10), 4.0)]))
    bev = nav.to_bev_height([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0])
    assert _bev_cell(bev, 0, 0) == 191.25
    # a road 10 m below encodes 127.5 - 10/20 * 127.5 = 63.75
    nav = Navigraph(_navigraph_data([((-50, 0, -10), (50, 0, -10), 4.0)]))
    bev = nav.to_bev_height([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0])
    assert _bev_cell(bev, 0, 0) == 63.75
    # beyond the span the encoding clamps (forced in with a wider z_tol):
    # 25 m below is the 0 floor, 25 m above the 255 ceiling
    nav = Navigraph(_navigraph_data([((-50, 0, -25), (50, 0, -25), 4.0)]))
    bev = nav.to_bev_height([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0], z_tol=30.0)
    assert _bev_cell(bev, 0, 0) == 0.0
    nav = Navigraph(_navigraph_data([((-50, 0, 25), (50, 0, 25), 4.0)]))
    bev = nav.to_bev_height([0, 0, 0], n=BEV_N, range_m=LIDAR_MAX_DIST,
                            fwd=[1, 0, 0], left=[0, 1, 0], z_tol=30.0)
    assert _bev_cell(bev, 0, 0) == 255.0
    # no road near: all zeros
    nav = Navigraph(_navigraph_data([((100, 0, 0), (120, 0, 0), 4.0)]))
    assert nav.to_bev_height([0, 0, 0]).sum() == 0.0


def test_navigraph_nearby_sloped_road_no_dead_zone():
    # A long slanted segment (z rises 0 -> 20 over y -50 -> 50). The car is on
    # it at y=20 (z=14). The segment's endpoint/midpoint z samples (0, 20, 10)
    # are all more than z_tol=3 from the car's z=14, so the old endpoint/midpoint
    # z gate dropped the segment (a dead zone on a long grade); the near-point z
    # gate keeps it (the road under the car is at the car's height).
    nav = Navigraph(_navigraph_data([((0, -50, 0), (0, 50, 20), 4.0)]))
    car = [0.0, 20.0, 14.0]   # on the segment (z at y=20 is 14)
    assert nav.nearby(car).any()
    bev = nav.to_bev(car, n=BEV_N, range_m=LIDAR_MAX_DIST,
                     fwd=[1, 0, 0], left=[0, 1, 0])
    assert _bev_cell(bev, 0, 0) == 1.0


def test_navigraph_surface_outlines():
    # a straight road along X, half-width 4, car at the origin; fwd=[1,0,0]
    # (row = forward), left=[0,1,0] (col = left = +Y). The road spans Y in
    # [-4, 4] -> col in [(40-4)/cell, (40+4)/cell]
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    cell = 2.0 * LIDAR_MAX_DIST / BEV_N
    outlines = nav.surface_outlines([0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    # one (outline, active) tuple per edge (left edge, then right edge); the
    # single road is the surface the car is on -> both edges are active
    assert len(outlines) == 2
    (left_edge, left_active), (right_edge, right_active) = outlines
    assert left_active and right_active
    assert np.allclose(left_edge[:, 1], (LIDAR_MAX_DIST - 4.0) / cell)
    assert np.allclose(right_edge[:, 1], (LIDAR_MAX_DIST + 4.0) / cell)
    # the road extends past the BEV range both ways: rows clip to the grid
    assert left_edge[:, 0].min() == 0.0 and left_edge[:, 0].max() == BEV_N
    # the outline matches the rasterization: just inside is drivable, just
    # outside is not
    bev = nav.to_bev([0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    assert bev[BEV_N // 2, int((LIDAR_MAX_DIST - 3.0) / cell)] == 1.0
    assert bev[BEV_N // 2, int((LIDAR_MAX_DIST - 5.0) / cell)] == 0.0
    # a road outside the range alters no BEV cell -> no outlines
    nav = Navigraph(_navigraph_data([((100, 0, 0), (120, 0, 0), 4.0)]))
    assert nav.surface_outlines([0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0]) == []


def test_navigraph_surface_outlines_closest_surface():
    # two parallel roads along X (the car's road at Y=0, another at Y=20),
    # the car on the first: only the car's road is the closest surface, so
    # its edges are active and the parallel road's are not
    nav = Navigraph(_navigraph_data([
        ((-50, 0, 0), (50, 0, 0), 4.0),    # the car's road
        ((-50, 20, 0), (50, 20, 0), 4.0),  # a parallel road, 20 m away
    ]))
    outlines = nav.surface_outlines([0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    assert len(outlines) == 4
    actives = [active for _, active in outlines]
    assert actives[:2] == [True, True]     # the car's road
    assert actives[2:] == [False, False]   # the parallel road


def test_navigraph_surface_outlines_fork_one_surface():
    # a Y fork: the stem (0,0)->(20,0) and two branches from the shared fork
    # node (20,0) to (40,10) and (40,-10). All three segments meet at the
    # fork node, so they are one road surface: the car on the stem sees every
    # edge as active.
    coords = {"s0": [0.0, 0.0, 0.0], "fork": [20.0, 0.0, 0.0],
              "b1": [40.0, 10.0, 0.0], "b2": [40.0, -10.0, 0.0]}
    widths = {name: 3.0 for name in coords}
    graph = {}
    for i, (na, nb) in enumerate((("s0", "fork"), ("fork", "b1"), ("fork", "b2"))):
        ea = {"eId": i, "inNode": na, "outNode": nb,
              "inPos": dict(zip("xyz", coords[na])),
              "outPos": dict(zip("xyz", coords[nb])),
              "inRadius": 3.0, "outRadius": 3.0}
        eb = dict(ea, inNode=nb, outNode=na,
                  inPos=dict(zip("xyz", coords[nb])),
                  outPos=dict(zip("xyz", coords[na])))
        graph.setdefault(na, {})[nb] = ea
        graph.setdefault(nb, {})[na] = eb
    nav = Navigraph({"graph": graph, "coords": coords, "widths": widths})
    outlines = nav.surface_outlines([10.0, 0.0, 0.0], fwd=[1, 0, 0],
                                    left=[0, 1, 0])
    assert len(outlines) == 6
    assert all(active for _, active in outlines)


def test_navigraph_nearby():
    # a road 30 m away is within 40 m; one 60 m away is not
    nav = Navigraph(_navigraph_data([((30, 0, 0), (40, 0, 0), 4.0)]))
    assert nav.nearby([0, 0, 0], 40.0).sum() == 1
    nav = Navigraph(_navigraph_data([((60, 0, 0), (70, 0, 0), 4.0)]))
    assert nav.nearby([0, 0, 0], 40.0).sum() == 0


def test_navigraph_nearby_height_filter():
    # only roads at the car's height are nearby: an overpass (same X/Y,
    # higher Z) and a road under a bridge are excluded, while a road a couple
    # of metres up (a hill) is still included. The car's state pos sits ~0.5 m
    # above the road surface.
    nav = Navigraph(_navigraph_data([
        ((-50, 0, 0), (50, 0, 0), 4.0),       # the road the car is on
        ((-50, 20, 8), (50, 20, 8), 4.0),     # an overpass 8 m up
        ((-50, -20, -8), (50, -20, -8), 4.0), # a road 8 m below
        ((-50, 30, 3), (50, 30, 3), 4.0),     # 2.5 m up (a hill)
    ]))
    sel = nav.nearby([0, 0, 0.5], 40.0)
    assert sel.sum() == 2
    # the other levels are absent from the BEV too
    bev = nav.to_bev([0, 0, 0.5], n=BEV_N, range_m=LIDAR_MAX_DIST,
                     fwd=[1, 0, 0], left=[0, 1, 0])
    assert _bev_cell(bev, 0, 0) == 1.0      # on the road
    assert _bev_cell(bev, 0, 30) == 1.0     # the hill road
    assert _bev_cell(bev, 0, 20) == 0.0     # the overpass's projection
    assert _bev_cell(bev, 0, -20) == 0.0    # the road below's projection


def test_navigraph_to_bev_no_road_near():
    nav = Navigraph(_navigraph_data([((100, 0, 0), (120, 0, 0), 4.0)]))
    assert nav.to_bev([0, 0, 0]).sum() == 0.0


def test_navigraph_road_surface_mask():
    # a straight road along X at Y=0, Z=0, half-width 4; the car is on it at
    # the origin. Returns within ROAD_SURFACE_TOL (0.5 m) of the road surface
    # are masked (dropped from the point cloud), the rest are kept: on the
    # surface the test is the vertical gap, off the edge it is the 3-D
    # distance to the nearest edge.
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    pos = [0.0, 0.0, 0.0]
    pc = np.array([
        [0.0, 0.0, 0.0],     # on the surface -> dropped
        [0.0, 0.0, 0.4],     # 0.4 m above -> dropped
        [0.0, 0.0, 0.6],     # 0.6 m above -> kept
        [0.0, 0.0, -0.4],    # 0.4 m below -> dropped
        [0.0, 3.0, 0.0],     # 3 m lateral, on the road -> dropped
        [0.0, 4.4, 0.0],     # 0.4 m off the edge -> dropped
        [0.0, 5.0, 0.0],     # 1 m off the edge -> kept
    ])
    assert nav.road_surface_mask(pc, pos).tolist() == \
        [True, True, False, True, True, True, False]


def test_navigraph_road_surface_mask_empty_and_far():
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    # an empty point cloud gives an empty mask
    assert nav.road_surface_mask(np.empty((0, 3)), [0, 0, 0]).shape == (0,)
    # a road 100 m away is not nearby: no point is dropped
    nav = Navigraph(_navigraph_data([((100, 0, 0), (120, 0, 0), 4.0)]))
    assert not nav.road_surface_mask(np.array([[0.0, 0.0, 0.0]]),
                                     [0, 0, 0]).any()


def test_navigraph_road_surface_mask_inclined():
    # A road sloping up at dz/dx = 0.5 (a 26.6 deg grade), half-width 4. The
    # car is on it at x=-25 (the road surface is at z=-12.5 there). The car and
    # its roof LiDAR are inclined relative to this road, so the gap to the
    # surface is measured along the road surface normal, not world Z: a return
    # 0.53 m straight up (world Z) from the surface is only 0.53*cos(26.6 deg)
    # = 0.474 m from it along the normal - within the 0.5 m tolerance, so it is
    # dropped, where a world-Z gap test would keep it (0.53 > 0.5). A return
    # 0.6 m straight up is 0.536 m along the normal - beyond the tolerance, so
    # it is kept.
    nav = Navigraph(_navigraph_data([((-50, 0, -25), (50, 0, 25), 4.0)]))
    pos = [-25.0, 0.0, -12.5]
    ab = np.array([100.0, 0.0, 50.0])
    normal = np.cross(ab / np.linalg.norm(ab), [0.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    s = np.array([-25.0, 0.0, -12.5])   # on the surface under the car
    pc = np.array([
        s,                    # on the surface -> dropped
        s + 0.4 * normal,     # 0.4 m along the normal -> dropped
        s + 0.6 * normal,     # 0.6 m along the normal -> kept
        s + [0.0, 0.0, 0.53], # 0.53 m world-Z up (0.474 m along the normal) -> dropped
        s + [0.0, 0.0, 0.6],  # 0.6 m world-Z up (0.536 m along the normal) -> kept
        s + [0.0, 3.8, 0.0],  # 3.8 m lateral, on the road -> dropped
        s + [0.0, 5.0, 0.0],  # 1 m off the edge -> kept
    ])
    assert nav.road_surface_mask(pc, pos).tolist() == \
        [True, True, False, True, False, True, False]


def test_navigraph_road_surface_mask_banked():
    # A straight road along X, half-width 4, banked 10 deg to the left (the
    # surface normal tilts toward +Y). The navigraph carries the per-node
    # surface normal, so the ribbon follows the bank: a return on the banked
    # surface 3 m to the left (0.52 m below the flat ribbon) is dropped,
    # where the flat ribbon (the fallback without node normals) would keep
    # it (0.52 > 0.5).
    bank = np.deg2rad(10.0)
    n = np.array([0.0, np.sin(bank), np.cos(bank)])
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)],
                                    normals={"n0a": list(n), "n0b": list(n)}))
    assert np.allclose(to_cpu(nav.normal), n, atol=1e-6)
    s = np.array([0.0, 0.0, 0.0])
    pc = np.array([
        s,                           # on the surface -> dropped
        [0.0, 3.0, -3.0 * np.tan(bank)],  # on the banked surface -> dropped
        s + 0.4 * n,                 # 0.4 m along the banked normal -> dropped
        s + 0.6 * n,                 # 0.6 m along the banked normal -> kept
        [0.0, 3.0, 0.0],             # 0.52 m above the banked surface -> kept
    ])
    assert nav.road_surface_mask(pc, [0, 0, 0]).tolist() == \
        [True, True, True, False, False]
    # without node normals the flat ribbon is the fallback: the banked-
    # surface return is 0.52 m from it -> kept
    nav_flat = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    assert nav_flat.road_surface_mask(pc, [0, 0, 0]).tolist() == \
        [True, False, True, False, True]


def test_navigraph_road_surface_mask_banked_vectorized(monkeypatch):
    # the vectorized pass (no cupy kernel) must agree with the fused kernel
    # on the banked ribbon
    from beamng_tech_env.core import roads
    bank = np.deg2rad(10.0)
    n = np.array([0.0, np.sin(bank), np.cos(bank)])
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)],
                                    normals={"n0a": list(n), "n0b": list(n)}))
    s = np.array([0.0, 0.0, 0.0])
    pc = np.array([
        s,
        [0.0, 3.0, -3.0 * np.tan(bank)],
        s + 0.4 * n,
        s + 0.6 * n,
        [0.0, 3.0, 0.0],
    ])
    expected = nav.road_surface_mask(pc, [0, 0, 0]).tolist()
    monkeypatch.setattr(roads, "_RSM_KERNEL", None)
    assert nav.road_surface_mask(pc, [0, 0, 0]).tolist() == expected


def test_navigraph_is_on_road():
    # a straight road along X, half-width 4: the car is on-road when its
    # horizontal distance to the centerline is within hw + tol
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    assert nav.is_on_road([0.0, 0.0, 0.5]) is True    # on the road
    assert nav.is_on_road([0.0, 3.5, 0.5]) is True    # 0.5 m inside the edge
    assert nav.is_on_road([0.0, 4.5, 0.5]) is True    # 0.5 m past the edge (within the 1 m tolerance)
    assert nav.is_on_road([0.0, 6.0, 0.5]) is False   # 2 m past the edge
    assert nav.is_on_road([100.0, 0.0, 0.5]) is False # no road nearby
    # the tolerance is configurable
    assert nav.is_on_road([0.0, 5.5, 0.5], tol=2.0) is True
    assert nav.is_on_road([0.0, 5.5, 0.5], tol=0.5) is False
    # an empty network is always off-road
    assert Navigraph(_navigraph_data([])).is_on_road([0.0, 0.0, 0.5]) is False


def test_rig_on_road_fast_track():
    """VehicleRig.on_road fast-track: within ON_ROAD_FAST_TRACK_DIST of the
    centerline the determination defaults to True (reusing a passed-in
    distance when given); beyond that, or with fast_track=False, the strict
    edge check decides."""
    from types import SimpleNamespace

    from beamng_tech_env.core.constants import ON_ROAD_FAST_TRACK_DIST
    from beamng_tech_env.core.rig import VehicleRig

    # straight road along X, half-width 4: the strict check is on-road up to
    # hw + ON_ROAD_TOL = 5 m from the centerline
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    rig = VehicleRig()
    rig.beamng = object()
    rig._navigraph = nav
    near = ON_ROAD_FAST_TRACK_DIST - 0.5   # inside the fast-track, off-road strictly
    far = ON_ROAD_FAST_TRACK_DIST + 0.5    # outside the fast-track
    computed = []
    rig.road_centerline_dist = lambda: computed.append(1) or 3.0

    def at(y):
        rig.vehicle = SimpleNamespace(state={"pos": [0.0, y, 0.5]})

    # fast-track: near the centerline -> True, even where the strict check
    # says off-road, and the passed distance is reused (no recompute)
    at(near)
    assert rig.on_road(centerline_dist=near) is True
    assert computed == []
    assert rig.on_road(fast_track=False, centerline_dist=near) is False

    # beyond the fast-track: the strict check decides (off-road here)
    at(near)
    assert rig.on_road(centerline_dist=far) is False
    at(far)
    assert rig.on_road(centerline_dist=far) is False

    # NaN (no road nearby / no navigraph) does not fire the fast-track
    at(0.0)
    assert rig.on_road(centerline_dist=float("nan")) is True   # strict: on the road
    at(near)
    assert rig.on_road(centerline_dist=float("nan")) is False  # strict: off-road

    # no distance passed: the fast-track computes it
    at(near)
    assert rig.on_road() is True
    assert computed == [1]


def test_navigraph_is_on_road_lanes_connected():
    # two adjacent lanes (each 1.75 m wide) with a 0.3 m gap between their
    # edges (the road surface is continuous, the navigraph lanes are not): a
    # car in the gap is still on-road (within the tolerance)
    nav = Navigraph(_navigraph_data([
        ((-50, -1.9, 0), (50, -1.9, 0), 1.75),  # left lane: y in [-3.65, -0.15]
        ((-50, 1.9, 0), (50, 1.9, 0), 1.75),    # right lane: y in [0.15, 3.65]
    ]))
    assert nav.is_on_road([0.0, 0.0, 0.5]) is True
    assert nav.is_on_road([0.0, 0.0, 0.5], tol=0.1) is False  # 0.15 m from either edge


def test_navigraph_is_on_road_height_filter():
    # an overpass 8 m up directly above the car: the car on the road below is
    # on-road (the road below), a car at the overpass's height is on-road (the
    # overpass), and a car in mid-air between the two is not
    nav = Navigraph(_navigraph_data([
        ((-50, 0, 0), (50, 0, 0), 4.0),   # the road the car is on
        ((-50, 0, 8), (50, 0, 8), 4.0),   # an overpass 8 m up
    ]))
    assert nav.is_on_road([0.0, 0.0, 0.5]) is True
    assert nav.is_on_road([0.0, 0.0, 8.5]) is True
    assert nav.is_on_road([0.0, 0.0, 4.0]) is False


def test_navigraph_save_load(tmp_path):
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0),
                                     ((0, -30, 0), (0, 30, 0), 3.0)]))
    path = tmp_path / "nav.npz"
    nav.save(path)
    nav2 = Navigraph.load(path)
    assert nav2.num_segments == nav.num_segments
    assert np.allclose(nav2.a, nav.a)
    assert np.allclose(nav2.b, nav.b)
    assert np.allclose(nav2.normal, nav.normal)
    assert np.array_equal(nav.to_bev([0, 0, 0]), nav2.to_bev([0, 0, 0]))


def test_navigraph_save_load_normals_round_trip(tmp_path):
    # the banked normals are persisted and restored: the road-surface mask is
    # identical after a save/load round trip
    bank = np.deg2rad(10.0)
    n = np.array([0.0, np.sin(bank), np.cos(bank)])
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)],
                                    normals={"n0a": list(n), "n0b": list(n)}))
    path = tmp_path / "nav.npz"
    nav.save(path)
    nav2 = Navigraph.load(path)
    assert np.allclose(to_cpu(nav2.normal), n, atol=1e-6)
    pc = np.array([[0.0, 3.0, -3.0 * np.tan(bank)], [0.0, 3.0, 0.0]])
    assert nav2.road_surface_mask(pc, [0, 0, 0]).tolist() == \
        nav.road_surface_mask(pc, [0, 0, 0]).tolist()


def test_navigraph_load_without_normals(tmp_path):
    # a file saved before the surface normals were added (no "normal" key)
    # loads with the flat (unbanked) normals as the fallback
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0),
                                     ((0, -30, 0), (0, 30, 0), 3.0)]))
    path = tmp_path / "old.npz"
    np.savez_compressed(path, a=to_cpu(nav.a), b=to_cpu(nav.b),
                        hw_a=to_cpu(nav.hw_a), hw_b=to_cpu(nav.hw_b))
    nav2 = Navigraph.load(path)
    assert nav2.num_segments == nav.num_segments
    assert np.allclose(nav2.normal, nav.normal)
    assert np.array_equal(nav.to_bev([0, 0, 0]), nav2.to_bev([0, 0, 0]))


def test_navigraph_to_bev_merge_margin_fills_gap():
    # two parallel 0.4 m surfaces 0.2 m apart (a lane divider): the gap is
    # off-road unmerged, drivable with merge_margin=0.25 (each surface grows
    # 0.25 m, so surfaces < 0.5 m apart merge)
    nav = Navigraph(_navigraph_data([
        ((0, -0.3, 0), (10, -0.3, 0), 0.2),
        ((0, 0.3, 0), (10, 0.3, 0), 0.2),
    ]))
    pos, fwd, left = [5, 0, 0], [1, 0, 0], [0, 1, 0]
    unmerged = nav.to_bev(pos, fwd=fwd, left=left)
    merged = nav.to_bev(pos, fwd=fwd, left=left, merge_margin=0.25)
    # the merge only grows the surface (a dilation), never shrinks it
    assert (merged > 0.5).sum() > (unmerged > 0.5).sum()
    # the gap cell (between the surfaces) is off-road unmerged...
    assert _bev_cell(unmerged, 5, 0) == 0.0
    # ...and filled by the merge
    assert _bev_cell(merged, 5, 0) == 1.0


# -- road_centerline (Gaussian-blur centerline of a merged mask) -----------

def test_road_centerline_middle_of_road():
    # a full-width strip centered on the car: the centerline sits at the
    # road's middle (center of mass, not an edge), a hair from the car
    bev = np.zeros((BEV_N, BEV_N), float)
    bev[:, 30:34] = 1.0   # cols 30..33, middle 31.5
    centerline, dist = road_centerline(bev)
    assert centerline.shape == (BEV_N, 2)
    assert np.allclose(centerline[:, 1], 31.5, atol=1.0)
    assert dist < 1.0


def test_road_centerline_off_center():
    # a strip 10.5 cells left of the car: the distance to the centerline
    # reflects the lateral offset
    bev = np.zeros((BEV_N, BEV_N), float)
    bev[:, 20:24] = 1.0   # cols 20..23, middle 21.5
    centerline, dist = road_centerline(bev)
    assert np.allclose(centerline[:, 1], 21.5, atol=1.0)
    assert dist > 10.0


def test_road_centerline_no_road():
    centerline, dist = road_centerline(np.zeros((BEV_N, BEV_N), float))
    assert centerline.shape == (0, 2)
    assert np.isnan(dist)


def test_road_centerline_segments_straight():
    # a straight road along X, half-width 4, the car at the origin on the
    # centerline (y=0); fwd=[1,0,0] (row = forward), left=[0,1,0] (col = left
    # = +Y). The centerline sits at the road's middle (col 32) and does not
    # wobble row to row.
    nav = Navigraph(_navigraph_data([((-50, 0, 0), (50, 0, 0), 4.0)]))
    centerline, dist = road_centerline_segments(
        nav, [0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    assert centerline.shape == (BEV_N, 2)
    assert np.allclose(centerline[:, 1], 32.0, atol=1.0)
    assert centerline[:, 1].max() - centerline[:, 1].min() < 1.0
    assert dist < 1.0


def test_road_centerline_segments_follows_car_surface():
    # two parallel roads (y=-5 and y=+5), the car on the y=-5 one. The grid is
    # centered on the car, so the car's road is at col 32 (the center) and the
    # other road at col 24; the middle between the roads (y=0) would be col 28.
    # The centerline follows the car's road (col 32), not the middle (col 28).
    nav = Navigraph(_navigraph_data([
        ((-50, -5, 0), (50, -5, 0), 4.0),
        ((-50, 5, 0), (50, 5, 0), 4.0),
    ]))
    centerline, dist = road_centerline_segments(
        nav, [0, -5, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    assert centerline.shape == (BEV_N, 2)
    assert np.allclose(centerline[:, 1], 32.0, atol=1.0)
    assert np.all(centerline[:, 1] > 30.0)   # not the middle (col 28)
    assert dist < 1.0


def test_road_centerline_segments_no_road():
    # a segment 100 m away (beyond LIDAR_MAX_DIST): no nearby segment, so an
    # empty centerline and a NaN distance.
    nav = Navigraph(_navigraph_data([((100, 0, 0), (120, 0, 0), 4.0)]))
    centerline, dist = road_centerline_segments(
        nav, [0, 0, 0], fwd=[1, 0, 0], left=[0, 1, 0])
    assert centerline.shape == (0, 2)
    assert np.isnan(dist)
