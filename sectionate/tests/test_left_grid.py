"""Native support for 'left'-staggered (MITgcm/ECCO) C-grids, including the
multi-tile lat-lon-cap (LLC) case. The 'outer' (symmetric) path is the proven
oracle: a section that stays interior must give the same transport on the matched
'left' grid, and a seam-crossing section on a split 'left' grid must match the
same section on the uncut grid."""

import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import (
    corner_position, corner_offset, check_outer, coord_dict, get_geo_corners,
)
from sectionate.section import grid_section
from sectionate.transports import convergent_transport
from sectionate.tracers import extract_tracer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _outer_grid(Nh=6, Ny=6, seed=0):
    """Single-tile symmetric ('outer') grid with random transports."""
    rng = np.random.default_rng(seed)
    xq = np.linspace(0., 90., Nh + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.linspace(-45., 45., Ny + 1); yh = 0.5 * (yq[:-1] + yq[1:])
    u = rng.standard_normal((Ny, Nh + 1))   # umo at (yh, xq)
    v = rng.standard_normal((Ny + 1, Nh))   # vmo at (yq, xh)
    ds = xr.Dataset(
        {"u": (("yh", "xq"), u), "v": (("yq", "xh"), v)},
        coords={
            "xq": (("xq",), np.arange(Nh + 1)), "yq": (("yq",), np.arange(Ny + 1)),
            "xh": (("xh",), np.arange(Nh)), "yh": (("yh",), np.arange(Ny)),
            "geolon_c": (("yq", "xq"), np.broadcast_to(xq, (Ny + 1, Nh + 1))),
            "geolat_c": (("yq", "xq"), np.broadcast_to(yq[:, None], (Ny + 1, Nh + 1))),
            "geolon": (("yh", "xh"), np.broadcast_to(xh, (Ny, Nh))),
            "geolat": (("yh", "xh"), np.broadcast_to(yh[:, None], (Ny, Nh))),
        })
    grid = xgcm.Grid(ds, coords={"X": {"outer": "xq", "center": "xh"},
                                 "Y": {"outer": "yq", "center": "yh"}},
                     boundary={"X": "extend", "Y": "extend"}, autoparse_metadata=False)
    return grid


def _left_from_outer(outer):
    """Equivalent 'left'-staggered grid: drop the high-side corner row/column and the
    matching high-side velocity column/row. left[k] == outer[k], so the interior
    velocity faces are identical and the transports must agree."""
    ds = outer._ds
    dsl = xr.Dataset(
        {"u": (("yh", "xq"), ds.u.values[:, :-1]),
         "v": (("yq", "xh"), ds.v.values[:-1, :])},
        coords={
            "xq": (("xq",), ds.xq.values[:-1]), "yq": (("yq",), ds.yq.values[:-1]),
            "xh": (("xh",), ds.xh.values), "yh": (("yh",), ds.yh.values),
            "geolon_c": (("yq", "xq"), ds.geolon_c.values[:-1, :-1]),
            "geolat_c": (("yq", "xq"), ds.geolat_c.values[:-1, :-1]),
            "geolon": (("yh", "xh"), ds.geolon.values),
            "geolat": (("yh", "xh"), ds.geolat.values),
        })
    return xgcm.Grid(dsl, coords={"X": {"left": "xq", "center": "xh"},
                                  "Y": {"left": "yq", "center": "yh"}},
                     boundary={"X": "extend", "Y": "extend"}, autoparse_metadata=False)


def _matched_single_and_split_left(Nh=4, Ny=4, seed=0):
    """'left'-staggered (MITgcm/ECCO) analogue of the non-symmetric split test: a
    single-tile left grid and the identical grid cut into two X faces (x<->x glued).
    This is the staggering+topology of the native LLC lat-lon-cap. A seam-crossing
    section on the split grid must match the same section on the uncut grid."""
    rng = np.random.default_rng(seed)
    nxh = 2 * Nh
    xh = np.linspace(5., 75., nxh); xq = xh - 5.       # centers; 'left' (west) edges
    yh = np.linspace(-30., 30., Ny); yq = yh - 10.
    u = rng.standard_normal((Ny, nxh))                # umo at (yh, xq)
    v = rng.standard_normal((Ny, nxh))                # vmo at (yq, xh)
    coords = {"X": {"left": "xq", "center": "xh"}, "Y": {"left": "yq", "center": "yh"}}

    ds_s = xr.Dataset(
        {"u": (("yh", "xq"), u), "v": (("yq", "xh"), v)},
        coords={
            "xq": (("xq",), np.arange(nxh)), "yq": (("yq",), np.arange(Ny)),
            "xh": (("xh",), np.arange(nxh)), "yh": (("yh",), np.arange(Ny)),
            "geolon_c": (("yq", "xq"), np.broadcast_to(xq, (Ny, nxh))),
            "geolat_c": (("yq", "xq"), np.broadcast_to(yq[:, None], (Ny, nxh))),
            "geolon": (("yh", "xh"), np.broadcast_to(xh, (Ny, nxh))),
            "geolat": (("yh", "xh"), np.broadcast_to(yh[:, None], (Ny, nxh))),
        })
    single = xgcm.Grid(ds_s, coords=coords, boundary={"X": "extend", "Y": "extend"},
                       autoparse_metadata=False)

    LONc = np.stack([np.broadcast_to(xq[0:Nh], (Ny, Nh)), np.broadcast_to(xq[Nh:2 * Nh], (Ny, Nh))])
    LATc = np.broadcast_to(yq[:, None], (Ny, Nh))[None].repeat(2, 0)
    LON = np.stack([np.broadcast_to(xh[0:Nh], (Ny, Nh)), np.broadcast_to(xh[Nh:2 * Nh], (Ny, Nh))])
    LAT = np.broadcast_to(yh[:, None], (Ny, Nh))[None].repeat(2, 0)
    ds_f = xr.Dataset(
        {"u": (("face", "yh", "xq"), np.stack([u[:, 0:Nh], u[:, Nh:2 * Nh]])),
         "v": (("face", "yq", "xh"), np.stack([v[:, 0:Nh], v[:, Nh:2 * Nh]]))},
        coords={
            "xq": (("xq",), np.arange(Nh)), "yq": (("yq",), np.arange(Ny)),
            "xh": (("xh",), np.arange(Nh)), "yh": (("yh",), np.arange(Ny)),
            "face": (("face",), [0, 1]),
            "geolon_c": (("face", "yq", "xq"), LONc), "geolat_c": (("face", "yq", "xq"), LATc),
            "geolon": (("face", "yh", "xh"), LON), "geolat": (("face", "yh", "xh"), LAT),
        })
    fc = {"face": {0: {"X": (None, (1, "X", False))}, 1: {"X": ((0, "X", False), None)}}}
    split = xgcm.Grid(ds_f, coords=coords, boundary="fill", fill_value=np.nan,
                      face_connections=fc, autoparse_metadata=False)
    return single, split


# ---------------------------------------------------------------------------
# Staggering-detection helpers
# ---------------------------------------------------------------------------

def test_corner_helpers_all_three_positions():
    outer = _outer_grid()
    left = _left_from_outer(outer)
    assert corner_position(outer) == "outer"
    assert corner_position(left) == "left"
    assert corner_offset(outer) == 0
    assert corner_offset(left) == 0
    # check_outer no longer raises on 'left'; it just reports False.
    assert check_outer(outer) is True
    assert check_outer(left) is False
    # coord_dict / get_geo_corners now succeed on 'left'.
    assert coord_dict(left)["X"]["corner"] == "xq"
    assert get_geo_corners(left)["X"].shape == left._ds.geolon_c.shape


def test_right_corner_offset():
    # A minimal 'right' grid still reports offset 1.
    N = 4
    ds = xr.Dataset(coords={
        "xq": (("xq",), np.arange(N)), "yq": (("yq",), np.arange(N)),
        "xh": (("xh",), np.arange(N)), "yh": (("yh",), np.arange(N)),
        "geolon_c": (("yq", "xq"), np.broadcast_to(np.arange(N) + 0.5, (N, N))),
        "geolat_c": (("yq", "xq"), np.broadcast_to((np.arange(N) + 0.5)[:, None], (N, N))),
    })
    grid = xgcm.Grid(ds, coords={"X": {"right": "xq", "center": "xh"},
                                 "Y": {"right": "yq", "center": "yh"}},
                     boundary="extend", autoparse_metadata=False)
    assert corner_position(grid) == "right"
    assert corner_offset(grid) == 1
    assert check_outer(grid) is False


# ---------------------------------------------------------------------------
# Transports: left matches the proven 'outer' oracle
# ---------------------------------------------------------------------------

def test_left_transport_matches_outer():
    outer = _outer_grid()
    left = _left_from_outer(outer)
    # A closed box well interior to both grids' corner spans (left lacks the high row/col).
    lonseg = np.array([15., 60., 60., 15., 15.])
    latseg = np.array([-30., -30., 15., 15., -30.])

    io, jo, lo, lao = grid_section(outer, lonseg, latseg)
    il, jl, ll, lal = grid_section(left, lonseg, latseg)
    # Same geographic section -> same corner-index path (left[k] == outer[k]).
    assert np.array_equal(io, il) and np.array_equal(jo, jl)

    conv_o = convergent_transport(
        outer, io, jo, utr="u", vtr="v", layer=None, geometry="cartesian"
    )["conv_mass_transport"].sum().values
    conv_l = convergent_transport(
        left, il, jl, utr="u", vtr="v", layer=None, geometry="cartesian"
    )["conv_mass_transport"].sum().values
    assert np.isclose(conv_o, conv_l, rtol=1e-12)


def test_left_multitile_mask_orientation():
    """`positive_in=mask` must resolve the inward sense on a 'left' + multi-tile grid.

    `is_mask_inside` previously located the "inside" tracer cell with the face-frame
    `Usign`/`Vsign`. On a multi-tile grid those diverge from the geometric `Lsign` (which
    carries the topology), so the mask path selected the wrong-side cell and inverted the
    sign of the returned transport -- exactly the failure on the native ECCO LLC grid.
    It now uses `Lsign`. (Single-tile grids are unaffected: there `Lsign` equals the
    face-frame sign by construction.) The geometric `positive_in=True` result and the
    uncut single-tile `positive_in=mask` result are the oracle the split grid must match."""
    single, split = _matched_single_and_split_left()
    lonseg = np.array([10., 60., 60., 10., 10.])
    latseg = np.array([-15., -15., 15., 15., -15.])

    def box_mask(grid):  # center cells enclosed by the box
        lon, lat = grid._ds["geolon"], grid._ds["geolat"]
        return (lon >= 10.) & (lon <= 60.) & (lat >= -15.) & (lat <= 15.)

    i, j, *_ = grid_section(single, lonseg, latseg)
    geom = convergent_transport(
        single, i, j, utr="u", vtr="v", layer=None, geometry="cartesian",
    )["conv_mass_transport"].sum().values
    assert np.abs(geom) > 1e-6               # non-trivial: the sign actually matters
    conv_single = convergent_transport(
        single, i, j, utr="u", vtr="v", layer=None, geometry="cartesian",
        positive_in=box_mask(single),
    )["conv_mass_transport"].sum().values

    i2, j2, f2, *_ = grid_section(split, lonseg, latseg)
    assert set(np.unique(f2).tolist()) == {0, 1}   # genuinely crosses the seam
    conv_split = convergent_transport(
        split, i2, j2, f2, utr="u", vtr="v", layer=None, geometry="cartesian",
        positive_in=box_mask(split),
    )["conv_mass_transport"].sum().values

    assert np.isclose(conv_single, geom, rtol=1e-12)
    assert np.isclose(conv_split, geom, rtol=1e-12)


def test_left_multitile_seam_transport_matches_uncut_grid():
    """The native LLC case: 'left' staggering + multi-tile face_connections."""
    single, split = _matched_single_and_split_left()
    lonseg = np.array([10., 60., 60., 10., 10.])
    latseg = np.array([-15., -15., 15., 15., -15.])

    i, j, lons, lats = grid_section(single, lonseg, latseg)
    conv_single = convergent_transport(
        single, i, j, utr="u", vtr="v", layer=None, geometry="cartesian"
    )["conv_mass_transport"].sum().values

    i2, j2, f2, l2, la2 = grid_section(split, lonseg, latseg)
    assert set(np.unique(f2).tolist()) == {0, 1}  # really crosses the seam
    conv_split = convergent_transport(
        split, i2, j2, f2, utr="u", vtr="v", layer=None, geometry="cartesian"
    )["conv_mass_transport"].sum().values
    assert np.isclose(conv_single, conv_split, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tracers: left matches 'outer'
# ---------------------------------------------------------------------------

def test_left_tracer_matches_outer():
    outer = _outer_grid()
    left = _left_from_outer(outer)
    # Identical tracer on the (shared) cell centers.
    theta = outer._ds.geolon * 0.3 - outer._ds.geolat * 0.1
    outer._ds["theta"] = theta
    left._ds["theta"] = (("yh", "xh"), theta.values)

    lonseg = np.array([15., 60., 60., 15., 15.])
    latseg = np.array([-30., -30., 15., 15., -30.])
    io, jo, lo, lao = grid_section(outer, lonseg, latseg)
    il, jl, ll, lal = grid_section(left, lonseg, latseg)
    tr_o = extract_tracer("theta", outer, io, jo).values
    tr_l = extract_tracer("theta", left, il, jl).values
    assert np.allclose(tr_o, tr_l, equal_nan=True)
