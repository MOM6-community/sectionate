"""Tests for sections on multi-tile grids defined by xgcm `face_connections`
(e.g. the lat-lon-cap and cubed-sphere grids)."""

import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import build_neighbor_maps, get_geo_corners, NEIGHBOR_DIRECTIONS
from sectionate.section import grid_section
from sectionate.transports import convergent_transport


def _make_grid(lon, lat, face_connections):
    """Build a symmetric ('outer' corner) multi-tile grid from (face, yg, xg) coords."""
    nf, ng, _ = lon.shape
    ds = xr.Dataset({}, coords={
        "xg": (("xg",), np.arange(ng)),
        "yg": (("yg",), np.arange(ng)),
        "face": (("face",), np.arange(nf)),
        "geolon_c": (("face", "yg", "xg"), lon),
        "geolat_c": (("face", "yg", "xg"), lat),
    })
    return xgcm.Grid(
        ds,
        coords={"X": {"outer": "xg"}, "Y": {"outer": "yg"}},
        padding="fill", fill_value=np.nan,
        face_connections=face_connections,
        autoparse_metadata=False,
    )


def two_face_x_to_x(Nc=6):
    """Two faces side-by-side in longitude: face0 [0,90], face1 [90,180]."""
    ng = Nc + 1
    lat1d = np.linspace(-45, 45, ng)
    lon = np.zeros((2, ng, ng)); lat = np.zeros((2, ng, ng))
    lon[0] = np.broadcast_to(np.linspace(0, 90, ng), (ng, ng))
    lon[1] = np.broadcast_to(np.linspace(90, 180, ng), (ng, ng))
    lat[0] = np.broadcast_to(lat1d[:, None], (ng, ng))
    lat[1] = lat[0]
    fc = {"face": {0: {"X": (None, (1, "X", False))},
                   1: {"X": ((0, "X", False), None)}}}
    return _make_grid(lon, lat, fc)


def left_two_tile_x_to_y(Nc=5):
    """A native 'left'-staggered (MITgcm/ECCO) 2-tile grid whose tiles meet at a
    rotated seam: face0's +X edge connects to face1's Y axis. Carries both
    tracer-center and corner coordinates (the 'left' correction in
    `build_neighbor_maps` needs the centers)."""
    ng = Nc  # native 'left' corners are the same size as centers
    LONc = np.zeros((2, ng, ng)); LATc = np.zeros((2, ng, ng))
    LON = np.zeros((2, Nc, Nc)); LAT = np.zeros((2, Nc, Nc))
    for f in range(2):
        off = 100 * f
        LONc[f] = off + np.arange(ng)[None, :]; LATc[f] = off + np.arange(ng)[:, None]
        LON[f] = off + np.arange(Nc)[None, :] + 0.5; LAT[f] = off + np.arange(Nc)[:, None] + 0.5
    ds = xr.Dataset(coords={
        "i": ("i", np.arange(Nc)), "j": ("j", np.arange(Nc)),
        "i_g": ("i_g", np.arange(ng)), "j_g": ("j_g", np.arange(ng)), "face": ("face", [0, 1]),
        "geolon": (("face", "j", "i"), LON), "geolat": (("face", "j", "i"), LAT),
        "geolon_c": (("face", "j_g", "i_g"), LONc), "geolat_c": (("face", "j_g", "i_g"), LATc)})
    fc = {"face": {0: {"X": (None, (1, "Y", False))}, 1: {"Y": ((0, "X", False), None)}}}
    return xgcm.Grid(
        ds, coords={"X": {"center": "i", "left": "i_g"}, "Y": {"center": "j", "left": "j_g"}},
        padding="fill", fill_value=np.nan, face_connections=fc, autoparse_metadata=False)


def two_face_x_to_y(Nc=4):
    """face0 right-X connects to face1 Y (a 90-degree rotation)."""
    ng = Nc + 1
    lon = np.zeros((2, ng, ng)); lat = np.zeros((2, ng, ng))
    # Encode (face, i) so we can read topology back out; geometry is not used here.
    for f in range(2):
        for j in range(ng):
            for i in range(ng):
                lon[f, j, i] = f * 1000 + i
                lat[f, j, i] = j
    fc = {"face": {0: {"X": (None, (1, "Y", False))},
                   1: {"Y": ((0, "X", False), None)}}}
    return _make_grid(lon, lat, fc)


def cubed_sphere(Nc=4):
    ng = Nc + 1
    lon = np.zeros((6, ng, ng)); lat = np.zeros((6, ng, ng))
    fc = {"face": {
        0: {"X": ((3, "X", False), (1, "X", False)), "Y": ((4, "Y", False), (5, "Y", False))},
        1: {"X": ((0, "X", False), (2, "X", False)), "Y": ((4, "X", False), (5, "X", True))},
        2: {"X": ((1, "X", False), (3, "X", False)), "Y": ((4, "Y", True), (5, "Y", True))},
        3: {"X": ((2, "X", False), (0, "X", False)), "Y": ((4, "X", True), (5, "X", False))},
        4: {"X": ((3, "Y", True), (1, "Y", False)), "Y": ((2, "Y", True), (0, "Y", False))},
        5: {"X": ((3, "Y", False), (1, "Y", True)), "Y": ((0, "Y", False), (2, "Y", True))},
    }}
    return _make_grid(lon, lat, fc)


def is_neighbor(prev, cur, maps):
    f, j, i = prev
    for d in NEIGHBOR_DIRECTIONS:
        fm, jm, im = maps[d]
        if (int(fm[f, j, i]), int(jm[f, j, i]), int(im[f, j, i])) == cur:
            return True
    return False


def assert_path_invariant(i_c, j_c, f_c, maps):
    """Every consecutive pair in the path must be a genuine grid neighbor."""
    for k in range(len(i_c) - 1):
        prev = (int(f_c[k]), int(j_c[k]), int(i_c[k]))
        cur = (int(f_c[k + 1]), int(j_c[k + 1]), int(i_c[k + 1]))
        if prev == cur:
            continue
        assert is_neighbor(prev, cur, maps), f"{prev} -> {cur} is not a grid neighbor"


# ---------------------------------------------------------------------------
# Neighbor-map unit tests (topology only)
# ---------------------------------------------------------------------------

def test_neighbor_maps_x_to_x_seam():
    grid = two_face_x_to_x()
    maps = build_neighbor_maps(grid, get_geo_corners(grid))
    nx = maps["right"][0].shape[-1]
    # Right edge of face 0 -> face 1, landing on its left edge (i=0), same row.
    fmap, jmap, imap = maps["right"]
    assert np.all(fmap[0][:, -1] == 1)
    assert np.all(imap[0][:, -1] == 0)
    assert np.all(jmap[0][:, -1] == np.arange(grid._ds.sizes["yg"]))
    # Interior steps right by one on the same face.
    assert np.all(fmap[0][:, :-1] == 0)
    assert np.all(imap[0][:, :-1] == np.arange(1, nx))


def test_neighbor_maps_x_to_y_rotation_and_reversal():
    grid = two_face_x_to_y()
    maps = build_neighbor_maps(grid, get_geo_corners(grid))
    fmap, jmap, imap = maps["right"]
    n = grid._ds.sizes["yg"]
    # Right edge of face0 lands on face1...
    assert np.all(fmap[0][:, -1] == 1)
    # ...on its bottom row (Y=0, the X->Y rotation)...
    assert np.all(jmap[0][:, -1] == 0)
    # ...with the tangential index reversed (j -> n-1-j).
    assert np.all(imap[0][:, -1] == (n - 1) - np.arange(n))


def test_neighbor_maps_left_grid_rotated_seam_are_edge_adjacent():
    """On a native 'left'-staggered grid, the vorticity lattice sits half a cell
    from the centers a face seam is defined on, so xgcm's corner-array padding lands
    cross-tile neighbors one corner off across a rotated/reversed seam (a *diagonal*
    instead of edge-adjacent corner). `build_neighbor_maps` must correct this: every
    interior corner's four neighbors must be EDGE-adjacent -- sharing the two cells
    of their common edge -- not merely diagonal. (Symmetric 'outer' grids never hit
    this; their seam vorticity points are shared.)"""
    from xgcm.padding import pad
    grid = left_two_tile_x_to_y(Nc=5)
    maps = build_neighbor_maps(grid, get_geo_corners(grid))

    nf, ny, nx = 2, 5, 5
    cid = xr.DataArray(np.arange(nf * ny * nx, dtype=float).reshape(nf, ny, nx),
                       dims=("face", "j", "i"), coords={"face": [0, 1]})
    bw = {ax: (1, 1) for ax in grid.axes}
    bdy = {ax: grid.axes[ax].padding for ax in grid.axes}
    C = pad(cid, grid, bw, padding=bdy, fill_value=np.nan).transpose("face", ..., "j", "i").values

    def cells(f, j, i):  # the (up to four) cells touching corner (f, j, i)
        v = (C[f, j, i], C[f, j, i + 1], C[f, j + 1, i], C[f, j + 1, i + 1])
        return frozenset(int(x) for x in v if not np.isnan(x))

    for d in NEIGHBOR_DIRECTIONS:
        fm, jm, im = maps[d]
        for f in range(nf):
            for j in range(ny):
                for i in range(nx):
                    nb = (int(fm[f, j, i]), int(jm[f, j, i]), int(im[f, j, i]))
                    if nb == (f, j, i):
                        continue  # wall
                    a, b = cells(f, j, i), cells(*nb)
                    # an interior corner (all four cells real) must share an EDGE (2 cells)
                    if len(a) == 4 and len(b) == 4:
                        assert len(a & b) >= 2, f"{d}: ({f},{j},{i}) -> {nb} only diagonally adjacent"


def test_cubed_sphere_reciprocal_or_raises():
    # xgcm's padding is unreliable (hash-seed dependent) for the cubed sphere's
    # complex reversed/rotated connections. build_neighbor_maps must therefore
    # EITHER produce a fully reciprocal (correct) topology OR refuse with a clear
    # error -- it must never silently return an inconsistent topology. This holds
    # regardless of which (buggy) halo xgcm happens to produce.
    grid = cubed_sphere()
    try:
        maps = build_neighbor_maps(grid, get_geo_corners(grid))
    except NotImplementedError:
        return
    # If it did not raise, the maps must be self-consistent: when face 0's right
    # neighbor is face 1, the connection definition (X right -> 1) must hold.
    assert np.all(maps["right"][0][0][:, -1] == 1)
    assert np.all(maps["left"][0][0][:, 0] == 3)
    assert np.all(maps["up"][0][0][-1, :] == 5)
    assert np.all(maps["down"][0][0][0, :] == 4)


# ---------------------------------------------------------------------------
# End-to-end section tests
# ---------------------------------------------------------------------------

def test_cross_face_section_returns_face_and_crosses_seam():
    grid = two_face_x_to_x()
    out = grid_section(grid, [15., 165.], [0., 0.])
    assert len(out) == 5  # multi-tile grids return the face index too
    i_c, j_c, f_c, lons_c, lats_c = out
    assert set(np.unique(f_c).tolist()) == {0, 1}  # the section spans both faces
    assert lons_c[0] == pytest.approx(15., abs=10.)
    assert lons_c[-1] == pytest.approx(165., abs=10.)


def test_cross_face_section_path_invariant():
    grid = two_face_x_to_x()
    maps = build_neighbor_maps(grid, get_geo_corners(grid))
    i_c, j_c, f_c, lons_c, lats_c = grid_section(grid, [15., 165.], [0., 0.])
    assert_path_invariant(i_c, j_c, f_c, maps)


# ---------------------------------------------------------------------------
# Fold guard
# ---------------------------------------------------------------------------

def _two_face_transport_grid(Nc=3):
    """2-face (X-X) grid with center+corner coords and U/V transports for transport tests."""
    ng = Nc + 1
    yq = np.linspace(-45, 45, ng); yh = 0.5 * (yq[:-1] + yq[1:])
    lonq = [np.linspace(0, 90, ng), np.linspace(90, 180, ng)]
    lonh = [0.5 * (l[:-1] + l[1:]) for l in lonq]

    LONc = np.stack([np.broadcast_to(lonq[f], (ng, ng)) for f in range(2)])
    LATc = np.stack([np.broadcast_to(yq[:, None], (ng, ng)) for f in range(2)])
    LON = np.stack([np.broadcast_to(lonh[f], (Nc, Nc)) for f in range(2)])
    LAT = np.stack([np.broadcast_to(yh[:, None], (Nc, Nc)) for f in range(2)])

    rng = np.arange(2 * Nc * ng, dtype=float)
    u = rng[: 2 * Nc * ng].reshape(2, Nc, ng)
    v = (rng[: 2 * ng * Nc].reshape(2, ng, Nc)) * 0.5
    ds = xr.Dataset(
        {"u": (("face", "yh", "xq"), u), "v": (("face", "yq", "xh"), v)},
        coords={
            "xq": (("xq",), np.arange(ng)), "yq": (("yq",), np.arange(ng)),
            "xh": (("xh",), np.arange(Nc)), "yh": (("yh",), np.arange(Nc)),
            "face": (("face",), [0, 1]),
            "geolon_c": (("face", "yq", "xq"), LONc), "geolat_c": (("face", "yq", "xq"), LATc),
            "geolon": (("face", "yh", "xh"), LON), "geolat": (("face", "yh", "xh"), LAT),
        },
    )
    fc = {"face": {0: {"X": (None, (1, "X", False))},
                   1: {"X": ((0, "X", False), None)}}}
    grid = xgcm.Grid(ds, coords={"X": {"outer": "xq", "center": "xh"},
                                 "Y": {"outer": "yq", "center": "yh"}},
                     padding="fill", fill_value=np.nan,
                     face_connections=fc, autoparse_metadata=False)
    return grid


def _single_face_slab(grid, face):
    """Standalone single-tile grid identical to one face of a multi-tile grid."""
    ds = grid._ds.isel({grid._facedim: face}).drop_vars(grid._facedim)
    return xgcm.Grid(ds, coords={"X": {"outer": "xq", "center": "xh"},
                                 "Y": {"outer": "yq", "center": "yh"}},
                     padding={"X": "extend", "Y": "extend"}, autoparse_metadata=False)


def test_within_face_transport_matches_single_tile():
    grid = _two_face_transport_grid()
    # A closed box wholly within face 0 (lon in [0, 90]).
    lonseg = np.array([15., 75., 75., 15., 15.])
    latseg = np.array([-15., -15., 15., 15., -15.])
    i, j, f, lons, lats = grid_section(grid, lonseg, latseg)
    assert np.all(f == 0)  # stays on face 0

    conv_mt = convergent_transport(
        grid, i, j, f, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values

    # Same section + transports on the standalone face-0 grid must agree exactly.
    slab = _single_face_slab(grid, 0)
    i0, j0, lons0, lats0 = grid_section(slab, lonseg, latseg)
    assert np.array_equal(i, i0) and np.array_equal(j, j0)
    conv_st = convergent_transport(
        slab, i0, j0, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values

    assert np.isclose(conv_mt, conv_st, rtol=1e-14)


def test_within_face_tracer_matches_single_tile():
    from sectionate.tracers import extract_tracer
    grid = _two_face_transport_grid()
    grid._ds["theta"] = grid._ds["u"].rename({"xq": "xh"}) + 7.0  # a (face, yh, xh) tracer
    lonseg = np.array([15., 75., 75., 15., 15.])
    latseg = np.array([-15., -15., 15., 15., -15.])
    i, j, f, lons, lats = grid_section(grid, lonseg, latseg)

    tr_mt = extract_tracer("theta", grid, i, j, f_c=f).values

    slab = _single_face_slab(grid, 0)
    slab._ds["theta"] = grid._ds["theta"].isel({grid._facedim: 0}).drop_vars(grid._facedim)
    i0, j0, lons0, lats0 = grid_section(slab, lonseg, latseg)
    tr_st = extract_tracer("theta", slab, i0, j0).values

    assert np.allclose(tr_mt, tr_st, equal_nan=True)


def _matched_single_and_split(Nh=4, Ny=4, seed=0):
    """A single-tile wide grid and the identical grid cut into two X faces (x<->x glued).

    A seam-crossing section on the split grid must give the same transport as the same
    section on the uncut grid -- the strongest oracle for seam attribution.
    """
    rng = np.random.default_rng(seed)
    nxh = 2 * Nh
    xq = np.linspace(0., 80., nxh + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.linspace(-40., 40., Ny + 1); yh = 0.5 * (yq[:-1] + yq[1:])
    u = rng.standard_normal((Ny, nxh + 1))      # umo at (yh, xq)
    v = rng.standard_normal((Ny + 1, nxh))      # vmo at (yq, xh)

    def grid2d(sl_q, sl_h, ds_extra):
        ds = xr.Dataset(ds_extra, coords={
            "xq": (("xq",), np.arange(sl_q)), "yq": (("yq",), np.arange(Ny + 1)),
            "xh": (("xh",), np.arange(sl_h)), "yh": (("yh",), np.arange(Ny)),
        })
        return ds

    # single-tile
    ds_s = xr.Dataset(
        {"u": (("yh", "xq"), u), "v": (("yq", "xh"), v)},
        coords={
            "xq": (("xq",), np.arange(nxh + 1)), "yq": (("yq",), np.arange(Ny + 1)),
            "xh": (("xh",), np.arange(nxh)), "yh": (("yh",), np.arange(Ny)),
            "geolon_c": (("yq", "xq"), np.broadcast_to(xq, (Ny + 1, nxh + 1))),
            "geolat_c": (("yq", "xq"), np.broadcast_to(yq[:, None], (Ny + 1, nxh + 1))),
            "geolon": (("yh", "xh"), np.broadcast_to(xh, (Ny, nxh))),
            "geolat": (("yh", "xh"), np.broadcast_to(yh[:, None], (Ny, nxh))),
        },
    )
    single = xgcm.Grid(ds_s, coords={"X": {"outer": "xq", "center": "xh"},
                                     "Y": {"outer": "yq", "center": "yh"}},
                       padding={"X": "extend", "Y": "extend"}, autoparse_metadata=False)

    # two faces: cols [0:Nh] and [Nh:2Nh]; faces share the boundary corner column at Nh.
    def face_slices(arr_q, arr_h):
        return (np.stack([arr_q[..., 0:Nh + 1], arr_q[..., Nh:2 * Nh + 1]]),
                np.stack([arr_h[..., 0:Nh], arr_h[..., Nh:2 * Nh]]))
    LONc = np.stack([np.broadcast_to(xq[0:Nh + 1], (Ny + 1, Nh + 1)),
                     np.broadcast_to(xq[Nh:2 * Nh + 1], (Ny + 1, Nh + 1))])
    LATc = np.broadcast_to(yq[:, None], (Ny + 1, Nh + 1))[None].repeat(2, 0)
    LON = np.stack([np.broadcast_to(xh[0:Nh], (Ny, Nh)),
                    np.broadcast_to(xh[Nh:2 * Nh], (Ny, Nh))])
    LAT = np.broadcast_to(yh[:, None], (Ny, Nh))[None].repeat(2, 0)
    uf = np.stack([u[:, 0:Nh + 1], u[:, Nh:2 * Nh + 1]])
    vf = np.stack([v[:, 0:Nh], v[:, Nh:2 * Nh]])
    ds_f = xr.Dataset(
        {"u": (("face", "yh", "xq"), uf), "v": (("face", "yq", "xh"), vf)},
        coords={
            "xq": (("xq",), np.arange(Nh + 1)), "yq": (("yq",), np.arange(Ny + 1)),
            "xh": (("xh",), np.arange(Nh)), "yh": (("yh",), np.arange(Ny)),
            "face": (("face",), [0, 1]),
            "geolon_c": (("face", "yq", "xq"), LONc), "geolat_c": (("face", "yq", "xq"), LATc),
            "geolon": (("face", "yh", "xh"), LON), "geolat": (("face", "yh", "xh"), LAT),
        },
    )
    fc = {"face": {0: {"X": (None, (1, "X", False))}, 1: {"X": ((0, "X", False), None)}}}
    split = xgcm.Grid(ds_f, coords={"X": {"outer": "xq", "center": "xh"},
                                    "Y": {"outer": "yq", "center": "yh"}},
                      padding="fill", fill_value=np.nan,
                      face_connections=fc, autoparse_metadata=False)
    return single, split


def test_seam_crossing_transport_matches_uncut_grid():
    single, split = _matched_single_and_split()
    # A closed box straddling the seam (lon 40).
    lonseg = np.array([10., 70., 70., 10., 10.])
    latseg = np.array([-20., -20., 20., 20., -20.])

    i, j, lons, lats = grid_section(single, lonseg, latseg)
    conv_single = convergent_transport(
        single, i, j, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values

    i2, j2, f2, lons2, lats2 = grid_section(split, lonseg, latseg)
    assert set(np.unique(f2).tolist()) == {0, 1}  # the section really crosses the seam
    conv_split = convergent_transport(
        split, i2, j2, f2, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values

    assert np.isclose(conv_single, conv_split, rtol=1e-12)


def _matched_single_and_split_nonsym(Nh=4, Ny=4, seed=0):
    """Non-symmetric ('right' corner) analogue of `_matched_single_and_split`: a single-tile
    grid and the identical grid cut into two X faces. Non-symmetric tilings do NOT share a
    boundary vorticity column, so seam crossings are real (non-degenerate) velocity edges."""
    rng = np.random.default_rng(seed)
    nxh = 2 * Nh
    xh = np.linspace(5., 75., nxh); xq = xh + 5.      # centers; 'right' edges
    yh = np.linspace(-30., 30., Ny); yq = yh + 10.
    u = rng.standard_normal((Ny, nxh))               # umo at (yh, xq)
    v = rng.standard_normal((Ny, nxh))               # vmo at (yq, xh)
    coords = {"X": {"right": "xq", "center": "xh"}, "Y": {"right": "yq", "center": "yh"}}

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
    single = xgcm.Grid(ds_s, coords=coords, padding={"X": "extend", "Y": "extend"},
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
    split = xgcm.Grid(ds_f, coords=coords, padding="fill", fill_value=np.nan,
                      face_connections=fc, autoparse_metadata=False)
    return single, split


def test_nonsymmetric_seam_transport_matches_uncut_grid():
    single, split = _matched_single_and_split_nonsym()
    lonseg = np.array([15., 65., 65., 15., 15.])
    latseg = np.array([-15., -15., 15., 15., -15.])
    i, j, lons, lats = grid_section(single, lonseg, latseg)
    conv_single = convergent_transport(
        single, i, j, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values
    i2, j2, f2, l2, la2 = grid_section(split, lonseg, latseg)
    assert set(np.unique(f2).tolist()) == {0, 1}
    conv_split = convergent_transport(
        split, i2, j2, f2, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values
    assert np.isclose(conv_single, conv_split, rtol=1e-12)


def _psi(lon, lat):
    """A smooth streamfunction. For a non-divergent flow (umo,vmo = its discrete curl), the
    transport across any section from P1 to P2 equals psi(P2) - psi(P1) -- a known, non-zero,
    rotation-independent answer, so it pins the per-edge signs without any external data."""
    return 1.3 * lon - 0.7 * lat + 0.02 * lon * lat


def _uv_from_psi(P):
    """umo = -dpsi/dy (at yh,xq), vmo = +dpsi/dx (at yq,xh), in the face's own frame."""
    return -(P[1:, :] - P[:-1, :]), (P[:, 1:] - P[:, :-1])


def rotated_two_face_streamfunction():
    """Two faces meeting at a 90-degree ROTATION (face 1's local axes: +y->east, +x->south),
    carrying a non-divergent streamfunction flow. face 0 is a normal lon[0,40] x lat[0,40]
    square; face 1 covers lon[40,80] x lat[0,40] rotated. A proper (orientation-preserving)
    rotation that attaches here needs no `reversed` flag on the connection."""
    a = np.arange(5)
    f0_lonc = np.broadcast_to(a * 10., (5, 5)); f0_latc = np.broadcast_to((a * 10.)[:, None], (5, 5))
    f1_lonc = np.broadcast_to((40. + a * 10.)[:, None], (5, 5)); f1_latc = np.broadcast_to(40. - a * 10., (5, 5))
    LONc = np.stack([f0_lonc, f1_lonc]); LATc = np.stack([f0_latc, f1_latc])
    c = np.arange(4)
    f0_lon = np.broadcast_to(5. + c * 10., (4, 4)); f0_lat = np.broadcast_to((5. + c * 10.)[:, None], (4, 4))
    f1_lon = np.broadcast_to((45. + c * 10.)[:, None], (4, 4)); f1_lat = np.broadcast_to(35. - c * 10., (4, 4))
    LON = np.stack([f0_lon, f1_lon]); LAT = np.stack([f0_lat, f1_lat])
    u0, v0 = _uv_from_psi(_psi(f0_lonc, f0_latc))
    u1, v1 = _uv_from_psi(_psi(f1_lonc, f1_latc))
    ds = xr.Dataset(
        {"u": (("face", "yh", "xq"), np.stack([u0, u1])), "v": (("face", "yq", "xh"), np.stack([v0, v1]))},
        coords={
            "xq": (("xq",), np.arange(5)), "yq": (("yq",), np.arange(5)),
            "xh": (("xh",), np.arange(4)), "yh": (("yh",), np.arange(4)), "face": (("face",), [0, 1]),
            "geolon_c": (("face", "yq", "xq"), LONc), "geolat_c": (("face", "yq", "xq"), LATc),
            "geolon": (("face", "yh", "xh"), LON), "geolat": (("face", "yh", "xh"), LAT),
        })
    fc = {"face": {0: {"X": (None, (1, "Y", False))}, 1: {"Y": ((0, "X", False), None)}}}
    return xgcm.Grid(ds, coords={"X": {"outer": "xq", "center": "xh"}, "Y": {"outer": "yq", "center": "yh"}},
                     padding="fill", fill_value=np.nan, face_connections=fc, autoparse_metadata=False)


def test_rotated_seam_transport_streamfunction():
    # Transport across a section crossing a ROTATED seam must equal the streamfunction
    # difference between its endpoints -- the geographic per-edge sign makes this hold even
    # though face 1's grid axes are rotated 90 degrees relative to face 0.
    grid = rotated_two_face_streamfunction()
    i, j, f, lons, lats = grid_section(grid, [10., 70.], [20., 20.])
    assert set(np.unique(f).tolist()) == {0, 1}  # the section really crosses the rotated seam
    conv = convergent_transport(
        grid, i, j, f, utr="u", vtr="v", geometry="cartesian"
    )["conv_mass_transport"].sum().values
    dpsi = _psi(lons[-1], lats[-1]) - _psi(lons[0], lats[0])
    assert np.isclose(abs(conv), abs(dpsi), rtol=1e-9)


def test_face_glued_to_itself_raises():
    """A face connected to itself is refused up front. Both sides of such a seam carry
    the same face index, so a crossing looks exactly like an ordinary step within the
    face -- nothing further down the pipeline notices, and a section is silently traced
    as if the seam were not there. (A periodic axis or a north fold belongs in the axis
    `padding` of a single-tile grid, which sectionate does support.)"""
    ng = 5
    lon = np.zeros((1, ng, ng)); lat = np.zeros((1, ng, ng))
    fc = {"face": {0: {"Y": ((0, "Y", True), (0, "Y", True))}}}
    grid = _make_grid(lon, lat, fc)
    with pytest.raises(NotImplementedError, match="glued to itself"):
        grid_section(grid, [0., 1.], [0., 1.])


@pytest.mark.parametrize("fc, exc", [
    # a neighbour face the grid does not have
    ({"face": {0: {"X": (None, (7, "X", False))},
               1: {"X": ((0, "X", False), None)}}}, KeyError),
    # a seam declared from only one side
    ({"face": {0: {"X": (None, (1, "X", False))},
               1: {"X": (None, None)}}}, TypeError),
])
def test_xgcm_rejects_broken_face_connections_at_construction(fc, exc):
    """xgcm -- not sectionate -- enforces that every face named as a neighbour exists and
    that every connection is mutual: `xgcm.Grid.__init__` aborts on both while it builds
    its face-connection table, so no such grid can reach `grid_section` in the first place.

    This is why `_check_supported_topology` does not re-check either one. The test pins
    that upstream contract, so that if xgcm ever stops enforcing it we find out here rather
    than by tracing a section through a topology nothing validated."""
    ng = 6
    lon = np.zeros((2, ng, ng)); lat = np.zeros((2, ng, ng))
    with pytest.raises(exc):
        _make_grid(lon, lat, fc)


def test_supported_multitile_topologies_pass_the_check():
    """The front-door check must not reject topologies sectionate really does support --
    including rotated seams (face 0's X axis glued to face 1's Y axis) and the real
    13-tile lat-lon-cap connections, whose seams are mutual across different axes."""
    import os, sys
    from sectionate.section import _check_supported_topology

    _check_supported_topology(two_face_x_to_x())
    _check_supported_topology(rotated_two_face_streamfunction())

    examples = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples")
    )
    if examples not in sys.path:
        sys.path.insert(0, examples)
    from load_example_ECCO_grid import LLC90_FACE_CONNECTIONS  # metadata only, no data

    ng = 4
    zeros = np.zeros((13, ng, ng))
    llc90 = _make_grid(zeros, zeros.copy(), {"face": LLC90_FACE_CONNECTIONS["tile"]})
    _check_supported_topology(llc90)


def _labelled_grid(face_connections, labels=(1, 2)):
    """A two-face 'outer' grid whose face coordinate carries arbitrary labels."""
    ng = 6
    lon = np.zeros((2, ng, ng)); lat = np.zeros((2, ng, ng))
    lon[0] = np.broadcast_to(np.linspace(0, 90, ng), (ng, ng))
    lon[1] = np.broadcast_to(np.linspace(90, 180, ng), (ng, ng))
    ds = xr.Dataset({}, coords={
        "xg": (("xg",), np.arange(ng)),
        "yg": (("yg",), np.arange(ng)),
        "face": (("face",), np.array(labels)),
        "geolon_c": (("face", "yg", "xg"), lon),
        "geolat_c": (("face", "yg", "xg"), lat),
    })
    return xgcm.Grid(
        ds, coords={"X": {"outer": "xg"}, "Y": {"outer": "yg"}},
        padding="fill", fill_value=np.nan,
        face_connections={"face": face_connections},
        autoparse_metadata=False,
    )


def test_check_reads_face_labels_not_positions():
    """Face labels need not be 0..N-1: a grid labelled `face = [1, 2]` is legal, and xgcm
    keys `face_connections` by those labels. The check must therefore compare labels, never
    positions along the face dimension.

    Both directions are pinned. A legal `[1, 2]` grid must pass -- an earlier version built
    its set of known faces as `range(n_faces)` and rejected this grid for naming a face 2
    it thought did not exist. And a `[1, 2]` grid whose face 2 is glued to itself must still
    be caught, and named as face 2, which is what would break if the surviving comparison
    ever drifted back to positions."""
    from sectionate.section import _check_supported_topology

    _check_supported_topology(_labelled_grid(
        {1: {"X": (None, (2, "X", False))},
         2: {"X": ((1, "X", False), None)}}
    ))

    glued = _labelled_grid(
        {1: {"X": (None, (2, "X", False))},
         2: {"X": ((1, "X", False), None), "Y": ((2, "Y", True), (2, "Y", True))}}
    )
    with pytest.raises(NotImplementedError, match="Face 2 .* glued to itself"):
        _check_supported_topology(glued)


def test_in_velocity_range_rejects_unknown_component():
    """`_in_velocity_range` decides whether a velocity index exists in a face's own
    arrays: in range for an edge stored on that face, out of range for one stored on the
    face across the seam. It only ever sees the "U"/"V" components `_anchor_velocity`
    produces, so anything else is a caller bug and is reported as one rather than
    silently falling through to the "U" branch."""
    from sectionate.transports import _in_velocity_range
    ranges = {"Xc": 4, "Yc": 4, "Xq": 5, "Yq": 5}

    assert _in_velocity_range("V", 3, 4, ranges)        # (X-center, Y-corner): both in range
    assert not _in_velocity_range("V", 4, 4, ranges)    # past the end of the X-center axis
    assert _in_velocity_range("U", 4, 3, ranges)        # (X-corner, Y-center): both in range
    assert not _in_velocity_range("U", 4, 4, ranges)    # past the end of the Y-center axis
    assert not _in_velocity_range("U", -1, 0, ranges)   # off the low end
    with pytest.raises(ValueError, match="'U' or 'V'"):
        _in_velocity_range("0", 0, 0, ranges)


def test_save_load_roundtrip_preserves_face_indices(tmp_path):
    """A multi-tile GriddedSection round-trips through save/load with its face indices
    intact. (f_c was previously not persisted, so a reloaded multi-tile section silently
    fell back to f_c=None and was mis-derived as single-tile.)"""
    from sectionate.section import Section, GriddedSection
    from sectionate.utils import save_gridded_section, load_gridded_section

    grid = _two_face_transport_grid()
    gs = GriddedSection(Section("seam", ([15., 165.], [0., 0.])), grid)
    assert gs.f_c is not None and set(np.unique(gs.f_c).tolist()) == {0, 1}  # truly multi-tile

    path = str(tmp_path / "gs.json")
    save_gridded_section(path, gs)
    gs2 = load_gridded_section(path, grid)

    assert gs2.f_c is not None
    np.testing.assert_array_equal(np.asarray(gs2.f_c), np.asarray(gs.f_c))
    np.testing.assert_array_equal(np.asarray(gs2.i_c), np.asarray(gs.i_c))
    np.testing.assert_array_equal(np.asarray(gs2.j_c), np.asarray(gs.j_c))

    # transport from the reloaded section matches the original
    kw = dict(utr="u", vtr="v", geometry="cartesian")
    t1 = convergent_transport(grid, gs.i_c, gs.j_c, gs.f_c, **kw)["conv_mass_transport"].sum().values
    t2 = convergent_transport(grid, gs2.i_c, gs2.j_c, gs2.f_c, **kw)["conv_mass_transport"].sum().values
    assert np.isclose(t1, t2)


# ---------------------------------------------------------------------------
# A registered vertical axis must not affect horizontal topology
# ---------------------------------------------------------------------------

def _with_vertical_axis(grid):
    """Rebuild `grid` with an extra registered Z axis and nothing else changed."""
    coords = {name: dict(axis.coords) for name, axis in grid.axes.items()}
    coords["Z"] = {"center": "z_l", "outer": "z_i"}
    ds = grid._ds.assign_coords({"z_l": ("z_l", np.array([5., 15.])),
                                 "z_i": ("z_i", np.array([0., 10., 20.]))})
    return xgcm.Grid(
        ds, coords=coords,
        padding={**{name: axis.padding for name, axis in grid.axes.items()}, "Z": "extend"},
        fill_value=np.nan,
        face_connections=grid._face_connections,
        autoparse_metadata=False,
    )


def assert_maps_equal(a, b):
    assert set(a) == set(b)
    for d in a:
        for x, y in zip(a[d], b[d]):
            if x is None or y is None:
                assert x is None and y is None
            else:
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("fixture", [
    two_face_x_to_x,        # 'outer' corners, no tracer centers -> _multitile_padded_maps
    left_two_tile_x_to_y,   # 'left' corners with centers        -> _OuterTopology
])
def test_vertical_axis_does_not_affect_multitile_neighbor_maps(fixture):
    """Both multi-tile neighbor-map paths padded their index arrays over *every* axis
    of the grid, so a Z axis -- which every real model grid registers -- made the grid
    untraceable. The maps must be identical with and without one."""
    grid = fixture()
    plain = build_neighbor_maps(grid, get_geo_corners(grid))
    with_z = _with_vertical_axis(grid)
    assert_maps_equal(plain, build_neighbor_maps(with_z, get_geo_corners(with_z)))
