"""A synthetic cubed-sphere in native 'left' (MITgcm) staggering.

The cube is generated geometry-first (gnomonic faces), its `face_connections`
are *derived* from corner coincidence, and the transports come from a
streamfunction defined on the physical corner points -- so the discrete flow is
exactly non-divergent and globally seam-consistent by construction. That gives
machine-precision oracles for the hardest multi-tile machinery: rotated seams
and the cube-corner junctions where three faces meet at a point that only one
face (at most) stores.
"""

import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import outer_topology, corner_position
from sectionate.section import grid_section
from sectionate.transports import convergent_transport

Nc = 8

# cube face frames: (center, local +x unit, local +y unit) of each face.
# Each face may be rotated in-plane; the rotations are chosen (below) so that
# every cube edge glues a low face side to a high face side -- the orientation
# xgcm's `face_connections` padding handles reliably (same-side gluings, the
# `reverse=True` case, are the configuration xgcm's halos are known to get
# wrong; the native LLC layout avoids them too).
_BASE_FRAMES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),    # +X
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),   # +Y
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),  # -X
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),   # -Y
    ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),   # +Z (top)
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),   # -Z (bottom)
]


def _rotated_frames(rots):
    frames = []
    for (c, u, v), k in zip(_BASE_FRAMES, rots):
        u, v = np.array(u, dtype=float), np.array(v, dtype=float)
        for _ in range(k):        # in-plane quarter turn (preserves handedness)
            u, v = v, -u
        frames.append((np.array(c, dtype=float), u, v))
    return frames


def _find_all_low_high_rotations():
    """Per-face quarter-turns making every cube edge a low<->high gluing."""
    import itertools
    for rots in itertools.product(range(4), repeat=6):
        frames = _rotated_frames(rots)
        ok = True
        for f in range(6):
            for f2 in range(f + 1, 6):
                sides = _shared_edge_sides(frames, f, f2)
                if sides is not None and sides[0] == sides[1]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return rots
    raise AssertionError("no all-low<->high cube orientation found")


def _edge_endpoints(frames, f, which):
    c, u, v = frames[f]
    lo, hi = -1.0, 1.0
    pts = {
        "Xlo": ((c - u - v), (c - u + v)),
        "Xhi": ((c + u - v), (c + u + v)),
        "Ylo": ((c - u - v), (c + u - v)),
        "Yhi": ((c - u + v), (c + u + v)),
    }[which]
    return tuple(tuple(np.round(p / np.linalg.norm(p), 9)) for p in pts)


def _shared_edge_sides(frames, f, f2):
    """If faces f and f2 share a cube edge, return ('lo'|'hi', 'lo'|'hi')."""
    for w in ("Xlo", "Xhi", "Ylo", "Yhi"):
        e = set(_edge_endpoints(frames, f, w))
        for w2 in ("Xlo", "Xhi", "Ylo", "Yhi"):
            if e == set(_edge_endpoints(frames, f2, w2)):
                return (w[-2:], w2[-2:])
    return None


_FRAMES = _rotated_frames(_find_all_low_high_rotations())


def _face_xyz(face, a, b):
    """Gnomonic point(s) on the unit sphere for local coords a, b in [-1, 1]."""
    c, u, v = _FRAMES[face]
    p = c[:, None, None] + a[None] * u[:, None, None] + b[None] * v[:, None, None]
    return p / np.linalg.norm(p, axis=0)


def _outer_corners_xyz():
    """Exact outer corner coordinates, (6, Nc+1, Nc+1, 3)."""
    e = np.linspace(-1.0, 1.0, Nc + 1)
    a, b = np.meshgrid(e, e, indexing="xy")   # a varies along i, b along j
    out = np.empty((6, Nc + 1, Nc + 1, 3))
    for f in range(6):
        out[f] = np.moveaxis(_face_xyz(f, a, b), 0, -1)
    return out


def _lonlat(xyz):
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    return lon, lat


def _derive_face_connections(cxyz):
    """Read the cube topology off the geometry: for every face edge, find the
    coincident edge of another face and encode it in xgcm's
    (neighbor_face, neighbor_axis, reverse) convention, where `reverse` marks a
    same-side gluing (low edge to low edge, or high to high)."""
    def edge_line(f, which):
        C = cxyz[f]
        return {
            "Xlo": C[:, 0], "Xhi": C[:, Nc], "Ylo": C[0, :], "Yhi": C[Nc, :],
        }[which]

    def key(p):
        return tuple(np.round(p, 9))

    lines = {(f, w): edge_line(f, w) for f in range(6)
             for w in ("Xlo", "Xhi", "Ylo", "Yhi")}
    conns = {f: {"X": [None, None], "Y": [None, None]} for f in range(6)}
    for (f, w), L in lines.items():
        match = None
        for (f2, w2), L2 in lines.items():
            if f2 == f:
                continue
            ends, ends2 = {key(L[0]), key(L[Nc])}, {key(L2[0]), key(L2[Nc])}
            if ends == ends2:
                match = (f2, w2)
                break
        assert match is not None, f"unmatched cube edge {(f, w)}"
        f2, w2 = match
        axis2 = "X" if w2 in ("Xlo", "Xhi") else "Y"
        side = 0 if w.endswith("lo") else 1
        side2 = 0 if w2.endswith("lo") else 1
        conns[f]["X" if w.startswith("X") else "Y"][side] = (f2, axis2, side == side2)
    return {"face": {f: {ax: tuple(v) for ax, v in d.items()}
                     for f, d in conns.items()}}


def _psi(xyz):
    """A smooth streamfunction of position (arbitrary; defined on the sphere,
    so its value at a shared corner is identical from every face's frame)."""
    return (1.3 * xyz[..., 0] - 0.7 * xyz[..., 1]
            + 0.4 * xyz[..., 2] + 0.9 * xyz[..., 0] * xyz[..., 1] * xyz[..., 2])


def cube_left_grid():
    """Native 'left'-staggered cubed-sphere grid carrying an exactly
    non-divergent transport field derived from a corner streamfunction:
    every velocity face's transport is the streamfunction difference between
    its two corner endpoints, so cell convergence telescopes to zero and any
    section's transport equals the endpoint streamfunction difference."""
    cxyz = _outer_corners_xyz()
    lon_o, lat_o = _lonlat(cxyz)
    psi = _psi(cxyz)                      # (6, Nc+1, Nc+1) on outer corners

    # native 'left' corners: drop the high row/column
    lonc, latc = lon_o[:, :Nc, :Nc], lat_o[:, :Nc, :Nc]
    # centers: gnomonic midpoints
    e = np.linspace(-1.0, 1.0, Nc + 1)
    m = 0.5 * (e[:-1] + e[1:])
    am, bm = np.meshgrid(m, m, indexing="xy")
    cen = np.stack([np.moveaxis(_face_xyz(f, am, bm), 0, -1) for f in range(6)])
    lonh, lath = _lonlat(cen)

    # u through the edge from corner (j,i) to (j+1,i); v from (j,i) to (j,i+1)
    u = psi[:, 1:, :Nc] - psi[:, :-1, :Nc]         # (6, Nc, Nc) at (j, i_g)
    v = -(psi[:, :Nc, 1:] - psi[:, :Nc, :-1])      # (6, Nc, Nc) at (j_g, i)

    ds = xr.Dataset(
        {"u": (("face", "j", "i_g"), u), "v": (("face", "j_g", "i"), v)},
        coords={
            "i": ("i", np.arange(Nc)), "j": ("j", np.arange(Nc)),
            "i_g": ("i_g", np.arange(Nc)), "j_g": ("j_g", np.arange(Nc)),
            "face": ("face", np.arange(6)),
            "geolon": (("face", "j", "i"), lonh), "geolat": (("face", "j", "i"), lath),
            "geolon_c": (("face", "j_g", "i_g"), lonc),
            "geolat_c": (("face", "j_g", "i_g"), latc),
        },
    )
    fc = _derive_face_connections(cxyz)
    grid = xgcm.Grid(
        ds, coords={"X": {"center": "i", "left": "i_g"},
                    "Y": {"center": "j", "left": "j_g"}},
        boundary="fill", fill_value=np.nan,
        face_connections=fc, autoparse_metadata=False,
    )
    return grid, psi


def test_cube_topology_is_complete_and_consistent():
    """Every cube corner resolves to a coherent node. A 'left'-staggered cube
    stores 6*Nc**2 corners for 6*Nc**2 + 2 physical points, so exactly two cube
    vertices live on no face: those become edge-less nodes (no native path
    corner can exist there, so their junction edges are honest walls), their
    six seam-neighbours lose that one edge, and the six stored vertices are
    ordinary 3-valent junction nodes -- crossable, unlike under the old
    junction-walling repair."""
    grid, _ = cube_left_grid()
    assert corner_position(grid) == "left"
    ot = outer_topology(grid)
    assert (ot.node_id >= 0).all()
    deg = np.array([len(a) for a in ot.node_adj])
    native = ot.node_native[:, 0] >= 0
    assert len(deg) == 6 * Nc * Nc + 2
    assert np.count_nonzero(~native) == 2             # the two unstored vertices
    assert np.array_equal(np.where(~native)[0], np.where(deg == 0)[0])
    nreps = np.array([len(r) for r in ot.node_reps])
    stored_vertices = (deg == 3) & (nreps == 3) & native
    assert np.count_nonzero(stored_vertices) == 6     # 8 cube vertices - 2 unstored
    # neighbours of the unstored vertices lose exactly the edge into them
    assert np.count_nonzero((deg == 3) & ~stored_vertices & native) == 6
    assert np.count_nonzero(deg == 4) == len(deg) - 14


def test_cube_padded_transports_telescope_exactly():
    """Cell convergence from the outer-padded transports is exactly zero for a
    streamfunction flow, cell by cell -- including along every rotated seam and
    around the cube vertices."""
    grid, _ = cube_left_grid()
    ot = outer_topology(grid)
    Uo, Vo = ot.padded_transports(grid._ds.u, grid._ds.v)
    conv = (Uo[:, :, :-1] - Uo[:, :, 1:]) + (Vo[:, :-1, :] - Vo[:, 1:, :])
    assert np.abs(conv).max() < 1e-12


@pytest.mark.parametrize("lonlat", [
    ((10., 80.), (5., 40.)),      # crosses the top-face seams (rotated)
    ((-150., 120.), (-30., 20.)), # crosses several equatorial seams
])
def test_cube_section_transport_is_streamfunction_difference(lonlat):
    """Transport across any section equals the streamfunction difference of its
    endpoints -- across rotated seams, in either direction."""
    grid, psi = cube_left_grid()
    lons, lats = lonlat
    i_c, j_c, f_c, lons_c, lats_c = grid_section(grid, list(lons), list(lats))
    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v",
                                layer=None)["conv_mass_transport"].sum().values
    # endpoint psi values, from the walked path's own native endpoints
    ot = outer_topology(grid)
    cxyz = _outer_corners_xyz()
    psi_o = _psi(cxyz)
    p = []
    for k in (0, -1):
        f, j, i = int(f_c[k]), int(j_c[k]), int(i_c[k])
        p.append(psi_o[f, j, i])
    dpsi = p[-1] - p[0]
    assert np.isclose(abs(conv), abs(dpsi), rtol=1e-9)


def test_cube_closed_section_around_vertex_carries_zero_net_transport():
    """A closed section encircling a cube vertex (crossing three faces and their
    rotated seams; the vertex itself is stored on at most one face) must carry
    exactly zero net transport for the non-divergent streamfunction flow."""
    grid, _ = cube_left_grid()
    ot = outer_topology(grid)
    deg = np.array([len(a) for a in ot.node_adj])
    nreps = np.array([len(r) for r in ot.node_reps])
    n = int(np.where((deg == 3) & (nreps == 3))[0][0])   # a stored cube vertex
    lon0, lat0 = float(ot.node_lon[n]), float(ot.node_lat[n])

    # four waypoints on a ring around the vertex (great-circle offsets)
    ring = []
    for az in (0., 90., 180., 270.):
        r = np.deg2rad(30.)
        la0, lo0 = np.deg2rad(lat0), np.deg2rad(lon0)
        th = np.deg2rad(az)
        la = np.arcsin(np.sin(la0) * np.cos(r) + np.cos(la0) * np.sin(r) * np.cos(th))
        lo = lo0 + np.arctan2(np.sin(th) * np.sin(r) * np.cos(la0),
                              np.cos(r) - np.sin(la0) * np.sin(la))
        ring.append((np.rad2deg(lo), np.rad2deg(la)))
    ring.append(ring[0])
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]

    i_c, j_c, f_c, _, _ = grid_section(grid, lons, lats)
    assert len(set(np.asarray(f_c).tolist())) >= 3   # truly encircles the vertex
    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v",
                                layer=None)["conv_mass_transport"].sum().values
    assert abs(float(conv)) < 1e-12
