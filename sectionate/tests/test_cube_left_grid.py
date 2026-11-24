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
    its two corner endpoints, so cell convergence sums to zero and any
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
        padding="fill", fill_value=np.nan,
        face_connections=fc, autoparse_metadata=False,
    )
    return grid, psi


def test_cube_topology_is_complete_and_consistent():
    """Every cube corner resolves to a coherent node. A 'left'-staggered cube
    stores 6*Nc**2 corners for 6*Nc**2 + 2 physical points, so exactly two cube
    vertices live on no face: those become edge-less nodes (no native path
    corner can exist there, so their junction edges are genuine closed
    boundaries (walls)), their
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


def _coordinate_jittered_cube_left_grid(eps=2.0):
    """`cube_left_grid` with a rigid, per-face random shift added to the CORNER
    coordinate arrays only (the transports, cell identities and
    `face_connections` are untouched).

    A point where three faces meet but that lives on no native face is stored
    once per touching face as a corner stored on no native face whose position
    each face must *extrapolate*. On the exact gnomonic cube all three faces
    extrapolate the same coordinate, so they merge by pure geometric coincidence.
    A real curvilinear grid instead extrapolates three slightly disagreeing
    positions -- the LLC90 tiles 2/6/10 rotated three-tile corner junction is the
    archetype -- and the
    coincidence merge fails. Shifting each face's corner coordinates rigidly by a
    fraction of a cell reproduces exactly that disagreement (the shift is well
    below half a cell, so genuine seam twins still coincide and the rest of the
    topology is unchanged)."""
    grid, psi = cube_left_grid()
    ds = grid._ds.copy(deep=True)
    rng = np.random.default_rng(1)
    off = rng.uniform(-eps, eps, size=(ds.sizes["face"], 2))
    lonc, latc = ds["geolon_c"].values.copy(), ds["geolat_c"].values.copy()
    for f in range(ds.sizes["face"]):
        lonc[f] += off[f, 0]
        latc[f] += off[f, 1]
    ds["geolon_c"] = (ds["geolon_c"].dims, lonc)
    ds["geolat_c"] = (ds["geolat_c"].dims, latc)
    grid2 = xgcm.Grid(
        ds, coords={"X": {"center": "i", "left": "i_g"},
                    "Y": {"center": "j", "left": "j_g"}},
        padding="fill", fill_value=np.nan,
        face_connections={grid._facedim: grid._face_connections[grid._facedim]},
        autoparse_metadata=False,
    )
    return grid2, psi


def test_disagreeing_extrapolation_junction_merges_and_conserves():
    """Regression for a three-tile corner junction that split into separate nodes
    (as the LLC90 tiles 2/6/10 junction once did). When the three faces meeting
    at a corner stored on no native face extrapolate *disagreeing* corner
    coordinates, the coordinate merges cannot see that they are one physical
    point, so without the topological (surrounding-cell) merge the junction
    splits into separate degree-0 nodes and the stored radial velocity faces
    around it read as zero from every neighbour's frame -- introducing a spurious
    convergence into the junction cells.

    Here the exact cube's two un-stored vertices are turned into that failure
    mode by jittering each face's corner coordinates. The topological merge must
    collapse each junction's three corner slots back into a single node (so the
    topology matches the un-jittered cube: exactly two non-native nodes, each the
    union of its three faces' slots) and the outer-padded transports must again
    sum to zero cell-by-cell for the streamfunction flow.
    """
    grid, _ = _coordinate_jittered_cube_left_grid()
    ot = outer_topology(grid)
    assert (ot.node_id >= 0).all()

    # (a) the disagreeing corner slots collapse to one node per junction: exactly
    #     two non-native nodes (as on the exact cube), each merging three slots
    #     from three distinct faces -- not one split-off node per face.
    nonnative = np.where(ot.node_native[:, 0] < 0)[0]
    assert len(nonnative) == 2
    for n in nonnative:
        reps = ot.node_reps[n]
        assert len(reps) == 3
        assert len({f for (f, _, _) in reps}) == 3

    # (b) with the junction unified, convergence sums to zero everywhere
    #     (before the fix it produces a spurious O(1e-2) convergence in the
    #     junction cells).
    Uo, Vo = ot.padded_transports(grid._ds.u, grid._ds.v)
    conv = (Uo[:, :, :-1] - Uo[:, :, 1:]) + (Vo[:, :-1, :] - Vo[:, 1:, :])
    assert np.abs(conv).max() < 1e-12


def test_cube_corner_resolution_is_coordinate_jitter_independent():
    """Corner identity is resolved from tracer-cell fingerprints, not coordinates:
    the topological diagonal recovery and the 3-cell junction match place every
    seam and cube-vertex corner combinatorially. So a rigid per-face coordinate
    jitter -- which shifts each face's every corner away from its true position and
    would defeat a nearest-native-corner snap of the cross-face stored vertices --
    must leave the resolved NATIVE corner topology and the (streamfunction)
    transports exactly unchanged.

    The jitter here (eps=2.0 deg, well above 0.35 of the ~0.2-1 deg local corner
    spacing near the vertices) exceeds the snap-acceptance band, so a purely
    coordinate-based resolution would mis-place the cross-face vertex slots."""
    grid0, _ = cube_left_grid()
    ot0 = outer_topology(grid0)

    def native_nodes(ot):
        return set(tuple(int(x) for x in r) for r in ot.node_native if r[0] >= 0)

    def native_edges(ot):
        key = {i: tuple(int(x) for x in ot.node_native[i])
               for i in range(len(ot.node_native))}
        E = set()
        for i, nbrs in enumerate(ot.node_adj):
            for j in nbrs:
                if ot.node_native[i][0] >= 0 and ot.node_native[j][0] >= 0:
                    E.add(frozenset((key[i], key[j])))
        return E

    grid, _ = _coordinate_jittered_cube_left_grid(eps=2.0)
    ot = outer_topology(grid)

    # native corner topology (the part that carries transports) is unchanged
    assert native_nodes(ot) == native_nodes(ot0)
    assert native_edges(ot) == native_edges(ot0)

    # and the streamfunction flow is still exactly non-divergent cell-by-cell
    Uo, Vo = ot.padded_transports(grid._ds.u, grid._ds.v)
    conv = (Uo[:, :, :-1] - Uo[:, :, 1:]) + (Vo[:, :-1, :] - Vo[:, 1:, :])
    assert np.abs(conv).max() < 1e-12


def test_cube_padded_transports_sum_to_zero_exactly():
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
    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v")["conv_mass_transport"].sum().values
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


@pytest.mark.parametrize("lonlat,faces_expected", [
    (((0., 178.), (55., 55.)), {4}),        # confined to the +Z (polar) face
    (((0., 179.), (30., 30.)), {0, 2, 4}),  # over the pole *and* across rotated seams
])
def test_cube_section_over_pole_has_stable_geometric_sign(lonlat, faces_expected):
    """A section whose great circle passes over the geographic north pole -- the one
    soft spot of the geometric per-edge sign that Geoff flagged in review.

    ``_left_sign`` (transports.py) orients each edge by a cross product in a local
    flat (east, north) frame, which degenerates exactly at the pole: ``cos(lat) -> 0``
    shrinks the east component and longitude is ill-defined at the +Z face-center
    corner, which this cube stores right on ``lat = 90``. The pole-adjacent edges
    carry real flux, so a sign that flipped there would break the exact
    equality by a finite amount. The single-tile bipolar fold signs via
    ``Usign``/``Vsign`` instead, so this multi-tile cube (whose polar cap actually
    invokes ``_left_sign``) is where the concern must be pinned down.
    """
    grid, psi = cube_left_grid()
    (lon0, lon1), (lat0, lat1) = lonlat
    i_c, j_c, f_c, lons_c, lats_c = grid_section(grid, [lon0, lon1], [lat0, lat1])

    # The walked path really does pass *through* the pole corner (else the near-pole
    # degeneracy is never exercised and the test is vacuous), and does so in its
    # interior -- so a pole-adjacent edge exists on either side of that corner.
    at_pole = np.where(np.asarray(lats_c) > 89.9)[0]
    assert at_pole.size >= 1
    assert 0 < int(at_pole[0]) < len(lats_c) - 1
    assert set(np.asarray(f_c).tolist()) == faces_expected

    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v")["conv_mass_transport"]
    # The most pole-adjacent velocity face genuinely carries flux, so its sign is
    # not free to be wrong-yet-harmless.
    polest = int(np.argmax(np.asarray(conv["lat"].values)))
    assert abs(float(conv.values[polest])) > 1e-6

    # ...and with every edge signed (the pole-adjacent ones included), the transport
    # sums to the endpoint streamfunction difference -- so the pole sign is right.
    dpsi = (psi[int(f_c[-1]), int(j_c[-1]), int(i_c[-1])]
            - psi[int(f_c[0]), int(j_c[0]), int(i_c[0])])
    assert np.isclose(abs(float(conv.sum().values)), abs(float(dpsi)), rtol=1e-9)


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
    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v")["conv_mass_transport"].sum().values
    assert abs(float(conv)) < 1e-12


# --------------------------------------------------------------------------
# Generalized builder for the coverage gaps flagged in review: right-staggering,
# an OPEN grid (walls / grid cuts), and same-side (reverse=True) gluings. The
# geometry is identical to `cube_left_grid`; only the staggering, the optional
# face subset, and the `face_connections` differ.
# --------------------------------------------------------------------------
def _cube_variant(rots, stagger="left", faces=None, fc=None):
    frames = _rotated_frames(rots)

    def face_xyz(f, a, b):
        c, u, v = frames[f]
        p = c[:, None, None] + a[None] * u[:, None, None] + b[None] * v[:, None, None]
        return p / np.linalg.norm(p, axis=0)

    e = np.linspace(-1.0, 1.0, Nc + 1)
    a, b = np.meshgrid(e, e, indexing="xy")
    m = 0.5 * (e[:-1] + e[1:])
    am, bm = np.meshgrid(m, m, indexing="xy")
    fs = list(range(6)) if faces is None else list(faces)
    cxyz_all = np.stack([np.moveaxis(face_xyz(f, a, b), 0, -1) for f in range(6)])
    cxyz = np.stack([np.moveaxis(face_xyz(f, a, b), 0, -1) for f in fs])
    cen = np.stack([np.moveaxis(face_xyz(f, am, bm), 0, -1) for f in fs])
    lon_o, lat_o = _lonlat(cxyz)
    lonh, lath = _lonlat(cen)
    psi = _psi(cxyz)

    if stagger == "left":
        lonc, latc = lon_o[:, :Nc, :Nc], lat_o[:, :Nc, :Nc]
        u = psi[:, 1:, :Nc] - psi[:, :-1, :Nc]
        v = -(psi[:, :Nc, 1:] - psi[:, :Nc, :-1])
        cpos = {"X": {"center": "i", "left": "i_g"},
                "Y": {"center": "j", "left": "j_g"}}
    elif stagger == "right":  # the stored vorticity corner is the NE corner of each cell
        lonc, latc = lon_o[:, 1:, 1:], lat_o[:, 1:, 1:]
        u = psi[:, 1:, 1:] - psi[:, :-1, 1:]      # east face of cell (i, j)
        v = -(psi[:, 1:, 1:] - psi[:, 1:, :-1])   # north face of cell (i, j)
        cpos = {"X": {"center": "i", "right": "i_g"},
                "Y": {"center": "j", "right": "j_g"}}
    else:  # outer (symmetric): every corner stored, both boundary velocity faces kept
        lonc, latc = lon_o, lat_o                 # full (Nc+1, Nc+1) corners
        u = psi[:, 1:, :] - psi[:, :-1, :]        # (Nc, Nc+1) on (center-Y, outer-X)
        v = -(psi[:, :, 1:] - psi[:, :, :-1])     # (Nc+1, Nc) on (outer-Y, center-X)
        cpos = {"X": {"center": "i", "outer": "i_g"},
                "Y": {"center": "j", "outer": "j_g"}}

    if fc is None:
        fc = _derive_face_connections(cxyz_all)
    nig, njg = u.shape[2], v.shape[1]             # Nc (left/right) or Nc+1 (outer)
    ds = xr.Dataset(
        {"u": (("face", "j", "i_g"), u), "v": (("face", "j_g", "i"), v)},
        coords={
            "i": np.arange(Nc), "j": np.arange(Nc),
            "i_g": np.arange(nig), "j_g": np.arange(njg), "face": np.arange(len(fs)),
            "geolon": (("face", "j", "i"), lonh), "geolat": (("face", "j", "i"), lath),
            "geolon_c": (("face", "j_g", "i_g"), lonc),
            "geolat_c": (("face", "j_g", "i_g"), latc),
        },
    )
    grid = xgcm.Grid(ds, coords=cpos, padding="fill", fill_value=np.nan,
                     face_connections=fc, autoparse_metadata=False)
    return grid, psi


def test_cube_right_staggered_padded_transports_sum_to_zero():
    """Same closed cube but native 'right' (NE-corner) staggering: the low-edge
    fill path (`I==0`/`J==0`) must be exercised and convergence must still be
    exactly zero cell-by-cell for the streamfunction flow."""
    grid, _ = _cube_variant(_find_all_low_high_rotations(), stagger="right")
    assert corner_position(grid) == "right"
    ot = outer_topology(grid)
    Uo, Vo = ot.padded_transports(grid._ds.u, grid._ds.v)
    conv = (Uo[:, :, :-1] - Uo[:, :, 1:]) + (Vo[:, :-1, :] - Vo[:, 1:, :])
    assert np.abs(conv).max() < 1e-12


def test_open_wall_padded_transports_zeroes_walls_and_reads_seam_twin():
    """An OPEN two-face grid (one shared seam, walls elsewhere) exercises the
    edge-stored-on-no-face path that is otherwise only hit by real LLC90 data:
    every wall halo slot must be exactly 0 (a wall carries no transport, not a
    spurious nonzero value), while the shared-seam halo slot must read the
    neighbour's *stored* twin velocity."""
    rots = _find_all_low_high_rotations()
    full_fc = _derive_face_connections(_outer_corners_xyz())["face"]
    # Deterministically pick a face whose Y-high edge glues, aligned and
    # non-reversed, to a neighbour's Y-low edge (exists in this fixed cube). In
    # 'left' staggering only the high edge gets an added halo slot, so f0's Y-high
    # slot becomes the seam (reads the neighbour's stored low-edge v) and every
    # other added slot on the two faces is a wall.
    f0 = nbr = None
    for f, axes in full_fc.items():
        conn = axes["Y"][1]                       # Y-high connection
        if conn is not None and conn[1] == "Y" and conn[2] is False:
            f0, nbr = f, conn[0]
            break
    assert f0 is not None, "no aligned Y-high<->Y-low seam in the cube fixture"
    # Two-face grid (remapped to faces 0, 1); only the f0<->nbr Y-seam is glued.
    fc = {"face": {
        0: {"X": (None, None), "Y": (None, (1, "Y", False))},
        1: {"X": (None, None), "Y": ((0, "Y", False), None)},
    }}
    grid, _ = _cube_variant(rots, stagger="left", faces=[f0, nbr], fc=fc)
    ot = outer_topology(grid)
    Uo, Vo = ot.padded_transports(grid._ds.u, grid._ds.v)
    v = np.asarray(grid._ds.v.values)

    # 'left' staggering adds only the high slot (column/row Nc).
    assert np.all(Uo[0, :, Nc] == 0.0)             # x-high edges are walls
    assert np.all(Uo[1, :, Nc] == 0.0)
    assert np.all(Vo[1, Nc, :] == 0.0)             # face 1's y-high edge is a wall
    assert np.allclose(Vo[0, Nc, :], v[1, 0, :])   # seam reads face 1's stored twin
    assert np.any(Vo[0, Nc, :] != 0.0)             # genuinely nonzero


def test_same_side_reverse_gluing_raises():
    """A same-side (`reverse=True`) gluing leaves a whole seam line of corners
    stored on no face (degenerate on a staggered grid); `outer_topology` must
    raise an error rather than return an invalid topology."""
    import itertools

    def has_same_side_gluing(rots):
        frames = _rotated_frames(rots)
        for f, f2 in itertools.combinations(range(6), 2):
            s = _shared_edge_sides(frames, f, f2)
            if s is not None and s[0] == s[1]:
                return True
        return False

    same_side = next(r for r in itertools.product(range(4), repeat=6)
                     if has_same_side_gluing(r))
    grid, _ = _cube_variant(same_side, stagger="left")
    with pytest.raises((NotImplementedError, ValueError)):
        outer_topology(grid)


def test_outer_reverse_gluing_matches_by_coincidence():
    """On an OUTER (symmetric) grid, a same-side ('reverse=True') gluing is a
    non-issue: every seam corner and velocity face is stored on both faces as a
    coincident twin (nothing dropped), so the topology builds with no unstored
    nodes and a section crossing the reverse seams carries transport equal to the
    streamfunction difference of its endpoints -- exactly, including the twin sign
    reconciliation. This is the same base-frames cube that FAILS in 'left'
    staggering (test_same_side_reverse_gluing_raises): there the same-side seam
    meets along the dropped edge so the data is absent, whereas 'outer' stores it
    as coincident twins that merge by geometric coincidence."""
    grid, psi = _cube_variant((0, 0, 0, 0, 0, 0), stagger="outer")   # base frames: has reverse
    assert corner_position(grid) == "outer"
    fc = grid._face_connections[grid._facedim]
    assert any(c is not None and c[2]                                # really has reverse=True
               for face in fc.values() for s in face.values() for c in s)

    ot = outer_topology(grid)                                        # builds; reverse is fine on outer
    assert (ot.node_id >= 0).all()
    assert np.all(ot.node_native[:, 0] >= 0)                        # every corner stored (0 unstored)

    # a section crossing several faces (hence reverse seams): transport == dpsi
    i_c, j_c, f_c, lons, lats = grid_section(grid, [10., -150.], [40., -30.])
    assert len(set(np.asarray(f_c).tolist())) >= 3
    conv = convergent_transport(grid, i_c, j_c, f_c, utr="u", vtr="v")["conv_mass_transport"].sum().values
    p0 = psi[int(f_c[0]), int(j_c[0]), int(i_c[0])]
    p1 = psi[int(f_c[-1]), int(j_c[-1]), int(i_c[-1])]
    assert np.isclose(abs(float(conv)), abs(p1 - p0), rtol=1e-9)
