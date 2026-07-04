import numpy as np
import xarray as xr
from xgcm.padding import pad as _module_pad


def get_facedim(grid):
    """
    Return the name of the `grid`'s face/tile dimension if it has `face_connections`
    metadata (multi-tile grids such as the lat-lon-cap or cubed-sphere), else None.

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    str or None
    """
    return getattr(grid, "_facedim", None)


def corner_position(grid):
    """
    Return the C-grid vorticity ("corner") position shared by the X and Y axes:
    "outer", "right", or "left".

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    str
        One of "outer", "right", "left".
    """
    for pos in ("outer", "right", "left"):
        if (pos in grid.axes["X"].coords) and (pos in grid.axes["Y"].coords):
            return pos
    raise ValueError(
        "Only C-grids with vorticity coordinates at a shared 'outer', 'right', or "
        "'left' position on both the X and Y axes are supported."
    )


def corner_offset(grid):
    """
    Integer index shift from a vorticity ("corner") point to its staggered velocity
    point, by corner position (see `corner_position`). 'outer' grids are the baseline
    (0); 'right' grids shift by +1; 'left' grids index like 'outer' (0), differing
    only in array length and in which boundary row/column is absent.

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    int
        0 for 'outer'/'left', 1 for 'right'.
    """
    return {"outer": 0, "right": 1, "left": 0}[corner_position(grid)]


def get_geo_corners(grid):
    """
    Find longitude and latitude coordinates from grid dataset, assuming the coordinate
    names contain the sub-strings "lon" and "lat", respectively.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of longitude and latitude coordinates.
    """
    pos = corner_position(grid)
    dims = {axis: grid.axes[axis].coords[pos] for axis in ["X", "Y"]}

    coords = grid._ds.coords

    geo_coord_dict = {
        axis: [
            coords[c] for c in coords
            if (
                (geoc in c.lower()) and
                (dims["X"] in coords[c].dims) and
                (dims["Y"] in coords[c].dims)
            )
        ]
        for axis, geoc in zip(["X", "Y"], ["lon", "lat"])
    }
    if any([len(v) == 0 for (k,v) in geo_coord_dict.items()]):
        raise ValueError("""grid._ds must contain two-dimensional ("X", "Y") coordinates including the strings "lon" and "lat", consistent with grid.coords.""")
    return {k:v[0] for (k,v) in geo_coord_dict.items()}


def coord_dict(grid):
    """
    Find names of "X" and "Y" dimension variables from grid dataset.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of "X" and "Y" dimension variables, at both cell 'center'
        position and the corner position ('outer', 'right', or 'left'; see `corner_position`).
    """
    corner_pos = corner_position(grid)

    return {
        "X": {
            "center": grid.axes["X"].coords["center"],
            "corner": grid.axes["X"].coords[corner_pos]},
        "Y": {
            "center": grid.axes["Y"].coords["center"],
            "corner": grid.axes["Y"].coords[corner_pos]},
    }
    
def check_outer(grid):
    """
    Check whether the grid's shared vorticity ("corner") position is 'outer'.
    'outer' C-grids have tracers on (M,N) 'center' positions and vorticity on
    (M+1, N+1) 'outer' positions. 'right' and 'left' grids have vorticity on (M,N)
    positions; both return False here. See `corner_position` for the general case.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.

    Returns
    -------
    bool
        True if the corner position is 'outer'; False otherwise ('right' or 'left').
    """
    return corner_position(grid) == "outer"


# ---------------------------------------------------------------------------
# Topology-aware neighbor maps
#
# The section pathfinder walks corner-to-corner across the grid. Rather than
# hard-coding how to step across periodic boundaries, multi-tile face seams, or
# the bipolar north fold, we precompute -- for every corner point -- the
# ([face,] j, i) index of each of its four neighbors ("right", "left", "up",
# "down"). Walls (no neighbor, e.g. a "fill"/"extend" edge) are represented as
# the point itself, so the pathfinder simply never finds them closer to the
# target -- reproducing clip-to-edge.
#
# For single-tile grids and multi-tile grids with shared ('outer') corners, the
# topology logic lives upstream in `xgcm`: we pad index-valued arrays with the
# grid's own boundary/`face_connections` metadata and read the halos (see
# `build_neighbor_maps`).
#
# Multi-tile grids with *staggered* ('left'/'right') corners -- MITgcm/ECCO
# lat-lon-cap, cubed-sphere -- are different: each vorticity point is stored
# exactly once globally, so a single face's corner lattice is incomplete along
# two of its edges, and xgcm's halo padding of corner-position arrays can land
# one corner off across rotated/reversed seams. For these grids the neighbor
# maps are instead built from the grid's *outer* (shared-corner) lattice: each
# face is extended by its missing corner row+column, every extended slot is
# resolved to the native corner that stores that physical point (by matching
# the surrounding tracer cells, which pad reliably), and the resulting global
# corner graph -- on which face seams, cube-vertex junctions, and even the
# south-pole grid cut are ordinary nodes -- is projected back onto native
# ([face,] j, i) indices. See `_OuterTopology`.
# ---------------------------------------------------------------------------

NEIGHBOR_DIRECTIONS = ("right", "left", "up", "down")


def build_neighbor_maps(grid, geocorners):
    """
    Topology-aware neighbor maps for any grid, derived entirely from `xgcm`.

    Builds index-valued DataArrays holding each corner point's own ([face,] j, i),
    pads them by one cell with `xgcm` -- which fills the halos using the grid's
    boundary metadata, whatever it encodes: a periodic wrap, a fill/extend wall,
    the connections of a multi-tile grid (`face_connections`, with any axis
    rotation and reversal), or a bipolar north fold (`boundary={"Y": {"fold": ...}}`,
    mirror + relabel across the seam) -- then reads the halos to obtain each
    point's four neighbors. All of the intricate topology logic therefore lives
    upstream in xgcm; sectionate only reads the resulting connectivity.

    Works for both single-tile grids (corner arrays with dims (Y, X); the returned
    `fmap` is None) and multi-tile grids (dims (face, Y, X)).

    Multi-tile grids do not go through xgcm's halo padding at all: on staggered
    ('left'/'right') corner lattices padding can land one corner off across
    rotated/reversed seams, and on shared-corner ('outer') tilings it produces
    seam-twin duplicates. Their maps are instead derived from the grid's outer
    (shared-corner) corner topology, resolved to native indices -- see
    `_OuterTopology`. Each physical corner appears under a single canonical
    native index (a shared 'outer' seam corner is not stepped through twice),
    and the returned maps have the same format and native index frame.

    Parameters
    ----------
    grid: xgcm.Grid
        Any C-grid. Topology is taken from its axis `boundary`/`face_connections`.
    geocorners: dict
        Output of `get_geo_corners`; `geocorners["X"]` is the corner-position
        longitude DataArray, dims (Y, X) or (face, Y, X).

    Returns
    -------
    dict
        Maps each of `NEIGHBOR_DIRECTIONS` to (fmap, jmap, imap). For multi-tile
        grids these are int arrays of shape (nf, ny, nx); for single-tile grids
        fmap is None and jmap/imap have shape (ny, nx). Walls (unconnected/fill
        edges) are represented as the point itself.
    """
    facedim = getattr(grid, "_facedim", None)
    multitile = facedim is not None

    if multitile:
        has_centers = all("center" in grid.axes[ax].coords for ax in ("X", "Y"))
        if has_centers:
            return outer_topology(grid).maps
        # Without tracer-center coordinates the cell-identity construction is
        # unavailable (and neither are velocities, so only walking is needed):
        # fall back to reading xgcm's corner halos directly. Only shared-corner
        # ('outer') tilings are safe here -- staggered corner arrays can pad one
        # corner off across rotated/reversed seams.
        if corner_position(grid) != "outer":
            raise ValueError(
                "Multi-tile grids with staggered ('left'/'right') corners require "
                "tracer-center coordinates to derive their corner topology."
            )
        return _multitile_padded_maps(grid, geocorners)
    da = geocorners["X"]
    Ydim, Xdim = da.dims[-2], da.dims[-1]
    ny, nx = da.sizes[Ydim], da.sizes[Xdim]

    dims = (Ydim, Xdim)
    shape = (ny, nx)
    iarr = xr.DataArray(np.broadcast_to(np.arange(nx), shape).astype(float), dims=dims)
    jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], shape).astype(float), dims=dims)
    own_j = np.broadcast_to(np.arange(ny)[:, None], shape)
    own_i = np.broadcast_to(np.arange(nx), shape)

    boundary = {ax: grid.axes[ax].boundary for ax in grid.axes}
    boundary_width = {ax: (1, 1) for ax in grid.axes}

    def pad(a):
        return _module_pad(a, grid, boundary_width, boundary=boundary, fill_value=np.nan)

    pj, pi = pad(jarr), pad(iarr)

    interior = slice(1, -1)
    slices = {
        "right": (interior, slice(2, None)),
        "left":  (interior, slice(0, -2)),
        "up":    (slice(2, None), interior),
        "down":  (slice(0, -2), interior),
    }

    maps = {}
    for d, (ysl, xsl) in slices.items():
        sel = {Ydim: ysl, Xdim: xsl}
        jmap = pj.isel(sel).values
        imap = pi.isel(sel).values
        # A NaN halo means "no neighbor" (an unconnected/fill edge) -- represent
        # the wall as the point itself, matching the clip-to-edge behavior.
        wall = np.isnan(jmap) | np.isnan(imap)
        jmap = np.where(wall, own_j, jmap).astype(np.int64)
        imap = np.where(wall, own_i, imap).astype(np.int64)
        maps[d] = (None, jmap, imap)

    return maps


def _multitile_padded_maps(grid, geocorners):
    """
    Fallback multi-tile neighbor maps for shared-corner ('outer') tilings that
    carry no tracer-center coordinates: pad index-valued corner arrays with
    xgcm's `face_connections` halos and read the neighbors off directly. A
    shared seam corner is stored on every face that touches it, so these maps
    step through each face's own copy (seam-twin semantics); `_validate_reciprocity`
    refuses topologies xgcm's padding cannot represent consistently.
    """
    facedim = grid._facedim
    da = geocorners["X"]
    Ydim, Xdim = da.dims[-2], da.dims[-1]
    ny, nx = da.sizes[Ydim], da.sizes[Xdim]
    nf = da.sizes[facedim]
    dims = (facedim, Ydim, Xdim)
    shape = (nf, ny, nx)
    iarr = xr.DataArray(np.broadcast_to(np.arange(nx), shape).astype(float), dims=dims)
    jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], shape).astype(float), dims=dims)
    farr = xr.DataArray(np.broadcast_to(np.arange(nf)[:, None, None], shape).astype(float), dims=dims)
    own_f = np.broadcast_to(np.arange(nf)[:, None, None], shape)
    own_j = np.broadcast_to(np.arange(ny)[:, None], shape)
    own_i = np.broadcast_to(np.arange(nx), shape)

    boundary = {ax: grid.axes[ax].boundary for ax in grid.axes}
    boundary_width = {ax: (1, 1) for ax in grid.axes}

    def pad(a):
        return _module_pad(a, grid, boundary_width, boundary=boundary, fill_value=np.nan)

    pj, pi, pf = pad(jarr), pad(iarr), pad(farr)

    interior = slice(1, -1)
    slices = {
        "right": (interior, slice(2, None)),
        "left":  (interior, slice(0, -2)),
        "up":    (slice(2, None), interior),
        "down":  (slice(0, -2), interior),
    }

    maps = {}
    for d, (ysl, xsl) in slices.items():
        sel = {Ydim: ysl, Xdim: xsl}
        jmap = pj.isel(sel).values
        imap = pi.isel(sel).values
        fmap = pf.isel(sel).values
        wall = np.isnan(fmap) | np.isnan(jmap) | np.isnan(imap)
        fmap = np.where(wall, own_f, fmap).astype(np.int64)
        jmap = np.where(wall, own_j, jmap).astype(np.int64)
        imap = np.where(wall, own_i, imap).astype(np.int64)
        maps[d] = (fmap, jmap, imap)

    _validate_reciprocity(maps, own_f, own_j, own_i)
    return maps


def _validate_reciprocity(maps, own_f, own_j, own_i):
    """
    Verify that the neighbor maps describe a consistent topology: if B is a
    (non-wall) neighbor of A, then A must be one of B's four neighbors. An
    inconsistent map would yield silently-wrong sections, so we detect it and
    refuse rather than return garbage neighbors.
    """
    for d, (fmap, jmap, imap) in maps.items():
        same = (fmap == own_f) & (jmap == own_j) & (imap == own_i)
        not_wall = ~same
        # Gather each neighbor's own four neighbors and look for the point back.
        reciprocated = np.zeros(jmap.shape, dtype=bool)
        for (f2, j2, i2) in maps.values():
            back = (
                (f2[fmap, jmap, imap] == own_f)
                & (j2[fmap, jmap, imap] == own_j)
                & (i2[fmap, jmap, imap] == own_i)
            )
            reciprocated |= back
        if np.any(not_wall & ~reciprocated):
            raise NotImplementedError(
                "Could not derive a consistent neighbor topology from this grid's "
                "`face_connections` metadata (the multi-tile neighbor maps are not "
                "reciprocal). Sections on grids with simpler topology are supported."
            )


def outer_topology(grid):
    """
    The complete corner-point topology of a multi-tile grid, built on its
    'outer' (shared-corner) lattice and resolved to native indices. Cached on
    the grid instance (construction pads the tracer cells once and resolves
    every extended corner slot, so it is worth reusing).

    Parameters
    ----------
    grid: xgcm.Grid
        A multi-tile grid (with `face_connections`).

    Returns
    -------
    _OuterTopology
    """
    cached = getattr(grid, "_sectionate_outer_topology", None)
    if cached is None:
        cached = _OuterTopology(grid)
        grid._sectionate_outer_topology = cached
    return cached


def _lonlat_to_xyz(lon, lat):
    """Unit-sphere Cartesian coordinates of (lon, lat) in degrees."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


class _OuterTopology:
    """
    Global corner-point topology of a multi-tile grid, built on the 'outer'
    (shared-corner) lattice.

    On a staggered ('left'/'right') multi-tile grid -- MITgcm/ECCO lat-lon-cap,
    cubed-sphere -- each vorticity point is stored exactly once globally, so a
    single face's corner array is incomplete along two of its edges. Any
    topology derived by padding the per-face corner lattice therefore breaks
    down exactly where regions are hardest to trace: rotated seams, 4-face
    cube-vertex junctions, and the south-pole grid cut. This class instead
    reconstructs, per face, the full `(Nyc+1, Nxc+1)` outer lattice of corner
    *slots* and resolves each slot to the native corner that stores its
    physical point:

    1. **Cell-identity fill.** Tracer *cells* pad reliably across any seam
       (rotation or reversal included), so each slot is keyed by the up-to-four
       global cell ids around it; a slot on a face seam has the same four cells
       as its native twin on the neighbouring face, giving an exact, purely
       combinatorial identification (no coordinate tolerance, no assumptions
       about xgcm's corner-halo alignment). Cells in the *diagonally* padded
       halo are excluded -- a double pad is unreliable across two seams -- so
       face-corner slots (where a diagonal cell is always involved) are left to
       step 2.
    2. **Extrapolate + snap.** Remaining slots (face corners, and wall edges
       such as the two sides of the LLC south-pole cut) are placed by linear
       extrapolation along their corner row/column and snapped to the nearest
       native corner within a fraction of the local grid spacing -- adopting
       that corner's exact stored coordinates and identity. This is what unites
       the two coincident-but-unconnected sides of a grid cut, and pins each
       face's representation of a cube-vertex to the single native corner that
       stores it. Slots that snap to nothing (a genuine open wall) keep the
       extrapolated coordinates and no native identity.
    3. **Nodes.** Slots are merged into physical corner *nodes* by shared
       native identity and by exact physical coincidence (a unit-sphere
       position key, which also collapses the pole's degenerate longitudes).
       Node adjacency is read off the outer lattices: two nodes are neighbours
       iff some face holds index-adjacent slots for them. Every physical
       velocity face is such an edge of exactly this graph, so cube-vertex
       junctions are ordinary 4-valent (Arctic cap: 3-valent) nodes.
    4. **Native maps.** The graph is projected back onto native corner indices
       as the standard `NEIGHBOR_DIRECTIONS` maps (`build_neighbor_maps`
       format): each native corner's neighbour in a direction is the adjacent
       *node*'s native corner. In-lattice directions are direct; directions
       that leave the face's outer lattice are resolved from the node graph,
       disambiguated where necessary by the tracer cells flanking the slot
       (again purely combinatorial). Walls map to the point itself.

    The section walker and the velocity-face attribution consume these native
    maps unchanged; sections and transports keep their native `(i_c, j_c, f_c)`
    contract with no seam special-cases left over.

    Attributes
    ----------
    maps : dict
        Native-frame neighbor maps, in the `build_neighbor_maps` format.
    node_id : np.ndarray of int, shape (nf, Nyc+1, Nxc+1)
        Physical corner node of each outer slot; -1 for slots that could not be
        placed (beyond an open wall).
    node_native : np.ndarray of int, shape (n_nodes, 3)
        The native `(face, j, i)` corner storing each node; a row of -1 for
        nodes with no native representation (open-wall points).
    node_lon, node_lat : np.ndarray of float, shape (n_nodes,)
        Physical coordinates of each node.
    node_adj : list of set of int
        Undirected node adjacency (each edge is a velocity face).
    t : int
        Index offset from native corner (j, i) to its outer slot (j+t, i+t):
        0 for 'left' (missing high row/column), 1 for 'right' (missing low).
    """

    # Snap acceptance: an extrapolated slot must land within this fraction of
    # the local corner spacing of a stored native corner to adopt its identity.
    # Extrapolation error is O(curvature * spacing**2), far below this; distinct
    # corners are a full spacing apart, far above it.
    SNAP_FRACTION = 0.35

    def __init__(self, grid):
        facedim = grid._facedim
        cd = coord_dict(grid)
        Xc, Yc = cd["X"]["center"], cd["Y"]["center"]
        ds = grid._ds
        nf = ds.sizes[facedim]
        Nyc, Nxc = ds.sizes[Yc], ds.sizes[Xc]
        nyq, nxq = ds.sizes[cd["Y"]["corner"]], ds.sizes[cd["X"]["corner"]]
        pos = corner_position(grid)
        # native corner (j, i) lives at outer slot (j+t, i+t)
        t = 1 if pos == "right" else 0
        nqy, nqx = Nyc + 1, Nxc + 1
        if (t + nyq > nqy) or (t + nxq > nqx):
            raise ValueError(
                "Corner arrays are larger than the grid's outer lattice; the "
                "corner/center coordinates are inconsistent."
            )
        self.facedim, self.nf, self.t = facedim, nf, t
        self.nqy, self.nqx = nqy, nqx
        self.nyq, self.nxq = nyq, nxq

        # --- tracer-cell identities, padded one cell with the grid's own
        # topology; every non-seam boundary pads NaN so walls stay walls ---
        def _seam_or_fill(b):
            return b if b == "periodic" else "fill"
        boundary = {ax: _seam_or_fill(grid.axes[ax].boundary) for ax in grid.axes}
        cid = xr.DataArray(
            np.arange(nf * Nyc * Nxc, dtype=float).reshape(nf, Nyc, Nxc),
            dims=(facedim, Yc, Xc),
        )
        bw = {ax: (1, 1) for ax in grid.axes}
        C = _module_pad(cid, grid, bw, boundary=boundary, fill_value=np.nan)
        C = C.transpose(facedim, ..., Yc, Xc).values  # (nf, Nyc+2, Nxc+2)
        # A diagonally-padded halo cell is a pad of a pad: across two seams it
        # is not reliable, so it never participates in identity matching.
        rel = np.ones(C.shape, dtype=bool)
        rel[:, (0, -1), 0] = False
        rel[:, (0, -1), -1] = False
        self._C, self._rel = C, rel

        # the four padded cells around outer slot (f, J, I)
        cells4 = np.stack(
            [C[:, :-1, :-1], C[:, :-1, 1:], C[:, 1:, :-1], C[:, 1:, 1:]], axis=-1
        )  # (nf, nqy, nqx, 4)
        rel4 = np.stack(
            [rel[:, :-1, :-1], rel[:, :-1, 1:], rel[:, 1:, :-1], rel[:, 1:, 1:]], axis=-1
        )
        usable4 = rel4 & ~np.isnan(cells4)

        # --- native identity of each outer slot ---
        ident = np.full((nf, nqy, nqx, 3), -1, dtype=np.int64)
        jn, xn = np.meshgrid(np.arange(nyq), np.arange(nxq), indexing="ij")
        for f in range(nf):
            ident[f, t:t + nyq, t:t + nxq, 0] = f
            ident[f, t:t + nyq, t:t + nxq, 1] = jn
            ident[f, t:t + nyq, t:t + nxq, 2] = xn

        geoc = get_geo_corners(grid)
        Yq, Xq = cd["Y"]["corner"], cd["X"]["corner"]
        nat_lon = geoc["X"].transpose(facedim, Yq, Xq).values.astype(float)
        nat_lat = geoc["Y"].transpose(facedim, Yq, Xq).values.astype(float)

        # index native corners by their four surrounding cells (all usable):
        # an exact key shared with any seam twin of the same physical point.
        cellkey_to_native = {}
        for f in range(nf):
            for j in range(nyq):
                for i in range(nxq):
                    J, I = j + t, i + t
                    if not usable4[f, J, I].all():
                        continue
                    key = tuple(sorted(int(c) for c in cells4[f, J, I]))
                    cellkey_to_native.setdefault(key, []).append((f, j, i))

        # fill non-native slots that have four usable cells (seam rows/columns)
        native_mask = ident[..., 0] >= 0
        for f, J, I in zip(*np.where(~native_mask)):
            if not usable4[f, J, I].all():
                continue
            key = tuple(sorted(int(c) for c in cells4[f, J, I]))
            match = cellkey_to_native.get(key)
            if match is not None:
                ident[f, J, I] = match[0]

        # --- slot coordinates: exact stored values wherever identified ---
        xyz = np.full((nf, nqy, nqx, 3), np.nan)
        has_id = ident[..., 0] >= 0
        idf, idj, idi = (ident[..., 0][has_id], ident[..., 1][has_id], ident[..., 2][has_id])
        xyz[has_id] = _lonlat_to_xyz(nat_lon[idf, idj, idi], nat_lat[idf, idj, idi])

        # --- extrapolate + snap the remaining slots (face corners, wall cuts) ---
        nat_xyz = _lonlat_to_xyz(nat_lon, nat_lat).reshape(-1, 3)
        nat_ok = np.isfinite(nat_xyz).all(axis=1)

        def _snap(p, h):
            d = np.linalg.norm(nat_xyz - p, axis=1)
            d[~nat_ok] = np.inf
            k = int(np.argmin(d))
            if d[k] < self.SNAP_FRACTION * h:
                return np.unravel_index(k, (nf, nyq, nxq))
            return None

        # Only extrapolate from *identified* slots: two faces meeting at a wall
        # corner then extrapolate from bit-identical sources and produce
        # bit-identical positions, so their wall corners still coincide exactly.
        # A final relaxed sweep accepts unidentified (already-extrapolated)
        # sources, so isolated wall corners still receive coordinates.
        pending = [tuple(idx) for idx in zip(*np.where(~has_id))]
        relaxed = False
        for _ in range(2 * (nqy + nqx)):
            progressed = False
            still = []
            for (f, J, I) in pending:
                placed = False
                extrap = None
                for dJ, dI in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    J1, I1, J2, I2 = J + dJ, I + dI, J + 2 * dJ, I + 2 * dI
                    if not (0 <= J2 < nqy and 0 <= I2 < nqx):
                        continue
                    if not relaxed and not (
                        ident[f, J1, I1, 0] >= 0 and ident[f, J2, I2, 0] >= 0
                    ):
                        continue
                    x1, x2 = xyz[f, J1, I1], xyz[f, J2, I2]
                    if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
                        continue
                    p = 2.0 * x1 - x2
                    n = np.linalg.norm(p)
                    if n == 0.0:
                        continue
                    p = p / n
                    h = np.linalg.norm(x1 - x2)
                    if h == 0.0:
                        continue
                    hit = _snap(p, h)
                    if hit is not None:
                        ident[f, J, I] = hit
                        xyz[f, J, I] = nat_xyz[np.ravel_multi_index(hit, (nf, nyq, nxq))]
                        placed = progressed = True
                        break
                    if extrap is None:
                        extrap = p
                if not placed:
                    if extrap is not None:
                        # open wall: keep the extrapolated position, no identity
                        xyz[f, J, I] = extrap
                        progressed = True
                    else:
                        still.append((f, J, I))
            pending = still
            if not pending:
                break
            if not progressed:
                if relaxed:
                    break
                relaxed = True
        # slots still unplaced (isolated beyond walls) stay NaN / no identity

        # --- merge slots into physical nodes ---
        # union-find keyed by native identity and by exact physical coincidence
        parent = {}

        def find(a):
            root = a
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(a, a) != a:
                parent[a], a = root, parent[a]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        def slot_index(f, J, I):
            return (f * nqy + J) * nqx + I

        by_ident = {}
        by_loc = {}
        by_cells3 = {}
        finite = np.isfinite(xyz).all(axis=-1)
        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    if not finite[f, J, I]:
                        continue
                    a = slot_index(f, J, I)
                    if ident[f, J, I, 0] >= 0:
                        k = tuple(int(v) for v in ident[f, J, I])
                        if k in by_ident:
                            union(a, by_ident[k])
                        else:
                            by_ident[k] = a
                    lk = tuple(np.round(xyz[f, J, I], 9))
                    if lk in by_loc:
                        union(a, by_loc[lk])
                    else:
                        by_loc[lk] = a
                    # Two distinct corners can share at most two cells, so slots
                    # sharing three or four (their full usable set) are the same
                    # physical point. Four-cell keys unify the twin storage of
                    # shared-corner ('outer') tilings' seam corners; three-cell
                    # keys unify the representations of a 3-valent cube vertex
                    # stored on *no* face (each face's view drops only its own
                    # unreliable diagonal cell).
                    nu = usable4[f, J, I].sum()
                    if nu >= 3:
                        ck = frozenset(
                            int(c) for c, u in zip(cells4[f, J, I], usable4[f, J, I]) if u
                        )
                        if ck in by_cells3:
                            union(a, by_cells3[ck])
                        else:
                            by_cells3[ck] = a

        # Identity-less slots (points beyond an open wall, stored on no face)
        # are merged by *approximate* coincidence: where a grid cut runs
        # between two faces' missing corner lines (the LLC lon=-115 cut under
        # Antarctica), both sides extrapolate to the same physical points up to
        # extrapolation error, and merging them lets a region tracer recognise
        # the cut's two sides as the same wall. Native corners are never
        # involved (exact identification already handled them).
        loose = [
            (f, J, I)
            for f in range(nf) for J in range(nqy) for I in range(nqx)
            if finite[f, J, I] and ident[f, J, I, 0] < 0
        ]
        if loose:
            pts = np.array([xyz[f, J, I] for (f, J, I) in loose])
            # local spacing: distance to the nearest finite in-face lattice neighbor
            for k, (f, J, I) in enumerate(loose):
                h = np.inf
                for dJ, dI in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    J1, I1 = J + dJ, I + dI
                    if 0 <= J1 < nqy and 0 <= I1 < nqx and finite[f, J1, I1]:
                        h = min(h, float(np.linalg.norm(xyz[f, J, I] - xyz[f, J1, I1])))
                if not np.isfinite(h):
                    continue
                d = np.linalg.norm(pts - pts[k], axis=1)
                for k2 in np.where(d < self.SNAP_FRACTION * h)[0]:
                    if k2 != k:
                        union(slot_index(*loose[k]), slot_index(*loose[int(k2)]))

        node_of_root = {}
        node_id = np.full((nf, nqy, nqx), -1, dtype=np.int64)
        node_native = []
        node_xyz = []
        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    if not finite[f, J, I]:
                        continue
                    r = find(slot_index(f, J, I))
                    n = node_of_root.get(r)
                    if n is None:
                        n = len(node_native)
                        node_of_root[r] = n
                        node_native.append((-1, -1, -1))
                        node_xyz.append(xyz[f, J, I])
                    node_id[f, J, I] = n
                    if ident[f, J, I, 0] >= 0:
                        cand = tuple(int(v) for v in ident[f, J, I])
                        cur = node_native[n]
                        if cur[0] < 0 or cand < cur:
                            node_native[n] = cand
                            node_xyz[n] = xyz[f, J, I]
        node_native = np.array(node_native, dtype=np.int64).reshape(-1, 3)
        node_xyz = np.array(node_xyz).reshape(-1, 3)
        n_nodes = node_native.shape[0]

        # --- node adjacency: index-adjacent slots on any face share an edge ---
        # each adjacency also records the (reliable) tracer cells its edge
        # separates, used below to disambiguate off-lattice direction slots.
        # Nodes with no native identity (points beyond an open wall) carry no
        # edges: the data stores no velocity there, so for walking and
        # transport attribution those directions are genuine walls.
        adj = [set() for _ in range(n_nodes)]
        edge_cells = {}
        is_native_node = node_native[:, 0] >= 0
        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    a = node_id[f, J, I]
                    if a < 0 or not is_native_node[a]:
                        continue
                    # rightward edge: shares padded-cell column I+1, rows J, J+1
                    if I + 1 < nqx and node_id[f, J, I + 1] >= 0:
                        b = node_id[f, J, I + 1]
                        if a != b and is_native_node[b]:
                            adj[a].add(b), adj[b].add(a)
                            key = (min(a, b), max(a, b))
                            cs = edge_cells.setdefault(key, set())
                            for (jp, ip) in ((J, I + 1), (J + 1, I + 1)):
                                if rel[f, jp, ip] and not np.isnan(C[f, jp, ip]):
                                    cs.add(int(C[f, jp, ip]))
                    # upward edge: shares padded-cell row J+1, columns I, I+1
                    if J + 1 < nqy and node_id[f, J + 1, I] >= 0:
                        b = node_id[f, J + 1, I]
                        if a != b and is_native_node[b]:
                            adj[a].add(b), adj[b].add(a)
                            key = (min(a, b), max(a, b))
                            cs = edge_cells.setdefault(key, set())
                            for (jp, ip) in ((J + 1, I), (J + 1, I + 1)):
                                if rel[f, jp, ip] and not np.isnan(C[f, jp, ip]):
                                    cs.add(int(C[f, jp, ip]))
        bad = [n for n in range(n_nodes) if len(adj[n]) > 4]
        if bad:
            fc = grid._face_connections[facedim]
            has_reverse = any(
                conn is not None and conn[2]
                for face in fc.values() for sides in face.values() for conn in sides
            )
            if pos != "outer" and has_reverse:
                raise NotImplementedError(
                    f"This grid is non-symmetric ('{pos}' staggering) and its "
                    "`face_connections` include same-side ('reverse=True') gluings. On "
                    "a 'left'/'right'-staggered grid a same-side seam meets along the "
                    "dropped edge, so the shared-seam corners and their transports are "
                    "stored on NO face -- they are simply absent from the arrays, and no "
                    "conservative corner topology can recover missing data. Only "
                    "opposite-side ('reverse=False') gluings, or a symmetric ('outer') "
                    "grid that stores every seam corner and velocity face on both sides, "
                    "carry the data such a topology needs."
                )
            raise NotImplementedError(
                "Could not derive a consistent corner topology from this grid's "
                "`face_connections` metadata (a corner point acquired more than four "
                "neighbors -- e.g. a face folded onto itself, which is degenerate on "
                "any staggering). Sections on grids with simpler topology are supported."
            )

        with np.errstate(invalid="ignore"):
            self.node_lat = np.rad2deg(np.arcsin(np.clip(node_xyz[:, 2], -1., 1.)))
            self.node_lon = np.rad2deg(np.arctan2(node_xyz[:, 1], node_xyz[:, 0]))
        self.node_id = node_id
        self.node_native = node_native
        self.node_adj = adj
        self._edge_cells = edge_cells
        self._Nyc, self._Nxc = Nyc, Nxc
        node_reps = [[] for _ in range(n_nodes)]
        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    n = node_id[f, J, I]
                    if n >= 0:
                        node_reps[n].append((f, J, I))
        self.node_reps = node_reps

        # No `_validate_reciprocity` here: the maps project an *undirected* node
        # graph, so node-level reciprocity holds by construction. Index-level
        # reciprocity deliberately does not: where one physical point is stored
        # natively more than once (e.g. the LLC south-boundary fold, whose rows
        # are bit-exact reversed copies of each other), every link targets the
        # node's single canonical native rep, so the other twins receive no
        # back-links -- the walker only ever visits canonical reps.
        self.maps = self._native_maps()

    def _edge_native_velocities(self, nA, nB):
        """
        All native storages of the velocity face between corner nodes `nA` and
        `nB`. Each entry is ``(var, f, j, i, to_cell, from_cell)``: the native
        velocity index, plus the global tracer-cell ids its positive direction
        points to and from in its own face's frame (either may be None at a
        face edge). A seam velocity is stored once (on the face whose low edge
        it is); a boundary-fold edge may be stored twice.
        """
        t, Nyc, Nxc = self.t, self._Nyc, self._Nxc
        out = []
        reps_B = {}
        for (f, J, I) in self.node_reps[nB]:
            reps_B.setdefault(f, []).append((J, I))
        for (f, Ja, Ia) in self.node_reps[nA]:
            for (Jb, Ib) in reps_B.get(f, ()):
                if abs(Ja - Jb) + abs(Ia - Ib) != 1:
                    continue

                def gid(jc, ic):
                    if 0 <= jc < Nyc and 0 <= ic < Nxc:
                        return (f * Nyc + jc) * Nxc + ic
                    return None

                if Ia == Ib:  # vertical outer edge: a U (X-direction) velocity
                    jc, I = min(Ja, Jb), Ia
                    i = I - t
                    if 0 <= i < self.nxq and 0 <= jc < Nyc:
                        out.append(("U", f, jc, i, gid(jc, I), gid(jc, I - 1)))
                else:         # horizontal outer edge: a V (Y-direction) velocity
                    J, ic = Ja, min(Ia, Ib)
                    j = J - t
                    if 0 <= j < self.nyq and 0 <= ic < Nxc:
                        out.append(("V", f, j, ic, gid(J, ic), gid(J - 1, ic)))
        return out

    def padded_transports(self, u, v):
        """
        Extend native staggered transports to the full outer lattice, reading
        each missing edge slot's value from the native storage of that physical
        edge (with the sign of the receiving face's own axis direction).

        This is the topology-exact, xgcm-independent analogue of
        ``xgcm.pad(..., other_component=...)``: every added slot is resolved
        through the corner-node graph to the *stored* velocity of the same
        physical face. Unlike a halo pad it needs no vector rotation and, more
        importantly, resolves edges that are stored on *no* face (open walls,
        grid cuts, a cap's un-stored vertex) to zero -- a wall carries no
        transport -- which no halo pad can supply a value for.

        Parameters
        ----------
        u, v : xr.DataArray or np.ndarray
            Native X- and Y-direction transports, dims ([face,] Y, X) with X/Y
            at (corner, center) for `u` and (center, corner) for `v`.

        Returns
        -------
        Uo, Vo : np.ndarray
            Outer-lattice transports, shapes (nf, Nyc, Nxc+1) and
            (nf, Nyc+1, Nxc). The convergence into cell (f, j, i) is
            ``Uo[f,j,i] - Uo[f,j,i+1] + Vo[f,j,i] - Vo[f,j+1,i]``, exactly
            consistent with boundary fluxes read from the same native arrays.
        """
        t, nf, Nyc, Nxc = self.t, self.nf, self._Nyc, self._Nxc
        U = np.asarray(getattr(u, "values", u), dtype=float)
        V = np.asarray(getattr(v, "values", v), dtype=float)
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)

        C = self._C

        def cellgid(f, jc, ic):
            return (f * Nyc + jc) * Nxc + ic

        def twin_value(nA, nB, to_gid, from_gid):
            """Stored value of the edge between nodes nA/nB, signed so that its
            positive direction points from cell `from_gid` to cell `to_gid`
            (the receiving face's own axis direction). Either may be None."""
            if nA < 0 or nB < 0:
                return 0.0
            for var, f2, j2, i2, to2, from2 in self._edge_native_velocities(nA, nB):
                val = U[f2, j2, i2] if var == "U" else V[f2, j2, i2]
                if (to2 is not None and to2 == to_gid) or (from2 is not None and from2 == from_gid):
                    return val
                if (to2 is not None and to2 == from_gid) or (from2 is not None and from2 == to_gid):
                    return -val
            return 0.0

        def halo(f, jp, ip):
            g = C[f, jp, ip]
            return None if np.isnan(g) else int(g)

        Uo = np.zeros((nf, Nyc, Nxc + 1))
        Uo[:, :, t:t + self.nxq] = U
        Vo = np.zeros((nf, Nyc + 1, Nxc))
        Vo[:, t:t + self.nyq, :] = V

        for f in range(nf):
            for I in (I for I in (0, Nxc) if not (t <= I < t + self.nxq)):
                icell = I if I == 0 else I - 1  # the face's own cell beside the slot
                for jc in range(Nyc):
                    nA = self.node_id[f, jc, I]
                    nB = self.node_id[f, jc + 1, I]
                    own = cellgid(f, jc, icell)
                    hal = halo(f, jc + 1, I if I == 0 else I + 1)
                    # positive Uo direction is f's +x: at the high edge it points
                    # from the in-face cell to the halo, at the low edge reverse.
                    to_gid, from_gid = (own, hal) if I == 0 else (hal, own)
                    Uo[f, jc, I] = twin_value(nA, nB, to_gid, from_gid)
            for J in (J for J in (0, Nyc) if not (t <= J < t + self.nyq)):
                jcell = J if J == 0 else J - 1
                for ic in range(Nxc):
                    nA = self.node_id[f, J, ic]
                    nB = self.node_id[f, J, ic + 1]
                    own = cellgid(f, jcell, ic)
                    hal = halo(f, J if J == 0 else J + 1, ic + 1)
                    to_gid, from_gid = (own, hal) if J == 0 else (hal, own)
                    Vo[f, J, ic] = twin_value(nA, nB, to_gid, from_gid)
        return Uo, Vo

    def _native_maps(self):
        """Project the node graph onto native-frame neighbor maps (the
        `build_neighbor_maps` format: for each native corner and direction,
        the native corner across that edge; walls map to the point itself)."""
        nf, nyq, nxq, t = self.nf, self.nyq, self.nxq, self.t
        nqy, nqx = self.nqy, self.nqx
        node_id, node_native, adj = self.node_id, self.node_native, self.node_adj
        C, rel = self._C, self._rel
        offsets = {"right": (0, 1), "left": (0, -1), "up": (1, 0), "down": (-1, 0)}
        # padded cells flanking the edge that leaves slot (J, I) in a direction
        flank = {
            "right": ((0, 1), (1, 1)),
            "left":  ((0, 0), (1, 0)),
            "up":    ((1, 0), (1, 1)),
            "down":  ((0, 0), (0, 1)),
        }

        maps = {
            d: (
                np.broadcast_to(np.arange(nf)[:, None, None], (nf, nyq, nxq)).copy(),
                np.broadcast_to(np.arange(nyq)[:, None], (nf, nyq, nxq)).copy(),
                np.broadcast_to(np.arange(nxq), (nf, nyq, nxq)).copy(),
            )
            for d in offsets
        }

        def assign(d, f, j, i, nat):
            maps[d][0][f, j, i] = nat[0]
            maps[d][1][f, j, i] = nat[1]
            maps[d][2][f, j, i] = nat[2]

        for f in range(nf):
            for j in range(self.nyq):
                for i in range(self.nxq):
                    J, I = j + t, i + t
                    n0 = node_id[f, J, I]
                    if n0 < 0:
                        continue
                    filled = {}
                    open_dirs = []
                    for d, (dJ, dI) in offsets.items():
                        J2, I2 = J + dJ, I + dI
                        n2 = node_id[f, J2, I2] if (0 <= J2 < nqy and 0 <= I2 < nqx) else -1
                        if n2 >= 0 and n2 != n0:
                            filled[d] = n2
                        else:
                            open_dirs.append(d)
                    if open_dirs:
                        # neighbours the face's own lattice cannot see: resolve
                        # from the node graph; when two directions are open (a
                        # face-corner slot) the tracer cells flanking each
                        # direction's edge identify which neighbour is which.
                        cand = [m for m in adj[n0] if m not in filled.values()]
                        if len(cand) > len(open_dirs):
                            raise NotImplementedError(
                                "Inconsistent corner topology: a corner point has "
                                "more graph neighbors than free directions."
                            )
                        if len(cand) == 1 and len(open_dirs) == 1:
                            filled[open_dirs[0]] = cand[0]
                        elif cand:
                            for m in cand:
                                cells_m = self._edge_cells.get((min(n0, m), max(n0, m)), set())
                                homes = []
                                for d in open_dirs:
                                    fc = set()
                                    for (oj, oi) in flank[d]:
                                        jp, ip = J + oj, I + oi
                                        if rel[f, jp, ip] and not np.isnan(C[f, jp, ip]):
                                            fc.add(int(C[f, jp, ip]))
                                    if fc & cells_m:
                                        homes.append(d)
                                if len(homes) == 1:
                                    filled[homes[0]] = m
                                    open_dirs.remove(homes[0])
                                # a neighbour whose edge no local cell can see is
                                # unreachable from this face's frame: leave the
                                # slot a wall rather than guess.
                    for d, m in filled.items():
                        nat = node_native[m]
                        if nat[0] >= 0:
                            assign(d, f, j, i, nat)
        return maps
