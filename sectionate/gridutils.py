import itertools
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


def _pad_axes(grid, dims):
    """
    Names of the `grid` axes that `dims` spans.

    `xgcm.padding.pad` iterates the whole `padding_width` mapping it is given and
    looks each axis' position up in the array being padded, so an axis the array
    has no dimension for is an error rather than a no-op. The arrays padded in
    this module are the horizontal corner/center index arrays built a few lines
    above each call, so passing `grid.axes` wholesale makes them fail on any grid
    that registers a vertical axis -- which every real model grid does. Deriving
    the list from the array's own dims keeps padding independent of whatever
    *other* axes the grid happens to carry.

    Parameters
    ----------
    grid: xgcm.Grid
    dims: iterable of str
        Dimension names of the array about to be padded.

    Returns
    -------
    list of str
    """
    dims = set(dims)
    return [
        name for name, axis in grid.axes.items()
        if dims & set(axis.coords.values())
    ]


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
# "down"). Where there is no neighbor -- a closed domain edge, i.e. a "fill" or
# "extend" boundary -- the point is recorded as its own neighbor. A step in that
# direction therefore stays put, so it never gets the walk closer to the target
# and is never taken: the section stops at the boundary instead of running off
# the array.
#
# For single-tile grids and multi-tile grids with shared ('outer') corners, the
# topology logic lives upstream in `xgcm`: we pad index-valued arrays with the
# grid's own boundary/`face_connections` metadata and read the halos (see
# `build_neighbor_maps`).
#
# Multi-tile grids with *staggered* ('left'/'right') corners -- MITgcm/ECCO
# lat-lon-cap, cubed-sphere -- are different: a single face's corner lattice is
# incomplete along two of its edges, a physical vorticity point may be stored
# once, not at all, or more than once, and xgcm's halo padding of corner-position
# arrays can land one corner off across rotated/reversed seams. For these grids
# the neighbor maps are instead built from the grid's *outer* (shared-corner)
# lattice: each face is extended by its missing corner row+column, and every
# extended slot is resolved to the native corner that stores that physical point
# by matching the surrounding tracer cells, which pad reliably (with a small
# coordinate fallback only where too few of those cells survive to identify a
# corner at all). The resulting global corner graph -- on which face seams, tile
# junctions and grid cuts are all ordinary nodes -- is projected back onto native
# ([face,] j, i) indices. See `_OuterTopology`.
# ---------------------------------------------------------------------------

NEIGHBOR_DIRECTIONS = ("right", "left", "up", "down")


def build_neighbor_maps(grid, geocorners):
    """
    Topology-aware neighbor maps for any grid, derived entirely from `xgcm`.

    Builds index-valued DataArrays holding each corner point's own ([face,] j, i),
    pads them by one cell with `xgcm` -- which fills the halos using the grid's
    padding metadata, whatever it encodes: a periodic wrap, a fill/extend wall,
    the connections of a multi-tile grid (`face_connections`, with any axis
    rotation and reversal), or a bipolar north fold (`padding={"Y": {"fold": ...}}`,
    mirror + relabel across the seam) -- then reads the halos to obtain each
    point's four neighbors. All of the intricate topology logic therefore lives
    upstream in xgcm; sectionate only reads the resulting connectivity.

    Works for both single-tile grids (corner arrays with dims (Y, X); the returned
    `fmap` is None) and multi-tile grids (dims (face, Y, X)).

    Multi-tile grids do not go through xgcm's halo padding at all: on staggered
    ('left'/'right') corner lattices padding can land one corner off across
    rotated/reversed seams, and on shared-corner ('outer') tilings it produces
    *seam twins* -- one physical corner stored once on each of the two faces that
    share a seam, so it carries two different native indices. Their maps are
    instead derived from the grid's outer (shared-corner) corner topology,
    resolved to native indices -- see `_OuterTopology`. Each physical corner
    appears under a single canonical native index (a shared 'outer' seam corner
    is not stepped through twice), and the returned maps have the same format
    and native index frame.

    Parameters
    ----------
    grid: xgcm.Grid
        Any C-grid. Topology is taken from its axis `padding`/`face_connections`.
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

    # --- single-tile grids (no face dimension) ---
    # One corner array of dims (Y, X), whose only topology is each axis' own
    # boundary condition: a periodic wrap, a closed ("fill"/"extend") edge, or a
    # bipolar north fold. All three are what xgcm's halo padding already encodes,
    # so the neighbor maps are read straight off the padded index arrays below and
    # the returned maps carry `fmap = None`.
    da = geocorners["X"]
    Ydim, Xdim = da.dims[-2], da.dims[-1]
    ny, nx = da.sizes[Ydim], da.sizes[Xdim]

    dims = (Ydim, Xdim)
    shape = (ny, nx)
    iarr = xr.DataArray(np.broadcast_to(np.arange(nx), shape).astype(float), dims=dims)
    jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], shape).astype(float), dims=dims)
    own_j = np.broadcast_to(np.arange(ny)[:, None], shape)
    own_i = np.broadcast_to(np.arange(nx), shape)

    axes = _pad_axes(grid, dims)
    padding = {ax: grid.axes[ax].padding for ax in axes}
    padding_width = {ax: (1, 1) for ax in axes}

    def pad(a):
        return _module_pad(a, grid, padding_width, padding=padding, fill_value=np.nan)

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
        # A NaN halo means "no neighbor" (a closed, unconnected/fill edge): record
        # the point as its own neighbor, so a step in that direction stays put and
        # the walk stops at the boundary rather than running off the array.
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

    axes = _pad_axes(grid, dims)
    padding = {ax: grid.axes[ax].padding for ax in axes}
    padding_width = {ax: (1, 1) for ax in axes}

    def pad(a):
        return _module_pad(a, grid, padding_width, padding=padding, fill_value=np.nan)

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
    cubed-sphere -- a single face's corner array is incomplete along two of its
    edges, and a physical vorticity point may be stored **once** (the common
    case), **not at all** (a junction that lives on no face -- a cube vertex, or
    a point where three tiles meet), or **more than once** (a *seam twin* -- the
    same physical corner stored once on each of two faces sharing an 'outer'
    seam; an index-reversed copy across a self-folded boundary; the coincident
    lips of a *grid cut*).

    **Grid cut**: a seam along which the mesh has been slit open, so one line of
    physical corners appears twice, as the cut's two coincident "lips". Unlike a
    face seam a cut is not declared in `face_connections`, so neither lip enters
    the other's halo and only their coordinates reveal that they are the same
    points. Either lip may be stored natively or not at all, and the two need not
    agree along the cut's length. (ECCO LLC90 is slit under Antarctica along the
    65E/115W great circle, leaving reversed-index lips: tiles 0/3 against 9/12.)

    Any topology derived by padding the per-face corner lattice therefore breaks
    down exactly where regions are hardest to trace: rotated seams, cube-vertex /
    tile junctions, and grid cuts. This class instead reconstructs, per face, the
    full `(Nyc+1, Nxc+1)` outer lattice of corner *slots* and resolves each slot
    to the native corner that stores its physical point:

    1. **Cell-identity fill (topological).** Tracer *cells* pad reliably across
       any seam (rotation or reversal included), so each slot is keyed by the
       up-to-four global cell ids around it -- its *fingerprint*, the set of
       tracer cells that meet at that corner. A fingerprint is a property of the
       physical point, not of any face's indexing, so a slot on a face seam has
       the same one as its native twin on the neighbouring face, giving an exact,
       purely combinatorial identification (no coordinate tolerance, no
       assumptions about xgcm's corner-halo alignment). The four *diagonally*
       padded corner cells are pads of pads (unreliable across two seams), so
       they are first **recovered topologically** as the unique common
       edge-neighbour of the two reliably-padded cells flanking each face corner
       -- giving normal-seam face corners a full four-cell fingerprint. A genuine
       3-face junction has no such diagonal but still meets three cells at a
       unique point, so if it has a native storage it is matched by that
       three-cell fingerprint (a 3-cell key is shared by at most one native
       corner). All of this is coordinate-free.
    2. **Coordinate fallback (only for the under-determined residue).** A few
       slots are left: points stored on NO face (open walls; the two coincident
       lips of a grid cut), and slots with only two usable cells that DO have a
       native storage -- corners where the surrounding cells are themselves
       incomplete, as where a face's edge runs into a cut that reaches a pole (on
       LLC90, the corners along its polar cut). Two cells cannot fix a corner
       topologically, so these last are matched to their stored native corner by
       extrapolating the slot's position and snapping to the nearest native
       corner within a fraction of the local spacing. This is the ONLY identity
       inferred from coordinates, confined to the genuinely under-determined
       residue. Wall / cut slots get an extrapolated position but no identity;
       those that snap to nothing keep it.
    3. **Nodes.** Slots are merged into physical corner *nodes* by shared
       native identity and by exact physical coincidence (a unit-sphere
       position key, which also collapses the pole's degenerate longitudes).
       Node adjacency is read off the outer lattices: two nodes are neighbours
       iff some face holds index-adjacent slots for them. Every physical
       velocity face is such an edge of exactly this graph, so a junction where
       several faces meet is an ordinary node of the usual valence (4 at a
       cube vertex; 3 at the corners of LLC90's Arctic cap, for example).
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

    # Coordinate-fallback acceptance threshold. Native identity is resolved
    # topologically wherever the tracer-cell fingerprint determines it; only a
    # small under-determined residue (slots with two usable cells that still have
    # a native storage -- on LLC90, the corners along its polar cut) is matched by
    # coordinates, by snapping the extrapolated slot to a native corner within this
    # fraction of the local corner spacing. It is also the tolerance of the `loose`
    # merge that unites the coincident lips of a grid cut. Extrapolation/round-off
    # error is O(curvature * spacing**2), far below this; distinct corners are a
    # full spacing apart, far above it -- though this margin narrows near
    # coordinate singularities (a pole, or a cut), which is why identity is snapped
    # only where no topological fingerprint exists.
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
        # A same-side ('reverse=True') gluing on a non-symmetric ('left'/'right')
        # grid meets along the *dropped* edge, so the shared-seam corners and their
        # transports are stored on NO face -- they are simply absent from the arrays
        # and no topology can recover missing data. Reject up front with a precise
        # reason. ('outer' grids store every seam corner/edge as a twin and are
        # handled below; opposite-side gluings store the seam on the low edge.)
        if pos != "outer":
            fc = grid._face_connections[facedim]
            if any(
                conn is not None and conn[2]
                for face in fc.values() for sides in face.values() for conn in sides
            ):
                raise NotImplementedError(
                    f"This grid is non-symmetric ('{pos}' staggering) and its "
                    "`face_connections` include same-side ('reverse=True') gluings. "
                    "On a 'left'/'right'-staggered grid such a seam meets along the "
                    "dropped edge, so the shared-seam corners and their transports are "
                    "stored on NO face -- they are absent from the arrays, and no "
                    "conservative corner topology can recover missing data. Provide "
                    "the grid in 'outer' (symmetric) staggering, which stores every "
                    "seam corner and velocity face as a twin on both sides."
                )
        # native corner (j, i) lives at outer slot (j+t, i+t)
        t = 1 if pos == "right" else 0
        nqy, nqx = Nyc + 1, Nxc + 1
        # Catches a grid whose declared corner *position* contradicts the length of its
        # corner arrays -- e.g. a symmetric (N+1)-length corner array declared 'right',
        # which xgcm builds without complaint (`Axis.__init__` validates position names
        # and duplicate dims, never dim lengths) -- or centers and corners taken from
        # different grids. Without it the failure surfaces ~100 lines below as an opaque
        # NumPy broadcast error. It is one-sided: corner arrays too *small* for the
        # declared position (a 'left'-sized array declared 'outer') still fit and pass.
        if (t + nyq > nqy) or (t + nxq > nqx):
            raise ValueError(
                f"This grid's corner coordinates are {nyq} x {nxq}, too large to sit at "
                f"the index offset that '{pos}' corner staggering implies. Its {Nyc} x "
                f"{Nxc} tracer cells have {nqy} x {nqx} corners in all, and at '{pos}' "
                f"the stored corner arrays sit at index offset {t} within that grid of "
                f"corners, so they can be at most {nqy - t} x {nqx - t}. "
                "Check that the corner coordinates are declared at the xgcm position "
                "they are really stored at -- a symmetric grid, whose corner arrays are "
                "one longer than its centers along each axis, is 'outer', not 'right' "
                "or 'left' -- and that the centers and corners come from the same grid."
            )
        self.facedim, self.nf, self.t = facedim, nf, t
        self.nqy, self.nqx = nqy, nqx
        self.nyq, self.nxq = nyq, nxq

        # --- tracer-cell identities, padded one cell with the grid's own
        # topology; every non-seam boundary pads NaN so walls stay walls ---
        def _seam_or_fill(b):
            return b if b == "periodic" else "fill"
        cid = xr.DataArray(
            np.arange(nf * Nyc * Nxc, dtype=float).reshape(nf, Nyc, Nxc),
            dims=(facedim, Yc, Xc),
        )
        axes = _pad_axes(grid, cid.dims)
        padding = {ax: _seam_or_fill(grid.axes[ax].padding) for ax in axes}
        bw = {ax: (1, 1) for ax in axes}
        C = _module_pad(cid, grid, bw, padding=padding, fill_value=np.nan)
        C = C.transpose(facedim, ..., Yc, Xc).values  # (nf, Nyc+2, Nxc+2)
        # A diagonally-padded halo cell is a pad of a pad: across two seams it
        # is not reliable, so it never participates in identity matching.
        rel = np.ones(C.shape, dtype=bool)
        rel[:, (0, -1), 0] = False
        rel[:, (0, -1), -1] = False

        # --- recover each face's four diagonal halo cells topologically ---
        # The four corners of the padded array, C[:, {0,-1}, {0,-1}], are pads of
        # pads: reaching them crosses two seams, which xgcm fills unreliably, so
        # `rel` excludes them. But each is simply the tracer cell diagonally across
        # a face corner, and that cell is the *unique common edge-neighbour* of the
        # corner cell's two flanking cells -- both of which the grid's single-axis
        # pads fill reliably (one seam each). Recovering it this way is purely
        # combinatorial (no coordinates, no thresholds, no axis-swap bookkeeping),
        # so a normal-seam face corner gains a full, reliable four-cell fingerprint
        # and is resolved by the same `cellkey_to_native` machinery as every other
        # slot. A genuine 3-face junction or an open-wall corner has no such
        # diagonal cell (empty/ambiguous intersection, or a wall neighbour): it is
        # left NaN and is resolved later by the 3-cell junction match (if it has a
        # native storage) or falls through to the coordinate-free `by_junction`.
        GX = _module_pad(
            cid, grid, {ax: (1, 1) if ax == "X" else (0, 0) for ax in axes},
            padding=padding, fill_value=np.nan,
        ).transpose(facedim, ..., Yc, Xc).values           # (nf, Nyc, Nxc+2)
        GY = _module_pad(
            cid, grid, {ax: (1, 1) if ax == "Y" else (0, 0) for ax in axes},
            padding=padding, fill_value=np.nan,
        ).transpose(facedim, ..., Yc, Xc).values           # (nf, Nyc+2, Nxc)

        def _cell_edge_neighbors(g):
            """Global ids of the (up to four) tracer cells sharing an edge with
            global cell `g`, read from the reliable single-axis seam pads."""
            f_ = g // (Nyc * Nxc)
            j_ = (g % (Nyc * Nxc)) // Nxc
            i_ = g % Nxc
            out = set()
            for val in (GX[f_, j_, i_], GX[f_, j_, i_ + 2],
                        GY[f_, j_, i_], GY[f_, j_ + 2, i_]):
                if not np.isnan(val):
                    out.add(int(val))
            return out

        def _fill_diagonal(f_, Jp, Ip, jcell, icell, ga, gb):
            if np.isnan(ga) or np.isnan(gb):
                return  # a flanking neighbour is a wall: no diagonal cell exists
            g0 = (f_ * Nyc + jcell) * Nxc + icell
            cand = (_cell_edge_neighbors(int(ga)) & _cell_edge_neighbors(int(gb))) - {g0}
            # Exactly one common neighbour is the diagonal cell; an empty or
            # ambiguous intersection means a genuine 3-face junction or open wall,
            # which has no diagonal cell -- leave it NaN for the later merges.
            if len(cand) == 1:
                C[f_, Jp, Ip] = float(next(iter(cand)))
                rel[f_, Jp, Ip] = True

        for f in range(nf):
            # SW corner: diagonal C[f,0,0]; flanks = west & south of native cell (0,0)
            _fill_diagonal(f, 0, 0, 0, 0, GX[f, 0, 0], GY[f, 0, 0])
            # SE corner: diagonal C[f,0,Nxc+1]; flanks = east & south of (0,Nxc-1)
            _fill_diagonal(f, 0, Nxc + 1, 0, Nxc - 1,
                           GX[f, 0, Nxc + 1], GY[f, 0, Nxc - 1])
            # NW corner: diagonal C[f,Nyc+1,0]; flanks = west & north of (Nyc-1,0)
            _fill_diagonal(f, Nyc + 1, 0, Nyc - 1, 0,
                           GX[f, Nyc - 1, 0], GY[f, Nyc + 1, 0])
            # NE corner: diagonal C[f,Nyc+1,Nxc+1]; flanks = east & north of (Nyc-1,Nxc-1)
            _fill_diagonal(f, Nyc + 1, Nxc + 1, Nyc - 1, Nxc - 1,
                           GX[f, Nyc - 1, Nxc + 1], GY[f, Nyc + 1, Nxc - 1])

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

        # fill 3-face junction slots that HAVE a native storage but no diagonal
        # cell. At a genuine 3-tile junction three tracer cells meet at the point;
        # it is stored natively on one touching face (with three or four usable
        # cells) but appears as a face-corner slot with only three usable cells on
        # the others (its diagonal cell does not exist, so the recovery above left
        # it NaN). Three tracer cells meet at a *unique* corner, so a 3-usable-cell
        # fingerprint identifies the slot exactly -- keyed against every native
        # corner's usable-cell 3-subsets (a subset is shared by at most one native
        # corner, so any match is unambiguous). This resolves such junctions with
        # no coordinates. A junction stored on NO face (for example a cube's
        # un-stored vertex, or LLC90's un-stored polar-cut corners) matches nothing
        # here and is left for the coordinate-free `by_junction` merge below.
        cell3_to_native = {}
        for f in range(nf):
            for j in range(nyq):
                for i in range(nxq):
                    J, I = j + t, i + t
                    u = usable4[f, J, I]
                    if u.sum() < 3:
                        continue
                    cu = sorted(int(c) for c, uu in zip(cells4[f, J, I], u) if uu)
                    for sub in itertools.combinations(cu, 3):
                        cell3_to_native.setdefault(frozenset(sub), set()).add((f, j, i))
        # Unlike the `by_cells3` *node* merge below, this identity match needs no
        # geometric `coincident` guard. That guard exists because a same-side
        # ('reverse') seam can make the *padded* cell neighbourhoods of two
        # mirror-distinct corners overlap. But such a seam is only ever allowed on
        # an 'outer' grid (staggered grids with a reverse gluing raise up front,
        # see above), and on an 'outer' grid every seam corner is a 4-usable-cell
        # twin already resolved by `cellkey_to_native` -- so nothing reaches this
        # 3-usable-cell path there (verified: 0 assignments on the outer+reverse
        # cube). Where this path does run (staggered grids, no reverse gluing),
        # three tracer cells meet at a single physical corner, so the 3-cell key is
        # unambiguous; a key that maps to more than one native corner (`len > 1`,
        # which a genuine grid never produces here) is skipped rather than guessed.
        for f, J, I in zip(*np.where(ident[..., 0] < 0)):
            u = usable4[f, J, I]
            if u.sum() != 3:
                continue
            key = frozenset(int(c) for c, uu in zip(cells4[f, J, I], u) if uu)
            match = cell3_to_native.get(key)
            if match is not None and len(match) == 1:
                ident[f, J, I] = next(iter(match))

        # --- slot coordinates: exact stored values wherever identified ---
        xyz = np.full((nf, nqy, nqx, 3), np.nan)
        has_id = ident[..., 0] >= 0
        idf, idj, idi = (ident[..., 0][has_id], ident[..., 1][has_id], ident[..., 2][has_id])
        xyz[has_id] = _lonlat_to_xyz(nat_lon[idf, idj, idi], nat_lat[idf, idj, idi])

        # --- coordinate fallback + extrapolation for the remaining slots ---
        # The topological steps above (four-cell fingerprint, diagonal recovery,
        # and 3-cell junction match) resolve every slot whose native identity is
        # combinatorially determined. What can be left are:
        #   (a) slots with only TWO usable cells that DO have a native storage --
        #       corners where a face's edge runs into a cut that reaches a pole, so
        #       that the cells which would complete the fingerprint are themselves
        #       missing. (On LLC90 these are its polar-cut corners, where a face's
        #       edge wraps onto the 65E/115W great circle right at the pole: 30
        #       corners in all.) Two cells do not fix a corner, so there is no
        #       topological fingerprint; these are matched to their stored native
        #       corner by snapping the extrapolated position to the nearest native
        #       corner within a small fraction of the local spacing
        #       (`SNAP_FRACTION`). This is the only place identity is inferred from
        #       coordinates, and only for this genuinely under-determined residue.
        #   (b) points stored on NO face -- open-wall corners and the two coincident
        #       lips of a grid cut -- which get an extrapolated position (no
        #       identity) so the cut's coincident lips can later be merged by
        #       location (`by_loc`/`loose`) and every wall corner takes part in the
        #       node graph.
        nat_xyz = _lonlat_to_xyz(nat_lon, nat_lat).reshape(-1, 3)
        nat_ok = np.isfinite(nat_xyz).all(axis=1)

        # A genuine, physically distinct corner sits at least ~half a spacing away,
        # while a true coincidence lands well within SNAP_FRACTION. (Measured on
        # LLC90: the closest distinct corner is 0.5 spacings away and the farthest
        # true coincidence 0.06 -- a ~6x margin.) The band between is a "no man's
        # land" that a well-posed grid never puts a two-cell fallback corner in:
        # `_snap` returns the nearest native corner's index and, for a
        # two-usable-cell slot, its distance ratio so the caller can raise if that
        # ratio is unexpectedly in the ambiguous band (see the raise below).
        DISTINCT_FRACTION = 0.5

        def _snap(p, h):
            d = np.linalg.norm(nat_xyz - p, axis=1)
            d[~nat_ok] = np.inf
            k = int(np.argmin(d))
            ratio = (d[k] / h) if h else np.inf
            if d[k] < self.SNAP_FRACTION * h:
                return np.unravel_index(k, (nf, nyq, nxq)), ratio
            return None, ratio

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
                best_fail_ratio = np.inf
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
                    hit, ratio = _snap(p, h)
                    if hit is not None:
                        ident[f, J, I] = hit
                        xyz[f, J, I] = nat_xyz[np.ravel_multi_index(hit, (nf, nyq, nxq))]
                        placed = progressed = True
                        break
                    best_fail_ratio = min(best_fail_ratio, ratio)
                    if extrap is None:
                        extrap = p
                if not placed:
                    if extrap is not None:
                        # A two-usable-cell slot has a native storage iff it snaps
                        # (its two cells cannot fingerprint it topologically). If it
                        # instead lands in the ambiguous band -- too far to match a
                        # native corner, too close to be a distinct one -- the
                        # geometry is pathological and silently demoting it to a wall
                        # could drop real transport. Raise instead (on LLC90 this
                        # never fires: its fallback corners snap with a ~6x margin).
                        if (usable4[f, J, I].sum() == 2
                                and self.SNAP_FRACTION <= best_fail_ratio < DISTINCT_FRACTION):
                            raise ValueError(
                                "A two-cell corner could not be resolved: its nearest "
                                "native corner lies at %.2f of the local corner spacing "
                                "-- beyond the snap-acceptance band (%.2f) yet closer "
                                "than a distinct corner (%.2f). The coordinate fallback "
                                "cannot safely resolve this ambiguous geometry, so "
                                "sectionate raises rather than silently demote the corner "
                                "to a transport-less wall." % (
                                    best_fail_ratio, self.SNAP_FRACTION, DISTINCT_FRACTION)
                            )
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

        def slot_index(f, J, I):
            return (f * nqy + J) * nqx + I

        def _unslot(s):
            return s // (nqx * nqy), (s // nqx) % nqy, s % nqx

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        by_ident = {}
        by_loc = {}
        by_cells3 = {}
        finite = np.isfinite(xyz).all(axis=-1)

        # Local spacing per slot (distance to the nearest finite in-face lattice
        # neighbour). The cell-based merges below identify two slots as the same
        # physical point from shared tracer cells; across a same-side ('reverse')
        # seam the padded cell neighbourhoods of mirror-distinct seam corners can
        # overlap in >=3 cells, so guard those merges with geometric coincidence:
        # two slots more than half a cell apart are never the same point.
        hscale = np.full((nf, nqy, nqx), np.inf)
        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    if not finite[f, J, I]:
                        continue
                    for dJ, dI in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        J1, I1 = J + dJ, I + dI
                        if 0 <= J1 < nqy and 0 <= I1 < nqx and finite[f, J1, I1]:
                            hscale[f, J, I] = min(
                                hscale[f, J, I],
                                float(np.linalg.norm(xyz[f, J, I] - xyz[f, J1, I1])),
                            )

        def coincident(a, b):
            fa, Ja, Ia = _unslot(a)
            fb, Jb, Ib = _unslot(b)
            h = min(hscale[fa, Ja, Ia], hscale[fb, Jb, Ib])
            if not np.isfinite(h):
                return True  # isolated slot: cannot judge, keep prior behaviour
            return (
                float(np.linalg.norm(xyz[fa, Ja, Ia] - xyz[fb, Jb, Ib])) < 0.5 * h
            )

        for f in range(nf):
            for J in range(nqy):
                for I in range(nqx):
                    if not finite[f, J, I]:
                        continue
                    a = slot_index(f, J, I)
                    if ident[f, J, I, 0] >= 0:
                        k = tuple(int(v) for v in ident[f, J, I])
                        if k in by_ident:
                            if coincident(a, by_ident[k]):
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
                    # physical point -- provided they are also geometrically
                    # coincident (a same-side seam can make mirror-distinct corners
                    # share cells; the guard rejects those). Four-cell keys unify
                    # the twin storage of shared-corner ('outer') tilings' seam
                    # corners; three-cell keys unify the representations of a
                    # 3-valent cube vertex stored on *no* face (each face's view
                    # drops only its own unreliable diagonal cell).
                    nu = usable4[f, J, I].sum()
                    if nu >= 3:
                        ck = frozenset(
                            int(c) for c, u in zip(cells4[f, J, I], usable4[f, J, I]) if u
                        )
                        if ck in by_cells3:
                            if coincident(a, by_cells3[ck]):
                                union(a, by_cells3[ck])
                        else:
                            by_cells3[ck] = a

        # Identity-less slots (points stored on no face) at the coincident lips of
        # a grid cut are merged by *approximate* coincidence. A cut is not declared
        # in `face_connections` (see the class docstring), so neither lip's cells
        # enter the other's halo: the coincidence carries NO shared tracer-cell
        # signal and can only be seen from coordinates. A cut's two lips need not
        # be uniform along its length -- part of a lip may have a native storage
        # (and so real transport) while the rest is stored on no face at all -- and
        # only the identity-less part is merged here; the stored part was already
        # resolved above, by exact identity or by the two-cell coordinate fallback.
        #
        # LLC90 illustrates both. Its southern domain is slit under Antarctica
        # along the 65E/115W polar great circle, leaving coincident,
        # reversed-index lips. On the 115W (tiles 9/12) side:
        #  - Equatorward (lat ~ -88 to -80), tile-9 and tile-12 corners are each
        #    stored on no face. This step merges those coincident pairs, so the
        #    cut's two sides read as one wall for a region tracer. These are the
        #    degree-0 lip and carry no transport.
        #  - Poleward (lat ~ -88 to -90), the cut meets the *stored* 65E lip: the
        #    tile-9 corners there have a native storage on tile 0 (with 3/12 at the
        #    junction itself), so the two-cell coordinate fallback above already
        #    resolved them to that native corner. They become NATIVE, degree-3/4
        #    nodes spanning tiles [0,9] (and [0,3,9]/[0,9,12] at the junction) and
        #    DO carry transport -- they are not part of `loose` (which is
        #    identity-less only) and are handled once, via their native storage.
        # The 65E (tiles 0/3) lip is native throughout and was merged exactly by
        # `by_loc` above (e.g. one node with reps on both tile 0 and tile 3).
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

        # Topological (coordinate-free) merge of a shared missing corner. A point
        # where three or more faces meet but that lives on NO native face (a
        # cube's un-stored vertex; a curvilinear multi-tile junction such as the
        # LLC90 tiles 2/6/10 corner) is stored once per touching face as a
        # corner slot stored on no native face. Every one of those slots looks out
        # on the SAME ring of surrounding tracer cells -- the faces share that ring
        # by construction -- so slots whose usable-cell set is identical are the
        # same physical point and must become one node. The coincidence merges
        # above already catch this when every face extrapolates the corner to the
        # same coordinate (the exact-geometry cube), but on real curvilinear data
        # each face's extrapolated lon/lat differs by up to a fraction of a cell,
        # so the geometric snap fails and the junction splits into degree-0
        # nodes that produce a spurious convergence (the stored radial velocity
        # faces around the junction then read as zero from every neighbour's
        # frame). Keying on the
        # surrounding cells instead is immune to that extrapolation error. A
        # genuine wall corner (no velocity stored on any surrounding face) still
        # collapses to one node but acquires no edges below, so it stays a wall.
        by_junction = {}
        for (f, J, I) in loose:
            u4 = usable4[f, J, I]
            if u4.sum() < 3:
                continue
            key = frozenset(int(c) for c, u in zip(cells4[f, J, I], u4) if u)
            if key in by_junction:
                union(slot_index(f, J, I), slot_index(*by_junction[key]))
            else:
                by_junction[key] = (f, J, I)

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

        This is a topology-exact analogue of ``xgcm.pad(..., other_component=...)``:
        every added slot is resolved through the corner-node graph to the *stored*
        velocity of the same physical face. That buys three things over a vector
        halo pad:

        * It needs no vector rotation. Each slot is filled with a value already
          stored in the frame it is read in, so nothing here can be wrong about
          how a rotated or reversed seam maps `u` onto `v`.
        * It takes plain arrays, rather than the ``{axis: component}`` mapping a
          vector pad needs to know which component is which.
        * It is independent of the axis `fill_value`. An edge stored on *no*
          face -- an open wall, a lip of a grid cut, a cap's un-stored vertex --
          is resolved to 0.0, where a halo pad can only insert whatever
          `fill_value` the axis declares. Where that `fill_value` is already 0
          the two agree exactly (on ECCO LLC90 this and a dict-form vector pad
          match bit for bit, both summing cell convergence to exactly 0.0
          globally); where it is not they diverge -- on the cubed-sphere test
          fixture, whose `fill_value` is NaN, a vector pad leaves 23 of 128
          cells' convergence NaN where this returns 0.

        For an open wall 0.0 is the true transport. For a cut lip or an un-stored
        cap vertex the physical edge exists but nothing in the dataset stores its
        value, so 0.0 is the conservative choice rather than a truth: it keeps
        cell convergence exactly conservative and confines the unknown flux to
        those edges, instead of letting a `fill_value` spread through the sum.

        This is not independent of xgcm's *scalar* padding: `_OuterTopology`
        builds the node graph in the first place by padding tracer-cell ids with
        the grid's own `face_connections` (`xgcm.padding.pad`), and this method
        reads that graph. What it removes is the dependence on the vector pad --
        the part that has to rotate and re-sign components across a seam -- and
        on the pad supplying a halo at all for edges no face stores, which it
        cannot do.

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
