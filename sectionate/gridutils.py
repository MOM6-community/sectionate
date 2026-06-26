import numpy as np
import xarray as xr


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
    "outer" (symmetric), "right" (non-symmetric), or "left" (MITgcm/ECCO).

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
        "Only C-grids with vorticity coordinates at a shared 'outer' (symmetric), "
        "'right' (non-symmetric), or 'left' (MITgcm/ECCO) position on both the X and "
        "Y axes are supported."
    )


def corner_offset(grid):
    """
    Integer index shift from a vorticity ("corner") point to its staggered velocity
    point, by corner position (see `corner_position`). Symmetric ('outer') grids are
    the baseline (0); non-symmetric ('right') grids shift by +1; 'left'-staggered
    (MITgcm/ECCO) grids index like 'outer' (0), differing only in array length and
    in which boundary row/column is absent.

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
    
def check_symmetric(grid):
    """
    Check whether the horizontal ocean model grid is symmetric, according to MOM6 conventions.
    Symmetric C-grids have tracers on (M,N) 'center' positions and vorticity on (M+1, N+1)
    'outer' positions. Non-symmetric ('right') and MITgcm/ECCO ('left') grids have vorticity
    on (M,N) positions; both return False here. See `corner_position` for the general case.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.

    Returns
    -------
    symmetric : bool
        True if symmetric ('outer'); False otherwise ('right' or 'left').
    """
    return corner_position(grid) == "outer"


# ---------------------------------------------------------------------------
# Topology-aware neighbor maps
#
# The section pathfinder walks corner-to-corner across the grid. Rather than
# hard-coding how to step across periodic boundaries, multi-tile face seams, or
# the bipolar north fold, we precompute -- for every corner point -- the
# ([face,] j, i) index of each of its four neighbors ("right", "left", "up",
# "down"). All of that topology logic lives upstream in `xgcm`: we pad
# index-valued arrays with the grid's own boundary/`face_connections` metadata
# and read the halos (see `build_neighbor_maps`). Walls (no neighbor, e.g. a
# "fill"/"extend" edge) are represented as the point itself, so the pathfinder
# simply never finds them closer to the target -- reproducing clip-to-edge.
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
    from xgcm.padding import pad as _module_pad

    facedim = getattr(grid, "_facedim", None)
    multitile = facedim is not None
    da = geocorners["X"]
    Ydim, Xdim = da.dims[-2], da.dims[-1]
    ny, nx = da.sizes[Ydim], da.sizes[Xdim]

    if multitile:
        nf = da.sizes[facedim]
        dims = (facedim, Ydim, Xdim)
        shape = (nf, ny, nx)
        iarr = xr.DataArray(np.broadcast_to(np.arange(nx), shape).astype(float), dims=dims)
        jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], shape).astype(float), dims=dims)
        farr = xr.DataArray(np.broadcast_to(np.arange(nf)[:, None, None], shape).astype(float), dims=dims)
        own_f = np.broadcast_to(np.arange(nf)[:, None, None], shape)
        own_j = np.broadcast_to(np.arange(ny)[:, None], shape)
        own_i = np.broadcast_to(np.arange(nx), shape)
    else:
        dims = (Ydim, Xdim)
        shape = (ny, nx)
        iarr = xr.DataArray(np.broadcast_to(np.arange(nx), shape).astype(float), dims=dims)
        jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], shape).astype(float), dims=dims)
        farr = None
        own_f = None
        own_j = np.broadcast_to(np.arange(ny)[:, None], shape)
        own_i = np.broadcast_to(np.arange(nx), shape)

    boundary = {ax: grid.axes[ax].boundary for ax in grid.axes}
    boundary_width = {ax: (1, 1) for ax in grid.axes}

    def pad(a):
        # Prefer the public Grid.pad (xgcm >= 0.9); fall back to the module-level
        # function in older releases (and dev builds) that lack the method.
        if hasattr(grid, "pad"):
            return grid.pad(a, boundary_width=boundary_width, boundary=boundary, fill_value=np.nan)
        return _module_pad(a, grid, boundary_width, boundary=boundary, fill_value=np.nan)

    pj, pi = pad(jarr), pad(iarr)
    pf = pad(farr) if multitile else None

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
        if multitile:
            fmap = pf.isel(sel).values
            wall = np.isnan(fmap) | np.isnan(jmap) | np.isnan(imap)
            fmap = np.where(wall, own_f, fmap).astype(np.int64)
        else:
            wall = np.isnan(jmap) | np.isnan(imap)
            fmap = None
        jmap = np.where(wall, own_j, jmap).astype(np.int64)
        imap = np.where(wall, own_i, imap).astype(np.int64)
        maps[d] = (fmap, jmap, imap)

    # Validate that multi-tile `face_connections` produced a self-consistent
    # neighbor map. Single-tile padding (periodic, fill/extend, bipolar fold) is
    # well-tested upstream, and the fold deliberately identifies seam corners
    # (i <-> mirror(i)) in a way that is not index-reciprocal, so we only validate
    # multi-tile maps.
    if multitile:
        _validate_reciprocity(maps, own_f, own_j, own_i)
    return maps


def _validate_reciprocity(maps, own_f, own_j, own_i):
    """
    Verify that the neighbor maps describe a consistent topology: if B is a
    (non-wall) neighbor of A, then A must be one of B's four neighbors. An
    inconsistent map would yield silently-wrong sections, so we detect it and
    refuse rather than return garbage neighbors. Handles both single-tile
    (`own_f is None`, 2-D index maps) and multi-tile (3-D) maps.
    """
    multitile = own_f is not None

    def gather(arr, fmap, jmap, imap):
        return arr[fmap, jmap, imap] if multitile else arr[jmap, imap]

    for d, (fmap, jmap, imap) in maps.items():
        same = (jmap == own_j) & (imap == own_i)
        if multitile:
            same &= (fmap == own_f)
        not_wall = ~same
        # Gather each neighbor's own four neighbors and look for the point back.
        reciprocated = np.zeros(jmap.shape, dtype=bool)
        for (f2, j2, i2) in maps.values():
            bj = gather(j2, fmap, jmap, imap)
            bi = gather(i2, fmap, jmap, imap)
            back = (bj == own_j) & (bi == own_i)
            if multitile:
                back &= (gather(f2, fmap, jmap, imap) == own_f)
            reciprocated |= back
        if np.any(not_wall & ~reciprocated):
            raise NotImplementedError(
                "Could not derive a consistent neighbor topology from this grid's "
                "`face_connections` metadata (the multi-tile neighbor maps are not "
                "reciprocal). Sections on grids with simpler topology are supported."
            )