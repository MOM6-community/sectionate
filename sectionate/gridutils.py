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
    dims = {}
    for axis in ["X", "Y"]:
        if "outer" in grid.axes[axis].coords:
            dims[axis] = grid.axes[axis].coords["outer"]
        elif "right" in grid.axes[axis].coords:
            dims[axis] = grid.axes[axis].coords["right"]
        else:
            raise ValueError("Only 'symmetric' and 'non-symmetric' grids\
            are currently supported. They require C-grid topology, i.e. with\
            vorticity coordinates at 'outer' and 'right' positions, respectively.")

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
        Dictionary containing names of "X" and "Y" dimension variables, at both cell 'center' position
        and either 'outer' or 'right' position. ('left' position not yet supported.)
    """
    if check_symmetric(grid):
        corner_pos = "outer"
    else:
        corner_pos = "right"
        
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
    Check whether the horizontal ocean model grid is symmetric or not, according to MOM6 conventions.
    Symmetric C-grids have tracers on (M,N) 'center' positions and vorticity on (M+1, N+1) 'outer' positions.
    Non-symmetric C-grids instead have vorticity on (M,N) 'right' positions.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    symmetric : bool
        True if symmetric; False if non-symmetric.
        
    """
    pos_dict = {
        p : ((p in grid.axes["X"].coords) and (p in grid.axes["Y"].coords))
        for p in ["outer", "right"]
    }
    if pos_dict["outer"]:
        return True
    elif pos_dict["right"]:
        return False
    else:
        raise ValueError("Horizontal grid axes ('X', 'Y') must be either both symmetric or both non-symmetric (by MOM6 conventions).")


# ---------------------------------------------------------------------------
# Topology-aware neighbor maps
#
# The section pathfinder walks corner-to-corner across the grid. Rather than
# hard-coding how to step across periodic boundaries or the connections between
# the tiles of a multi-tile grid, we precompute, for every corner point, the
# (face, j, i) index of each of its four neighbors ("right", "left", "up",
# "down"). Walls (no neighbor, e.g. a "fill"/"extend" edge) are represented as
# the point itself, so the pathfinder simply never finds them closer to the
# target -- reproducing the previous clip-to-edge behavior.
#
#   - Single-tile grids: neighbors follow from each axis' `boundary` metadata
#     ("periodic" -> wrap, otherwise clip). Computed with plain numpy on the
#     (already-trimmed) coordinate array the pathfinder walks.
#   - Multi-tile grids (`face_connections`): edge-crossings require xgcm's
#     topology logic (axis rotation + reversal across face seams). We reuse it
#     directly by padding index-valued arrays with `xgcm` and reading the halos.
# ---------------------------------------------------------------------------

NEIGHBOR_DIRECTIONS = ("right", "left", "up", "down")


def simple_neighbor_maps(shape, boundary):
    """
    Neighbor maps for a single-tile grid, derived from `boundary` metadata.

    Parameters
    ----------
    shape: tuple of int
        (ny, nx) shape of the corner coordinate array the pathfinder walks.
    boundary: dict
        Maps grid axis ("X", "Y") to its xgcm boundary condition. "periodic"
        wraps; anything else clips to the edge (so edge points are their own
        neighbor across that boundary).

    Returns
    -------
    dict
        Maps each of `NEIGHBOR_DIRECTIONS` to (fmap, jmap, imap), where fmap is
        None (no face dimension) and jmap/imap are int arrays of shape (ny, nx)
        giving the neighbor's (j, i) for every point.
    """
    ny, nx = shape
    I = np.broadcast_to(np.arange(nx), (ny, nx))
    J = np.broadcast_to(np.arange(ny)[:, None], (ny, nx))

    def step(idx, n, b, delta):
        if b == "periodic":
            return np.mod(idx + delta, n)
        return np.clip(idx + delta, 0, n - 1)

    bx, by = boundary.get("X"), boundary.get("Y")
    return {
        "right": (None, J, step(I, nx, bx, +1)),
        "left":  (None, J, step(I, nx, bx, -1)),
        "up":    (None, step(J, ny, by, +1), I),
        "down":  (None, step(J, ny, by, -1), I),
    }


def build_neighbor_maps(grid, geocorners):
    """
    Neighbor maps for a multi-tile grid, derived from `face_connections`.

    Builds index-valued DataArrays holding each corner point's own (face, j, i),
    pads them by one cell with `xgcm` (which fills the halos using the grid's
    face connections, applying any axis rotation and reversal), then reads the
    halos to obtain each point's four neighbors. This delegates the intricate
    cross-face index translation to xgcm's tested padding logic.

    Parameters
    ----------
    grid: xgcm.Grid
        A grid with `face_connections` metadata (multi-tile).
    geocorners: dict
        Output of `get_geo_corners`; `geocorners["X"]` is the corner-position
        longitude DataArray with dims (facedim, Y, X).

    Returns
    -------
    dict
        Maps each of `NEIGHBOR_DIRECTIONS` to (fmap, jmap, imap), int arrays of
        shape (nf, ny, nx). Walls are represented as the point itself.
    """
    from xgcm.padding import pad as _module_pad

    facedim = grid._facedim
    da = geocorners["X"]
    Ydim, Xdim = da.dims[-2], da.dims[-1]
    nf, ny, nx = da.sizes[facedim], da.sizes[Ydim], da.sizes[Xdim]

    dims = (facedim, Ydim, Xdim)
    iarr = xr.DataArray(np.broadcast_to(np.arange(nx), (nf, ny, nx)).astype(float), dims=dims)
    jarr = xr.DataArray(np.broadcast_to(np.arange(ny)[:, None], (nf, ny, nx)).astype(float), dims=dims)
    farr = xr.DataArray(np.broadcast_to(np.arange(nf)[:, None, None], (nf, ny, nx)).astype(float), dims=dims)

    boundary = {ax: grid.axes[ax].boundary for ax in grid.axes}
    boundary_width = {ax: (1, 1) for ax in grid.axes}

    def pad(a):
        # Prefer the public Grid.pad (xgcm >= 0.9); fall back to the module-level
        # function in older releases that lack the method.
        if hasattr(grid, "pad"):
            return grid.pad(a, boundary_width=boundary_width, boundary=boundary, fill_value=np.nan)
        return _module_pad(a, grid, boundary_width, boundary=boundary, fill_value=np.nan)

    pf, pj, pi = pad(farr), pad(jarr), pad(iarr)

    interior = slice(1, -1)
    slices = {
        "right": (interior, slice(2, None)),
        "left":  (interior, slice(0, -2)),
        "up":    (slice(2, None), interior),
        "down":  (slice(0, -2), interior),
    }

    own_f = np.broadcast_to(np.arange(nf)[:, None, None], (nf, ny, nx))
    own_j = np.broadcast_to(np.arange(ny)[:, None], (nf, ny, nx))
    own_i = np.broadcast_to(np.arange(nx), (nf, ny, nx))

    maps = {}
    for d, (ysl, xsl) in slices.items():
        fmap = pf.isel({Ydim: ysl, Xdim: xsl}).values
        jmap = pj.isel({Ydim: ysl, Xdim: xsl}).values
        imap = pi.isel({Ydim: ysl, Xdim: xsl}).values
        # A NaN halo means "no neighbor" (an unconnected/fill edge) -- represent
        # the wall as the point itself, matching the single-tile clip behavior.
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
    (non-wall) neighbor of A, then A must be one of B's four neighbors.

    xgcm's padding can mis-fill halos for complex reversed/rotated face
    connections (its behavior is even hash-seed dependent for some
    configurations, e.g. a full cubed sphere). Such a failure would otherwise
    yield silently-wrong sections, so we detect it here and refuse rather than
    return garbage neighbors.
    """
    for d, (fmap, jmap, imap) in maps.items():
        not_wall = ~((fmap == own_f) & (jmap == own_j) & (imap == own_i))
        # Gather each neighbor's own four neighbors and look for the point back.
        reciprocated = np.zeros(fmap.shape, dtype=bool)
        for (f2, j2, i2) in maps.values():
            bf = f2[fmap, jmap, imap]
            bj = j2[fmap, jmap, imap]
            bi = i2[fmap, jmap, imap]
            reciprocated |= (bf == own_f) & (bj == own_j) & (bi == own_i)
        if np.any(not_wall & ~reciprocated):
            raise NotImplementedError(
                "Could not derive a consistent neighbor topology from this grid's "
                "`face_connections`. This is a known limitation of xgcm's padding for "
                "complex rotated/reversed connections (e.g. a full cubed sphere or the "
                "lat-lon-cap arctic cap). Sections on grids with simpler face connections "
                "are supported."
            )