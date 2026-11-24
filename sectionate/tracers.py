import numpy as np
import xarray as xr
from .transports import (
    uvindices_from_qindices
)

from .gridutils import (
    check_symmetric,
    coord_dict,
    get_geo_corners
)
from .section import fold_partner_offset

def extract_tracer(
    name,
    grid,
    i_c,
    j_c,
    sect_coord="sect",
    topology="latlon"
    ):
    """
    Extract tracer data on cell thickness grid along the grid path
    of (i_c, j_c) for plotting.

    PARAMETERS:
    -----------
    name:
        name of variable in `grid._ds`
    grid: xgcm.Grid
        grid describing model and containing data
    i_c: int
        vorticity point indices along 'X' dimension 
    j_c: int
        vorticity point indices along 'Y' dimension
    sect_coord: str
        Name of the dimension describing along-section data in the output. Default: 'sect'.
    topology: str
        Default: "latlon". Grid topology, as passed to `sectionate.grid_section` when the section was
        created. For "MOM-tripolar", the cell on the far side of a velocity point on the top row of
        vorticity points is the cell across the tripolar fold.

    RETURNS:
    --------

    xarray.DataArray with data interpolated to the U and V points along the section: the mean of the
    two tracer cells either side of each velocity point, or the value of the one cell there is if the
    other is missing (NaN) or would lie beyond a non-periodic edge of the grid.
    """
    
    da=grid._ds[name]
    coords = coord_dict(grid)
    symmetric = check_symmetric(grid)
    
    # get indices of UV points from broken line
    uvindices = uvindices_from_qindices(grid, i_c, j_c)
        
    section = xr.Dataset()
    section["i"] = xr.DataArray(uvindices["i"], dims=sect_coord)
    section["j"] = xr.DataArray(uvindices["j"], dims=sect_coord)
    section["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=sect_coord)
    section["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=sect_coord)

    Xc, Yc = coords["X"]["center"], coords["Y"]["center"]
    nx, ny = da[Xc].size, da[Yc].size
    periodic = {axis: grid.axes[axis]._boundary == "periodic" for axis in ["X", "Y"]}

    fold_offset = None
    if topology == "MOM-tripolar":
        # Top-row vorticity point i is the same point as (s - i) % nx across the fold. Vorticity points
        # i and i+1 (symmetric) or i-1 and i (non-symmetric) bound tracer cell i, so the cell across
        # the fold from top-row cell i is cell (s - 1 - i) % nx or (s + 1 - i) % nx, respectively.
        geo_corners = get_geo_corners(grid)
        gridlon, gridlat = geo_corners["X"].values, geo_corners["Y"].values
        if symmetric and periodic["X"]:
            gridlon, gridlat = gridlon[:, :-1], gridlat[:, :-1]
        fold_offset = fold_partner_offset(gridlon, gridlat) + (-1 if symmetric else 1)

    def cell(i, j):
        """Selector of the tracer cell (j, i), which may lie one cell beyond an edge of the grid."""
        if fold_offset is not None:
            across = j > ny-1
            i = xr.where(across, fold_offset - i, i)
        i = np.mod(i, nx) if periodic["X"] else np.clip(i, 0, nx-1)
        j = np.mod(j, ny) if periodic["Y"] else np.clip(j, 0, ny-1)
        return {Xc: i, Yc: j}

    increment = 1 if symmetric else 0
    usel_left  = cell(section["i"]-increment  , section["j"])
    usel_right = cell(section["i"]-increment+1, section["j"])
    vsel_left  = cell(section["i"], section["j"]-increment  )
    vsel_right = cell(section["i"], section["j"]-increment+1)

    tracer = sum([
        xr.where(~np.isnan(da.isel(usel_right)), 0.5*da.isel(usel_left),  da.isel(usel_left) ).fillna(0.) * section["Umask"],
        xr.where(~np.isnan(da.isel(usel_left )), 0.5*da.isel(usel_right), da.isel(usel_right)).fillna(0.) * section["Umask"],
        xr.where(~np.isnan(da.isel(vsel_right)), 0.5*da.isel(vsel_left),  da.isel(vsel_left) ).fillna(0.) * section["Vmask"],
        xr.where(~np.isnan(da.isel(vsel_left )), 0.5*da.isel(vsel_right), da.isel(vsel_right)).fillna(0.) * section["Vmask"],
    ])
    tracer = tracer.where(tracer!=0., np.nan)
    tracer.name = da.name
    tracer.attrs = da.attrs

    return tracer
