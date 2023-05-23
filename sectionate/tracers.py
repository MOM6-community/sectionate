import numpy as np
import xarray as xr
from .transports import (
    uvindices_from_qindices
)

from .gridutils import (
    check_symmetric,
    coord_dict
)

def extract_tracer(
    name,
    grid,
    isec,
    jsec,
    sect_coord="sect"
    ):
    """
    Extract tracer data on cell thickness grid along the grid path
    of (isec, jsec) for plotting.

    PARAMETERS:
    -----------
    name:
        name of variable in `grid._ds`
    grid: xgcm.Grid
        grid describing model and containing data
    isec: int
        vorticity point indices along 'X' dimension 
    jsec: int
        vorticity point indices along 'Y' dimension
    sect_coord: str
        Name of the dimension describing along-section data in the output. Default: 'sect'.

    RETURNS:
    --------

    xarray.DataArray with data interpolated to the U and V points along the section.
    """
    
    da=grid._ds[name]
    coords = coord_dict(grid)
    symmetric = check_symmetric(grid)
    
    # get indices of UV points from broken line
    uvindices = uvindices_from_qindices(grid, isec, jsec)
        
    section = xr.Dataset()
    section["i"] = xr.DataArray(uvindices["i"], dims=sect_coord)
    section["j"] = xr.DataArray(uvindices["j"], dims=sect_coord)
    section["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=sect_coord)
    section["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=sect_coord)

    usel_left  = {coords["X"]["h"]: np.mod(section["i"]-np.int64(symmetric), da[coords["X"]["h"]].size),
                  coords["Y"]["h"]: np.mod(section["j"]                    , da[coords["Y"]["h"]].size)}
    usel_right = {coords["X"]["h"]: np.mod(section["i"]+1-np.int64(symmetric), da[coords["X"]["h"]].size),
                  coords["Y"]["h"]: np.mod(section["j"]                      , da[coords["Y"]["h"]].size)}

    vsel_left  = {coords["X"]["h"]: np.mod(section["i"]                    , da[coords["X"]["h"]].size),
                  coords["Y"]["h"]: np.mod(section["j"]-np.int64(symmetric), da[coords["Y"]["h"]].size)}
    vsel_right = {coords["X"]["h"]: np.mod(section["i"]                      , da[coords["X"]["h"]].size),
                  coords["Y"]["h"]: np.mod(section["j"]+1-np.int64(symmetric), da[coords["Y"]["h"]].size)}

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
