import numpy as np
import xarray as xr
from sectionate.transports import uvindices_from_qindices

def extract_tracer(
    da,
    isec,
    jsec,
    symmetric,
    dim_names={'xh':'xh', 'yh':'yh'},
    section_coord="sect"
    ):
    """extract tracer data on cell thickness grid along the broken line of (isec, jsec) for plotting

    PARAMETERS:
    -----------

    da: xarray.DataArray
        data to sample
    isec: int
        indices of i-points of broken line
    jsec: int
        indices of j-points of broken line
    xdim: str
        name of the x-dimension of tracer array. Defaults to 'xh'.
    ydim: str
        name of the y-dimension of tracer array. Defaults to 'yh'.
    section_coord: str
        name of the produced axis for along section data. Defaults to 'sect'.

    RETURNS:
    --------

    xarray.DataArray with data sampled on U and V points of the section.
    """

    # get indices of UV points from broken line
    uvindices = uvindices_from_qindices(isec, jsec, symmetric)
        
    section = xr.Dataset()
    section["i"] = xr.DataArray(uvindices["i"], dims=section_coord)
    section["j"] = xr.DataArray(uvindices["j"], dims=section_coord)
    section["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=section_coord)
    section["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=section_coord)

    usel = {dim_names["xh"]: np.mod(section["i"]-np.int64(symmetric), da[dim_names["xh"]].size),
            dim_names["yh"]: np.mod(section["j"]                    , da[dim_names["yh"]].size)}
    usel_next = {dim_names["xh"]: np.mod(section["i"]+1-np.int64(symmetric), da[dim_names["xh"]].size),
                 dim_names["yh"]: np.mod(section["j"]                      , da[dim_names["yh"]].size)}

    vsel = {dim_names["xh"]: np.mod(section["i"]                    , da[dim_names["xh"]].size),
            dim_names["yh"]: np.mod(section["j"]-np.int64(symmetric), da[dim_names["yh"]].size)}
    vsel_next = {dim_names["xh"]: np.mod(section["i"]                      , da[dim_names["xh"]].size),
                 dim_names["yh"]: np.mod(section["j"]+1-np.int64(symmetric), da[dim_names["yh"]].size)}

    tracer = (
         ( 0.5*(da.isel(usel) + da.isel(usel_next)).fillna(0.) * section["Umask"])
        +( 0.5*(da.isel(vsel) + da.isel(vsel_next)).fillna(0.) * section["Vmask"])
    )
    tracer = tracer.where(tracer!=0., np.nan)

    return tracer
