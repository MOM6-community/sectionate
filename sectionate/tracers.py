import numpy as np
import xarray as xr
from sectionate.transports import uvindices_from_qindices

def extract_tracer(
    da,
    isec,
    jsec,
    symmetric,
    dim_names={'xh':'xh', 'yh':'yh'},
    section_coord="sect",
    new_algo=False,
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

    if new_algo:
        
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
        
    else:
        def sample_pt(uvindices, i):
            return {k:v[i] for (k,v) in uvindices.items()}

        # interp onto U or V point
        def extract_1pt(da, uvindex, dim_names=dim_names):
            i, j = uvindex["i"], uvindex["j"]
            if uvindex["var"] == "U":
                interp_data = da.isel({dim_names["xh"]: slice(
                        i   - np.int64(symmetric),
                        i+2 - np.int64(symmetric)
                    ), dim_names["yh"]: j}).mean(
                    dim=[dim_names["xh"]], skipna=False
                )
            elif uvindex["var"] == "V":
                interp_data = da.isel({dim_names["yh"]: slice(
                        j   - np.int64(symmetric),
                        j+2 - np.int64(symmetric)
                    ), dim_names["xh"]: i}).mean(
                    dim=[dim_names["yh"]], skipna=False
                )
            else:
                raise ValueError("point-type can only be U or V")
            if dim_names["xh"] in interp_data.coords:
                interp_data = interp_data.reset_coords(names=dim_names["xh"], drop=True)
            if dim_names["yh"] in interp_data.coords:
                interp_data = interp_data.reset_coords(names=dim_names["yh"], drop=True)
            return interp_data.expand_dims(section_coord)

        tracer = extract_1pt(da, sample_pt(uvindices, 0))
        for i in range(1, len(uvindices['var'])):
            interp_data = extract_1pt(da, sample_pt(uvindices, i))
            # concat over new dimension
            tracer = xr.concat([tracer, interp_data], dim=section_coord)

        # transpose
        tracer = tracer.transpose(*(..., section_coord))
        # rechunk
        tracer = tracer.chunk({section_coord: len(tracer[section_coord])})

    return tracer
