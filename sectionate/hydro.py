import xarray as xr


def get_all_points_from_section(isec, jsec):
    """turn (i,j) points into xarray.dataarrays

    PARAMETERS:
    -----------

    isec: int
        indices of i-points of broken line
    jsec: int
        indices of j-points of broken line

    RETURNS:
    --------

    xarray.Dataset with data arrays for i and j points for all section points
    """

    # write results in dataset so we can later use to index data
    pts = xr.Dataset()
    pts["i"] = xr.DataArray(isec, dims=("sect"))
    pts["j"] = xr.DataArray(jsec, dims=("sect"))

    return pts


def MOM6_extract_hydro(da, isec, jsec):
    """extract data along the broken line of (isec, jsec) for plotting

    PARAMETERS:
    -----------

    da: xarray.DataArray
        data to sample
    isec: int
        indices of i-points of broken line
    jsec: int
        indices of j-points of broken line

    RETURNS:
    --------

    xarray.DataArray with data sampled at i and j points on section
    """

    idx = get_all_points_from_section(isec, jsec)
    return da.isel(yh=idx["j"], xh=idx["i"])
