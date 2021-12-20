import xarray as xr
from sectionate.transports_C import MOM6_UVpoints_from_section


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


def MOM6_extract_hydro(da, isec, jsec, xdim="xh", ydim="yh", section="sect",
                       offset_center_x=0, offset_center_y=0):
    """extract data along the broken line of (isec, jsec) for plotting

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
    section: str
        name of the produced axis for along section data. Defaults to 'sect'.
    offset_center_x: int
        offset in x-direction between center and corner points
    offset_center_y: int
        offset in y-direction between center and corner points

    RETURNS:
    --------

    xarray.DataArray with data sampled on U and V points of the section.
    """

    # get U, V points from broken line
    uvpoints = MOM6_UVpoints_from_section(isec, jsec)

    # interp onto U or V point
    def extract_1pt(da, uvpoint, xdim=xdim, ydim=ydim):
        pttype, i, j, _ = uvpoint
        if pttype == 'U':
           j = j + offset_center_y
           #interp_data = 0.5 * (da.isel({xdim:i, ydim:j}) + da.isel({xdim:i+1, ydim:j}))
           interp_data = da.isel({xdim:slice(i,i+2), ydim:j}).mean(dim=[xdim], skipna=True)
        elif pttype == 'V':
           i = i + offset_center_x
           #interp_data = 0.5 * (da.isel({xdim:i, ydim:j}) + da.isel({xdim:i, ydim:j+1}))
           interp_data = da.isel({ydim:slice(j,j+2), xdim:i}).mean(dim=[ydim], skipna=True)
        else:
           raise ValueError("point-type can only be U or V")
        if xdim in interp_data.coords:
            interp_data = interp_data.reset_coords(names=xdim, drop=True)
        if ydim in interp_data.coords:
            interp_data = interp_data.reset_coords(names=ydim, drop=True)
        return interp_data.expand_dims(section)

    hydro = extract_1pt(da, uvpoints[0])
    for pt in uvpoints[1:]:
        interp_data = extract_1pt(da, pt)
        # concat over new dimension
        hydro = xr.concat([hydro, interp_data], dim=section)

    return hydro
