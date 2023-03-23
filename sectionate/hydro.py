import xarray as xr
from sectionate.transports_C import MOM6_UVpoints_from_section

def MOM6_extract_hydro(
    da,
    isec,
    jsec,
    xdim="xh",
    ydim="yh",
    section="sect"
    ):
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

    RETURNS:
    --------

    xarray.DataArray with data sampled on U and V points of the section.
    """

    # get U, V points from broken line
    uvpoints = MOM6_UVpoints_from_section(isec, jsec)

    #
    def sample_pt(uvpoints, p):
        return {k:v[p] for (k,v) in uvpoints.items()}
    
    # interp onto U or V point
    def extract_1pt(da, uvpoint, xdim=xdim, ydim=ydim):
        i, j = uvpoint["i"], uvpoint["j"]
        if uvpoint["var"] == "U":
            interp_data = da.isel({xdim: slice(i, i + 2), ydim: j}).mean(
                dim=[xdim], skipna=True
            )
        elif uvpoint["var"] == "V":
            interp_data = da.isel({ydim: slice(j, j + 2), xdim: i}).mean(
                dim=[ydim], skipna=True
            )
        else:
            raise ValueError("point-type can only be U or V")
        if xdim in interp_data.coords:
            interp_data = interp_data.reset_coords(names=xdim, drop=True)
        if ydim in interp_data.coords:
            interp_data = interp_data.reset_coords(names=ydim, drop=True)
        return interp_data.expand_dims(section)

    hydro = extract_1pt(da, sample_pt(uvpoints, 0))
    for p in range(1, len(uvpoints['var'])):
        interp_data = extract_1pt(da, sample_pt(uvpoints, p))
        # concat over new dimension
        hydro = xr.concat([hydro, interp_data], dim=section)

    # transpose
    hydro = hydro.transpose(*(..., section))
    # rechunk
    hydro = hydro.chunk({section: len(hydro[section])})

    return hydro
