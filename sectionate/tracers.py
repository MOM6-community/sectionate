import xarray as xr
from sectionate.transports import uvindices_from_qindices

def extract_tracer(
    da,
    isec,
    jsec,
    symmetric,
    xdim="xh",
    ydim="yh",
    section="sect"
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
    section: str
        name of the produced axis for along section data. Defaults to 'sect'.

    RETURNS:
    --------

    xarray.DataArray with data sampled on U and V points of the section.
    """

    # get indices of UV points from broken line
    uvindices = uvindices_from_qindices(isec, jsec, symmetric)

    #
    def sample_pt(uvindices, i):
        return {k:v[i] for (k,v) in uvindices.items()}
    
    # interp onto U or V point
    def extract_1pt(da, uvindex, xdim=xdim, ydim=ydim):
        i, j = uvindex["i"], uvindex["j"]
        if uvindex["var"] == "U":
            interp_data = da.isel({xdim: slice(i, i + 2), ydim: j}).mean(
                dim=[xdim], skipna=True
            )
        elif uvindex["var"] == "V":
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

    tracer = extract_1pt(da, sample_pt(uvindices, 0))
    for i in range(1, len(uvindices['var'])):
        interp_data = extract_1pt(da, sample_pt(uvindices, i))
        # concat over new dimension
        tracer = xr.concat([tracer, interp_data], dim=section)

    # transpose
    tracer = tracer.transpose(*(..., section))
    # rechunk
    tracer = tracer.chunk({section: len(tracer[section])})

    return tracer
