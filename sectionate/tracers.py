import xarray as xr
from sectionate.transports import uvindices_from_qindices

def extract_tracer(
    da,
    isec,
    jsec,
    symmetric,
    xdim="xh",
    ydim="yh",
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
        pass
#         section = xr.Dataset()
#         section["i"] = xr.DataArray(uvindices["i"], dims=section_coord)
#         section["j"] = xr.DataArray(uvindices["j"], dims=section_coord)
#         section["Usign"] = xr.DataArray(np.float64(~uvindices['nward'])*2-1, dims=section_coord)
#         section["Vsign"] = xr.DataArray(np.float64(uvindices['eward'])*2-1, dims=section_coord)
#         section["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=section_coord)
#         section["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=section_coord)

#         usel = {dim_names["yh"]: np.mod(section["j"], ds[dim_names["yh"]].size), dim_names["xq"]: section["i"]}
#         vsel = {dim_names["xh"]: np.mod(section["i"], ds[dim_names["xh"]].size), dim_names["yq"]: section["j"]}

#         dsout = xr.Dataset({outname: (
#              (ds[utr].isel(usel).fillna(0.) * section["Usign"] * section["Umask"])
#             +(ds[vtr].isel(vsel).fillna(0.) * section["Vsign"] * section["Vmask"])
#         ) * orient_fact})

#         if (cell_widths['U'] in ds.coords) and (cell_widths['V'] in ds.coords):
#             dsout = dsout.assign_coords({'dl': xr.DataArray((
#                  (ds[cell_widths['U']].isel(usel).fillna(0.) * section["Umask"])
#                 +(ds[cell_widths['V']].isel(vsel).fillna(0.) * section["Vmask"])
#             ), dims=(section_coord,), attrs={'units':'m'})})

        
    else:
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
