import numpy as np
import xarray as xr


avail_models = ["MOM6"]


def MOM6_UVpoints_from_section(isec, jsec):
    """from q points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/dev-gfdl/api/generated/pages/Horizontal_indexing.html
    """
    nsec = len(isec)
    uvpoints = []
    for k in range(1, nsec):
        if isec[k] == isec[k - 1]:
            # U-point
            if jsec[k] > jsec[k - 1]:
                point = (
                    "U",
                    isec[k],
                    jsec[k],
                    "up",
                )
            elif jsec[k] < jsec[k - 1]:
                point = (
                    "U",
                    isec[k - 1],
                    jsec[k - 1],
                    "down",
                )
            else:
                raise ValueError(
                    f"cannot find U-V point between 2 identical Q-points at i,j = {isec[k]}, {jsec[k]}"
                )
        elif jsec[k] == jsec[k - 1]:
            # V-point
            if isec[k] > isec[k - 1]:
                point = (
                    "V",
                    isec[k],
                    jsec[k],
                    "up",
                )
            elif isec[k] < isec[k - 1]:
                point = (
                    "V",
                    isec[k - 1],
                    jsec[k - 1],
                    "down",
                )
            else:
                raise ValueError(
                    f"cannot find U-V point between 2 identical Q-points at i,j = {isec[k]}, {jsec[k]}"
                )
        uvpoints.append(point)
    return uvpoints


def MOM6_UVpoints_tolonlat(uvpoints, dsgrid):
    """ get longitude/latitude of UV points """
    lons = np.array([])
    lats = np.array([])
    for point in uvpoints:
        pointtype, i, j = point
        if pointtype == "U":
            londim = "xq"
            latdim = "yh"
        elif pointtype == "V":
            londim = "xh"
            latdim = "yq"
        lon = dsgrid[londim].isel({londim: int(i)}).values
        lat = dsgrid[latdim].isel({latdim: int(j)}).values
        lons = np.append(lons, lon)
        lats = np.append(lats, lat)
    return lons, lats


def get_U_points_from_section(isec, jsec, model="MOM6"):
    """find and return indices of U points along a given broken line

    PARAMETERS:
    -----------

    isec: int
        indices of i-points of broken line
    jsec: int
        indices of j-points of broken line
    model: str
        name of the ocean model used

    RETURNS:
    --------

    xarray.Dataset with data arrays for i and j points for U points
    """

    # find U-V points along given section
    if model == "MOM6":
        uvpoints = MOM6_UVpoints_from_section(isec, jsec)
    else:
        raise NotImplementedError(f"available models are {avail_models}")

    # filter U-points
    ipts_u = []
    jpts_u = []
    for pt in uvpoints:
        if pt[0] == "U":
            ipts_u.append(pt[1])
            jpts_u.append(pt[2])

    # write results in dataset so we can later use to index data
    upts = xr.Dataset()
    upts["i_u"] = xr.DataArray(ipts_u, dims=("sect"))
    upts["j_u"] = xr.DataArray(jpts_u, dims=("sect"))

    return upts


def get_V_points_from_section(isec, jsec, model="MOM6"):
    """find and return indices of V points along a given broken line

    PARAMETERS:
    -----------

    isec: int
        indices of i-points of broken line
    jsec: int
        indices of j-points of broken line
    model: str
        name of the ocean model used

    RETURNS:
    --------

    xarray.Dataset with data arrays for i and j points for V points
    """

    # find U-V points along given section
    if model == "MOM6":
        uvpoints = MOM6_UVpoints_from_section(isec, jsec)
    else:
        raise NotImplementedError(f"available models are {avail_models}")

    # filter V-points
    ipts_v = []
    jpts_v = []
    for pt in uvpoints:
        if pt[0] == "V":
            ipts_v.append(pt[1])
            jpts_v.append(pt[2])

    # write results in dataset so we can later use to index data
    vpts = xr.Dataset()
    vpts["i_v"] = xr.DataArray(ipts_v, dims=("sect"))
    vpts["j_v"] = xr.DataArray(jpts_v, dims=("sect"))

    return vpts


def MOM6_compute_transport(ds, isec, jsec, utr="umo", vtr="vmo", vertdim="z_l"):
    """Compute the transport (MOM6)

    Args:
        ds (xarray.Dataset): dataset containing transport variables
        isec (list of int): i-indices of section broken line
        jsec (list of int): j-indices of section broken line
        utr (str, optional): Override for zonal transport. Defaults to "umo".
        vtr (str, optional): Override for merid transport. Defaults to "vmo".
        vertdim (str, optional): Vertical dimension for integration. Defaults to "z_l".

    Returns:
        xr.DataArray: transport across section
    """

    upts = get_U_points_from_section(isec, jsec, model="MOM6")
    vpts = get_V_points_from_section(isec, jsec, model="MOM6")

    Trp = ds[utr].isel(yh=upts["j_u"], xq=upts["i_u"]).sum(dim=(vertdim, "sect")) + ds[
        vtr
    ].isel(yq=vpts["j_v"], xh=vpts["i_v"]).sum(dim=(vertdim, "sect"))

    Trp.attrs.update(ds[utr].attrs)

    return Trp


def MOM6_normal_transport(
    ds,
    isec,
    jsec,
    utr="umo",
    vtr="vmo",
    layer="z_l",
    interface="z_i",
    outname="uvnormal",
):

    if layer.replace("_", " ").split()[0] != interface.replace("_", " ").split()[0]:
        raise ValueError("Inconsistent layer and interface depth variables")

    uvpoints = MOM6_UVpoints_from_section(isec, jsec)
    norm = []
    out = None

    for pt in uvpoints:
        if pt[0] == "U":

            fact = -1 if pt[3] == "up" else 1

            tmp = (
                ds[utr]
                .isel(xq=pt[1], yh=pt[2])
                .rename({"yh": "ysec", "xq": "xsec"})
                .expand_dims(dim="sect", axis=-1)
            ) * fact

            norm.append(fact)

        if pt[0] == "V":
            tmp = (
                ds[vtr]
                .isel(xh=pt[1], yq=pt[2])
                .rename({"yq": "ysec", "xh": "xsec"})
                .expand_dims(dim="sect", axis=-1)
            )
            norm.append(np.nan)
        if out is None:
            out = tmp.copy()
        else:
            out = xr.concat([out, tmp], dim="sect")

    dsout = xr.Dataset()
    dsout[outname] = out
    dsout[layer] = ds[layer]
    dsout[interface] = ds[interface]
    dsout["norm"] = xr.DataArray(norm, dims=('sect'))

    return dsout
