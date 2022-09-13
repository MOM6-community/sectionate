import numpy as np
import xarray as xr
from scipy.spatial import KDTree


avail_models = ["MOM6"]


def MOM6_UVmask_from_section(uvpoints):
    """decode section points into dataset"""

    usign = []
    ipts = []
    jpts = []
    umask = []
    vmask = []

    for pt in uvpoints:
        ipts.append(pt["i"])
        jpts.append(pt["j"])
        usign.append(-1 if (pt["nward"] == "up") else 1)
        umask.append(1 if (pt["var"] == "U") else 0)
        vmask.append(1 if (pt["var"] == "V") else 0)

    section = xr.Dataset()
    section["ipts"] = xr.DataArray(ipts, dims=("sect"))
    section["jpts"] = xr.DataArray(jpts, dims=("sect"))
    section["usign"] = xr.DataArray(usign, dims=("sect"))
    section["umask"] = xr.DataArray(umask, dims=("sect"))
    section["vmask"] = xr.DataArray(vmask, dims=("sect"))

    return section


def MOM6_UVpoints_from_section(isec, jsec):
    """from q points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html
    """
    nsec = len(isec)
    uvpoints = []
    for k in range(1, nsec):
        nward = jsec[k] > jsec[k - 1]
        eward = isec[k] > isec[k - 1]
        point = {
            'var': 'U' if (jsec[k] != jsec[k - 1]) else 'V', 
            'i': isec[k - np.int64(not(eward))],
            'j': jsec[k - np.int64(not(nward))],
            'nward': nward,
            'eward': eward,
        }
        uvpoints.append(point)
    return uvpoints


def MOM6_UVpoints_tolonlat(uvpoints, dsgrid):
    """get longitude/latitude of UV points"""
    lons = np.array([])
    lats = np.array([])
    for point in uvpoints:
        if point["var"] == "U":
            londim = "xq"
            latdim = "yh"
        elif point["var"] == "V":
            londim = "xh"
            latdim = "yq"
        lon = dsgrid[londim].isel({londim: int(point["i"])}).values
        lat = dsgrid[latdim].isel({latdim: int(point["j"])}).values
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
        if pt["var"] == "U":
            ipts_u.append(pt["i"])
            jpts_u.append(pt["j"])

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
        if pt["var"] == "V":
            ipts_v.append(pt["i"])
            jpts_v.append(pt["j"])

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
        section="sect",
        old_algo=True,
        offset_center_x=0,
        offset_center_y=0,
    ):

    if layer.replace("_", " ").split()[0] != interface.replace("_", " ").split()[0]:
        raise ValueError("Inconsistent layer and interface depth variables")

    uvpoints = MOM6_UVpoints_from_section(isec, jsec)

    if old_algo:
        norm = []
        out = None

        for pt in uvpoints:
            if pt["var"] == "U":

                fact = -1 if pt["nward"] else 1
                tmp = (
                    ds[utr]
                    .isel(xq=pt["i"], yh=pt["j"] + offset_center_y)
                    .rename({"yh": "ysec", "xq": "xsec"})
                    .expand_dims(dim="sect", axis=-1)
                ) * fact
                norm.append(fact)

            if pt["var"] == "V":
                tmp = (
                    ds[vtr]
                    .isel(xh=pt["i"] + offset_center_x, yq=pt["j"])
                    .rename({"yq": "ysec", "xh": "xsec"})
                    .expand_dims(dim=section, axis=-1)
                )
                norm.append(np.nan)
            if out is None:
                out = tmp.copy()
            else:
                out = xr.concat([out, tmp], dim=section)

        dsout = xr.Dataset()
        dsout[outname] = out
        dsout[layer] = ds[layer]
        dsout[interface] = ds[interface]
        dsout["norm"] = xr.DataArray(norm, dims=(section))

    else:

        section = MOM6_UVmask_from_section(uvpoints)

        normal_transport = (
            ds[utr].isel(yh=section["jpts"] + offset_center_y, xq=section["ipts"])
            * section["usign"]
            * section["umask"]
            + ds[vtr].isel(yq=section["jpts"], xh=section["ipts"] + offset_center_x)
            * section["vmask"]
        )

        dsout = xr.Dataset()
        dsout[outname] = normal_transport
        dsout[layer] = ds[layer]
        dsout[interface] = ds[interface]
        dsout["norm"] = section["usign"].where(section["umask"] == 1)

    return dsout


def find_offset_center_corner(
    lon_center, lat_center, lon_corner, lat_corner, debug=False
):
    """find the cell center ih,jh indexes that fall between (iq-1, jq-1) and (iq, jq) corners
    this is critical for when the grid is symetric or a subset
    """

    # transform into numpy arrays
    lon_center = (
        lon_center.values if not isinstance(lon_center, np.ndarray) else lon_center
    )
    lat_center = (
        lat_center.values if not isinstance(lat_center, np.ndarray) else lat_center
    )
    lon_corner = (
        lon_corner.values if not isinstance(lon_corner, np.ndarray) else lon_corner
    )
    lat_corner = (
        lat_corner.values if not isinstance(lat_corner, np.ndarray) else lat_corner
    )

    # define the bounds where the cell centers have to fall between
    # pick some distance from the edge so we have some leeway
    ny, nx = lon_corner.shape
    bottom = np.ceil(ny / 4).astype("int")
    left = np.ceil(nx / 4).astype("int")
    top = bottom + 1
    right = left + 1

    if debug:
        print(f"irange={left},{right}")
        print(f"jrange={bottom},{top}")

    lon_corner_bottom_left = lon_corner[bottom, left]
    lat_corner_bottom_left = lat_corner[bottom, left]
    lon_corner_bottom_right = lon_corner[bottom, right]
    lat_corner_bottom_right = lat_corner[bottom, right]
    lon_corner_top_left = lon_corner[top, left]
    lat_corner_top_left = lat_corner[top, left]
    lon_corner_top_right = lon_corner[top, right]
    lat_corner_top_right = lat_corner[top, right]

    if debug:
        print(
            f"bottom left corner is {lon_corner_bottom_left:.2f},{lat_corner_bottom_left:.2f}"
        )
        print(
            f"top right corner is {lon_corner_top_right:.2f},{lat_corner_top_right:.2f}"
        )

    # make a guess of what the lon/lat is the cell center values
    lon_center_guess = 0.25 * (
        lon_corner_bottom_left
        + lon_corner_bottom_right
        + lon_corner_top_left
        + lon_corner_top_right
    )
    lat_center_guess = 0.25 * (
        lat_corner_bottom_left
        + lat_corner_bottom_right
        + lat_corner_top_left
        + lat_corner_top_right
    )

    if debug:
        print(f"guess = {lon_center_guess},{lat_center_guess}")

    # use a KD Tree to find the closest center point to our guess
    tree = KDTree(list(zip(lon_center.ravel(), lat_center.ravel())))
    distance, ji_index = tree.query([lon_center_guess, lat_center_guess], k=1)
    jcenter, icenter = np.unravel_index(ji_index, lon_center.shape)

    if debug:
        print(f"found {icenter}, {jcenter}")

    assert lon_center[jcenter, icenter] <= lon_corner_bottom_right
    assert lon_center[jcenter, icenter] <= lon_corner_top_right
    assert lon_center[jcenter, icenter] >= lon_corner_bottom_left
    assert lon_center[jcenter, icenter] >= lon_corner_top_left

    assert lat_center[jcenter, icenter] <= lat_corner_top_left
    assert lat_center[jcenter, icenter] <= lat_corner_top_right
    assert lat_center[jcenter, icenter] >= lat_corner_bottom_left
    assert lat_center[jcenter, icenter] >= lat_corner_bottom_right

    ioffset = icenter - right
    joffset = jcenter - top

    return ioffset, joffset
