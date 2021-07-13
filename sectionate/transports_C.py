import numpy as np


def MOM6_UVpoints_from_section(isec, jsec):
    """ from q points given by section, infer u-v points using MOM6 conventions:
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
                )
            elif jsec[k] < jsec[k - 1]:
                point = (
                    "U",
                    isec[k],
                    jsec[k - 1],
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
                )
            elif isec[k] < isec[k - 1]:
                point = (
                    "V",
                    isec[k],
                    jsec[k - 1],
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
