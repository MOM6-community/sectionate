import numpy as np
import xarray as xr

avail_models = ["MOM6"]

def MOM6_UVpoints_from_section(isec, jsec, symmetric=True):
    """from q points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html
    """
    nsec = len(isec)
    uvpoints = {'var':[], 'i':[], 'j':[], 'nward':[], 'eward':[]}
    for k in range(1, nsec):
        zonal = not(jsec[k] != jsec[k - 1])
        nward = jsec[k] > jsec[k - 1]
        eward = isec[k] > isec[k - 1]
        point = {
            'var': 'V' if zonal else 'U', 
            'i': isec[k - np.int64(not(eward))],
            'j': jsec[k - np.int64(not(nward))],
            'nward': nward,
            'eward': eward,
        }
        point['i'] -= np.int64(not(symmetric) and (point['var']=="V"))
        point['j'] -= np.int64(not(symmetric) and (point['var']=="U"))
        for (key, v) in uvpoints.items():
            v.append(point[key])
    return uvpoints

def MOM6_UVcoords_from_points_uv(gridlon_u, gridlat_u, gridlon_v, gridlat_v, uvpoints, symmetric=True):
    lons, lats = np.zeros(len(uvpoints['var'])), np.zeros(len(uvpoints['var']))
    for p in range(len(uvpoints['var'])):
        var, i, j = uvpoints['var'][p], uvpoints['i'][p], uvpoints['j'][p]
        if var == 'U':
            lon = gridlon_u.isel(xq=i, yh=j).values
            lat = gridlat_u.isel(xq=i, yh=j).values
        elif var == 'V':
            lon = gridlon_v.isel(xh=i, yq=j).values
            lat = gridlat_v.isel(xh=i, yq=j).values
        lons[p] = lon
        lats[p] = lat
    return lons, lats

def MOM6_UVcoords_from_points_hc(
    gridlon_h, gridlat_h, gridlon_c, gridlat_c, uvpoints, symmetric=True,
    xh="xh", yh="yh", xq="xq", yq="yq"
):
    lons, lats = np.zeros(len(uvpoints['var'])), np.zeros(len(uvpoints['var']))
    for p in range(len(uvpoints['var'])):
        var, i, j = uvpoints['var'][p], uvpoints['i'][p], uvpoints['j'][p]
        if var == 'U':
            lon = gridlon_c.isel({xq:i, yq:j}).values
            lat = gridlat_h.isel({xh:i, yh:j}).values
        elif var == 'V':
            lon = gridlon_h.isel({xh:i, yh:j}).values
            lat = gridlat_c.isel({xq:i, yq:j}).values
        lons[p] = lon
        lats[p] = lat
    return lons, lats

def MOM6_convergent_transport(
        ds,
        isec,
        jsec,
        utr="umo",
        vtr="vmo",
        layer="z_l",
        interface="z_i",
        outname="uvnormal",
        section="sect",
        counterclockwise=True,
        symmetric=True
    ):

    if layer.replace("_", " ").split()[0] != interface.replace("_", " ").split()[0]:
        raise ValueError("Inconsistent layer and interface depth variables")

    uvpoints = MOM6_UVpoints_from_section(isec, jsec, symmetric=symmetric)

    norm = []
    out = None

    if counterclockwise:
        orient_fact = 1.
    else:
        orient_fact = -1.
    
    for p in range(len(uvpoints['var'])):
        pt = {k:v[p] for (k,v) in uvpoints.items()}
        if pt["var"] == "U":
            fact = -1 if pt["nward"] else 1
            tmp = (
                ds[utr]
                .isel(xq=pt["i"], yh=pt["j"]).fillna(0.)
                .rename({"yh": "ysec", "xq": "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact * orient_fact
            norm.append(fact)

        if pt["var"] == "V":
            fact = -1 if not(pt["eward"]) else 1
            tmp = (
                ds[vtr]
                .isel(xh=pt["i"], yq=pt["j"]).fillna(0.)
                .rename({"yq": "ysec", "xh": "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact * orient_fact
            norm.append(fact)
        if out is None:
            out = tmp.copy()
        else:
            out = xr.concat([out, tmp], dim=section)

    dsout = xr.Dataset()
    dsout[outname] = out
    dsout[layer] = ds[layer]
    dsout[interface] = ds[interface]
    dsout["norm"] = xr.DataArray(norm, dims=(section))

    return dsout