import numpy as np
import xarray as xr

avail_models = ["MOM6"]

def MOM6_UVpoints_from_section(isec, jsec, symmetric=True):
    """from q points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html
    """
    nsec = len(isec)
    uvpoints = []
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
        uvpoints.append(point)
    return uvpoints

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

    for pt in uvpoints:
        if pt["var"] == "U":
            fact = -1 if pt["nward"] else 1
            tmp = (
                ds[utr]
                .isel(xq=pt["i"], yh=pt["j"]).fillna(0.)
                .rename({"yh": "ysec", "xq": "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact
            norm.append(fact)

        if pt["var"] == "V":
            fact = -1 if not(pt["eward"]) else 1
            tmp = (
                ds[vtr]
                .isel(xh=pt["i"], yq=pt["j"]).fillna(0.)
                .rename({"yq": "ysec", "xh": "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact
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