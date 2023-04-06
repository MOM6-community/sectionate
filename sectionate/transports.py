import numpy as np
import xarray as xr

def uvindices_from_qindices(isec, jsec, symmetric):
    """From vorticity (q) points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html
    """
    nsec = len(isec)
    uvindices = {'var':[], 'i':[], 'j':[], 'nward':[], 'eward':[]}
    for k in range(1, nsec):
        zonal = not(jsec[k] != jsec[k - 1])
        nward = jsec[k] > jsec[k - 1]
        eward = isec[k] > isec[k - 1]
        # Handle corner cases for wrapping boundaries
        if (isec[k] - isec[k - 1])>1: eward = False
        elif (isec[k] - isec[k - 1])<-1: eward = True
        uvindex = {
            'var': 'V' if zonal else 'U', 
            'i': isec[k - np.int64(not(eward))],
            'j': jsec[k - np.int64(not(nward))],
            'nward': nward,
            'eward': eward,
        }
        uvindex['i'] -= np.int64(not(symmetric) and (uvindex['var']=="V"))
        uvindex['j'] -= np.int64(not(symmetric) and (uvindex['var']=="U"))
        for (key, v) in uvindices.items():
            v.append(uvindex[key])
    return uvindices

def uvcoords_from_uvindices(grid, uvindices, coord_prefix="geo", dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'}):
    lons, lats = np.zeros(len(uvindices['var'])), np.zeros(len(uvindices['var']))

    geo_coords = [c for c in list(grid.coords)+list(grid.data_vars) if coord_prefix in c]
    hnames = {f"{coord_prefix}{d}_h":c for d,c in {d:c for d in ['lon', 'lat'] for c in geo_coords
                                                   if (dim_names['xh'] in grid[c].dims) and (dim_names['yh'] in grid[c].dims) if d in c}.items()}
    unames = {f"{coord_prefix}{d}_u":c for d,c in {d:c for d in ['lon', 'lat'] for c in geo_coords
                                                   if (dim_names['xq'] in grid[c].dims) and (dim_names['yh'] in grid[c].dims) if d in c}.items()}
    vnames = {f"{coord_prefix}{d}_v":c for d,c in {d:c for d in ['lon', 'lat'] for c in geo_coords
                                                   if (dim_names['xh'] in grid[c].dims) and (dim_names['yq'] in grid[c].dims) if d in c}.items()}
    qnames = {f"{coord_prefix}{d}_q":c for d,c in {d:c for d in ['lon', 'lat'] for c in geo_coords
                                                   if (dim_names['xq'] in grid[c].dims) and (dim_names['yq'] in grid[c].dims) if d in c}.items()}
    
    for p in range(len(uvindices['var'])):
        var, i, j = uvindices['var'][p], uvindices['i'][p], uvindices['j'][p]
        if var == 'U':
            if (f"{coord_prefix}lon_u" in unames) and (f"{coord_prefix}lat_u" in unames):
                lon = grid[unames[f"{coord_prefix}lon_u"]].isel({dim_names['xq']:i, dim_names['yh']:j}).values
                lat = grid[unames[f"{coord_prefix}lat_u"]].isel({dim_names['xq']:i, dim_names['yh']:j}).values
            elif (f"{coord_prefix}lon_q" in qnames) and (f"{coord_prefix}lat_h" in hnames):
                lon = grid[qnames[f"{coord_prefix}lon_q"]].isel({dim_names['xq']:i, dim_names['yq']:j}).values
                lat = grid[hnames[f"{coord_prefix}lat_h"]].isel({dim_names['xh']:i, dim_names['yh']:j}).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to identify U-velociy faces.")
        elif var == 'V':
            if (f"{coord_prefix}lon_v" in vnames) and (f"{coord_prefix}lat_v" in vnames):
                lon = grid[vnames[f"{coord_prefix}lon_v"]].isel({dim_names['xh']:i, dim_names['yq']:j}).values
                lat = grid[vnames[f"{coord_prefix}lat_v"]].isel({dim_names['xh']:i, dim_names['yq']:j}).values
            elif (f"{coord_prefix}lon_h" in hnames) and (f"{coord_prefix}lat_q" in qnames):
                lon = grid[hnames[f"{coord_prefix}lon_h"]].isel({dim_names['xh']:i, dim_names['yh']:j}).values
                lat = grid[qnames[f"{coord_prefix}lat_q"]].isel({dim_names['xq']:i, dim_names['yq']:j}).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to identify V-velociy faces.")
        lons[p] = lon
        lats[p] = lat
    return lons, lats
    
def uvcoords_from_qindices(grid, isec, jsec, symmetric, coord_prefix="geo", dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'}):
    return uvcoords_from_uvindices(grid, uvindices_from_qindices(isec, jsec, symmetric), coord_prefix=coord_prefix, dim_names=dim_names)

def convergent_transport(
        ds,
        isec,
        jsec,
        symmetric,
        utr="umo",
        vtr="vmo",
        layer="z_l",
        interface="z_i",
        outname="conv_mass_transport",
        section="sect",
        counterclockwise=True,
        dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'},
        cell_widths={'U':'dyCu', 'V':'dxCv'}
    ):
    
    if (layer is not None) and (interface is not None):
        if layer.replace("l", "i") != interface:
            raise ValueError("Inconsistent layer and interface grid variables!")

    uvindices = uvindices_from_qindices(isec, jsec, symmetric)

    sign = []
    out = None
    
    if counterclockwise:
        orient_fact = 1.
    else:
        orient_fact = -1.
    
    dist = []
    cum_dist = 0.
    for i in range(len(uvindices['var'])):
        pt = {k:v[i] for (k,v) in uvindices.items()}
        if pt["var"] == "U":
            fact = -1 if pt["nward"] else 1
            tmp = (
                ds[utr]
                .isel(xq=pt["i"], yh=pt["j"]).fillna(0.)
                .rename({dim_names["yh"]: "ysec", dim_names["xq"]: "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact * orient_fact
            sign.append(fact)
            if cell_widths['U'] in ds.coords:
                tmp = tmp.rename({cell_widths['U']: 'dl'})
                d = ds[cell_widths['U']].isel(xq=pt["i"], yh=pt["j"])
                cum_dist += d
                dist.append(cum_dist - d/2.)

        elif pt["var"] == "V":
            fact = -1 if not(pt["eward"]) else 1
            tmp = (
                ds[vtr]
                .isel(xh=pt["i"], yq=pt["j"]).fillna(0.)
                .rename({dim_names["yq"]: "ysec", dim_names["xh"]: "xsec"})
                .expand_dims(dim=section, axis=-1)
            ) * fact * orient_fact
            sign.append(fact)
            if cell_widths['V'] in ds.coords:
                tmp = tmp.rename({cell_widths['V']: 'dl'})
                d = ds[cell_widths['V']].isel(xh=pt["i"], yq=pt["j"])
                
            if (cell_widths['U'] in ds.coords) or (cell_widths['V'] in ds.coords):
                cum_dist += d
                dist.append(cum_dist - d/2.)
        
        if out is None:
            out = tmp.copy()
        else:
            out = xr.concat([out, tmp], dim=section)

    dsout = xr.Dataset({section: np.arange(0, out[section].size)})
    dsout[outname] = out
    if layer is not None:
        dsout[layer] = ds[layer]
        if interface is not None:
            dsout[interface] = ds[interface]
    dsout["sign"] = xr.DataArray(sign, dims=(section))
    if len(dist)>0:
        dsout=dsout.assign_coords({"distance": xr.DataArray(dist, dims=(section), attrs={'units':'m'})})

    return dsout
