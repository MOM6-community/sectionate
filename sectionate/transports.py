import numpy as np
import xarray as xr

def uvindices_from_qindices(isec, jsec, symmetric):
    """From vorticity (q) points given by section, infer u-v points using MOM6 conventions:
    https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html
    """
    nsec = len(isec)
    uvindices = {
        'var':np.zeros(nsec-1, dtype='<U2'),
        'i':np.zeros(nsec-1, dtype=np.int64),
        'j':np.zeros(nsec-1, dtype=np.int64),
        'nward':np.zeros(nsec-1, dtype=bool),
        'eward':np.zeros(nsec-1, dtype=bool)
    }
    for k in range(0, nsec-1):
        zonal = not(jsec[k+1] != jsec[k])
        eward = isec[k+1] > isec[k]
        nward = jsec[k+1] > jsec[k]
        # Handle corner cases for wrapping boundaries
        if (isec[k+1] - isec[k])>1: eward = False
        elif (isec[k+1] - isec[k])<-1: eward = True
        uvindex = {
            'var': 'V' if zonal else 'U', 
            'i': isec[k+np.int64(not(eward) and zonal)],
            'j': jsec[k+np.int64(not(nward) and not(zonal))],
            'nward': nward,
            'eward': eward,
        }
        uvindex['i'] += np.int64(not(symmetric) and zonal)
        uvindex['j'] += np.int64(not(symmetric) and not(zonal))
        for (key, v) in uvindices.items():
            v[k] = uvindex[key]
    return uvindices

def uvcoords_from_uvindices(ds, grid, uvindices, coord_prefix="geo"):
    lons, lats = np.zeros(len(uvindices['var'])), np.zeros(len(uvindices['var']))

    coords = coord_dict(grid)
    geo_coords = [c for c in list(ds.coords) if coord_prefix in c]
    hnames = {f"{coord_prefix}{d}_h":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (coords["X"]["h"] in ds[c].coords) and (coords["Y"]["h"] in ds[c].coords) if d in c}.items()}
    unames = {f"{coord_prefix}{d}_u":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (coords["X"]["q"] in ds[c].coords) and (coords["Y"]["h"] in ds[c].coords) if d in c}.items()}
    vnames = {f"{coord_prefix}{d}_v":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (coords["X"]["h"] in ds[c].coords) and (coords["Y"]["q"] in ds[c].coords) if d in c}.items()}
    qnames = {f"{coord_prefix}{d}_q":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (coords["X"]["q"] in ds[c].coords) and (coords["Y"]["q"] in ds[c].coords) if d in c}.items()}
    
    for p in range(len(uvindices['var'])):
        var, i, j = uvindices['var'][p], uvindices['i'][p], uvindices['j'][p]
        i = np.mod(i, ds[coords["X"]["h"]].size)
        if var == 'U':
            if (f"{coord_prefix}lon_u" in unames) and (f"{coord_prefix}lat_u" in unames):
                lon = ds[unames[f"{coord_prefix}lon_u"]].isel({coords["X"]["q"]:i, coords["Y"]["h"]:j}).values
                lat = ds[unames[f"{coord_prefix}lat_u"]].isel({coords["X"]["q"]:i, coords["Y"]["h"]:j}).values
            elif (f"{coord_prefix}lon_q" in qnames) and (f"{coord_prefix}lat_h" in hnames):
                lon = ds[qnames[f"{coord_prefix}lon_q"]].isel({coords["X"]["q"]:i, coords["Y"]["q"]:j}).values
                lat = ds[hnames[f"{coord_prefix}lat_h"]].isel({coords["X"]["h"]:i, coords["Y"]["h"]:j}).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to identify U-velociy faces.")
        elif var == 'V':
            if (f"{coord_prefix}lon_v" in vnames) and (f"{coord_prefix}lat_v" in vnames):
                lon = ds[vnames[f"{coord_prefix}lon_v"]].isel({coords["X"]["h"]:i, coords["Y"]["q"]:j}).values
                lat = ds[vnames[f"{coord_prefix}lat_v"]].isel({coords["X"]["h"]:i, coords["Y"]["q"]:j}).values
            elif (f"{coord_prefix}lon_h" in hnames) and (f"{coord_prefix}lat_q" in qnames):
                lon = ds[hnames[f"{coord_prefix}lon_h"]].isel({coords["X"]["h"]:i, coords["Y"]["h"]:j}).values
                lat = ds[qnames[f"{coord_prefix}lat_q"]].isel({coords["X"]["q"]:i, coords["Y"]["q"]:j}).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to identify V-velociy faces.")
        lons[p] = lon
        lats[p] = lat
    return lons, lats
    
def uvcoords_from_qindices(
    ds,
    grid,
    isec,
    jsec,
    coord_prefix="geo",
    ):
    return uvcoords_from_uvindices(
        ds,
        grid,
        uvindices_from_qindices(isec, jsec, check_symmetric(grid)),
        coord_prefix=coord_prefix,
    )

def check_symmetric(grid):
    x_sym = grid.axes['X']._default_shifts == {'center': 'outer', 'outer': 'center'}
    y_sym = grid.axes['Y']._default_shifts == {'center': 'outer', 'outer': 'center'}
    if x_sym and y_sym:
        return True
    elif not(x_sym) and not(y_sym):
        return False
    else:
        raise ValueError("Horizontal grid axes ('X', 'Y') must be either both symmetric or both non-symmetric.")
        
def coord_dict(grid):
    if check_symmetric(grid):
        q_pos = "outer"
    else:
        q_pos = "right"
        
    return {
        "X": {"h": grid.axes['X'].coords["center"], "q": grid.axes["X"].coords[q_pos]},
        "Y": {"h": grid.axes['Y'].coords["center"], "q": grid.axes["Y"].coords[q_pos]},
    }
        

def convergent_transport(
    ds,
    grid,
    isec,
    jsec,
    utr="umo",
    vtr="vmo",
    layer="z_l",
    interface="z_i",
    outname="conv_mass_transport",
    section_coord="sect",
    counterclockwise=True,
    dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'},
    cell_widths={'U':'dyCu', 'V':'dxCv'}
    ):
    
    if (layer is not None) and (interface is not None):
        if layer.replace("l", "i") != interface:
            raise ValueError("Inconsistent layer and interface grid variables!")

    uvindices = uvindices_from_qindices(isec, jsec, check_symmetric(grid))
    
    if counterclockwise:
        orient_fact = 1.
    else:
        orient_fact = -1.
    
    section = xr.Dataset()
    section["i"] = xr.DataArray(uvindices["i"], dims=section_coord)
    section["j"] = xr.DataArray(uvindices["j"], dims=section_coord)
    section["Usign"] = xr.DataArray(np.float64(~uvindices['nward'])*2-1, dims=section_coord)
    section["Vsign"] = xr.DataArray(np.float64(uvindices['eward'])*2-1, dims=section_coord)
    section["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=section_coord)
    section["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=section_coord)
    
    coords = coord_dict(grid)
    usel = {coords["Y"]["h"]: np.mod(section["j"], ds[coords["Y"]["h"]].size), coords["X"]["q"]: section["i"]}
    vsel = {coords["X"]["h"]: np.mod(section["i"], ds[coords["X"]["h"]].size), coords["Y"]["q"]: section["j"]}
    
    dsout = xr.Dataset({outname: (
         (ds[utr].isel(usel).fillna(0.) * section["Usign"] * section["Umask"])
        +(ds[vtr].isel(vsel).fillna(0.) * section["Vsign"] * section["Vmask"])
    ) * orient_fact})
    
    if (cell_widths['U'] in ds.coords) and (cell_widths['V'] in ds.coords):
        dsout = dsout.assign_coords({'dl': xr.DataArray((
             (ds[cell_widths['U']].isel(usel).fillna(0.) * section["Umask"])
            +(ds[cell_widths['V']].isel(vsel).fillna(0.) * section["Vmask"])
        ), dims=(section_coord,), attrs={'units':'m'})})

    if layer is not None:
        dsout[layer] = ds[layer]
        if interface is not None:
            dsout[interface] = ds[interface]

    return dsout