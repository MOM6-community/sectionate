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

def uvcoords_from_uvindices(grid, uvindices, coord_prefix="geo", dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'}):
    lons, lats = np.zeros(len(uvindices['var'])), np.zeros(len(uvindices['var']))

    geo_coords = [c for c in list(grid.coords)+list(grid.data_vars) if coord_prefix in c]
    hnames = {f"{coord_prefix}{d}_h":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (dim_names['xh'] in grid[c].dims) and (dim_names['yh'] in grid[c].dims) if d in c}.items()}
    unames = {f"{coord_prefix}{d}_u":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (dim_names['xq'] in grid[c].dims) and (dim_names['yh'] in grid[c].dims) if d in c}.items()}
    vnames = {f"{coord_prefix}{d}_v":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (dim_names['xh'] in grid[c].dims) and (dim_names['yq'] in grid[c].dims) if d in c}.items()}
    qnames = {f"{coord_prefix}{d}_q":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if (dim_names['xq'] in grid[c].dims) and (dim_names['yq'] in grid[c].dims) if d in c}.items()}
    
    for p in range(len(uvindices['var'])):
        var, i, j = uvindices['var'][p], uvindices['i'][p], uvindices['j'][p]
        i = np.mod(i, grid[dim_names['xh']].size)
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
        section_coord="sect",
        counterclockwise=True,
        dim_names={'xh':'xh', 'yh':'yh', 'xq':'xq', 'yq':'yq'},
        cell_widths={'U':'dyCu', 'V':'dxCv'}
    ):
    
    if (layer is not None) and (interface is not None):
        if layer.replace("l", "i") != interface:
            raise ValueError("Inconsistent layer and interface grid variables!")

    uvindices = uvindices_from_qindices(isec, jsec, symmetric)
    
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
    
    usel = {dim_names["yh"]: np.mod(section["j"], ds[dim_names["yh"]].size), dim_names["xq"]: section["i"]}
    vsel = {dim_names["xh"]: np.mod(section["i"], ds[dim_names["xh"]].size), dim_names["yq"]: section["j"]}
    
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