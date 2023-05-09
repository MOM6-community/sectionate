import numpy as np
import xarray as xr
import dask

from .gridutils import check_symmetric

def uvindices_from_qindices(grid, isec, jsec):
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
    symmetric = check_symmetric(grid)
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

def uvcoords_from_uvindices(grid, uvindices):
    lons, lats = np.zeros(len(uvindices['var'])), np.zeros(len(uvindices['var']))

    ds = grid._ds
    coords = coord_dict(grid)
    geo_coords = [c for c in list(ds.coords) if "geo" in c]
    hnames = {f"geo{d}_h":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if ((coords["X"]["h"] in ds[c].coords) and
                   (coords["Y"]["h"] in ds[c].coords))
               if d in c}.items()}
    unames = {f"geo{d}_u":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if ((coords["X"]["q"] in ds[c].coords) and 
                   (coords["Y"]["h"] in ds[c].coords))
               if d in c}.items()}
    vnames = {f"geo{d}_v":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if ((coords["X"]["h"] in ds[c].coords) and
                   (coords["Y"]["q"] in ds[c].coords))
               if d in c}.items()}
    qnames = {f"geo{d}_q":c for d,c in
              {d:c for d in ['lon', 'lat'] for c in geo_coords
               if ((coords["X"]["q"] in ds[c].coords) and
                   (coords["Y"]["q"] in ds[c].coords))
               if d in c}.items()}

    for p in range(len(uvindices['var'])):
        var, i, j = uvindices['var'][p], uvindices['i'][p], uvindices['j'][p]
        if var == 'U':
            if (f"geolon_u" in unames) and (f"geolat_u" in unames):
                lon = ds[unames[f"geolon_u"]].isel({
                    coords["X"]["q"]:i,
                    coords["Y"]["h"]:j
                }).values
                lat = ds[unames[f"geolat_u"]].isel({
                    coords["X"]["q"]:i,
                    coords["Y"]["h"]:j
                }).values
            elif (f"geolon_q" in qnames) and (f"geolat_h" in hnames):
                lon = ds[qnames[f"geolon_q"]].isel({
                    coords["X"]["q"]:i,
                    coords["Y"]["q"]:j
                }).values
                lat = ds[hnames[f"geolat_h"]].isel({
                    coords["X"]["h"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["h"]:wrap_idx(j, grid, "Y")
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify U-velociy faces.")
        elif var == 'V':
            if (f"geolon_v" in vnames) and (f"geolat_v" in vnames):
                lon = ds[vnames[f"geolon_v"]].isel({
                    coords["X"]["h"]:i,
                    coords["Y"]["q"]:j
                }).values
                lat = ds[vnames[f"geolat_v"]].isel({
                    coords["X"]["h"]:i,
                    coords["Y"]["q"]:j
                }).values
            elif (f"geolon_h" in hnames) and (f"geolat_q" in qnames):
                lon = ds[hnames[f"geolon_h"]].isel({
                    coords["X"]["h"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["h"]:wrap_idx(j, grid, "Y")
                }).values
                lat = ds[qnames[f"geolat_q"]].isel({
                    coords["X"]["q"]:i,
                    coords["Y"]["q"]:j
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify V-velociy faces.")
        lons[p] = lon
        lats[p] = lat
    return lons, lats
    
def uvcoords_from_qindices(grid, isec, jsec):
    return uvcoords_from_uvindices(
        grid,
        uvindices_from_qindices(grid, isec, jsec),
    )
        
def coord_dict(grid):
    if check_symmetric(grid):
        q_pos = "outer"
    else:
        q_pos = "right"
        
    return {
        "X": {
            "h": grid.axes['X'].coords["center"],
            "q": grid.axes["X"].coords[q_pos]},
        "Y": {
            "h": grid.axes['Y'].coords["center"],
            "q": grid.axes["Y"].coords[q_pos]},
    }

def convergent_transport(
    grid,
    isec,
    jsec,
    utr="umo",
    vtr="vmo",
    layer="z_l",
    interface="z_i",
    outname="conv_mass_transport",
    sect_coord="sect",
    counterclockwise=True,
    mask_inside=True,
    cell_widths={'U':'dyCu', 'V':'dxCv'},
    ):
    
    if (layer is not None) and (interface is not None):
        if layer.replace("l", "i") != interface:
            raise ValueError("Inconsistent layer and interface grid variables!")

    uvindices = uvindices_from_qindices(grid, isec, jsec)
    uvcoords = uvcoords_from_qindices(grid, isec, jsec)
    
    sect = xr.Dataset()
    sect = sect.assign_coords({
        sect_coord: xr.DataArray(
            np.arange(uvindices["i"].size),
            dims=(sect_coord,)
        )
    })
    sect["i"] = xr.DataArray(uvindices["i"], dims=sect_coord)
    sect["j"] = xr.DataArray(uvindices["j"], dims=sect_coord)
    sect["Usign"] = xr.DataArray(
        np.float32(~uvindices['nward'])*2-1,
        dims=sect_coord
    )
    sect["Vsign"] = xr.DataArray(
        np.float32(uvindices['eward'])*2-1,
        dims=sect_coord
    )
    sect["var"] = xr.DataArray(uvindices["var"], dims=sect_coord)
    sect["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=sect_coord)
    sect["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=sect_coord)
    
    mask_types = (np.ndarray, dask.array.Array, xr.DataArray)
    if isinstance(mask_inside, mask_types):
        mask_inside = is_mask_inside(mask_inside, grid, sect)
    else:
        mask_inside = mask_inside ^ (not(counterclockwise))
    orient_fact = np.int32(mask_inside)*2-1
    
    coords = coord_dict(grid)
    usel = {
        coords["X"]["q"]: sect["i"],
        coords["Y"]["h"]: wrap_idx(sect["j"], grid, "Y")
    }
    vsel = {
        coords["X"]["h"]: wrap_idx(sect["i"], grid, "X"),
        coords["Y"]["q"]: sect["j"]
    }
    
    conv_umo_masked = (
        grid._ds[utr].isel(usel).fillna(0.)
        *sect["Usign"]*sect["Umask"]
    )
    conv_vmo_masked = (
        grid._ds[vtr].isel(vsel).fillna(0.)
        *sect["Vsign"]*sect["Vmask"]
    )
    conv_transport = xr.DataArray(
        (conv_umo_masked + conv_vmo_masked)*orient_fact,
    )
    dsout = xr.Dataset({outname: conv_transport})
    
    if ((cell_widths['U'] in grid._ds.coords) and
        (cell_widths['V'] in grid._ds.coords)):
        
        dsout = dsout.assign_coords({
            'dl': xr.DataArray(
                (
                    (grid._ds[cell_widths['U']].isel(usel).fillna(0.)
                     *sect["Umask"])+
                    (grid._ds[cell_widths['V']].isel(vsel).fillna(0.)
                     *sect["Vmask"])
                ),
                dims=(sect_coord,),
                attrs={'units':'m'}
            )
        })

    dsout = dsout.assign_coords({
        'sign': orient_fact*(
            sect["Usign"]*sect["Umask"] +
            sect["Vsign"]*sect["Vmask"]
        ),
        'dir': xr.DataArray(
            np.array(['U' if u else 'V' for u in sect["Umask"]]),
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        'lon': xr.DataArray(
            uvcoords[0],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        'lat': xr.DataArray(
            uvcoords[1],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
    })
    dsout[outname].attrs = {**dsout[outname].attrs, **{
        'orient_fact':orient_fact,
        'mask_inside':mask_inside,
        'counterclockwise':counterclockwise
    }}
    dsout[outname].attrs
    
    if layer is not None:
        dsout[layer] = grid._ds[layer]
        if interface is not None:
            dsout[interface] = grid._ds[interface]

    return dsout

def is_mask_inside(mask, grid, sect, idx=0):
    symmetric = check_symmetric(grid)
    coords = coord_dict(grid)
    if sect['var'][idx]=="U":
        i = (
            sect['i'][idx]
            - int(sect['Usign'][idx].values==-1.)
            + int(not(symmetric))
        )
        j = sect['j'][idx]
        if 0<=i<=grid._ds[coords["X"]["h"]].size-1:
            mask_inside = mask.isel({
                coords["X"]["h"]: i,
                coords["Y"]["h"]: j
            }).values
        elif i==-1:
            mask_inside = not(mask.isel({
                coords["X"]["h"]: i+1,
                coords["Y"]["h"]: j
            })).values
        elif i==grid._ds[coords["X"]["h"]].size:
            mask_inside = not(mask.isel({
                coords["X"]["h"]: i-1,
                coords["Y"]["h"]: j
            })).values
    elif sect['var'][idx]=="V":
        i = sect['i'][idx]
        j = (
            sect['j'][idx]
            - int(sect['Vsign'][idx].values==-1.)
            + int(not(symmetric))
        )
        if 0<=j<=grid._ds[coords["Y"]["h"]].size-1:
            mask_inside = mask.isel({
                coords["X"]["h"]: i,
                coords["Y"]["h"]: j
            }).values
        elif j==-1:
            mask_inside = not(mask.isel({
                coords["X"]["h"]: i,
                coords["Y"]["h"]: j+1,
            })).values
        elif j==grid._ds[coords["Y"]["h"]].size:
            mask_inside = not(mask.isel({
                coords["X"]["h"]: i,
                coords["Y"]["h"]: j-1
            })).values
    return mask_inside


def wrap_idx(idx, grid, axis):
    coords = coord_dict(grid)
    if grid.axes[axis]._periodic == axis:
        idx = np.mod(idx, grid._ds[coords[axis]["h"]].size)
    else:
        idx = np.minimum(idx, grid._ds[coords[axis]["h"]].size-1)
    return idx