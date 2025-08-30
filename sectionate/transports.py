import warnings
import numpy as np
import xarray as xr
import dask

from .gridutils import check_symmetric, coord_dict, get_geo_corners
from .section import distance_on_unit_sphere

def uvindices_from_qindices(grid, i_c, j_c):
    """
    Find the `grid` indices of the N-1 velocity points defined by the consecutive indices of
    N vorticity points. Follows MOM6 conventions (https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Indexing.html),
    automatically checking `grid` metadata to determine whether the grid is symmetric or non-symmetric.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    i_c: int
        vorticity point indices along "X" dimension 
    j_c: int
        vorticity point indices along "Y" dimension

    RETURNS:
    --------
    uvindices : dict
        Dictionary containing:
          - "var" : "U" if corresponding to "X"-direction velocity (usually nominally zonal), "V" otherwise
          - "i" : "X"-dimension index of appropriate "U" or "V" velocity
          - "j" : "Y"-dimension index of appropriate "U" or "V" velocity
          - "nward" : True if point was passed through while going in positive "j"-index direction
          - "eward" : True if point was passed through while going in positive "i"-index direction
    """
    nsec = len(i_c)
    uvindices = {
        "var":np.zeros(nsec-1, dtype="<U2"),
        "i":np.zeros(nsec-1, dtype=np.int64),
        "j":np.zeros(nsec-1, dtype=np.int64),
        "nward":np.zeros(nsec-1, dtype=bool),
        "eward":np.zeros(nsec-1, dtype=bool)
    }
    symmetric = check_symmetric(grid)
    for k in range(0, nsec-1):
        zonal = not(j_c[k+1] != j_c[k])
        eward = i_c[k+1] > i_c[k]
        nward = j_c[k+1] > j_c[k]
        # Handle corner cases for wrapping boundaries
        if (i_c[k+1] - i_c[k])>1: eward = False
        elif (i_c[k+1] - i_c[k])<-1: eward = True
        uvindex = {
            "var": "V" if zonal else "U", 
            "i": i_c[k+(1 if not(eward) and zonal else 0)],
            "j": j_c[k+(1 if not(nward) and not(zonal) else 0)],
            "nward": nward,
            "eward": eward,
        }
        uvindex["i"] += (1 if not(symmetric) and zonal else 0)
        uvindex["j"] += (1 if not(symmetric) and not(zonal) else 0)
        for (key, v) in uvindices.items():
            v[k] = uvindex[key]
    return uvindices

def uvcoords_from_uvindices(grid, uvindices):
    """
    Find the (lons,lats) coordinates of the N-1 velocity points defined by `uvindices` (returned by `uvindices_from_qindices`).
    Assumes the names of longitude and latitude coordinates in `grid` contain the sub-strings "lon" and "lat", respectively,
    but otherwise finds the names using `grid` metadata.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    uvindices : dict
        Dictionary returned by `sectionate.transports.uvindices_from_qindices`, containing:
          - "var" : "U" if corresponding to "X"-direction velocity (usually nominally zonal), "V" otherwise
          - "i" : "X"-dimension index of appropriate "U" or "V" velocity
          - "j" : "Y"-dimension index of appropriate "U" or "V" velocity
          - "nward" : True if point was passed through while going in positive "j"-index direction
          - "eward" : True if point was passed through while going in positive "i"-index direction

    RETURNS:
    --------
    lons : np.ndarray(float)
    lats : np.ndarray(float)
    """
    lons, lats = np.zeros(len(uvindices["var"])), np.zeros(len(uvindices["var"]))

    ds = grid._ds
    coords = coord_dict(grid)
    geo_coords = [c for c in list(ds.coords) if ("lon" in c) or ("lat" in c)]
    center_names = {f"geo{d}_center":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["center"] in ds[c].coords) and
                   (coords["Y"]["center"] in ds[c].coords))
               if d in c}.items()}
    u_names = {f"geo{d}_u":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["corner"] in ds[c].coords) and 
                   (coords["Y"]["center"] in ds[c].coords))
               if d in c}.items()}
    v_names = {f"geo{d}_v":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["center"] in ds[c].coords) and
                   (coords["Y"]["corner"] in ds[c].coords))
               if d in c}.items()}
    corner_names = {f"geo{d}_corner":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["corner"] in ds[c].coords) and
                   (coords["Y"]["corner"] in ds[c].coords))
               if d in c}.items()}

    for p in range(len(uvindices["var"])):
        var, i, j = uvindices["var"][p], uvindices["i"][p], uvindices["j"][p]
        if var == "U":
            if (f"geolon_u" in u_names) and (f"geolat_u" in u_names):
                lon = ds[u_names[f"geolon_u"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["center"]:j
                }).values
                lat = ds[u_names[f"geolat_u"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["center"]:j
                }).values
            elif (f"geolon_corner" in corner_names) and (f"geolat_center" in center_names):
                lon = ds[corner_names[f"geolon_corner"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["corner"]:j
                }).values
                lat = ds[center_names[f"geolat_center"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["center"]:wrap_idx(j, grid, "Y")
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify U-velociy faces.")
        elif var == "V":
            if (f"geolon_v" in v_names) and (f"geolat_v" in v_names):
                lon = ds[v_names[f"geolon_v"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["corner"]:j
                }).values
                lat = ds[v_names[f"geolat_v"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["corner"]:j
                }).values
            elif (f"geolon_center" in center_names) and (f"geolat_corner" in corner_names):
                lon = ds[center_names[f"geolon_center"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["center"]:wrap_idx(j, grid, "Y")
                }).values
                lat = ds[corner_names[f"geolat_corner"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["corner"]:j
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify V-velociy faces.")
        lons[p] = lon
        lats[p] = lat
    return lons, lats
    
def uvcoords_from_qindices(grid, i_c, j_c):
    """
    Directly finds coordinates of velocity points from vorticity point indices, wrapping other functions.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    i_c: int
        vorticity point indices along "X" dimension 
    j_c: int
        vorticity point indices along "Y" dimension

    RETURNS:
    --------
    lons : np.ndarray(float)
    lats : np.ndarray(float)
    """
    return uvcoords_from_uvindices(
        grid,
        uvindices_from_qindices(grid, i_c, j_c),
    )

def convergent_transport(
    grid,
    i_c,
    j_c,
    utr="umo",
    vtr="vmo",
    layer="z_l",
    interface="z_i",
    outname="conv_mass_transport",
    sect_coord="sect",
    geometry="spherical",
    positive_in=True,
    cell_widths={"U":"dyCu", "V":"dxCv"},
    ):
    """
    Lazily calculates extensive transports normal to a section, with the sign convention of positive into the spherical polygon
    defined by the section, unless overridden by changing the "positive_in=True" keyword argument. Supports curvlinear geometries
    and complicated grid topologies, as long as the grid is *locally* orthogonal (as in MOM6).
    Lazily broadcasts the calculation in all dimensions except ("X", "Y").

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables. Must include variables "utr" and "vtr" (see kwargs).
    i_c: int
        Vorticity point indices along "X" dimension. 
    j_c: int
        Vorticity point indices along "Y" dimension.
    utr: str
        Name of "X"-direction tracer transport
    vtr: str
        Name of "Y"-direction tracer transport
    layer : str or None
    interface : str or None
    outname : str
        Name of output xr.DataArray variable. Default: "conv_mass_transport".
    sect_coord: str
        Name of the dimension describing along-section data in the output. Default: "sect".
    geometry : str
        Geometry to use to check orientation of the section. Supported geometries are ["cartesian", "spherical"].
        Default: "spherical".
    positive_in : bool or xr.DataArray of type bool
        If True, convergence is defined as "inwards" with respect to the corresponding "geometry".
        If False, convergence is defined as "outwards" (equivalently, negative the inward convergence).
        If a boolean xr.DataArray, get value of positive_in by selecting the value of the mask on the inside
        of an arbitrary velocity face.
    cell_widths : dict
        Values of "U" and "V" items in `cell_widths` correspond to the names of the coordinates describing the width of
        velocity cells. If they are both present in `grid._ds`, accumulate distance along the section and add it to the
        returned xr.DataArray.

    RETURNS:
    --------
    dsout : xr.DataArray
        Contains the calculated normal transport and the coordinates of the contributing velocity points (`lon`, `lat`),
        as well as some useful metadata, such as whether each point corresponds to a "U" or "V" velocity and whether
        the sign of the transport had to be flipped to make it point inwards.
    """
    
    if (layer is not None) and (interface is not None):
        if layer.replace("l", "i") != interface:
            raise ValueError("Inconsistent layer and interface grid variables!")
            
    uvindices = uvindices_from_qindices(grid, i_c, j_c)
    uvcoords = uvcoords_from_qindices(grid, i_c, j_c)
    
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
        np.array([1 if i else -1 for i in ~uvindices["nward"]]),
        dims=sect_coord
    )
    sect["Vsign"] = xr.DataArray(
        np.array([1 if i else -1 for i in uvindices["eward"]]),
        dims=sect_coord
    )
    sect["var"] = xr.DataArray(uvindices["var"], dims=sect_coord)
    sect["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=sect_coord)
    sect["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=sect_coord)
    
    mask_types = (np.ndarray, dask.array.Array, xr.DataArray)
    if isinstance(positive_in, mask_types):
        positive_in = is_mask_inside(positive_in, grid, sect)
        
    else:
        if (geometry == "cartesian") and (grid.axes["X"]._boundary == "periodic"):
            raise ValueError("Periodic cartesian domains are not yet supported!")
        coords = coord_dict(grid)
        geo_corners = get_geo_corners(grid)
        idx = {
            coords["X"]["corner"]:xr.DataArray(i_c, dims=("pt",)),
            coords["Y"]["corner"]:xr.DataArray(j_c, dims=("pt",)),
        }
        counterclockwise = is_section_counterclockwise(
            geo_corners["X"].isel(idx).values,
            geo_corners["Y"].isel(idx).values,
            geometry=geometry
        )
        positive_in = positive_in ^ (not(counterclockwise))
    orient_fact = 1 if positive_in else -1
    
    coords = coord_dict(grid)
    usel = {
        coords["X"]["corner"]: sect["i"],
        coords["Y"]["center"]: wrap_idx(sect["j"], grid, "Y")
    }
    vsel = {
        coords["X"]["center"]: wrap_idx(sect["i"], grid, "X"),
        coords["Y"]["corner"]: sect["j"]
    }
    
    u = grid._ds[utr]
    v = grid._ds[vtr]
    
    conv_umo_masked = (
        u.isel(usel).fillna(0.)
        *sect["Usign"]*sect["Umask"]
    )
    conv_vmo_masked = (
        v.isel(vsel).fillna(0.)
        *sect["Vsign"]*sect["Vmask"]
    )
    conv_transport = xr.DataArray(
        (conv_umo_masked + conv_vmo_masked)*orient_fact,
    )
    dsout = xr.Dataset({outname: conv_transport})
    
    if ((cell_widths["U"] in grid._ds.coords) and
        (cell_widths["V"] in grid._ds.coords)):
        
        dsout = dsout.assign_coords({
            "dl": xr.DataArray(
                (
                    (grid._ds[cell_widths["U"]].isel(usel).fillna(0.)
                     *sect["Umask"])+
                    (grid._ds[cell_widths["V"]].isel(vsel).fillna(0.)
                     *sect["Vmask"])
                ),
                dims=(sect_coord,),
                attrs={"units":"m"}
            )
        })

    dsout = dsout.assign_coords({
        "sign": orient_fact*(
            sect["Usign"]*sect["Umask"] +
            sect["Vsign"]*sect["Vmask"]
        ),
        "dir": xr.DataArray(
            np.array(["U" if u else "V" for u in sect["Umask"]]),
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        "lon": xr.DataArray(
            uvcoords[0],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        "lat": xr.DataArray(
            uvcoords[1],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
    })
    dsout[outname].attrs = {**dsout[outname].attrs, **{
        "orient_fact":orient_fact,
        "positive_in":positive_in,
    }}
    dsout[outname].attrs
    
    if layer is not None:
        dsout[layer] = grid._ds[layer]
        if interface is not None:
            dsout[interface] = grid._ds[interface]

    return dsout

def is_section_counterclockwise(lons_c, lats_c, geometry="spherical"):
    """
    Check if the polygon defined by the consecutive (lons_c, lats_c) is `counterclockwise` (with respect to a given
    `geometry`). Under the hood, it does this by checking whether the signed area (or determinant) of the polygon
    is negative (counterclockwise) or positive (clockwise). This is only a meaningful calculation if the section
    is closed, i.e. (lons_c[-1], lats_c[-1]) == (lons_c[0], lats_c[0]), and therefore defines a polygon.
    
    For the case `geometry="spherical"`, the periodic nature of the longitude coordinate complicates things;
    instead of working in spherical coordinates, we use a South-Pole stereographic projection of the surface of the sphere
    and evaluate the orientation of the projected polygon with respect to the stereographic plane.

    PARAMETERS:
    -----------
    lons_c : np.ndarray(float), longitude of corner points, in degrees
    lats_c : np.ndarray(float), latitude of corner points, in degrees
    geometry : str
        Geometry to use to check orientation of the section. Supported geometries are ["cartesian", "spherical"].
        Default: "spherical".

    RETURNS:
    --------
    counterclockwise : bool
    """
    if distance_on_unit_sphere(lons_c[0], lats_c[0], lons_c[-1], lats_c[-1]) > 10.:
        warnings.warn("The orientation of open sections is ambiguous–verify that it matches expectations!")
        lons_c = np.append(lons_c, lons_c[0])
        lats_c = np.append(lats_c, lats_c[0])
    
    if geometry == "spherical":
        X, Y = stereographic_projection(lons_c, lats_c)
    elif geometry == "cartesian":
        X, Y = lons_c, lats_c
    else:
        raise ValueError("""Only "spherical" and "cartesian" geometries are currently supported.""")
    
    signed_area = 0.
    for i in range(X.size-1):
        signed_area += (X[i+1]-X[i])*(Y[i+1]+Y[i])
    return signed_area < 0.

def stereographic_projection(lons, lats):
    """
    Projects longitudes and latitudes onto the South-Polar stereographic plane.

    PARAMETERS:
    -----------
    lons : np.ndarray(float), in degrees
    lats : np.ndarray(float), in degrees

    RETURNS:
    --------
    X : np.ndarray(float)
    Y : np.ndarray(float)
    """
    lats = np.clip(lats, -90. +1.e-3, 90. -1.e-3)
    varphi = np.deg2rad(-lats+90.)
    theta = np.deg2rad(lons)
    
    R = np.sin(varphi)/(1. - np.clip(np.cos(varphi), -1. +1.e-14, 1. -1.e-14))
    Theta = -theta
    
    X, Y = R*np.cos(Theta), R*np.sin(Theta)
    
    return X, Y

def is_mask_inside(mask, grid, sect, idx=0):
    """
    Find the `(i,j)` indices of the `grid` tracer-cell point "inside" of the velocity face at index `idx`,
    and evaluate the value of the `mask` there.

    PARAMETERS:
    -----------
    mask : xr.DataArray
    grid : xgcm.Grid
    sect : dict
        Dictionary of uvindices (see `sectionate.transports.uvindices_from_qindices`).

    RETURNS:
    --------
    positive_in : bool
    """
    symmetric = check_symmetric(grid)
    coords = coord_dict(grid)
    if sect["var"][idx]=="U":
        i = (
            sect["i"][idx]
            - (1 if sect["Usign"][idx].values==-1. else 0)
            + (1 if not(symmetric) else 0)
        )
        j = sect["j"][idx]
        if 0<=i<=grid._ds[coords["X"]["center"]].size-1:
            positive_in = mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j
            }).values
        elif i==-1:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i+1,
                coords["Y"]["center"]: j
            })).values
        elif i==grid._ds[coords["X"]["center"]].size:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i-1,
                coords["Y"]["center"]: j
            })).values
    elif sect["var"][idx]=="V":
        i = sect["i"][idx]
        j = (
            sect["j"][idx]
            - (1 if sect["Vsign"][idx].values==-1. else 0)
            + (1 if not(symmetric) else 0)
        )
        if 0<=j<=grid._ds[coords["Y"]["center"]].size-1:
            positive_in = mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j
            }).values
        elif j==-1:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j+1,
            })).values
        elif j==grid._ds[coords["Y"]["center"]].size:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j-1
            })).values
    return positive_in


def wrap_idx(idx, grid, axis):
    coords = coord_dict(grid)
    if grid.axes[axis]._boundary == "periodic":
        idx = np.mod(idx, grid._ds[coords[axis]["center"]].size)
    else:
        idx = np.minimum(idx, grid._ds[coords[axis]["center"]].size-1)
    return idx