import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from .gridutils import get_geo_corners, check_symmetric

def grid_section(grid, lons, lats):
    geocorners = get_geo_corners(grid)
    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        check_symmetric(grid),
        periodic=grid.axes['X']._periodic == ("X")
    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    symmetric,
    periodic=True
    ):
    """create section from list of segments

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude
    gridlat: np.ndarray
        2d array of latitude
    lons: list of float
        longitude of section starting, intermediate and end points
    lats: list of float
        latitude of section starting, intermediate and end points


    RETURNS:
    -------

    isect, jsect: list of int
        list of (i,j) pairs for section
    lonsect, latsect: list of float
        corresponding longitude and latitude for isect, jsect

    """

    isect = np.array([], dtype=np.int64)
    jsect = np.array([], dtype=np.int64)
    lonsect = np.array([], dtype=np.float64)
    latsect = np.array([], dtype=np.float64)

    if len(lons) != len(lats):
        raise ValueError("lons and lats should have the same length")
        
    for k in range(len(lons) - 1):
        iseg, jseg, lonseg, latseg = create_section(
            gridlon,
            gridlat,
            lons[k],
            lats[k],
            lons[k + 1],
            lats[k + 1],
            symmetric,
            periodic=periodic
        )

        isect = np.concatenate([isect, iseg[:-1]], axis=0)
        jsect = np.concatenate([jsect, jseg[:-1]], axis=0)
        lonsect = np.concatenate([lonsect, lonseg[:-1]], axis=0)
        latsect = np.concatenate([latsect, latseg[:-1]], axis=0)
        
    isect = np.concatenate([isect, [iseg[-1]]], axis=0)
    jsect = np.concatenate([jsect, [jseg[-1]]], axis=0)
    lonsect = np.concatenate([lonsect, [lonseg[-1]]], axis=0)
    latsect = np.concatenate([latsect, [latseg[-1]]], axis=0)

    return isect.astype(np.int64), jsect.astype(np.int64), lonsect, latsect

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, symmetric, periodic=True):
    """ replacement function for the old create_section """

    if symmetric and periodic:
        gridlon=gridlon[:,1:]
        gridlat=gridlat[:,1:]
        
    iseg, jseg, lonseg, latseg = infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        periodic=periodic
    )
    return (
        iseg+np.int64(symmetric and periodic),
        jseg,
        lonseg,
        latseg
    )

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, periodic=True):
    """find the grid path joining (lonstart, latstart) and (lonend, latend) pairs

    PARAMETERS:
    -----------

    lonstart: float
        longitude of section starting point
    latstart: float
        latitude of section starting point
    lonend: float
        longitude of section end point
    latend: float
        latitude of section end point

    gridlon: np.ndarray
        2d array of longitude
    gridlat: np.ndarray
        2d array of latitude

    RETURNS:
    -------

    iseg, jseg: list of int
        list of (i,j) pairs bounded by (i1, j1) and (i2, j2)
    lonseg, latseg: list of float
        corresponding longitude and latitude for iseg, jseg
    """

    istart, jstart = find_closest_grid_point(
        lonstart,
        latstart,
        gridlon,
        gridlat
    )
    iend, jend = find_closest_grid_point(
        lonend,
        latend,
        gridlon,
        gridlat
    )
    iseg, jseg, lonseg, latseg = infer_grid_path(
        istart,
        jstart,
        iend,
        jend,
        gridlon,
        gridlat,
        periodic=periodic
    )

    return iseg, jseg, lonseg, latseg


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, periodic=True):
    """find the grid path joining (i1, j1) and (i2, j2) pairs

    PARAMETERS:
    -----------

    i1: integer
        i-coord of point1
    j1: integer
        j-coord of point1
    i2: integer
        i-coord of point2
    j2: integer
        j-coord of point2

    gridlon: np.ndarray
        2d array of longitude
    gridlat: np.ndarray
        2d array of latitude

    nitmax: int
        max number of iteration allowed

    RETURNS:
    -------

    iseg, jseg: list of int
        list of (i,j) pairs bounded by (i1, j1) and (i2, j2)
    lonseg, latseg: list of float
        corresponding longitude and latitude for iseg, jseg
    """
    ny, nx = gridlon.shape
    
    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values

    # target coordinates
    lon_start, lat_start = gridlon[j1, i1], gridlat[j1, i1]
    lon_stop, lat_stop = gridlon[j2, i2], gridlat[j2, i2]
    d_total = distance_on_unit_sphere(lon_start, lat_start, lon_stop, lat_stop)
    
    # init loop index to starting position
    i = i1
    j = j1

    iseg = [i]  # add first point to list of points
    jseg = [j]  # add first point to list of points
    
    # check if shortest path crosses periodic boundary (if applicable)
    wraplons = False
    if periodic:
        wraplons = np.abs(lon_stop - lon_start) > 180.
    wrapsign = int(not(wraplons))*2-1
        
    # iterate through the grid path steps until we reach end of section
    ct = 0 # grid path step counter

    print((i1, j1), (i2, j2))
    while (i%nx != i2) or (j != j2):
        # safety precaution, exit after N iterations
        if ct > 1000:
            raise RuntimeError(f"Should have reached the endpoint by now: {ct}/{Nsteps} steps.")

        if j!=ny-1:
            j_up, i_up = j+1, i%nx
        else:
            j_up = j
            i_up = (nx-1) - (i%nx)
            print((i,j), (i_up, j_up))
            
        neighbors = [
            (j, (i+1)%nx),
            (j, (i-1)%nx),
            (j-1, i%nx),
            (j_up, i_up)
        ]
        
        shortest = np.inf
        for (_j, _i) in neighbors:
            d_step = distance_on_unit_sphere(
                gridlon[j,i],
                gridlat[j,i],
                gridlon[_j,_i],
                gridlat[_j,_i]
            )
            d_remaining = distance_on_unit_sphere(
                gridlon[j,i],
                gridlat[j,i],
                lon_stop,
                lat_stop
            )
            
            lon_pred, lat_pred = arc_path(
                lon_start,
                lat_start,
                lon_stop,
                lat_stop,
                1-(d_remaining-d_step*np.sqrt(2.))/d_total
            )
            d = distance_on_unit_sphere(
                gridlon[_j,_i],
                gridlat[_j,_i],
                lon_pred,
                lat_pred
            )
            print(d_remaining/d_total, -d_step/d_total, d/d_total)
            if d < shortest:
                j_next, i_next = _j, _i
                shortest = d
                
        print((i_next, j_next), "\n")
        
        j = j_next
        i = i_next
        
        j_last, i_last = j,i

        # add new point to list
        iseg.append(i)
        jseg.append(j)
        
        ct+=1

    # create lat/lon vectors from i,j pairs
    lonseg = []
    latseg = []
    for jj, ji in zip(jseg, iseg):
        lonseg.append(gridlon[jj, ji])
        latseg.append(gridlat[jj, ji])
    return np.array(iseg), np.array(jseg), np.array(lonseg), np.array(latseg)


def find_closest_grid_point(lon, lat, gridlon, gridlat):
    """find integer indices of closest grid point in grid of coordinates
    gridlon, gridlat for a given geographical lon/lat.

    PARAMETERS:
    -----------
        lon (float): longitude of point to find
        lat (float): latitude of point to find
        gridlon (numpy.ndarray): grid longitudes
        gridlat (numpy.ndarray): grid latitudes

    RETURNS:
    --------

    iclose, jclose: integer
        grid indices for geographical point of interest
    """

    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values
    dist = distance_on_unit_sphere(lon, lat, gridlon, gridlat)
    jclose, iclose = np.unravel_index(dist.argmin(), gridlon.shape)
    return iclose, jclose

def distance_on_unit_sphere(lon1, lat1, lon2, lat2, method="vincenty", R=6.371e6):
    
    # phi = latitude
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.abs(phi2-phi1)
    
    # lam = longitude
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    dlam = np.abs(lam2-lam1)
    
    if method=="vincenty":
        numerator = np.sqrt(
            (np.cos(phi2)*np.sin(dlam))**2 +
            (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam))**2
        )
        denominator = np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        arc = np.arctan2(numerator, denominator)
        
    elif method=="haversine":
        arc = 2*np.arcsin(np.sqrt(
            np.sin(dphi/2.)**2 + (1. - np.sin(dphi/2.)**2 - np.sin((phi1+phi2)/2.)**2)*np.sin(dlam/2.)**2
        ))
    
        
    elif method=="law of cosines":
        arc = np.arccos(
            np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return R * arc

def arc_path_range(lon1, lat1, lon2, lat2, N, start=0., stop=1.):
    r = np.linspace(start, stop, N)
    return arc_path(lon1, lat1, lon2, lat2, r)
    
def arc_path(lon1, lat1, lon2, lat2, r):
    r = np.clip(r, 0., 1.)
    
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    
    arc = distance_on_unit_sphere(lam1, lam2, phi1, phi2, R=1.)
    
    A = np.sin((1-r)*arc)/np.sin(arc)
    B = np.sin(r*arc)/np.sin(arc)
    x = A*np.cos(phi1)*np.cos(lam1) + B*np.cos(phi2)*np.cos(lam2)
    y = A*np.cos(phi1)*np.sin(lam1) + B*np.cos(phi2)*np.sin(lam2)
    z = A*np.sin(phi1) + B*np.sin(phi2)

    lons = np.rad2deg(np.arctan2(y, x))
    lats = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    
    return lons, lats