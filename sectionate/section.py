import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from .gridutils import get_geo_corners, check_symmetric

def grid_section(grid, lons, lats, topology="cartesian"):
    geocorners = get_geo_corners(grid)
    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        check_symmetric(grid),
        periodic=[ax for ax in grid.axes if grid.axes[ax]._periodic],
        topology=topology
    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    symmetric,
    periodic=["X"],
    topology="cartesian"
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
            periodic=periodic,
            topology=topology
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

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, symmetric, periodic=["X"], topology="cartesian"):
    """ replacement function for the old create_section """

    if symmetric and periodic==["X"]:
        gridlon=gridlon[:,:-1]
        gridlat=gridlat[:,:-1]

    iseg, jseg, lonseg, latseg = infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        periodic=periodic,
        topology=topology
    )
    return (
        iseg,
        jseg,
        lonseg,
        latseg
    )

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, periodic=["X"], topology="cartesian"):
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
        periodic=periodic,
        topology=topology
    )

    return iseg, jseg, lonseg, latseg


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, periodic=["X"], topology="cartesian"):
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
    lon1, lat1 = gridlon[j1, i1], gridlat[j1, i1]
    lon2, lat2 = gridlon[j2, i2], gridlat[j2, i2]
    
    # init loop index to starting position
    i = i1
    j = j1

    iseg = [i]  # add first point to list of points
    jseg = [j]  # add first point to list of points

    # iterate through the grid path steps until we reach end of section
    ct = 0 # grid path step counter

    # Grid-agnostic algorithm:
    # First, find all four neighbors (subject to grid topology)
    # Second, throw away any that are further from the destination than the current point
    # Third, go to the valid neighbor that has the smallest angle from the arc path between the
    # start and end points (the shortest geodesic path)
    j_prev, i_prev = j,i
    while (i%nx != i2) or (j != j2):
        # safety precaution: exit after taking enough steps to have crossed the entire model grid
        if ct > (nx+ny+1):
            raise RuntimeError(f"Should have reached the endpoint by now: {ct}/{Nsteps} steps.")

        d_current = distance_on_unit_sphere(
                gridlon[j,i],
                gridlat[j,i],
                lon2,
                lat2
            )
        
        if d_current < 1.e-9:
            break
        
        if periodic==["X"]:
            right = (j, (i+1)%nx)
            left = (j, (i-1)%nx)
        else:
            right = (j, np.clip(i+1, 0, nx-1))
            left = (j, np.clip(i-1, 0, nx-1))
        down = (np.clip(j-1, 0, ny-1), i)
        
        if topology=="MOM-tripolar":
            if j!=ny-1:
                up = (j+1, i%nx)
            else:
                up = (j-1, (nx-1) - (i%nx))
                
        elif topology=="cartesian":
                up = (np.clip(j+1, 0, ny-1), i)
        else:
            raise ValueError("Only 'cartesian' and 'MOM-tripolar' grid topologies are currently supported.")
        
        neighbors = [right, left, down, up]
        
        smallest_angle = np.inf
        d_list = []
        for (_j, _i) in neighbors:
            d = distance_on_unit_sphere(
                gridlon[_j,_i],
                gridlat[_j,_i],
                lon2,
                lat2
            )
            d_list.append(d/d_current)
            if d==0.:
                j_next, i_next = _j, _i
                smallest_angle = 0.
                break
            elif d < d_current:
                angle = spherical_angle(
                    lon2,
                    lat2,
                    lon1,
                    lat1,
                    gridlon[_j,_i],
                    gridlat[_j,_i],
                )
                if angle < smallest_angle:
                    j_next, i_next = _j, _i
                    smallest_angle = angle
                    
        # TODO: Better handling of the edge cases where none of the neighbors
        # get us closer to the target coordinates! This seems to only happen when
        # approaching the top corners of the grid (edges of the tripolar seams),
        # but sectionate behaves quite strangely in these cases.
        # Here, although none of the points get us closer to our target point, we 
        # pick the one that is the closest and hope that will put us on the right path.
        # Maybe it is the wrong choice.
        if smallest_angle == np.inf:
            if (j_prev, i_prev) in neighbors:
                idx = neighbors.index((j_prev, i_prev))
                del neighbors[idx]
                del d_list[idx]
            
            (j_next, i_next ) = neighbors[np.argmin(d_list)]

        j_prev, i_prev = j,i
        
        j = j_next
        i = i_next

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

def spherical_angle(lonA, latA, lonB, latB, lonC, latC):
    
    a = distance_on_unit_sphere(lonB, latB, lonC, latC, R=1.)
    b = distance_on_unit_sphere(lonC, latC, lonA, latA, R=1.)
    c = distance_on_unit_sphere(lonA, latA, lonB, latB, R=1.)
        
    return np.arccos(np.clip((np.cos(a) - np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)), -1., 1.))