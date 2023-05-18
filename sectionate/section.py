import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from .gridutils import get_geo_corners, check_symmetric

def grid_section(grid, lons, lats, topology="cartesian"):
    """
    Compute composite section along model `grid` velocity faces that approximates geodesic paths
    between consecutive points defined by (lons, lats).

    Parameters
    ----------
    grid: xgcm.Grid
        Object describing the geometry of the ocean model grid, including metadata about variable names for
        the staggered C-grid dimensions and c oordinates.
    lons: list or np.ndarray
        Longitudes, in degrees, of consecutive vertices defining a piece-wise geodesic section.
    lats: list or np.ndarray
        Latitudes, in degrees (in range [-90, 90]), of consecutive vertices defining a piece-wise geodesic section.
    topology: str
        Default: 'cartesian'. Currently only supports the following options: ['cartesian', 'MOM-tripolar'].
        
    Returns
    -------
    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """
    geocorners = get_geo_corners(grid)
    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        check_symmetric(grid),
        periodic=(grid.axes['X']._periodic),
        topology=topology
    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    symmetric,
    periodic=("X"),
    topology="cartesian"
    ):
    """
    Compute composite section along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates geodesic paths between consecutive points defined by (lons, lats).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude (with dimensions ('Y', 'X')), in degrees
    gridlat: np.ndarray
        2d array of latitude (with dimensions ('Y', 'X')), in degrees
    lons: list of float
        longitude of section starting, intermediate and end points, in degrees
    lats: list of float
        latitude of section starting, intermediate and end points, in degrees
    symmetric: bool
        True if symmetric (vorticity on "outer" positions); False if non-symmetric (assuming "right" positions).
    periodic: ("X") or False
        Default: ("X"). Set to False if using a non-periodic regional domain. For "periodic=False", the algorithm will
        break if shortest paths between two points in the domain leaves the domain!
    topology: str
        Default: 'cartesian'. Currently only supports the following options: ['cartesian', 'MOM-tripolar'].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
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

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, symmetric, periodic=("X"), topology="cartesian"):
    """
    Compute a section segment along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates the geodesic path between points (lonstart, latstart) and (lonend, latend).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude (with dimensions ('Y', 'X')), in degrees
    gridlat: np.ndarray
        2d array of latitude (with dimensions ('Y', 'X')), in degrees
    lonstart: float
        longitude of starting point, in degrees
    lonend: float
        longitude of end point, in degrees
    latstart: float
        latitude of starting point, in degrees
    latend: float
        latitude of end point, in degrees
    symmetric: bool
        True if symmetric (vorticity on "outer" positions); False if non-symmetric (assuming "right" positions).
    periodic: ("X") or False
        Default: ("X"). Set to False if using a non-periodic regional domain. For "periodic=False", the algorithm will
        break if shortest paths between two points in the domain leaves the domain!
    topology: str
        Default: 'cartesian'. Currently only supports the following options: ['cartesian', 'MOM-tripolar'].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """

    if symmetric and periodic==("X"):
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

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, periodic=("X"), topology="cartesian"):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximates
    the geodesic path between points (lonstart, latstart) and (lonend, latend).

    PARAMETERS:
    -----------

    lonstart: float
        longitude of section starting point, in degrees
    latstart: float
        latitude of section starting point, in degrees
    lonend: float
        longitude of section end point, in degrees
    latend: float
        latitude of section end point, in degrees
    gridlon: np.ndarray
        2d array of longitude, in degrees
    gridlat: np.ndarray
        2d array of latitude, in degrees
    periodic: ("X") or False
        Default: ("X"). Set to False if using a non-periodic regional domain. For "periodic=False", the algorithm will
        break if shortest paths between two points in the domain leaves the domain!
    topology: str
        Default: 'cartesian'. Currently only supports the following options: ['cartesian', 'MOM-tripolar'].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
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


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, periodic=("X"), topology="cartesian"):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximate
    the geodesic path between points (gridlon[j1,i1], gridlat[j1,i1]) and
    (gridlon[j2,i2], gridlat[j2,i2]).

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
        2d array of longitude, in degrees
    gridlat: np.ndarray
        2d array of latitude, in degrees
    periodic: ("X") or False
        Default: ("X"). Set to False if using a non-periodic regional domain. For "periodic=False", the algorithm will
        break if shortest paths between two points in the domain leaves the domain!
    topology: str
        Default: 'cartesian'. Currently only supports the following options: ['cartesian', 'MOM-tripolar'].

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
        
        if d_current == 0.:
            break
        
        if periodic==("X"):
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
    """
    Find integer indices of closest grid point in grid of coordinates
    (gridlon, gridlat), for a given point (lon, at).

    PARAMETERS:
    -----------
        lon (float): longitude of point to find, in degrees
        lat (float): latitude of point to find, in degrees
        gridlon (numpy.ndarray): grid longitudes, in degrees
        gridlat (numpy.ndarray): grid latitudes, in degrees

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

def distance_on_unit_sphere(lon1, lat1, lon2, lat2, R=6.371e6, method="vincenty"):
    """
    Calculate geodesic arc distance between points (lon1, lat1) and (lon2, lat2).

    PARAMETERS:
    -----------
        lon1 : float
            Start longitude(s), in degrees
        lat1 : float
            Start latitude(s), in degrees
        lon2 : float
            End longitude(s), in degrees
        lat2 : float
            End latitude(s), in degrees
        R : float
            Radius of sphere. Default: 6.371e6 (realistic Earth value). Set to 1 for
            arc distance in radius.
        method : str
            Name of method. Supported methods: ["vincenty", "haversine", "law of cosines"].
            Default: "vincenty", which is the most robust. Note, however, that it still can result in
            vanishingly small (but crucially non-zero) errors; such as that the distance between (0., 0.)
            and (360., 0.) is 1.e-16 meters when it should be identically zero.

    RETURNS:
    --------

    dist : float
        Geodesic distance between points (lon1, lat1) and (lon2, lat2).
    """
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.abs(phi2-phi1)
    
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

    return R * arc

def spherical_angle(lonA, latA, lonB, latB, lonC, latC):
    """
    Calculate the spherical triangle angle alpha between geodesic arcs AB and AC defined by
    [(lonA, latA), (lonB, latB)] and [(lonA, latA), (lonC, latC)], respectively.

    PARAMETERS:
    -----------
        lonA : float
            Longitude of point A, in degrees
        latA : float
            Latitude of point A, in degrees
        lonB : float
            Longitude of point B, in degrees
        latB : float
            Latitude of point B, in degrees
        lonC : float
            Longitude of point C, in degrees
        latC : float
            Latitude of point C, in degrees

    RETURNS:
    --------

    angle : float
        Spherical absolute value of triangle angle alpha, in radians.
    """
    a = distance_on_unit_sphere(lonB, latB, lonC, latC, R=1.)
    b = distance_on_unit_sphere(lonC, latC, lonA, latA, R=1.)
    c = distance_on_unit_sphere(lonA, latA, lonB, latB, R=1.)
        
    return np.arccos(np.clip((np.cos(a) - np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)), -1., 1.))

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