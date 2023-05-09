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

    # init loop index to starting position
    i = i1
    j = j1

    iseg = [i]  # add first point to list of points
    jseg = [j]  # add first point to list of points
    ct = 0  # counter for max iteration safeguard
    
    # check if shortest path crosses periodic boundary (if applicable)
    wraplons = False
    if periodic:
        wraplons = np.abs(gridlon[j2, i2] - gridlon[j1, i1]) > 180.
    wrapsign = int(not(wraplons))*2-1

    # find direction of iteration
    idir = np.sign(i2 - i1)*wrapsign
    jdir = np.sign(j2 - j1)

    # find length of segment, as both straight line (L) and gri path following grid (Nsteps)
    di = (((i2 - i1)*idir)%nx)*idir
    dj = j2-j1
    L = np.sqrt( di**2 + dj**2 )
    Nsteps = np.abs(di) + np.abs(dj)
    
    # compute slope of the section straight line segment
    if idir != 0:
        slope = dj / di
        vertical = False
    else:
        vertical = True
        
    # iterate through the broken line segments until we reach end of section
    while (i%nx != i2) or (j != j2):
        # safety precaution, exit after N iterations
        if ct > Nsteps:
            raise RuntimeError(f"Should have reached the endpoint by now: {ct}/{Nsteps} steps.")
        ct += 1

        # get 2 possible next points
        inext = i + idir
        jnext = j + jdir

        # increment prediction along shortest path (straight line)
        if vertical:
            ipred, jpred = inext, jnext
        else:
            ipred = i1 + di * ct/Nsteps
            jpred = j1 + dj * ct/Nsteps

        # compute squared distances from predicted j to j and jnext
        di_pred = ((inext - ipred)**2 + (j     - jpred)**2)
        dj_pred = ((i     - ipred)**2 + (jnext - jpred)**2)

        # increment direction that ends up closer to predicted point along shorted path
        if np.isclose(dj_pred, di_pred):
            if idir == 1:
                i = inext
            else:
                j = jnext
        elif dj_pred < di_pred:
            j = jnext
        elif dj_pred > di_pred:
            i = inext

        # add new point to list
        iseg.append(i%nx)
        jseg.append(j)

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
    dist = distance_on_unit_sphere(lat, lon, gridlat, gridlon)
    jclose, iclose = np.unravel_index(dist.argmin(), gridlon.shape)
    return iclose, jclose


def distance_on_unit_sphere(lat1, lon1, lat2, lon2, method="law of cosines", R=6.371e6):

    if method=="law of cosines":
        # phi = 90 - latitude
        phi1 = np.deg2rad(90.0 - lat1)
        phi2 = np.deg2rad(90.0 - lat2)

        # theta = longitude
        theta1 = np.deg2rad(lon1)
        theta2 = np.deg2rad(lon2)
        # Compute spherical distance from spherical coordinates.
        # For two locations in spherical coordinates
        # (1, theta, phi) and (1, theta, phi)
        # cosine( arc length ) =
        #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
        # distance = rho * arc length
        cos = np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(
            phi2
        )
        arc = np.arccos(cos)/2

    if method=="haversine":
        lon1, lat1, lon2, lat2 = np.deg2rad(lon1), np.deg2rad(lat1), np.deg2rad(lon2), np.deg2rad(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
        arc = np.arcsin(np.sqrt(a))
    
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return 2 * R * arc