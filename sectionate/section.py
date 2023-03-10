import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def create_section_composite(gridlon, gridlat, segment_lons, segment_lats, closed=False, periodic=True):
    """create section from list of segments

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude
    gridlat: np.ndarray
        2d array of latitude
    segment_lons: list of float
        longitude of section starting, intermediate and end points
    segment_lats: list of float
        latitude of section starting, intermediate and end points


    RETURNS:
    -------

    isect, jsect: list of int
        list of (i,j) pairs for section
    xsect, ysect: list of float
        corresponding longitude and latitude for isect, jsect

    """

    isect = np.array([], dtype=np.int64)
    jsect = np.array([], dtype=np.int64)
    xsect = np.array([], dtype=np.float64)
    ysect = np.array([], dtype=np.float64)

    if len(segment_lons) != len(segment_lats):
        raise ValueError("segment_lons and segment_lats should have the same length")

    for k in range(len(segment_lons) - 1):
        iseg, jseg, xseg, yseg = create_section(
            gridlon,
            gridlat,
            segment_lons[k],
            segment_lats[k],
            segment_lons[k + 1],
            segment_lats[k + 1],
            periodic=periodic
        )

        isect = np.concatenate([isect, iseg[:-1]], axis=0)
        jsect = np.concatenate([jsect, jseg[:-1]], axis=0)
        xsect = np.concatenate([xsect, xseg[:-1]], axis=0)
        ysect = np.concatenate([ysect, yseg[:-1]], axis=0)
        
    if not closed:
        isect = np.concatenate([isect, [iseg[-1]]], axis=0)
        jsect = np.concatenate([jsect, [jseg[-1]]], axis=0)
        xsect = np.concatenate([xsect, [xseg[-1]]], axis=0)
        ysect = np.concatenate([ysect, [yseg[-1]]], axis=0)
        
    if closed:
        isect = np.append(isect, isect[0])
        jsect = np.append(jsect, jsect[0])
        xsect = np.append(xsect, xsect[0])
        ysect = np.append(ysect, ysect[0])

    return isect.astype(np.int64), jsect.astype(np.int64), xsect, ysect


def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, periodic=True):
    """ replacement function for the old create_section """

    iseg, jseg, lonseg, latseg = infer_broken_line_from_geo(
        lonstart, latstart, lonend, latend, gridlon, gridlat, periodic=periodic
    )
    return iseg, jseg, lonseg, latseg


def infer_broken_line_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, periodic=True):
    """find the broken line joining (lonstart, latstart) and (lonend, latend) pairs

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

    istart, jstart = find_closest_grid_point(lonstart, latstart, gridlon, gridlat)
    iend, jend = find_closest_grid_point(lonend, latend, gridlon, gridlat)
    iseg, jseg, lonseg, latseg = infer_broken_line(
        istart, jstart, iend, jend, gridlon, gridlat, periodic=periodic
    )

    return iseg, jseg, lonseg, latseg


def infer_broken_line(i1, j1, i2, j2, gridlon, gridlat, periodic=True):
    """find the broken line joining (i1, j1) and (i2, j2) pairs

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

    # find length of segment, as straight line and broken line following grid
    #print(idir, jdir, wrapsign, nx)
    di = (((i2 - i1)*idir)%nx)*idir
    dj = j2-j1
    L = np.sqrt( di**2 + dj**2 )
    Nsteps = np.abs(di) + np.abs(dj)
    
    # compute slope of the section
    if idir != 0:
        slope = dj / di
        vertical = False
    else:
        vertical = True
        
    # iterate until we reach end of section
    #print([i1, j1], "to", [i2, j2], [di, dj], end="\n")
    while (i%nx != i2) or (j != j2):
        # safety precaution, exit after N iterations
        if ct > Nsteps:
            raise RuntimeError("max iterations reached")
        ct += 1

        # get 2 possible next points
        inext = i + idir
        jnext = j + jdir

        # increment prediction along shortest path line
        if vertical:
            ipred, jpred = inext, jnext
        else:
            ipred = i1 + di * ct/Nsteps
            jpred = j1 + dj * ct/Nsteps

        # compute squared distances from predicted j to j and jnext
        di_pred = ((inext - ipred)**2 + (j     - jpred)**2)
        dj_pred = ((i     - ipred)**2 + (jnext - jpred)**2)

        #print([i,j], [inext, jnext], [ipred, jpred])
        # increment direction that ends up closer to predicted point along shorted path
        if dj_pred < di_pred:
            j = jnext
        elif dj_pred > di_pred:
            i = inext
        elif dj_pred == di_pred:
            if idir == 1:
                i = inext
            else:
                j = jnext

        # add new point to list
        iseg.append(i%nx)
        jseg.append(j)

    # create lat/lon vectors from i,j pairs
    lonseg = []
    latseg = []
    for jj, ji in zip(jseg, iseg):
        lonseg.append(gridlon[jj, ji])
        latseg.append(gridlat[jj, ji])
    return iseg, jseg, lonseg, latseg


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


def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = np.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    cos = np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(
        phi2
    )
    arc = np.arccos(cos)
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc


# ------------------------- older functions, here for legacy purposes ------------------
def linear_fit(x, y, x1, y1, x2, y2, eps=1e-12):
    """generate the function for which we want to find zero contour

    PARAMETERS:
    -----------
    x: numpy.ndarray
        2d array of coordinate1 (e.g. longitude)
    y: numpy.ndarray
        2d array of coordinate2 (e.g. latitude)
    x1: float
        x-coord of point1
    y1: float
        y-coord of point1
    x2: float
        x-coord of point2
    y2: float
        y-coord of point2

    """
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    alpha = (y2 - y1) / (x2 - x1 + eps)
    func = y - y1 - alpha * (x - x1)
    return func


def create_zero_contour(func, debug=False):
    plt.figure()
    cont = plt.contour(func, 0)
    if debug:
        plt.show()
    plt.close()
    return cont


def get_broken_line_from_contour(
    contour, rounding="down", debug=False, maxdist=10000000
):
    """return a broken line from contour, suitable to integrate transport

    PARAMETERS:
    -----------

    contour : matplotlib contour object
        output of contour(func, [0])

    RETURNS:
    --------

    iseg, jseg: numpy.ndarray
        i,j indices of broken line
    """
    # first guess of our broken line is the contour position in (x,y)
    # raw array has float grid indices
    for item in contour.allsegs:
        if len(item) != 0:
            # discontinuous contours are stored in multiple arrays
            # so we need to concatenate them
            contourxy = None
            for cont in item:
                if len(cont) != 0:
                    if contourxy is None:
                        contourxy = cont
                    else:
                        contourxy = np.concatenate([contourxy, cont])
    xseg_raw = contourxy[:, 0]
    yseg_raw = contourxy[:, 1]

    # 1st guess: convert to integer
    nseg = len(xseg_raw)
    if rounding == "down":
        iseg_fg = np.floor(xseg_raw).astype("int")
        jseg_fg = np.floor(yseg_raw).astype("int")
    elif rounding == "up":
        iseg_fg = np.ceil(xseg_raw).astype("int")
        jseg_fg = np.ceil(yseg_raw).astype("int")
    else:
        raise ValueError("Unkown rounding, only up or down")

    # 2nd guess: create broken line along cell edges
    iseg_sg = [iseg_fg[0]]
    jseg_sg = [jseg_fg[0]]

    for kseg in np.arange(1, nseg):
        if (iseg_fg[kseg] - iseg_fg[kseg - 1] == 0) and (
            jseg_fg[kseg] - jseg_fg[kseg - 1] == 0
        ):
            pass  # we don't want to double count points
        elif (iseg_fg[kseg] - iseg_fg[kseg - 1] == 0) or (
            jseg_fg[kseg] - jseg_fg[kseg - 1] == 0
        ):
            # we are along one face of the cell
            # check for "missing" points
            if maxdist > np.abs(iseg_fg[kseg] - iseg_fg[kseg - 1]) > 1:
                if debug:
                    print(
                        f"info: filling {iseg_fg[kseg] - iseg_fg[kseg-1]} points in i between {iseg_fg[kseg-1]} and {iseg_fg[kseg]}"
                    )
                # add missing points
                for kpt in range(iseg_fg[kseg - 1] + 1, iseg_fg[kseg] + 1):
                    iseg_sg.append(kpt)
                    jseg_sg.append(jseg_fg[kseg])
            elif maxdist > np.abs(jseg_fg[kseg] - jseg_fg[kseg - 1]) > 1:
                if debug:
                    print(
                        f"info: filling {jseg_fg[kseg] - jseg_fg[kseg-1]} points in j between {jseg_fg[kseg-1]} and {jseg_fg[kseg]}"
                    )
                # add missing points
                for kpt in range(jseg_fg[kseg - 1] + 1, jseg_fg[kseg] + 1):
                    iseg_sg.append(iseg_fg[kseg])
                    jseg_sg.append(kpt)
            else:
                iseg_sg.append(iseg_fg[kseg])
                jseg_sg.append(jseg_fg[kseg])
        else:
            # we need to create two segments stopping by
            # an intermediate cell corner
            iseg_sg.append(iseg_fg[kseg])
            jseg_sg.append(jseg_fg[kseg - 1])
            iseg_sg.append(iseg_fg[kseg])
            jseg_sg.append(jseg_fg[kseg])

    iseg = np.array(iseg_sg, np.dtype("i"))
    jseg = np.array(jseg_sg, np.dtype("i"))

    return iseg, jseg


def bound_broken_line(x, y, x1, y1, x2, y2, iseg, jseg, tol=1.0):
    """subset the broken line between the bounds

    PARAMETERS:
    -----------
    x: numpy.ndarray
        2d array of coordinate1 (e.g. longitude)
    y: numpy.ndarray
        2d array of coordinate2 (e.g. latitude)
    x1: float
        x-coord of point1
    y1: float
        y-coord of point1
    x2: float
        x-coord of point2
    y2: float
        y-coord of point2
    iseg: numpy.ndarray
        1d vector of section points i-index
    jseg: numpy.ndarray
        1d vector of section points j-index

    RETURNS:
    --------

    iseg_bnd, j_seg_bnd: numpy.ndarray
        iseg, jseg bounded by (x1, y1) and (x2, y2)
    xseg_bnd, yseg_bnd: numpy.ndarray
        corresponding geographical coordinates
    """
    nseg = len(iseg)
    iseg_bnd = []  # indices along x
    jseg_bnd = []  # indices along y
    xseg_bnd = []  # x coord values
    yseg_bnd = []  # y coord values
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)
    # We should test for numpy/xaray type for x and y and
    # use .values if xarray
    xbndlow = xmin
    xbndhigh = xmax
    if np.isclose(xmin, xmax, atol=tol):
        xbndlow -= tol
        xbndhigh += tol
    ybndlow = ymin
    ybndhigh = ymax
    if np.isclose(ymin, ymax, atol=tol):
        ybndlow -= tol
        ybndhigh += tol

    for k in range(nseg):
        x_pt = x[jseg[k], iseg[k]]
        y_pt = y[jseg[k], iseg[k]]
        if (xbndlow <= x_pt <= xbndhigh) and (ybndlow <= y_pt <= ybndhigh):
            iseg_bnd.append(iseg[k])
            jseg_bnd.append(jseg[k])
            xseg_bnd.append(x_pt)
            yseg_bnd.append(y_pt)

    iseg_bnd = np.array(iseg_bnd)
    jseg_bnd = np.array(jseg_bnd)
    xseg_bnd = np.array(xseg_bnd)
    yseg_bnd = np.array(yseg_bnd)

    return iseg_bnd, jseg_bnd, xseg_bnd, yseg_bnd


def create_section_old(
    x, y, x1, y1, x2, y2, method="linear", tol=1.0, rounding="best", debug=False
):
    if method == "linear":
        func = linear_fit(x, y, x1, y1, x2, y2)
    else:
        ValueError("only linear is available now")
    cont = create_zero_contour(func, debug=debug)
    # generate both contours (rounding up and down)
    iseg_u, jseg_u = get_broken_line_from_contour(cont, rounding="up", debug=debug)
    isec_u, jsec_u, xsec_u, ysec_u = bound_broken_line(
        x, y, x1, y1, x2, y2, iseg_u, jseg_u, tol=tol
    )
    iseg_d, jseg_d = get_broken_line_from_contour(cont, rounding="down", debug=debug)
    isec_d, jsec_d, xsec_d, ysec_d = bound_broken_line(
        x, y, x1, y1, x2, y2, iseg_d, jseg_d, tol=tol
    )

    if rounding == "down":
        isec, jsec, xsec, ysec = isec_d, jsec_d, xsec_d, ysec_d
    elif rounding == "up":
        isec, jsec, xsec, ysec = isec_u, jsec_u, xsec_u, ysec_u
    elif rounding == "best":
        dist_d1 = []
        dist_d2 = []
        dist_u1 = []
        dist_u2 = []
        for k in range(len(xsec_d)):
            dist_d1.append(distance_on_unit_sphere(y1, x1, ysec_d, xsec_d))
            dist_d2.append(distance_on_unit_sphere(y2, x2, ysec_d, xsec_d))
        for k in range(len(xsec_u)):
            dist_u1.append(distance_on_unit_sphere(y1, x1, ysec_u, xsec_u))
            dist_u2.append(distance_on_unit_sphere(y2, x2, ysec_u, xsec_u))

        dist_d1 = np.array(dist_d1)
        dist_d2 = np.array(dist_d2)
        dist_u1 = np.array(dist_u1)
        dist_u2 = np.array(dist_u2)
        # start counting points
        down = 0
        up = 0
        if dist_d1.min() < dist_u1.min():
            down += 1
        if dist_d2.min() < dist_u2.min():
            down += 1
        if dist_d1.min() > dist_u1.min():
            up += 1
        if dist_d2.min() > dist_u2.min():
            up += 1

        if (up > down) and (up != 0):
            print("best fit is rounding up")
            isec, jsec, xsec, ysec = isec_u, jsec_u, xsec_u, ysec_u
        elif (down > up) and (down != 0):
            print("best fit is rounding down")
            isec, jsec, xsec, ysec = isec_d, jsec_d, xsec_d, ysec_d
        else:
            raise ValueError("cannot choose...")
    else:
        raise ValueError("unknown roundup type, only up, down and best")
    return isec, jsec, xsec, ysec
