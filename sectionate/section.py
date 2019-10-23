import numpy as np
import matplotlib.pyplot as plt


def linear_fit(x, y, x1, y1, x2, y2, eps=1e-12):
    """ generate the function for which we want to find zero contour

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

    alpha = (y2-y1) / (x2-x1+eps)
    func = y - y1 - alpha * (x-x1)
    return func


def create_zero_contour(func):
    plt.figure()
    cont = plt.contour(func, 0)
    #plt.show()
    plt.close()
    return cont


def get_broken_line_from_contour(contour):
    ''' return a broken line from contour, suitable to integrate transport

    PARAMETERS:
    -----------

    contour : matplotlib contour object
        output of contour(func, [0])

    RETURNS:
    --------

    iseg, jseg: numpy.ndarray
        i,j indices of broken line
    '''
    # first guess of our broken line is the contour position in (x,y)
    # raw array has float grid indices
    for item in contour.allsegs:
        if len(item) !=0:
            contourxy = item[0]
    xseg_raw = contourxy[:, 0]
    yseg_raw = contourxy[:, 1]

    # 1st guess: convert to integer
    nseg = len(xseg_raw)
    iseg_fg = np.floor(xseg_raw)
    jseg_fg = np.floor(yseg_raw)

    # 2nd guess: create broken line along cell edges
    iseg_sg = [iseg_fg[0]]
    jseg_sg = [jseg_fg[0]]

    for kseg in np.arange(1, nseg):
        if (iseg_fg[kseg] - iseg_fg[kseg-1] == 0) or \
           (jseg_fg[kseg] - jseg_fg[kseg-1] == 0):
            # we are along one face of the cell so we're good
            iseg_sg.append(iseg_fg[kseg])
            jseg_sg.append(jseg_fg[kseg])
        else:
            # we need to create two segments stopping by
            # an intermediate cell corner
            iseg_sg.append(iseg_fg[kseg])
            jseg_sg.append(jseg_fg[kseg-1])
            iseg_sg.append(iseg_fg[kseg])
            jseg_sg.append(jseg_fg[kseg])

    iseg = np.array(iseg_sg, np.dtype('i'))
    jseg = np.array(jseg_sg, np.dtype('i'))

    return iseg, jseg


def bound_broken_line(x, y, x1, y1, x2, y2, iseg, jseg, tol=1.):
    """ subset the broken line between the bounds

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


def create_section(x, y, x1, y1, x2, y2, method='linear'):
    if method == 'linear':
        func = linear_fit(x, y, x1, y1, x2, y2)
    else:
        ValueError('only linear is available now')
    cont = create_zero_contour(func)
    iseg, jseg = get_broken_line_from_contour(cont)
    isec, jsec, xsec, ysec = bound_broken_line(x, y, x1, y1, x2, y2,
                                               iseg, jseg)
    return isec, jsec, xsec, ysec
