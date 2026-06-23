import numpy as np
import xarray as xr

from .gridutils import (
    get_geo_corners,
    check_symmetric,
    get_facedim,
    build_neighbor_maps,
    simple_neighbor_maps,
)

class Section():
    """A named hydrographic section"""
    def __init__(self, name, coords, children = {}, parent = None):
        """Initiate named hydrographic section

        Arguments
        ---------
        name [str] -- name of the section
        coords [list or tuple] -- coordinates that define the section
        
            If type is list, elements of the list must all be 2-tuples
            of the form (lon, lat).
            
            If type is tuple, it must be of the form (lons, lats), where
            lons and lats are lists of np.ndarray instances of the same
            length with elements of type float.

        Keyword Arguments
        -----------------
        children [mapping from str to Section (default: {})] -- dictionary
            mapping the names of child sections to their Section instances.
            This attribute will generally be populated automatically from
            the function `join_sections`.

        parent [Section (default: None)] -- TO DO

        Returns
        -------
        instance of Section

        Examples
        --------
        >>> sec.Section("Bering Strait", [(-170.3, 66.1), (-167.6,65.7)])
        Section(Bering Strait, [(-170.3, 66.1), (-167.6, 65.7)])
        """
        self.name = name
        if type(coords) is tuple:
            if len(coords) == 2:
                self.update_coords(coords_from_lonlat(coords[0], coords[1]))
            else:
                raise ValueError("If coords is a tuple, must be (lons, lats)")
        elif type(coords) is list:
            if all([(type(c) is tuple) and (len(c)==2) for c in coords]):
                self.coords = coords.copy()
                self.lons_c, self.lats_c = lonlat_from_coords(self.coords)
            else:
                raise ValueError("If coords is a list, its elements must be (lon,lat) 2-tuples")
        else:
            raise ValueError("coords must be a 2-tuple of lists/arrays or a list of 2-tuples")
            
        self.children = children.copy() # need this to be a copy or get a recursion error in __repr__...
        self.parent = parent
        self.save = {}

    def reverse(self):
        """Reverse the section's direction"""
        self.update_coords(self.coords[::-1])
        return self

    def update_coords(self, coords):
        """Update coordinates (including longitude and latitude arrrays)"""
        self.coords = coords
        self.lons_c, self.lats_c = lonlat_from_coords(self.coords)

    def copy(self):
        """Create a deep copy of the Section instance"""
        section = Section(
            self.name,
            self.coords,
            children=self.children,
            parent=self.parent
        )
        section.save = self.save.copy()
        return section
    
    def __repr__(self, indent=0, show_attributes=True):
        indent_str = "  " * indent
        summary = f"{indent_str}Section({self.name}, {self.coords})"
        
        # Automatically extract and add attributes
        if show_attributes:
            summary += f"\n{indent_str}attributes:"
            for attr, value in vars(self).items():
                if attr not in ['children', 'save'] and not attr.startswith('_'):
                    summary += f"\n{indent_str}  {attr}"
    
        if len(self.children) > 0:
            summary += f"\n{indent_str}  children:"
            for child in self.children.values():
                child_repr = child.__repr__(indent + 3, show_attributes=False)
                summary += f"\n{indent_str}    - {child_repr.lstrip()}"
        return summary
        
class GriddedSection(Section):
    """Initiate named hydrographic section specific to an ocean model grid

    Arguments
    ---------
    section [sectionate.Section] -- named Sectionate section
    grid [xgcm.Grid] -- ocean model grid object

    Keyword Arguments
    -----------------
    i_c [list or np.ndarray] -- x corner indices
    j_c [list or np.ndarray] -- y corner indices

    Returns
    -------
    instance of GriddedSection
    """
    def __init__(self, section, grid, i_c=None, j_c=None, f_c=None):
        super().__init__(
            section.name,
            section.coords,
            children = section.children,
            parent = section.parent
        )
        self.grid = grid
        self.f_c = f_c
        if isinstance(i_c, (list, np.ndarray)) & isinstance(j_c, (list, np.ndarray)):
            self.i_c = i_c
            self.j_c = j_c
        else:
            self.grid_section()

    def grid_section(self, **kwargs):
        """Pass this Section's coordinates to sectionate.grid_section

        Arguments
        ---------
        grid

        Keyword Arguments
        -----------------
        **kwargs passed directly to sectionate.grid_section
        """
        out = grid_section(
            self.grid,
            self.lons_c,
            self.lats_c,
            **kwargs
        )
        if len(out) == 5:
            self.i_c, self.j_c, self.f_c, self.lons_c, self.lats_c = out
        else:
            self.i_c, self.j_c, self.lons_c, self.lats_c = out
            self.f_c = None

        return out
    
    def copy(self):
        """Creates a copy of a GriddedSection, with deep copies of all attributes except the grid."""
        super().copy()
        self.i_c = self.i_c.copy()
        self.j_c = self.j_c.copy()


def join_sections(name, *sections, **kwargs):
    """
    Joins child Sections together to create a parent Section.

    Arguments
    ---------
    name [str] -- name of the parent section
    *sections [Section] -- the sequence of child sections to be joined

    Keyword Arguments
    -----------------
    align [bool (Default : True)] -- reverse sections as needed to minimize
        the distance between end/start points of consecutive sections

    Returns
    -------
    instance of Section

    Example
    -------
    >>> section1 = sec.Section("section1", ([0., 100.], [0., 0.]))
    >>> section2 = sec.Section("section2", ([100., 200.], [0., 0.]))
    >>> section = sec.join_sections("section", section1, section2)
    >>> section
    Section(section, [(0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (200.0, 0.0)])
 Children:
  - Section(section1, [(0.0, 0.0), (100.0, 0.0)])
  - Section(section2, [(100.0, 0.0), (200.0, 0.0)])
    """
    if type(name) is not str:
        raise ValueError("first argument (name) must be a str.")
    elif any([not(isinstance(s, Section)) for s in sections]):
        raise ValueError("all positional arguments after the first must be instances of Section")
    align = kwargs["align"] if "align" in kwargs else True
    extend = kwargs["extend"] if "extend" in kwargs else False
    
    section = Section(name, sections[0].coords)
    if len(sections) > 1:
        for i, s in enumerate(sections[1:], start=1):
            if not(align):
                coords1, coords2 = section.coords, s.coords
            else:
                coords1, coords2 = align_coords(
                    section.coords,
                    s.coords,
                    extend=extend
                )

            s.update_coords(coords2)
            section.update_coords(coords1 + coords2)
            
            if i == 1:
                sections[0].update_coords(coords1)
                section.children[sections[0].name] = sections[0]
            section.children[s.name] = s
            
    return section
        
def grid_section(grid, lons, lats):
    """
    Compute composite section along model `grid` velocity faces that approximates geodesic paths
    between consecutive points defined by (lons, lats).

    The grid topology is inferred entirely from the `grid` metadata: each axis' `boundary`
    condition ("periodic" wraps, otherwise clip) for single-tile grids, and `face_connections`
    for multi-tile grids (e.g. the lat-lon-cap or cubed-sphere).

    Parameters
    ----------
    grid: xgcm.Grid
        Object describing the geometry of the ocean model grid, including metadata about variable names for
        the staggered C-grid dimensions and coordinates.
    lons: list or np.ndarray
        Longitudes, in degrees, of consecutive vertices defining a piece-wise geodesic section.
    lats: list or np.ndarray
        Latitudes, in degrees (in range [-90, 90]), of consecutive vertices defining a piece-wise geodesic section.

    Returns
    -------
    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c) correspond to indices of vorticity points that define velocity faces. For
        multi-tile grids, the face index f_c of each point is returned as well.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """
    geocorners = get_geo_corners(grid)
    boundary = {ax:grid.axes[ax].boundary for ax in grid.axes}

    facedim = get_facedim(grid)
    if facedim is not None:
        _check_supported_topology(grid)
        neighbor_maps = build_neighbor_maps(grid, geocorners)
    else:
        neighbor_maps = None

    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        check_symmetric(grid),
        boundary=boundary,
        neighbor_maps=neighbor_maps,
    )


def _check_supported_topology(grid):
    """
    Raise if the multi-tile `grid` requires topology features sectionate does not yet support.

    Specifically, the tripolar/bipolar north fold is represented as a face that connects to
    itself (a self-connection) along the "Y" axis. xgcm does not support this either (see
    https://github.com/xgcm/xgcm/issues/194), and sectionate's pathfinder cannot currently
    cross such a seam, so we refuse it explicitly rather than silently produce a wrong path.
    """
    facedim = grid._facedim
    for axis in grid.axes:
        connections = getattr(grid.axes[axis], "_connections", None) or {}
        for face, sides in connections.items():
            for side in sides:
                if side is None:
                    continue
                neighbor_face = side[0]
                if neighbor_face == face:
                    raise NotImplementedError(
                        "Grids with a face that connects to itself (e.g. the tripolar/bipolar "
                        "north fold) are not yet supported. Sections that do not cross the fold "
                        "work with a single-tile grid using boundary={'X':'periodic','Y':'extend'}."
                    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    symmetric,
    boundary={"X":"periodic", "Y":"extend"},
    neighbor_maps=None
    ):
    """
    Compute composite section along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates geodesic paths between consecutive points defined by (lons, lats).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    lons: list of float
        longitude of section starting, intermediate and end points, in degrees
    lats: list of float
        latitude of section starting, intermediate and end points, in degrees
    symmetric: bool
        True if symmetric (vorticity on "outer" positions); False if non-symmetric (assuming "right" positions).
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    neighbor_maps: dict or None
        Precomputed topology-aware neighbor maps for multi-tile grids (see
        `sectionate.gridutils.build_neighbor_maps`); None for single-tile grids.

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    multitile = neighbor_maps is not None

    i_c = np.array([], dtype=np.int64)
    j_c = np.array([], dtype=np.int64)
    f_c = np.array([], dtype=np.int64)
    lons_c = np.array([], dtype=np.float64)
    lats_c = np.array([], dtype=np.float64)

    if len(lons) != len(lats):
        raise ValueError("lons and lats should have the same length")

    for k in range(len(lons) - 1):
        seg = create_section(
            gridlon,
            gridlat,
            lons[k],
            lats[k],
            lons[k + 1],
            lats[k + 1],
            symmetric,
            boundary=boundary,
            neighbor_maps=neighbor_maps,
        )
        if multitile:
            i_c_seg, j_c_seg, f_c_seg, lons_c_seg, lats_c_seg = seg
        else:
            i_c_seg, j_c_seg, lons_c_seg, lats_c_seg = seg

        i_c = np.concatenate([i_c, i_c_seg[:-1]], axis=0)
        j_c = np.concatenate([j_c, j_c_seg[:-1]], axis=0)
        lons_c = np.concatenate([lons_c, lons_c_seg[:-1]], axis=0)
        lats_c = np.concatenate([lats_c, lats_c_seg[:-1]], axis=0)
        if multitile:
            f_c = np.concatenate([f_c, f_c_seg[:-1]], axis=0)

    i_c = np.concatenate([i_c, [i_c_seg[-1]]], axis=0)
    j_c = np.concatenate([j_c, [j_c_seg[-1]]], axis=0)
    lons_c = np.concatenate([lons_c, [lons_c_seg[-1]]], axis=0)
    lats_c = np.concatenate([lats_c, [lats_c_seg[-1]]], axis=0)
    if multitile:
        f_c = np.concatenate([f_c, [f_c_seg[-1]]], axis=0)
        return i_c.astype(np.int64), j_c.astype(np.int64), f_c.astype(np.int64), lons_c, lats_c

    return i_c.astype(np.int64), j_c.astype(np.int64), lons_c, lats_c

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, symmetric, boundary={"X":"periodic", "Y":"extend"}, neighbor_maps=None):
    """
    Compute a section segment along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates the geodesic path between points (lonstart, latstart) and (lonend, latend).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude (with dimensions ("Y", "X")), in degrees
    gridlat: np.ndarray
        2d array of latitude (with dimensions ("Y", "X")), in degrees
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
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    neighbor_maps: dict or None
        Precomputed topology-aware neighbor maps for multi-tile grids (see
        `sectionate.gridutils.build_neighbor_maps`); None for single-tile grids.

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    # Symmetric periodic single-tile grids carry a redundant final corner column
    # (the periodic wrap of the first); drop it so periodicity is expressed purely
    # by the modulo step. Multi-tile periodicity is handled by `face_connections`.
    if symmetric and boundary["X"] == "periodic" and neighbor_maps is None:
        gridlon=gridlon[:,:-1]
        gridlat=gridlat[:,:-1]

    return infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        boundary=boundary,
        neighbor_maps=neighbor_maps,
    )

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, boundary={"X":"periodic", "Y":"extend"}, neighbor_maps=None):
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
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    neighbor_maps: dict or None
        Precomputed topology-aware neighbor maps for multi-tile grids (see
        `sectionate.gridutils.build_neighbor_maps`); None for single-tile grids.

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    multitile = neighbor_maps is not None
    if multitile:
        istart, jstart, fstart = find_closest_grid_point(lonstart, latstart, gridlon, gridlat)
        iend, jend, fend = find_closest_grid_point(lonend, latend, gridlon, gridlat)
    else:
        istart, jstart = find_closest_grid_point(lonstart, latstart, gridlon, gridlat)
        iend, jend = find_closest_grid_point(lonend, latend, gridlon, gridlat)
        fstart, fend = None, None

    return infer_grid_path(
        istart,
        jstart,
        iend,
        jend,
        gridlon,
        gridlat,
        boundary=boundary,
        neighbor_maps=neighbor_maps,
        f1=fstart,
        f2=fend,
    )


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, boundary={"X":"periodic", "Y":"extend"}, neighbor_maps=None, f1=None, f2=None):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximate
    the geodesic path between the starting and ending corner points.

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
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
        Only used for single-tile grids (when `neighbor_maps is None`).
    neighbor_maps: dict or None
        Precomputed topology-aware neighbor maps (see `sectionate.gridutils.build_neighbor_maps`).
        Required for multi-tile grids. If None, single-tile maps are built from `boundary`.
    f1, f2: integer or None
        Face indices of the starting and ending points (multi-tile grids only); None otherwise.

    RETURNS:
    -------

    For single-tile grids:
        i_c_seg, j_c_seg, lons_c_seg, lats_c_seg
    For multi-tile grids, additionally the face index of each point:
        i_c_seg, j_c_seg, f_c_seg, lons_c_seg, lats_c_seg

    (i_c_seg, j_c_seg[, f_c_seg]) are the vorticity-point indices bounded by the start and end
    points; (lons_c_seg, lats_c_seg) are the corresponding longitude and latitude.
    """
    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values

    multitile = neighbor_maps is not None
    if multitile:
        nfaces, ny, nx = gridlon.shape
    else:
        # Single-tile grid: neighbors follow directly from `boundary` metadata.
        # Build the lookup maps from the (already-trimmed) coordinate array so
        # they stay consistent with the array we actually walk.
        ny, nx = gridlon.shape
        nfaces = 1
        neighbor_maps = simple_neighbor_maps((ny, nx), boundary)

    def coord(arr, f, j, i):
        return arr[j, i] if f is None else arr[f, j, i]

    def neighbor(direction, f, j, i):
        fmap, jmap, imap = neighbor_maps[direction]
        if fmap is None:
            return (None, int(jmap[j, i]), int(imap[j, i]))
        return (int(fmap[f, j, i]), int(jmap[f, j, i]), int(imap[f, j, i]))

    # target coordinates
    lon1, lat1 = coord(gridlon, f1, j1, i1), coord(gridlat, f1, j1, i1)
    lon2, lat2 = coord(gridlon, f2, j2, i2), coord(gridlat, f2, j2, i2)

    # init loop position to starting point
    f, j, i = f1, j1, i1

    i_c_seg = [i]  # add first point to list of points
    j_c_seg = [j]  # add first point to list of points
    f_c_seg = [f]  # add first point to list of points

    # iterate through the grid path steps until we reach end of section
    ct = 0 # grid path step counter
    # safety bound: enough steps to cross the whole grid (all faces) once
    nstep_max = (nx + ny + 1) * nfaces

    # Grid-agnostic algorithm:
    # First, find all four neighbors (using grid topology via `neighbor_maps`)
    # Second, throw away any that are further from the destination than the current point
    # Third, go to the valid neighbor that has the smallest angle from the arc path between the
    # start and end points (the shortest geodesic path)
    f_prev, j_prev, i_prev = f, j, i
    while (f, j, i) != (f2, j2, i2):

        # safety precaution: exit after taking enough steps to have crossed the entire model grid
        if ct > nstep_max:
            raise RuntimeError(f"Should have reached the endpoint by now.")

        d_current = distance_on_unit_sphere(
                coord(gridlon, f, j, i),
                coord(gridlat, f, j, i),
                lon2,
                lat2
            )

        if d_current < 1.e-12:
            break

        neighbors = [
            neighbor("right", f, j, i),
            neighbor("left",  f, j, i),
            neighbor("down",  f, j, i),
            neighbor("up",    f, j, i),
        ]

        next_pt = None
        smallest_angle = np.inf
        d_list = []
        for (_f, _j, _i) in neighbors:
            d = distance_on_unit_sphere(
                coord(gridlon, _f, _j, _i),
                coord(gridlat, _f, _j, _i),
                lon2,
                lat2
            )
            d_list.append(d/d_current)
            if d < d_current:
                if d==0.: # We're done!
                    next_pt = (_f, _j, _i)
                    smallest_angle = 0.
                    break
                # Instead of simply moving to the point that gets us closest to the target,
                # a more robust approach is to pick, among the points that do get us closer,
                # the one that most closely follows the great circle between the start and
                # end points of the section. We average the angles relative to both end
                # points so that the shortest path is unique and insensitive to which direction
                # the section is traveled.
                else:
                    angle1 = spherical_angle(
                        lon2,
                        lat2,
                        lon1,
                        lat1,
                        coord(gridlon, _f, _j, _i),
                        coord(gridlat, _f, _j, _i),
                    )
                    angle2 = spherical_angle(
                        lon1,
                        lat1,
                        lon2,
                        lat2,
                        coord(gridlon, _f, _j, _i),
                        coord(gridlat, _f, _j, _i),
                    )
                    angle = (angle1+angle2)/2.
                    if angle < smallest_angle:
                        next_pt = (_f, _j, _i)
                        smallest_angle = angle

        # There can be some strange edge cases in which none of the neighboring points
        # actually get us closer to the target (e.g. when closing folds in the grid).
        # In these cases, simply pick the adjacent point that gets us closest, as long as
        # it was not our previous point (to avoid endless loops). This algorithm should be
        # guaranteed to always get us to the target point.
        if (smallest_angle == np.inf) or (next_pt == (f_prev, j_prev, i_prev)):
            if (f_prev, j_prev, i_prev) in neighbors:
                idx = neighbors.index((f_prev, j_prev, i_prev))
                del neighbors[idx]
                del d_list[idx]

            next_pt = neighbors[int(np.argmin(d_list))]

        f_prev, j_prev, i_prev = f, j, i

        f, j, i = next_pt

        i_c_seg.append(i)
        j_c_seg.append(j)
        f_c_seg.append(f)

        ct+=1

    # create lat/lon vectors from (f,j,i) triples
    lons_c_seg = []
    lats_c_seg = []
    for ff, jj, ji in zip(f_c_seg, j_c_seg, i_c_seg):
        lons_c_seg.append(coord(gridlon, ff, jj, ji))
        lats_c_seg.append(coord(gridlat, ff, jj, ji))

    i_c, j_c = np.array(i_c_seg), np.array(j_c_seg)
    lons_c, lats_c = np.array(lons_c_seg), np.array(lats_c_seg)
    if multitile:
        return i_c, j_c, np.array(f_c_seg), lons_c, lats_c
    return i_c, j_c, lons_c, lats_c


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

    For 2d (single-tile) grids:
        iclose, jclose: integer grid indices for the geographical point of interest
    For 3d (multi-tile) grids, additionally the face index:
        iclose, jclose, fclose
    """

    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values
    dist = distance_on_unit_sphere(lon, lat, gridlon, gridlat)
    idx = np.unravel_index(np.nanargmin(dist), gridlon.shape)
    if gridlon.ndim == 3:
        fclose, jclose, iclose = idx
        return iclose, jclose, fclose
    jclose, iclose = idx
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

def align_coords(coords1, coords2, extend=False):
    """Align coords1 and coords2 by minimizing distance between coords[-1] and coords[0]

    Arguments
    ---------
    coords1 [list of (lon,lat) tuples]
    coords2 [list of (lon,lat) tuples]

    Keyword Arguments
    -----------------
    extend [bool (Default : False)] -- extends coords1 so that its starting point is 
        equal to the end point of coords2 and its end point is the starting point of
        coords2.

    Returns
    -------
    (coords1, coords2)

    Examples
    --------
    >>> coords1 = [(-100, 0), (-50, 0)]
    >>> coords2 = [(   0, 0), (-40, 0)]
    >>> sec.align_coords(coords1, coords2)
    
    """
    coords_options = [
        [coords1      , coords2      ],
        [coords1[::-1], coords2      ],
        [coords1      , coords2[::-1]],
        [coords1[::-1], coords2[::-1]]
    ]
    dists = np.array([
        coord_distance(c1[-1], c2[0])
        for (c1,c2) in coords_options
    ])
    coords1, coords2 = coords_options[np.argmin(dists)]
    if extend:
        coords1 = [coords2[-1]] + coords1 + [coords2[0]]
    return coords1, coords2

def coord_distance(coord1, coord2):
    """Spherical distance between coord1 and coord2"""
    return distance_on_unit_sphere(
        coord1[0],
        coord1[1],
        coord2[0],
        coord2[1]
    )

def lonlat_from_coords(coords):
    """Turns list of coordinate pairs into arrays of longitudes and latitudes"""
    return (
            np.array([lon for (lon, lat) in coords]),
            np.array([lat for (lon, lat) in coords])
        )

def coords_from_lonlat(lons, lats):
    """Turns iterable longitudes and latitudes into a list of coordinate pairs"""
    return [(lon, lat) for (lon, lat) in zip(lons, lats)]
