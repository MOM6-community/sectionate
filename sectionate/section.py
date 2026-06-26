import numpy as np
import xarray as xr

from .gridutils import (
    get_geo_corners,
    get_facedim,
    build_neighbor_maps,
)

# Two corner indices map to the same physical point on seams that fold or wrap (e.g.
# the bipolar north fold). Geodesic distances below this many metres are treated as
# the same point: far below any real grid spacing, far above float round-trip error.
COINCIDENT_TOLERANCE_M = 1.e-3

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
        """Create a copy of this GriddedSection, with deep copies of all attributes
        except the `grid`, which is shared (not copied)."""
        new = GriddedSection(
            self,                       # supplies name, coords, children, parent
            self.grid,                  # shared, not copied
            i_c=self.i_c.copy(),
            j_c=self.j_c.copy(),
            f_c=None if self.f_c is None else self.f_c.copy(),
        )
        # Carry over the gridded path coordinates (grid_section overwrites lons_c/lats_c
        # in place; `__init__` would otherwise reset them to the raw waypoint coords).
        new.lons_c = self.lons_c.copy()
        new.lats_c = self.lats_c.copy()
        new.save = self.save.copy()
        return new


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
    facedim = get_facedim(grid)

    if facedim is not None:
        _check_supported_topology(grid)

    # All topologies -- periodic wrap, fill/extend walls, multi-tile face
    # connections, and the bipolar north fold -- are derived uniformly from the
    # grid's xgcm metadata by padding index arrays (see `build_neighbor_maps`).
    # Where a single physical corner carries two indices (a periodic seam, a shared
    # multi-tile boundary corner, or the fold seam), the walk simply steps through
    # both; the resulting zero-length section edge carries no flux and is dropped when
    # faces are derived (see `transports.uvindices_from_qindices`).
    neighbor_maps = build_neighbor_maps(grid, geocorners)

    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        neighbor_maps=neighbor_maps,
    )


def _check_supported_topology(grid):
    """
    Raise if the multi-tile `grid` requires topology features sectionate does not yet support.

    A tripolar/bipolar north fold expressed as a `face_connections` self-connection (a face that
    connects to itself along the "Y" axis) is not supported. Use the single-tile bipolar-fold
    boundary instead -- ``boundary={"X": "periodic", "Y": {"fold": "corner"}}`` (xgcm >= the
    bipolar-fold release) -- which sectionate handles natively via xgcm's fold padding.
    """
    facedim = grid._facedim
    connections = (getattr(grid, "_face_connections", None) or {}).get(facedim, {})
    for face, axis_sides in connections.items():
        for sides in axis_sides.values():
            for side in sides:
                if side is None:
                    continue
                neighbor_face = side[0]
                if neighbor_face == face:
                    raise NotImplementedError(
                        "Grids that represent the tripolar/bipolar north fold as a face that "
                        "connects to itself (a `face_connections` self-connection) are not "
                        "supported. Build the grid as a single tile with the fold expressed as a "
                        "boundary instead: boundary={'X':'periodic','Y':{'fold':'corner'}}."
                    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    neighbor_maps,
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
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile). Sections are always built from an `xgcm.Grid`, so these
        are always supplied; the usual entry point is `sectionate.grid_section`.

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    # A face dimension (3-D corner arrays) marks a multi-tile grid, whose sections
    # carry a per-point face index. This is independent of `neighbor_maps`, which is
    # now always provided for grid-derived sections (single- and multi-tile alike).
    multitile = np.ndim(gridlon) == 3

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

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, neighbor_maps):
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
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile).

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    return infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        neighbor_maps=neighbor_maps,
    )

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, neighbor_maps):
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
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile).

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    multitile = np.ndim(gridlon) == 3
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
        neighbor_maps=neighbor_maps,
        f1=fstart,
        f2=fend,
    )


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, neighbor_maps, f1=None, f2=None):
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
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`. The
        grid's full topology (periodic wrap, fill/extend walls, multi-tile face seams, and
        the bipolar north fold) is encoded here, so a grid is always required to produce
        them -- typically via `sectionate.grid_section`.
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

    if neighbor_maps is None:
        raise ValueError(
            "infer_grid_path requires topology-aware `neighbor_maps`. Build them from an "
            "xgcm.Grid with sectionate.gridutils.build_neighbor_maps(grid, get_geo_corners(grid)), "
            "or use the high-level sectionate.grid_section(grid, lons, lats)."
        )

    multitile = gridlon.ndim == 3
    if multitile:
        nfaces, ny, nx = gridlon.shape
    else:
        ny, nx = gridlon.shape
        nfaces = 1

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

        # We are done once we reach the endpoint. On grids where two distinct
        # corner indices map to the same physical point -- the bipolar fold seam,
        # where corner i is identified with its mirror image -- the walker can land
        # on the endpoint's twin index rather than the endpoint index itself, so we
        # also stop on physical coincidence. The tolerance (in metres) sits far below
        # any real grid spacing yet well above the fold's float round-trip error.
        if d_current < COINCIDENT_TOLERANCE_M:
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
