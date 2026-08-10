import numpy as np
import pytest
import xarray as xr
import xgcm

# define simple xgcm grid
xq = np.array([0., 60, 120, 180, 240, 300., 360.])
yq = np.array([-80., -40, 0, 40, 80.])

lon_c, lat_c = np.meshgrid(xq, yq)
ds = xr.Dataset({}, coords={
    "xq":xr.DataArray(xq, dims=("xq",)),
    "yq":xr.DataArray(yq, dims=("yq",)),
    "lon_c":xr.DataArray(lon_c, dims=("yq", "xq",)),
    "lat_c":xr.DataArray(lat_c, dims=("yq", "xq",))
})
coords = {
    'X': {'outer': 'xq'},
    'Y': {'outer': 'yq'}
}
boundary = {
    'X': 'periodic',
    'Y': 'extend'
}
grid = xgcm.Grid(ds, coords=coords, padding=boundary, autoparse_metadata=False)

def modequal(a,b):
    return np.equal(np.mod(a, 360.), np.mod(b, 360.))

def test_open_grid_section():
    from sectionate.section import grid_section
    lonseg = np.array([0., 120, 120, 0])
    latseg = np.array([-80., -80, 0, 0])
    i, j, lons, lats = grid_section(grid, lonseg, latseg)
    assert np.all([
        modequal(i, np.array([0, 1, 2, 2, 2, 1, 0])),
        modequal(j, np.array([0, 0, 0, 1, 2, 2, 2])),
        modequal(lons, np.array([0.,  60., 120., 120., 120.,  60., 0.])),
        modequal(lats, np.array([-80., -80., -80., -40.,   0.,   0.,   0.]))
    ])
    
def test_closed_grid_section():
    from sectionate.section import grid_section
    lonseg = np.array([0., 120, 120, 0, 0])
    latseg = np.array([-80., -80, 0, 0, -80.])
    i, j, lons, lats = grid_section(grid, lonseg, latseg)
    assert np.all([
        modequal(i, np.array([0, 1, 2, 2, 2, 1, 0, 0, 0])),
        modequal(j, np.array([0, 0, 0, 1, 2, 2, 2, 1, 0])),
        modequal(lons, np.array([0.,  60., 120., 120., 120.,  60., 0., 0., 0.])),
        modequal(lats, np.array([-80., -80., -80., -40.,   0.,   0.,   0., -40., -80.]))
    ])
    
def test_periodic_grid_section():
    from sectionate.section import grid_section
    lonseg = np.array([300, 60])
    latseg = np.array([0, 0])
    i, j, lons, lats = grid_section(grid, lonseg, latseg)
    # The seam vertex (360 == 0) is symmetric+periodic, so it carries two indices (6 and 0):
    # the path steps through both, leaving a doubled corner. The zero-length edge between
    # them carries no flux and is dropped when faces are derived (see uvindices_from_qindices).
    assert np.all([
        modequal(i, np.array([5, 6, 0, 1])),
        modequal(j, np.array([2, 2, 2, 2])),
        modequal(lons, np.array([300., 360., 0., 60.])),
        modequal(lats, np.array([0.,   0.,   0., 0.]))
    ])


def test_latitude_circle_zero_length_closure():
    """A latitude-circle section closing back to its start (a 360 -> 0 segment) must
    trace as a zero-length step, not be rejected as a 360-degree arc. This is the
    globe-encircling boundary case (endpoints coincide modulo 360 degrees)."""
    from sectionate.section import grid_section, _check_segment_span
    # Unit level: a segment whose endpoints coincide modulo 360 carries no east/west
    # ambiguity and must be allowed under latitude circle.
    _check_segment_span(360., 0., 0., 0., "latitude circle")   # zero-length closure
    _check_segment_span(10., 0., 370., 0., "latitude circle")  # same, offset
    # End to end: a zonal line explicitly closed with a 360 -> 0 segment traces the
    # full latitude circle rather than raising.
    lonseg = np.array([0., 120., 240., 360., 0.])
    latseg = np.array([0., 0., 0., 0., 0.])
    i, j, lons, lats = grid_section(grid, lonseg, latseg, curve="latitude circle")
    assert np.all([
        modequal(i, np.array([0, 1, 2, 3, 4, 5, 6])),
        modequal(lons, np.array([0., 60., 120., 180., 240., 300., 360.])),
        modequal(lats, np.array([0., 0., 0., 0., 0., 0., 0.])),
    ])


def test_latitude_circle_takes_shortest_path_west():
    """Sectionate always walks the *shortest* path between two waypoints, so a
    latitude-circle segment written 0 -> 270 travels 90 degrees WEST rather than 270
    degrees east. Raw longitudes carry no intent to go the long way round."""
    from sectionate.section import grid_section, _check_segment_span

    # Unit level: a raw 270-degree change is a 90-degree westward one; not an error.
    _check_segment_span(0., 0., 270., 0., "latitude circle")

    # End to end: 0 -> 270 leaves lon=0 westward across the periodic seam (the seam
    # vertex carries both index 6 (lon 360) and index 0 (lon 0)); the endpoint snaps to
    # the nearest corner at lon=240. The path never visits the eastern half.
    i, j, lons, lats = grid_section(grid, [0., 270.], [0., 0.], curve="latitude circle")
    assert np.all([
        modequal(i, np.array([0, 6, 5, 4])),
        modequal(j, np.array([2, 2, 2, 2])),
        modequal(lons, np.array([0., 360., 300., 240.])),
        modequal(lats, np.array([0., 0., 0., 0.])),
    ])
    assert not np.any(np.isin(np.mod(lons, 360.), [60., 120., 180.]))

    # The default great circle already behaved this way; both curves now agree.
    i_gc, j_gc, lons_gc, lats_gc = grid_section(grid, [0., 270.], [0., 0.])
    assert np.all([modequal(i_gc, i), modequal(j_gc, j)])


def test_half_circle_separation_is_the_only_ambiguity_error():
    """The single remaining ambiguity error covers both curve types: endpoints exactly
    half a circle apart have two equally short paths between them, so the shortest path
    is not unique. Anything short of that resolves silently to the shortest path."""
    from sectionate.section import grid_section, _check_segment_span

    ambiguous = [
        # (lon1, lat1, lon2, lat2, curve)
        (0., 0., 180., 0., "latitude circle"),    # exactly +180 degrees of longitude
        (0., 0., -180., 0., "latitude circle"),   # exactly -180 degrees of longitude
        (10., 30., 190., 30., "latitude circle"), # 180 degrees away from lon=10
        (0., 0., 180., 0., "great circle"),       # antipodal on the equator
        (0., 30., 180., -30., "great circle"),    # antipodal off the equator
    ]
    for lon1, lat1, lon2, lat2, curve in ambiguous:
        with pytest.raises(ValueError, match="half a circle apart"):
            _check_segment_span(lon1, lat1, lon2, lat2, curve)

    # Just short of half a circle is unambiguous for both curves.
    _check_segment_span(0., 0., 179.9, 0., "latitude circle")
    _check_segment_span(0., 0., 179.9, 0., "great circle")

    # The same single error reaches the user through the public API.
    with pytest.raises(ValueError, match="half a circle apart"):
        grid_section(grid, [0., 180.], [0., 0.], curve="latitude circle")
    with pytest.raises(ValueError, match="half a circle apart"):
        grid_section(grid, [0., 180.], [0., 0.])


def test_latitude_circle_meridional_segment_walks_the_meridian():
    """A latitude-circle segment whose endpoints share a longitude is the meridional
    connector used to join zonal arcs at different latitudes. It is unambiguous, so it
    is allowed, and it walks straight down the meridian."""
    from sectionate.section import grid_section

    i, j, lons, lats = grid_section(grid, [60., 60.], [-80., 80.], curve="latitude circle")
    assert np.all([
        modequal(i, np.array([1, 1, 1, 1, 1])),
        modequal(j, np.array([0, 1, 2, 3, 4])),
        modequal(lons, np.array([60., 60., 60., 60., 60.])),
        modequal(lats, np.array([-80., -40., 0., 40., 80.])),
    ])


def test_latitude_circle_oblique_segment_rejected():
    """No circle of constant latitude passes through two points that differ in both
    latitude and longitude, so such a segment is rejected rather than traced as an
    L-shaped, direction-dependent path. Great circle handles it instead."""
    from sectionate.section import grid_section, _check_segment_span

    with pytest.raises(ValueError, match="oblique"):
        _check_segment_span(0., 0., 120., 40., "latitude circle")
    with pytest.raises(ValueError, match="oblique"):
        grid_section(grid, [0., 120.], [0., 40.], curve="latitude circle")

    # The same waypoints are fine as a great circle, and as a zonal leg plus a
    # meridional leg under latitude circle.
    grid_section(grid, [0., 120.], [0., 40.])
    grid_section(grid, [0., 120., 120.], [0., 0., 40.], curve="latitude circle")
