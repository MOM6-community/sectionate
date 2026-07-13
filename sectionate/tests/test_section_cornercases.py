import numpy as np
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


def test_latitude_circle_long_arc_still_rejected():
    """The zero-length exemption must not relax the genuine-ambiguity guard: a real
    arc spanning >= 180 degrees of longitude (taken as given) is still rejected."""
    from sectionate.section import _check_segment_span
    import pytest
    for lon2 in (180., 270., 200.):       # genuine arcs, endpoints do NOT coincide
        with pytest.raises(ValueError, match="spans"):
            _check_segment_span(0., 0., lon2, 0., "latitude circle")
