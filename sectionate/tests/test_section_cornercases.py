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
    """The single remaining ambiguity error covers every curve: endpoints written exactly
    half a circle apart have two equally long ways round, so neither is the shorter.
    Anything short of that resolves silently to the shortest path."""
    from sectionate.section import grid_section, _check_segment_span

    ambiguous = [
        # (lon1, lat1, lon2, lat2, curve)
        (0., 0., 180., 0., "latitude circle"),    # exactly +180 degrees of longitude
        (0., 0., -180., 0., "latitude circle"),   # exactly -180 degrees of longitude
        (10., 30., 190., 30., "latitude circle"), # 180 degrees away from lon=10
        (0., 0., 180., 0., "great circle"),       # antipodal on the equator
        (0., 30., 180., -30., "great circle"),    # antipodal off the equator
        (0., 0., 180., 0., "latitude and great circle"),    # resolves to the parallel
        (0., 30., 180., -30., "latitude and great circle"), # resolves to the geodesic
    ]
    for lon1, lat1, lon2, lat2, curve in ambiguous:
        with pytest.raises(ValueError, match="half a circle apart"):
            _check_segment_span(lon1, lat1, lon2, lat2, curve)

    # Just short of half a circle is unambiguous for every curve.
    _check_segment_span(0., 0., 179.9, 0., "latitude circle")
    _check_segment_span(0., 0., 179.9, 0., "great circle")
    _check_segment_span(0., 0., 179.9, 0., "latitude and great circle")

    # The same single error reaches the user through the public API.
    with pytest.raises(ValueError, match="half a circle apart"):
        grid_section(grid, [0., 180.], [0., 0.], curve="latitude circle")
    with pytest.raises(ValueError, match="half a circle apart"):
        grid_section(grid, [0., 180.], [0., 0.])


def test_half_circle_error_names_the_measure_it_reports():
    """The separation the error quotes is degrees of LONGITUDE for a segment along a
    parallel and degrees of ARC for a geodesic one -- two different quantities that
    coincide only on the equator. At 89N, 180 degrees of longitude is a 222 km hop, so
    the number is only meaningful if the message says which measure it is."""
    from sectionate.section import _check_segment_span

    with pytest.raises(ValueError, match="degrees of longitude along the parallel"):
        _check_segment_span(0., 89., 180., 89., "latitude circle")
    with pytest.raises(ValueError, match="degrees of arc"):
        _check_segment_span(0., 89., 180., -89., "great circle")


def test_latitude_circle_requires_constant_latitude():
    """Under curve="latitude circle" every segment must lie on a circle of constant
    latitude. Meridional and oblique segments alike lie on none, so both raise."""
    from sectionate.section import grid_section, _check_segment_span

    not_constant_latitude = [
        (60., -80., 60., 80.),   # meridional
        (0., 0., 120., 40.),     # oblique
    ]
    for lon1, lat1, lon2, lat2 in not_constant_latitude:
        with pytest.raises(ValueError, match="constant latitude"):
            _check_segment_span(lon1, lat1, lon2, lat2, "latitude circle")

    with pytest.raises(ValueError, match="constant latitude"):
        grid_section(grid, [60., 60.], [-80., 80.], curve="latitude circle")
    with pytest.raises(ValueError, match="constant latitude"):
        grid_section(grid, [0., 120.], [0., 40.], curve="latitude circle")
    # Rejected even as one leg of a section whose other legs are constant-latitude.
    with pytest.raises(ValueError, match="constant latitude"):
        grid_section(grid, [0., 120., 120.], [0., 0., 40.], curve="latitude circle")


def test_latitude_and_great_circle_chooses_per_segment():
    """curve="latitude and great circle" decides segment by segment: constant-latitude
    segments follow the parallel, everything else follows the geodesic. Segments that
    "latitude circle" rejects are therefore traced, not raised on."""
    from sectionate.section import grid_section, _check_segment_span

    combined = "latitude and great circle"
    assert _check_segment_span(0., 40., 120., 40., combined) == "latitude circle"
    assert _check_segment_span(60., -80., 60., 80., combined) == "great circle"
    assert _check_segment_span(0., 0., 120., 40., combined) == "great circle"

    # A meridional segment walks straight down the meridian.
    i, j, lons, lats = grid_section(grid, [60., 60.], [-80., 80.], curve=combined)
    assert np.all([
        modequal(i, np.array([1, 1, 1, 1, 1])),
        modequal(j, np.array([0, 1, 2, 3, 4])),
        modequal(lons, np.array([60., 60., 60., 60., 60.])),
        modequal(lats, np.array([-80., -40., 0., 40., 80.])),
    ])

    # An oblique segment traces exactly the great-circle path.
    oblique = grid_section(grid, [0., 120.], [0., 40.], curve=combined)
    assert np.all([np.array_equal(a, b)
                   for a, b in zip(oblique, grid_section(grid, [0., 120.], [0., 40.]))])

    # A zonal segment traces exactly the latitude-circle path.
    zonal = grid_section(grid, [0., 120.], [0., 0.], curve=combined)
    assert np.all([
        np.array_equal(a, b) for a, b in
        zip(zonal, grid_section(grid, [0., 120.], [0., 0.], curve="latitude circle"))
    ])

    # And a section mixing the two kinds of leg is traced end to end.
    i, j, lons, lats = grid_section(grid, [0., 120., 120.], [0., 0., 40.], curve=combined)
    assert np.all([
        modequal(lons, np.array([0., 60., 120., 120.])),
        modequal(lats, np.array([0., 0., 0., 40.])),
    ])


def test_unknown_curve_is_rejected():
    """An unrecognized `curve` names all three accepted values."""
    from sectionate.section import grid_section

    with pytest.raises(ValueError, match="curve must be one of"):
        grid_section(grid, [0., 120.], [0., 0.], curve="rhumb line")


def _fine_global_grid(dlon=2., dlat=2.):
    """A 2-degree global 'outer' grid, fine enough that a parallel and a geodesic between
    the same two waypoints are visibly different paths. The 7x5 fixture above cannot show
    this: its corner rows are 40 degrees apart, so both curves snap to the same one."""
    xq = np.arange(0., 360. + dlon, dlon)
    yq = np.arange(-90., 90. + dlat, dlat)
    lon_f, lat_f = np.meshgrid(xq, yq)
    ds_f = xr.Dataset({}, coords={
        "xq": xr.DataArray(xq, dims=("xq",)),
        "yq": xr.DataArray(yq, dims=("yq",)),
        "lon_c": xr.DataArray(lon_f, dims=("yq", "xq")),
        "lat_c": xr.DataArray(lat_f, dims=("yq", "xq")),
    })
    return xgcm.Grid(
        ds_f,
        coords={"X": {"outer": "xq"}, "Y": {"outer": "yq"}},
        padding={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )


def test_parallel_is_held_where_the_geodesic_bows():
    """The test that actually separates the two metrics. Between (0, 40N) and (120, 40N)
    the geodesic bows a long way poleward -- it is the shorter path -- while the parallel
    does not. On a 2-degree grid the constant-latitude walk holds lat 40 for all 61 of its
    points; the great-circle walk climbs to 60N and takes 83."""
    from sectionate.section import grid_section

    fine = _fine_global_grid()

    i, j, lons, lats = grid_section(fine, [0., 120.], [40., 40.], curve="latitude circle")
    assert np.all(lats == 40.)
    assert len(i) == 61
    assert np.array_equal(lons, np.arange(0., 122., 2.))

    i_gc, j_gc, lons_gc, lats_gc = grid_section(fine, [0., 120.], [40., 40.])
    assert lats_gc.max() == 60.                 # bows ~20 degrees poleward
    assert len(i_gc) == 83

    # The combined option classifies this segment as constant-latitude, so it must
    # reproduce the parallel exactly, not the geodesic.
    combined = grid_section(fine, [0., 120.], [40., 40.],
                            curve="latitude and great circle")
    assert np.all([np.array_equal(a, b) for a, b in zip(combined, (i, j, lons, lats))])


def test_float32_waypoints_still_count_as_constant_latitude():
    """Model corner coordinates are routinely stored as float32 (ECCO's are), so
    latitudes that a user reads off a grid come back a few 1e-7 degrees -- a few cm --
    from the value they were written as. That must not turn a zonal segment into an
    oblique one, which is what a nanodegree classification tolerance would do."""
    from sectionate.section import (
        grid_section, _is_constant_latitude, _check_segment_span,
        CONSTANT_LATITUDE_ATOL_DEG,
    )

    lat32 = float(np.float32(26.4))
    offset = abs(lat32 - 26.4)
    assert 1.e-9 < offset < CONSTANT_LATITUDE_ATOL_DEG   # ~4.e-7 degrees, ~4 cm
    assert _is_constant_latitude(lat32, 26.4)
    assert _check_segment_span(0., lat32, 120., 26.4, "latitude circle") == "latitude circle"
    assert _check_segment_span(0., lat32, 120., 26.4,
                               "latitude and great circle") == "latitude circle"

    # End to end: both waypoints snap to the same corner row and the path holds it.
    fine = _fine_global_grid()
    i, j, lons, lats = grid_section(fine, [0., 120.], [lat32, 26.4], curve="latitude circle")
    assert np.all(lats == lats[0])
    assert len(i) == 61
