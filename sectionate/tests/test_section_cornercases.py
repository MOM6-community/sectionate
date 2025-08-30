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
grid = xgcm.Grid(ds, coords=coords, boundary=boundary, autoparse_metadata=False)

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
    assert np.all([
        modequal(i, np.array([5, 0, 1])),
        modequal(j, np.array([2, 2, 2])),
        modequal(lons, np.array([300.,  0., 60.])),
        modequal(lats, np.array([0.,   0.,   0.]))
    ])
