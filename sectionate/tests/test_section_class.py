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
    from sectionate.section import Section, GriddedSection
    lonseg = np.array([0., 120, 120, 0])
    latseg = np.array([-80., -80, 0, 0])
    sec = Section("testsec", (lonseg, latseg))
    sec_gridded = GriddedSection(sec, grid)

    assert np.all([
        modequal(sec_gridded.i_c, np.array([0, 1, 2, 2, 2, 1, 0])),
        modequal(sec_gridded.j_c, np.array([0, 0, 0, 1, 2, 2, 2])),
        modequal(sec_gridded.lons_c, np.array([0.,  60., 120., 120., 120.,  60., 0.])),
        modequal(sec_gridded.lats_c, np.array([-80., -80., -80., -40.,   0.,   0.,   0.]))
    ])