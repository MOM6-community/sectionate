import numpy as np
import xarray as xr
import xgcm

# define simple xgcm grid
def initialize_outer_grid():
    xh = np.array([0.5])
    yh = np.array([0.5])
    xq = np.array([0., 1.])
    yq = np.array([0., 1.])

    lon, lat = np.meshgrid(xh, yh)
    lon_c, lat_c = np.meshgrid(xq, yq)
    ds = xr.Dataset({}, coords={
        "xh":xr.DataArray(xh, dims=("xh",)),
        "yh":xr.DataArray(yh, dims=("yh",)),
        "xq":xr.DataArray(xq, dims=("xq",)),
        "yq":xr.DataArray(yq, dims=("yq",)),
        "geolon":xr.DataArray(lon, dims=("yh", "xh")),
        "geolat":xr.DataArray(lat, dims=("yh", "xh")),
        "geolon_c":xr.DataArray(lon_c, dims=("yq", "xq",)),
        "geolat_c":xr.DataArray(lat_c, dims=("yq", "xq",))
    })
    coords = {
        'X': {'outer': 'xq', 'center': 'xh'},
        'Y': {'outer': 'yq', 'center': 'yh'}
    }
    grid = xgcm.Grid(ds, coords=coords, periodic=False)
    return grid

def test_convergent_transport():
    from sectionate.section import grid_section
    from sectionate.transports import convergent_transport
    grid = initialize_outer_grid()
    grid._ds['u'] = xr.DataArray(np.array([[1., -np.sqrt(2.)]]), dims=("yh","xq",))
    grid._ds['v'] = xr.DataArray(np.array([[0], [np.pi]]), dims=("yq","xh",))
    
    # closed path around the whole square domain
    lonseg = np.array([0, 1, 1, 0, 0])
    latseg = np.array([0, 0, 1, 1, 0])
    i, j, lons, lats = grid_section(grid, lonseg, latseg)

    conv = convergent_transport(
        grid,
        i,
        j,
        utr="u",
        vtr="v",
        layer=None
    )['conv_mass_transport'].sum().values
    
    assert np.isclose(1. + 0. + np.sqrt(2.) - np.pi, conv, rtol=1.e-14)