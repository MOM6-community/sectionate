import numpy as np
import xarray as xr
import xgcm

# define simple xgcm grid
def initialize_minimal_outer_grid():
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
    boundary = {
        'X': "extend", # Specifying None causes this to default to `periodic` for some reason
        'Y': "extend"  # Specifying None causes this to default to `periodic` for some reason
    }
    grid = xgcm.Grid(ds, coords=coords, boundary=boundary, autoparse_metadata=False)
    return grid

def test_convergent_transport():
    from sectionate.section import grid_section
    from sectionate.transports import convergent_transport
    grid = initialize_minimal_outer_grid()
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
        layer=None,
        geometry="cartesian"
    )['conv_mass_transport'].sum().values
    
    assert np.isclose(1. + 0. + np.sqrt(2.) - np.pi, conv, rtol=1.e-14)
    
    
def initialize_minimal_spherical_grid():
    xq = np.array([0., 120., 240., 360.])
    yq = np.array([-80, 0., 80.])
    xh = np.array([60., 180., 300.])
    yh = np.array([-40., 40.])

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
    boundary = {
        'X': "periodic",
        'Y': "extend"
    }
    grid = xgcm.Grid(ds, coords=coords, boundary=boundary, autoparse_metadata=False)
    return grid
    
def test_convergent_transport_convention():
    from sectionate.section import grid_section
    from sectionate.transports import convergent_transport
    grid = initialize_minimal_spherical_grid()
    u = np.zeros((grid._ds.yh.size, grid._ds.xq.size))
    v = np.ones((grid._ds.yq.size, grid._ds.xh.size))
    grid._ds['u'] = xr.DataArray(u, dims=("yh", "xq"))
    grid._ds['v'] = xr.DataArray(v, dims=("yq", "xh"))

    lonseg = np.array([0., -120., -240., -360.])
    latseg = np.array([0., 0., 0., 0.])
    
    i, j, lons, lats = grid_section(grid, lonseg, latseg)
    conv = convergent_transport(
        grid,
        i,
        j,
        utr="u",
        vtr="v",
        layer=None
    )['conv_mass_transport'].sum().values
    
    conv_rev = convergent_transport(
        grid,
        i[::-1],
        j[::-1],
        utr="u",
        vtr="v",
        layer=None
    )['conv_mass_transport'].sum().values
    
    assert np.equal(-3., conv) and np.equal(-3, conv_rev)
