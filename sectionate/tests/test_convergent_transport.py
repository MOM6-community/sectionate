import numpy as np
import xarray as xr
import xgcm
import pytest

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
    
def initialize_minimal_cartesian_grid():
    # 3 x 2 cells; corners at x=0..3, y=0..2.
    xq = np.array([0., 1., 2., 3.])
    yq = np.array([0., 1., 2.])
    xh = np.array([0.5, 1.5, 2.5])
    yh = np.array([0.5, 1.5])

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
    grid = xgcm.Grid(ds, coords=coords, boundary={"X":"extend", "Y":"extend"},
                     autoparse_metadata=False)
    return grid


def test_open_section_left_of_transect():
    """An open section has no interior, so `positive_in` follows the left-of-transect
    convention: positive transport points to the LEFT of travel (first waypoint -> last)."""
    import warnings
    from sectionate.section import grid_section
    from sectionate.transports import convergent_transport
    grid = initialize_minimal_cartesian_grid()
    # Purely northward flow (v = +1); no zonal flow.
    grid._ds['u'] = xr.DataArray(np.zeros((2, 4)), dims=("yh", "xq"))
    grid._ds['v'] = xr.DataArray(np.ones((3, 3)), dims=("yq", "xh"))

    # Open, eastward (west -> east) zonal section along the interior corner row y=1.
    i, j, lons, lats = grid_section(grid, [0., 3.], [1., 1.])

    def total(i_c, j_c, positive_in=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # open-section UserWarning is expected here
            return convergent_transport(
                grid, i_c, j_c, utr="u", vtr="v", layer=None,
                geometry="cartesian", positive_in=positive_in,
            )["conv_mass_transport"].sum().values

    # Heading east, "left" is north, so the northward flow is positive (3 unit V-faces).
    assert np.isclose(total(i, j), 3.)
    # positive_in=False flips to right-of-transect (south is positive) -> negative.
    assert np.isclose(total(i, j, positive_in=False), -3.)
    # Reversing the waypoint order (heading west) puts north on the right -> negative.
    assert np.isclose(total(i[::-1], j[::-1]), -3.)

    # The open-section orientation warning must be raised.
    with pytest.warns(UserWarning, match="left-of-transect"):
        convergent_transport(
            grid, i, j, utr="u", vtr="v", layer=None, geometry="cartesian",
        )


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
