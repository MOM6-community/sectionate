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
    grid = xgcm.Grid(ds, coords=coords, padding=boundary, autoparse_metadata=False)
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
    grid = xgcm.Grid(ds, coords=coords, padding=boundary, autoparse_metadata=False)
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
    grid = xgcm.Grid(ds, coords=coords, padding={"X":"extend", "Y":"extend"},
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
                grid, i_c, j_c, utr="u", vtr="v",
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
            grid, i, j, utr="u", vtr="v", geometry="cartesian",
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
    )['conv_mass_transport'].sum().values
    
    conv_rev = convergent_transport(
        grid,
        i[::-1],
        j[::-1],
        utr="u",
        vtr="v",
    )['conv_mass_transport'].sum().values
    
    assert np.equal(-3., conv) and np.equal(-3, conv_rev)



# ---------------------------------------------------------------------------
# Vertical coordinates on the output: the layer coordinate comes along with the
# transports, the interface coordinate comes from the grid.
# ---------------------------------------------------------------------------

def initialize_minimal_vertical_grid(
    layer="z_l", interface="z_i", register_z=True, axis="Z"
):
    """The minimal 1x1-cell 'outer' grid of `initialize_minimal_outer_grid`, plus a
    single-layer vertical coordinate named (`layer`, `interface`) that the transports
    "u"/"v" are resolved over. `register_z` selects whether the grid declares that
    vertical axis or (as most grids handed to sectionate do, since sections are only
    ever traced horizontally) only its horizontal ones; `axis` names it."""
    xh, yh = np.array([0.5]), np.array([0.5])
    xq, yq = np.array([0., 1.]), np.array([0., 1.])

    lon, lat = np.meshgrid(xh, yh)
    lon_c, lat_c = np.meshgrid(xq, yq)
    ds = xr.Dataset({}, coords={
        "xh": xr.DataArray(xh, dims=("xh",)),
        "yh": xr.DataArray(yh, dims=("yh",)),
        "xq": xr.DataArray(xq, dims=("xq",)),
        "yq": xr.DataArray(yq, dims=("yq",)),
        layer: xr.DataArray(np.array([0.5]), dims=(layer,)),
        interface: xr.DataArray(np.array([0., 1.]), dims=(interface,)),
        "geolon": xr.DataArray(lon, dims=("yh", "xh")),
        "geolat": xr.DataArray(lat, dims=("yh", "xh")),
        "geolon_c": xr.DataArray(lon_c, dims=("yq", "xq",)),
        "geolat_c": xr.DataArray(lat_c, dims=("yq", "xq",)),
    })
    ds["u"] = xr.DataArray(np.array([[[1., -np.sqrt(2.)]]]), dims=(layer, "yh", "xq"))
    ds["v"] = xr.DataArray(np.array([[[0.], [np.pi]]]), dims=(layer, "yq", "xh"))
    coords = {
        'X': {'outer': 'xq', 'center': 'xh'},
        'Y': {'outer': 'yq', 'center': 'yh'},
    }
    if register_z:
        coords[axis] = {'center': layer, 'outer': interface}
    return xgcm.Grid(ds, coords=coords, padding={'X': "extend", 'Y': "extend"},
                     autoparse_metadata=False)


# The closed path of vorticity points around the single cell of the minimal grid,
# i.e. what `grid_section` returns for the square [0,1]x[0,1] (see
# `test_convergent_transport`); hardcoded so these tests exercise only how
# `convergent_transport` labels its output vertically.
MINIMAL_LOOP_I = np.array([0, 1, 1, 0, 0])
MINIMAL_LOOP_J = np.array([0, 0, 1, 1, 0])

# The transport around that cell, independent of how it is resolved vertically.
MINIMAL_LOOP_TRANSPORT = 1. + 0. + np.sqrt(2.) - np.pi


def transport_around_minimal_cell(grid):
    from sectionate.transports import convergent_transport
    return convergent_transport(
        grid,
        MINIMAL_LOOP_I,
        MINIMAL_LOOP_J,
        utr="u",
        vtr="v",
        geometry="cartesian",
    )


def test_layer_coordinate_comes_from_the_transports():
    """The layer (cell-center) coordinate is a dimension of `utr`/`vtr` themselves, so it
    reaches the output without being named. A grid declaring only its horizontal axes --
    the usual case -- has no interface coordinate to add."""
    grid = initialize_minimal_vertical_grid("sigma2_l", "sigma2_i", register_z=False)
    dsout = transport_around_minimal_cell(grid)

    assert "sigma2_l" in dsout.coords
    np.testing.assert_array_equal(dsout["sigma2_l"].values, np.array([0.5]))
    assert "sigma2_i" not in dsout.coords
    assert np.isclose(
        dsout['conv_mass_transport'].sum().values, MINIMAL_LOOP_TRANSPORT, rtol=1.e-14
    )


def test_interface_coordinate_comes_from_the_grid():
    """The interface coordinate is one point longer than the layer coordinate, so it is a
    dimension of nothing on the section and cannot ride along. A grid that registers its
    vertical axis names it, and it is attached from there."""
    grid = initialize_minimal_vertical_grid("sigma2_l", "sigma2_i", register_z=True)
    dsout = transport_around_minimal_cell(grid)

    assert "sigma2_l" in dsout.coords and "sigma2_i" in dsout.coords
    np.testing.assert_array_equal(dsout["sigma2_i"].values, np.array([0., 1.]))
    # Only the vertical axis contributes: the horizontal ones are indexed away along the
    # section, so no "xq"/"yq" dimension is dragged back onto the output.
    assert set(dsout.sizes) == {"sect", "sigma2_l", "sigma2_i"}
    assert np.isclose(
        dsout['conv_mass_transport'].sum().values, MINIMAL_LOOP_TRANSPORT, rtol=1.e-14
    )


@pytest.mark.parametrize("layer,interface", [
    ("z_l", "z_i"),             # the canonical MOM6 depth coordinate
    ("lam_l", "lam_i"),         # a stem containing an "l" ...
    ("level_l", "level_i"),     # ... and one containing two
    ("MyCenters", "MyEdges"),   # no shared stem and no suffix convention at all
])
def test_vertical_coordinate_names_need_no_convention(layer, interface):
    """The pair is matched through the grid axis that registers it, so it works whatever
    the names spell. Matching them by rewriting every "l" of the layer name into an "i"
    rejected any stem that itself contained an "l"."""
    grid = initialize_minimal_vertical_grid(layer, interface, register_z=True)
    dsout = transport_around_minimal_cell(grid)
    assert layer in dsout.coords and interface in dsout.coords


def test_vertical_axis_need_not_be_named_Z():
    """The axis is found through the dimension the transports carry, not by looking up an
    axis called "Z", so a differently named vertical axis works the same way."""
    grid = initialize_minimal_vertical_grid(
        "sigma2_l", "sigma2_i", register_z=True, axis="sigma2"
    )
    dsout = transport_around_minimal_cell(grid)
    assert "sigma2_l" in dsout.coords and "sigma2_i" in dsout.coords


def test_interface_dimension_without_coordinate_values_attaches_nothing():
    """An axis position may name a bare dimension carrying no coordinate values; there is
    then nothing to attach, and the layer coordinate still arrives with the transports."""
    grid = initialize_minimal_vertical_grid("z_l", "z_i", register_z=True)
    grid._ds = grid._ds.drop_vars("z_i")  # keep the dim, drop its values
    dsout = transport_around_minimal_cell(grid)
    assert "z_l" in dsout.coords
    assert "z_i" not in dsout.coords


def test_two_dimensional_transports_get_no_vertical_coordinates():
    """Transports with no vertical dimension pick up no vertical coordinate -- and, unlike
    the old `layer="z_l"` default, trigger no lookup of a name the grid never heard of."""
    from sectionate.transports import convergent_transport
    grid = initialize_minimal_outer_grid()
    grid._ds['u'] = xr.DataArray(np.array([[1., -np.sqrt(2.)]]), dims=("yh", "xq",))
    grid._ds['v'] = xr.DataArray(np.array([[0], [np.pi]]), dims=("yq", "xh",))
    dsout = transport_around_minimal_cell(grid)
    assert set(dsout.sizes) == {"sect"}
    assert np.isclose(
        dsout['conv_mass_transport'].sum().values, MINIMAL_LOOP_TRANSPORT, rtol=1.e-14
    )


@pytest.mark.parametrize("kwarg", ["layer", "interface"])
def test_layer_and_interface_keywords_are_gone(kwarg):
    """Both were removed: the layer coordinate comes from the data and the interface
    coordinate from the grid, so there is nothing left for a caller to name."""
    from sectionate.transports import convergent_transport
    grid = initialize_minimal_vertical_grid("z_l", "z_i", register_z=True)
    with pytest.raises(TypeError, match=kwarg):
        convergent_transport(
            grid, MINIMAL_LOOP_I, MINIMAL_LOOP_J, utr="u", vtr="v",
            geometry="cartesian", **{kwarg: "z_l"},
        )
