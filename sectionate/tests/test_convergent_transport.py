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


def initialize_minimal_vertical_grid(layer, interface, register_z=True):
    """The minimal 1x1-cell 'outer' grid of `initialize_minimal_outer_grid`, plus a
    single-layer vertical coordinate named (`layer`, `interface`). `register_z` selects
    whether the grid declares that vertical axis or (as most grids handed to sectionate
    do, since sections are only ever traced horizontally) only its horizontal ones."""
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
        coords['Z'] = {'center': layer, 'outer': interface}
    return xgcm.Grid(ds, coords=coords, padding={'X': "extend", 'Y': "extend"},
                     autoparse_metadata=False)


# The closed path of vorticity points around the single cell of the minimal grid,
# i.e. what `grid_section` returns for the square [0,1]x[0,1] (see
# `test_convergent_transport`); hardcoded so these tests exercise only
# `convergent_transport`'s layer/interface handling.
MINIMAL_LOOP_I = np.array([0, 1, 1, 0, 0])
MINIMAL_LOOP_J = np.array([0, 0, 1, 1, 0])


def transport_around_minimal_cell(grid, layer, interface):
    from sectionate.transports import convergent_transport
    return convergent_transport(
        grid,
        MINIMAL_LOOP_I,
        MINIMAL_LOOP_J,
        utr="u",
        vtr="v",
        layer=layer,
        interface=interface,
        geometry="cartesian",
    )


@pytest.mark.parametrize("register_z", [True, False])
@pytest.mark.parametrize("layer,interface", [
    ("z_l", "z_i"),           # the canonical MOM6 depth coordinate
    ("sigma2_l", "sigma2_i"),  # a stem with no "l" in it
    ("lam_l", "lam_i"),        # a stem containing an "l" ...
    ("level_l", "level_i"),    # ... and one containing two
])
def test_layer_interface_pairs_accepted(layer, interface, register_z):
    """A consistent (layer, interface) pair must be accepted whatever its stem spells.
    Matching them by rewriting every "l" in `layer` into an "i" spuriously rejected any
    stem that itself contains an "l"."""
    grid = initialize_minimal_vertical_grid(layer, interface, register_z=register_z)
    dsout = transport_around_minimal_cell(grid, layer, interface)

    # both vertical coordinates are carried through to the output ...
    assert layer in dsout.coords
    assert interface in dsout.coords
    # ... and the transport is the depth-independent result of `test_convergent_transport`
    conv = dsout['conv_mass_transport'].sum().values
    assert np.isclose(1. + 0. + np.sqrt(2.) - np.pi, conv, rtol=1.e-14)


def test_layer_interface_from_grid_axis_without_naming_convention():
    """Names the grid itself pairs on one axis are accepted even when they follow no
    "_l"/"_i" naming convention: the grid, not the spelling, is the authority."""
    grid = initialize_minimal_vertical_grid("MyCenters", "MyEdges", register_z=True)
    dsout = transport_around_minimal_cell(grid, "MyCenters", "MyEdges")
    assert "MyCenters" in dsout.coords and "MyEdges" in dsout.coords


@pytest.mark.parametrize("layer,interface", [
    ("z_l", "sigma2_i"),  # two different vertical coordinates
    ("ml_l", "mi_i"),     # different stems ("ml" vs "mi"), but a whole-string
                          # "l"->"i" substitution would have accepted them
    ("z_l", "z_l"),       # the layer coordinate passed twice
])
def test_inconsistent_layer_interface_rejected(layer, interface):
    """Genuinely mismatched pairs are still rejected, and the message names them. The
    grid registers no vertical axis here, so the names are all there is to go on; a
    grid that *does* register one rejects a contradicting name earlier and more
    specifically (see `test_layer_interface_contradicting_z_axis_raises`)."""
    grid = initialize_minimal_vertical_grid("z_l", "z_i", register_z=False)
    with pytest.raises(ValueError, match="do not describe the same vertical axis"):
        transport_around_minimal_cell(grid, layer, interface)


# ---------------------------------------------------------------------------
# layer/interface are taken from the grid's vertical axis when it has one
# ---------------------------------------------------------------------------

def test_layer_interface_derived_from_grid_z_axis():
    """A grid that registers a "Z" axis already names its vertical coordinates, so the
    caller should not have to repeat them."""
    grid = initialize_minimal_vertical_grid("sigma2_l", "sigma2_i", register_z=True)
    dsout = transport_around_minimal_cell(grid, None, None)  # nothing passed
    assert "sigma2_l" in dsout.coords
    assert "sigma2_i" in dsout.coords
    conv = dsout['conv_mass_transport'].sum().values
    assert np.isclose(1. + 0. + np.sqrt(2.) - np.pi, conv, rtol=1.e-14)


def test_layer_interface_derived_matches_explicitly_passed():
    """Passing the same names the "Z" axis carries -- what a caller that read them off
    `grid.axes["Z"].coords` does -- is equivalent to passing nothing."""
    grid = initialize_minimal_vertical_grid("lam_l", "lam_i", register_z=True)
    zc = grid.axes["Z"].coords["center"]
    zi = grid.axes["Z"].coords["outer"]
    explicit = transport_around_minimal_cell(grid, zc, zi)
    derived = transport_around_minimal_cell(grid, None, None)
    xr.testing.assert_identical(explicit, derived)


@pytest.mark.parametrize("layer,interface", [
    ("z_l", None),        # layer contradicts the axis
    (None, "z_i"),        # interface contradicts the axis
    ("z_l", "z_i"),       # both do
])
def test_layer_interface_contradicting_z_axis_raises(layer, interface):
    """An explicit name that disagrees with the grid's own vertical axis is an error,
    not an override: the output would be labelled with a coordinate that need not
    describe the data."""
    grid = initialize_minimal_vertical_grid("sigma2_l", "sigma2_i", register_z=True)
    with pytest.raises(ValueError, match="contradicts the grid"):
        transport_around_minimal_cell(grid, layer, interface)


def test_no_vertical_axis_and_no_names_attaches_nothing():
    """Without a "Z" axis and without explicit names there is no vertical coordinate to
    attach -- and, unlike the old `layer="z_l"` default, no lookup of a name the grid
    has never heard of."""
    grid = initialize_minimal_outer_grid()
    grid._ds['u'] = xr.DataArray(np.array([[1., -np.sqrt(2.)]]), dims=("yh", "xq",))
    grid._ds['v'] = xr.DataArray(np.array([[0], [np.pi]]), dims=("yq", "xh",))
    dsout = transport_around_minimal_cell(grid, None, None)
    assert not any(c.endswith(("_l", "_i")) for c in dsout.coords)
    conv = dsout['conv_mass_transport'].sum().values
    assert np.isclose(1. + 0. + np.sqrt(2.) - np.pi, conv, rtol=1.e-14)


def test_z_axis_center_without_coordinate_values_is_not_derived():
    """A "Z" axis position may name a bare dimension carrying no coordinate values;
    such a name cannot be attached to the output, so it is not derived."""
    grid = initialize_minimal_vertical_grid("z_l", "z_i", register_z=True)
    grid._ds = grid._ds.drop_vars("z_i")  # keep the dim, drop its values
    dsout = transport_around_minimal_cell(grid, None, None)
    assert "z_l" in dsout.coords
    assert "z_i" not in dsout.coords
