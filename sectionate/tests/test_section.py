import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import (
    get_geo_corners,
    build_neighbor_maps,
)


# define simple lat-lon grid
lon, lat = np.meshgrid(np.arange(360), np.arange(-80, 81))
ds = xr.Dataset()
ds["lon"] = xr.DataArray(lon, dims=("y", "x"))
ds["lat"] = xr.DataArray(lat, dims=("y", "x"))


def _latlon_grid(vertical_axis=False):
    """The lat-lon grid above (X-periodic, Y clipped), optionally also registering a
    vertical axis -- as every real model grid does, and as sections never traverse."""
    ny, nx = lat.shape
    ds = xr.Dataset(coords={
        "xq": np.arange(nx), "yq": np.arange(ny),
        "xh": np.arange(nx) + 0.5, "yh": np.arange(ny) + 0.5,
        "geolon_c": (("yq", "xq"), lon.astype(float)),
        "geolat_c": (("yq", "xq"), lat.astype(float)),
        "geolon": (("yh", "xh"), lon.astype(float)),
        "geolat": (("yh", "xh"), lat.astype(float)),
    })
    coords = {"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}}
    if vertical_axis:
        ds = ds.assign_coords({"z_l": ("z_l", np.array([5., 15.])),
                               "z_i": ("z_i", np.array([0., 10., 20.]))})
        coords["Z"] = {"center": "z_l", "outer": "z_i"}
    return xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": "extend"},
                     autoparse_metadata=False)


def _latlon_neighbor_maps(vertical_axis=False):
    """Neighbor maps for the lat-lon grid above, built the same way `grid_section`
    does -- from an xgcm.Grid. The low-level pathfinder always requires these; a grid
    is the only source of topology-aware connectivity."""
    g = _latlon_grid(vertical_axis=vertical_axis)
    return build_neighbor_maps(g, get_geo_corners(g))


def test_vertical_axis_does_not_affect_neighbor_maps():
    """Horizontal connectivity must not depend on whether the grid also registers a
    vertical axis. `build_neighbor_maps` padded its index arrays over *every* axis of
    the grid, and xgcm's `pad` raises on an axis the array has no dimension for, so a
    Z axis made an otherwise ordinary grid untraceable."""
    plain = _latlon_neighbor_maps()
    with_z = _latlon_neighbor_maps(vertical_axis=True)
    assert set(plain) == set(with_z)
    for d in plain:
        for a, b in zip(plain[d], with_z[d]):
            if a is None or b is None:
                assert a is None and b is None
            else:
                np.testing.assert_array_equal(a, b)


def test_grid_section_on_grid_with_vertical_axis():
    """End-to-end: a section traced on a grid registering X, Y and Z is identical to
    the same section traced on the horizontal-only view of that grid."""
    from sectionate.section import grid_section
    lons, lats = [10., 40.], [-10., 20.]
    plain = grid_section(_latlon_grid(), lons, lats)
    with_z = grid_section(_latlon_grid(vertical_axis=True), lons, lats)
    for a, b in zip(plain, with_z):
        np.testing.assert_array_equal(a, b)


def test_distance_on_unit_sphere():
    from sectionate.section import distance_on_unit_sphere

    # test of few points with unit radius
    d = distance_on_unit_sphere(0, 0, 1.e-20, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 360, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 45, 0, -45, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 180, 0, R=1.)
    assert np.isclose(d, np.pi, atol=1.e-14)
    d = distance_on_unit_sphere(180, 0, 90, 0, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)
    d = distance_on_unit_sphere(180, 45, 180, 0, R=1.)
    assert np.isclose(d, np.pi/4, atol=1.e-14)


def test_find_closest_grid_point():
    from sectionate.section import find_closest_grid_point

    # check it works with numpy arrays
    i, j = find_closest_grid_point(0, 0, lon, lat)
    assert np.equal(i, 0)
    assert np.equal(j, 80)

    # and xarray
    i, j = find_closest_grid_point(0, 0, ds["lon"], ds["lat"])
    assert np.equal(i, 0)
    assert np.equal(j, 80)

    i, j = find_closest_grid_point(180, 80, ds["lon"], ds["lat"])
    assert np.equal(i, 180)
    assert np.equal(j, 160)


def test_grid_path():
    from sectionate.section import infer_grid_path
    maps = _latlon_neighbor_maps()

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path(0, 80, 179, 80, lon, lat, maps)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 0, 160, lon, lat, maps)
    assert len(isec) == 161
    assert lonsec[0] == 0.
    assert lonsec[-1] == 0.
    assert latsec[0] == -80.0
    assert latsec[-1] == 80.0

    # test diagonal
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 100, 100, lon, lat, maps)
    assert len(isec) == 201  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 50, 100, lon, lat, maps)
    assert len(isec) == 151  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(10, 10, 100, 50, lon, lat, maps)
    assert len(isec) == 131  # expect ni+nj+1 values


def test_infer_grid_path_from_geo():
    from sectionate.section import infer_grid_path_from_geo
    maps = _latlon_neighbor_maps()

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(0, 0, 179, 0, lon, lat, maps)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(180, -80, 180, 80, lon, lat, maps)
    assert len(isec) == 161
    assert lonsec[0]  == 180.0
    assert lonsec[-1] == 180.0
    assert latsec[0]  == -80.0
    assert latsec[-1] == 80.0


def test_infer_grid_path_requires_neighbor_maps():
    """The low-level pathfinder needs grid-derived connectivity; there is no fallback."""
    from sectionate.section import infer_grid_path, infer_grid_path_from_geo
    with pytest.raises(ValueError):
        infer_grid_path(0, 0, 1, 1, lon, lat, None)
    with pytest.raises(ValueError):
        infer_grid_path_from_geo(0, 0, 1, 1, lon, lat, None)


def test_zero_length_seam_face_dropped():
    """On a symmetric periodic grid the seam vertex (360 == 0) carries two indices, so a
    seam-crossing section has a doubled corner in i_c. The zero-length edge between the two
    identities is not emitted as a velocity face, while the real faces on either side are."""
    from sectionate.section import grid_section, distance_on_unit_sphere
    from sectionate.transports import uvindices_from_qindices
    N = 8
    xq = np.linspace(0., 360., N + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.array([-30., 0., 30.]); yh = np.array([-15., 15.])
    lon_c, lat_c = np.meshgrid(xq, yq); lon2, lat2 = np.meshgrid(xh, yh)
    g = xgcm.Grid(
        xr.Dataset(coords={
            "xq": np.arange(N + 1), "yq": np.arange(3), "xh": np.arange(N), "yh": np.arange(2),
            "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
            "geolon": (("yh", "xh"), lon2), "geolat": (("yh", "xh"), lat2),
        }),
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": "extend"}, autoparse_metadata=False,
    )
    # short way from 315 deg to 45 deg runs east across the periodic seam
    i_c, j_c, lons_c, lats_c = grid_section(g, [315., 45.], [0., 0.])

    edge_len = distance_on_unit_sphere(lons_c[:-1], lats_c[:-1], lons_c[1:], lats_c[1:])
    n_degenerate = int(np.sum(edge_len < 1.e-3))
    assert n_degenerate == 1                          # one zero-length edge at the seam vertex
    assert i_c.size >= 4                               # the doubled corner is retained in i_c

    uv = uvindices_from_qindices(g, i_c, j_c)
    # the degenerate edge is not a face; the flanking real faces are
    assert uv["var"].size == (i_c.size - 1) - n_degenerate
    assert np.all(uv["var"] != "0")


def test_coincident_twin_termination():
    """When the walker reaches a corner that is a different index but the same physical
    point as the target (a fold-seam twin), it terminates there via COINCIDENT_TOLERANCE_M
    rather than oscillating to the step-count cap. Pre-fix this looped to a RuntimeError."""
    from sectionate.section import (
        infer_grid_path, COINCIDENT_TOLERANCE_M, distance_on_unit_sphere,
    )
    # col 3 is a near-exact twin of col 0 (the target): a sub-tolerance offset that is
    # nonetheless above the old exact-coincidence threshold.
    glon = np.array([[0.0, 2.0, 1.0, 1.e-12]])
    glat = np.zeros((1, 4))
    d_twin = distance_on_unit_sphere(glon[0, 0], glat[0, 0], glon[0, 3], glat[0, 3])
    assert 1.e-12 < d_twin < COINCIDENT_TOLERANCE_M

    def m(imap):
        return (None, np.zeros((1, 4), dtype=int), np.array([imap], dtype=int))
    # a line  col1 -- col2 -- col3 (~col0);  col0 hangs only off col3
    maps = {
        "right": m([3, 1, 1, 2]),
        "left":  m([0, 2, 3, 0]),
        "up":    m([0, 1, 2, 3]),
        "down":  m([0, 1, 2, 3]),
    }
    i_c, j_c, lons_c, lats_c = infer_grid_path(1, 0, 0, 0, glon, glat, maps)
    # stopped on the twin (index 3), never needing to reach the target's index (0)
    assert i_c[-1] == 3
    assert distance_on_unit_sphere(lons_c[-1], lats_c[-1], glon[0, 0], glat[0, 0]) < COINCIDENT_TOLERANCE_M
