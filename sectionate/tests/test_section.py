import numpy as np
import xarray as xr


# define simple grid
lon, lat = np.meshgrid(np.arange(360), np.arange(-90, 90))
ds = xr.Dataset()
ds["lon"] = xr.DataArray(lon, dims=("y", "x"))
ds["lat"] = xr.DataArray(lat, dims=("y", "x"))

def test_distance_on_unit_sphere():
    from sectionate.section import distance_on_unit_sphere

    # test of few points with unit radius
    d = distance_on_unit_sphere(0, 0, 1.e-20, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 360, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 90, 0, -90, R=1.)
    assert np.isclose(d, np.pi, atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 180, 0, R=1.)
    assert np.isclose(d, np.pi, atol=1.e-14)
    d = distance_on_unit_sphere(180, 0, 90, 0, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)
    d = distance_on_unit_sphere(180, 90, 180, 0, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)


def test_find_closest_grid_point():
    from sectionate.section import find_closest_grid_point

    # check it works with numpy arrays
    i, j = find_closest_grid_point(0, 0, lon, lat)
    assert np.equal(i, 0)
    assert np.equal(j, 90)

    # and xarray
    i, j = find_closest_grid_point(0, 0, ds["lon"], ds["lat"])
    assert np.equal(i, 0)
    assert np.equal(j, 90)

    i, j = find_closest_grid_point(180, 89, ds["lon"], ds["lat"])
    assert np.equal(i, 180)
    assert np.equal(j, 179)


def test_grid_path():
    from sectionate.section import infer_grid_path

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path(0, 90, 179, 90, lon, lat)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path(180, 0, 180, 179, lon, lat)
    assert len(isec) == 180
    assert lonsec[0] == 180.0
    assert lonsec[-1] == 180.0
    assert latsec[0] == -90.0
    assert latsec[-1] == 89.0

    # test diagonal
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 100, 100, lon, lat)
    assert len(isec) == 201  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 50, 100, lon, lat)
    assert len(isec) == 151  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(10, 10, 100, 50, lon, lat)
    assert len(isec) == 131  # expect ni+nj+1 values


def test_infer_grid_path_from_geo():
    from sectionate.section import infer_grid_path_from_geo

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(0, 0, 179, 0, lon, lat)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(180, -89, 180, 89, lon, lat)
    assert len(isec) == 179
    assert lonsec[0] == 180.0
    assert lonsec[-1] == 180.0
    assert latsec[0] == -89.0
    assert latsec[-1] == 89.0

    # test diagonal
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(0, -89, 180, 0, lon, lat)
    assert len(isec) == 272
