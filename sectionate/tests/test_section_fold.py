"""Tests for sections on single-tile grids with a bipolar north-fold boundary.

The fold is expressed purely as an xgcm boundary, e.g.
``padding={"X": "periodic", "Y": {"fold": "corner"}}``; sectionate derives the
seam connectivity from xgcm's padding with no fold-specific code of its own.
"""
import os

import numpy as np
import pytest
import xarray as xr
import xgcm

from sectionate.gridutils import get_geo_corners, build_neighbor_maps
from sectionate.section import grid_section


def _fold_grid(nx=8, ny=5, pivot="corner"):
    """A minimal single-tile C-grid, X-periodic with a bipolar north fold."""
    xq, yq = np.arange(nx), np.arange(ny)
    xh, yh = np.arange(nx) + 0.5, np.arange(ny) + 0.5
    lon_c, lat_c = np.meshgrid(xq * (360.0 / nx), yq)
    lon, lat = np.meshgrid(xh * (360.0 / nx), yh)
    ds = xr.Dataset(coords={
        "xq": xq, "yq": yq, "xh": xh, "yh": yh,
        "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
        "geolon": (("yh", "xh"), lon), "geolat": (("yh", "xh"), lat),
    })
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}},
        padding={"X": "periodic", "Y": {"fold": pivot}},
        autoparse_metadata=False,
    )


def _xgcm_supports_fold():
    """The bipolar-fold boundary is only available in newer xgcm (xgcm/xgcm#711)."""
    try:
        _fold_grid(4, 3)
        return True
    except Exception:
        return False


# Skip the whole module on xgcm releases without the bipolar-fold boundary.
pytestmark = pytest.mark.skipif(
    not _xgcm_supports_fold(),
    reason="installed xgcm lacks the bipolar north-fold boundary (xgcm/xgcm#711)",
)


def test_fold_neighbor_connectivity():
    """The top row's `up` neighbors cross the seam (fold), unlike a clipped edge."""
    nx, ny = 8, 5
    fold = _fold_grid(nx, ny)
    fold_maps = build_neighbor_maps(fold, get_geo_corners(fold))
    top = ny - 1
    up_f, up_j, up_i = fold_maps["up"]

    # Single-tile: no face dimension.
    assert up_f is None
    # The fold folds the top row down by one row onto mirrored columns: every
    # seam corner's `up` neighbor sits on row top-1 (not a self-wall) at a
    # different column (the zonal mirror).
    assert np.all(up_j[top] == top - 1)
    assert np.any(up_i[top] != np.arange(nx))

    # Contrast with a plain extend edge, where `up` of the top row is a wall
    # (the point itself) -- i.e. the fold connectivity is genuinely being used.
    extend = _fold_grid(nx, ny)
    extend = xgcm.Grid(
        extend._ds,
        coords={"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}},
        padding={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )
    ext_maps = build_neighbor_maps(extend, get_geo_corners(extend))
    assert np.all(ext_maps["up"][1][top] == top)        # clipped to self
    assert np.all(ext_maps["up"][2][top] == np.arange(nx))


def _mom6_example_path():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(
        here, "..", "..", "data", "MOM6_global_example_sigma2_budgets_v0_0_6.nc"
    ))


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_fold_section_crosses_arctic_on_real_grid():
    """A high-latitude zonal section that clips on an `extend` grid traces the full
    latitude circle once the same grid declares its bipolar fold."""
    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    lons = np.arange(0.0, 360.0 + 5.0, 5.0)
    lats = [80.0] * len(lons)

    extend = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": "extend"},
                       autoparse_metadata=False)
    with pytest.raises(RuntimeError):
        grid_section(extend, lons, lats)  # cannot cross the fold without it

    fold = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)
    i_c, j_c, lons_c, lats_c = grid_section(fold, lons, lats)
    # The traced path hugs the requested latitude (it did not wander off the seam).
    assert np.all(np.abs(np.asarray(lats_c) - 80.0) < 1.0)
    # It reaches the northern fold row.
    assert j_c.max() == ds.yq.size - 1
