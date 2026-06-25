"""Tests for `gridutils.symmetrize`, which converts non-symmetric ('left' or
'right') C-grids -- e.g. native MITgcm/ECCO lat-lon-cap grids -- into the
symmetric ('outer') grids the rest of sectionate consumes."""

import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import symmetrize, check_symmetric, get_geo_corners


def _symmetric_periodic_grid(N=6):
    """Small doubly-periodic symmetric ('outer') grid with corner coords."""
    xg = np.arange(N + 1, dtype=float)
    yg = np.arange(N + 1, dtype=float)
    xc = 0.5 * (xg[:-1] + xg[1:])
    yc = 0.5 * (yg[:-1] + yg[1:])
    LONc, LATc = np.meshgrid(xg, yg)
    LON, LAT = np.meshgrid(xc, yc)
    ds = xr.Dataset(coords={
        "xc": ("xc", xc), "yc": ("yc", yc),
        "xg": ("xg", xg), "yg": ("yg", yg),
        "geolon": (("yc", "xc"), LON), "geolat": (("yc", "xc"), LAT),
        "geolon_c": (("yg", "xg"), LONc), "geolat_c": (("yg", "xg"), LATc),
    })
    return xgcm.Grid(ds, coords={"X": {"center": "xc", "outer": "xg"},
                                 "Y": {"center": "yc", "outer": "yg"}},
                     boundary="periodic", autoparse_metadata=False)


def _left_from_symmetric(sym):
    """Derive an equivalent 'left'-staggered grid by dropping the high-side
    corner row/column (the SW corner of each cell)."""
    ds = sym._ds
    N = ds.sizes["xc"]
    dsl = xr.Dataset(coords={
        "xc": ("xc", np.arange(N, dtype=float)),
        "yc": ("yc", np.arange(N, dtype=float)),
        "xg": ("xg", np.arange(N, dtype=float)),
        "yg": ("yg", np.arange(N, dtype=float)),
        "geolon": (("yc", "xc"), ds.geolon.values),
        "geolat": (("yc", "xc"), ds.geolat.values),
        "geolon_c": (("yg", "xg"), ds.geolon_c.values[:-1, :-1]),
        "geolat_c": (("yg", "xg"), ds.geolat_c.values[:-1, :-1]),
    })
    return xgcm.Grid(dsl, coords={"X": {"center": "xc", "left": "xg"},
                                  "Y": {"center": "yc", "left": "yg"}},
                     boundary="periodic", autoparse_metadata=False)


def test_symmetrize_is_noop_on_symmetric_grid():
    sym = _symmetric_periodic_grid()
    assert symmetrize(sym) is sym


def test_left_grid_is_not_directly_supported():
    left = _left_from_symmetric(_symmetric_periodic_grid())
    # 'left' is unsupported until symmetrized.
    with pytest.raises(ValueError):
        check_symmetric(left)


def test_symmetrize_recovers_symmetric_corners():
    sym = _symmetric_periodic_grid(N=6)
    left = _left_from_symmetric(sym)
    rebuilt = symmetrize(left)

    assert check_symmetric(rebuilt)
    geo = get_geo_corners(rebuilt)
    # symmetrize restores the 'outer' corner grid (one more point per axis).
    assert geo["X"].shape == sym._ds.geolon_c.shape == (7, 7)
    # interior corners are identical to the original symmetric grid.
    assert np.allclose(geo["X"].values[:-1, :-1], sym._ds.geolon_c.values[:-1, :-1])
    # the regenerated high-side corner comes from the grid's own topology; for a
    # periodic axis that is the wrap of the low-side corner (same coordinate
    # value). (For multi-tile face_connections seams it is the neighbour tile's
    # corner, with its own distinct value -- exercised by the ECCO example.)
    assert np.allclose(geo["X"].values[:, -1], geo["X"].values[:, 0])
