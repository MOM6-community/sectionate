"""Walking a section from A to B must give the exact reverse of walking it from B to A,
and the transport through it must have the same magnitude either way. This exercises the
deterministic, direction-independent neighbor selection on non-trivial (diagonal,
multi-segment) sections -- not just axis-aligned ones."""

import warnings
import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.section import grid_section
from sectionate.transports import convergent_transport


def _fine_grid(nx=24, ny=20, seed=0):
    """A single-tile ('outer') grid with random transports, fine enough that an oblique
    section traces a long, genuinely 2-D staircase."""
    rng = np.random.default_rng(seed)
    xq = np.arange(nx + 1, dtype=float); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.arange(ny + 1, dtype=float) - ny / 2.; yh = 0.5 * (yq[:-1] + yq[1:])
    u = rng.standard_normal((ny, nx + 1))   # umo at (yh, xq)
    v = rng.standard_normal((ny + 1, nx))   # vmo at (yq, xh)
    ds = xr.Dataset(
        {"u": (("yh", "xq"), u), "v": (("yq", "xh"), v)},
        coords={
            "xq": (("xq",), np.arange(nx + 1)), "yq": (("yq",), np.arange(ny + 1)),
            "xh": (("xh",), np.arange(nx)), "yh": (("yh",), np.arange(ny)),
            "geolon_c": (("yq", "xq"), np.broadcast_to(xq, (ny + 1, nx + 1))),
            "geolat_c": (("yq", "xq"), np.broadcast_to(yq[:, None], (ny + 1, nx + 1))),
            "geolon": (("yh", "xh"), np.broadcast_to(xh, (ny, nx))),
            "geolat": (("yh", "xh"), np.broadcast_to(yh[:, None], (ny, nx))),
        })
    return xgcm.Grid(ds, coords={"X": {"outer": "xq", "center": "xh"},
                                 "Y": {"outer": "yq", "center": "yh"}},
                     padding={"X": "extend", "Y": "extend"}, autoparse_metadata=False)


def _assert_reverse_equal(grid, lon, lat, curve="great circle"):
    fwd = grid_section(grid, lon, lat, curve=curve)
    rev = grid_section(grid, lon[::-1], lat[::-1], curve=curve)
    i, j, lo, la = fwd
    i2, j2, lo2, la2 = rev
    # The reverse traversal yields the forward path, reversed -- exactly.
    assert np.array_equal(i, i2[::-1]), f"i not reversible:\n{i}\n{i2[::-1]}"
    assert np.array_equal(j, j2[::-1]), f"j not reversible:\n{j}\n{j2[::-1]}"
    assert np.allclose(lo, lo2[::-1]) and np.allclose(la, la2[::-1])
    return i, j


def test_oblique_section_reversible_and_equal_transport():
    grid = _fine_grid()
    # A ~30-degree, multi-segment section through the grid interior (slope ~tan(30 deg)).
    lon = np.array([2.0, 8.0, 14.0, 21.0])
    lat = np.array([-8.0, -4.5, -1.0, 3.0])

    i, j = _assert_reverse_equal(grid, lon, lat)

    # Genuinely 2-D: the path steps in both i and j (not an axis-aligned line).
    assert np.any(np.diff(i) != 0) and np.any(np.diff(j) != 0)
    assert len(i) >= 20

    # Transport magnitude is the same regardless of traversal direction (the sign flips
    # with the left-of-transect normal for an open section).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # open-section UserWarning
        c_fwd = convergent_transport(
            grid, i, j, utr="u", vtr="v", geometry="cartesian"
        )["conv_mass_transport"].sum().values
        c_rev = convergent_transport(
            grid, i[::-1], j[::-1], utr="u", vtr="v", geometry="cartesian"
        )["conv_mass_transport"].sum().values
    assert np.isclose(abs(c_fwd), abs(c_rev), rtol=1e-12)


def test_steep_and_shallow_oblique_sections_reversible():
    grid = _fine_grid(seed=3)
    # A steeper (~60-degree) and a shallow (~20-degree) multi-segment section.
    _assert_reverse_equal(grid, np.array([3.0, 6.0, 9.0, 12.0]),
                                np.array([-9.0, -4.0, 1.0, 6.0]))   # steep
    _assert_reverse_equal(grid, np.array([1.0, 9.0, 17.0, 23.0]),
                                np.array([-7.0, -5.0, -3.0, -1.0]))  # shallow


def test_zonal_and_meridional_segments_reversible():
    grid = _fine_grid(seed=5)
    # Constant-latitude hops at two different latitudes, joined by a meridional step.
    # Under "latitude and great circle" the two zonal legs follow their parallels and the
    # meridional leg follows the geodesic; both metrics must reverse exactly, and so must
    # the per-segment choice between them.
    lon = np.array([2.0, 10.0, 10.0, 20.0])
    lat = np.array([-6.0, -6.0, 4.0, 4.0])
    _assert_reverse_equal(grid, lon, lat, curve="latitude and great circle")

    # A purely zonal section reverses exactly under "latitude circle" too.
    _assert_reverse_equal(grid, np.array([2.0, 10.0, 20.0]), np.array([-6.0, -6.0, -6.0]),
                          curve="latitude circle")
