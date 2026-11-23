"""Sections across the fold of a MOM-style tripolar grid.

The test grid is regular in latitude and longitude south of LAT0. North of it is a bipolar cap, built from
bipolar coordinates in the north-polar stereographic plane, whose two poles sit on vorticity points i = 0
and i = NX/2 of the cap's southern edge, as in MOM5/MOM6. The top row of vorticity points is the fold:
(top, i) and (top, NX - i) are the same point.

Transports are made divergence-free by deriving them from a streamfunction on vorticity points,
u = psi[j+1, i] - psi[j, i] and v = -(psi[j, i+1] - psi[j, i]). The transport across any section is then
psi(end) - psi(start), whatever path the section takes: exactly zero around a closed section. That checks
both that the section is a connected chain of velocity faces and that each face is the right one with the
right sign, including where the section crosses the fold.
"""
import numpy as np
import pytest
import xarray as xr
import xgcm

from sectionate.section import grid_section
from sectionate.transports import convergent_transport

NX = 72        # vorticity points around a latitude circle
NYC = 12       # rows of cells in the bipolar cap
LAT_SOUTH = 30.
LAT0 = 60.     # southern edge of the bipolar cap
DLAT = 2.
LON0 = 80.     # longitude of the first pole


def tripolar_supergrid():
    """Longitude and latitude on a supergrid with twice the resolution of the model grid."""
    m = 2*NX
    lam = 2*np.pi*np.arange(m+1)/m          # angle from the first pole, including the periodic repeat
    lam_deg = np.rad2deg(lam)

    # regular latitude-longitude grid up to and including the edge of the cap
    lat_s = np.arange(LAT_SOUTH, LAT0 + DLAT/4, DLAT/2)
    x_s = np.broadcast_to(LON0 + lam_deg, (lat_s.size, m+1))
    y_s = np.broadcast_to(lat_s[:, None], (lat_s.size, m+1))

    # bipolar cap: z = tan(colatitude/2) exp(i longitude) in the north-polar stereographic plane, with the
    # poles at z = +a and -a on the cap's edge |z| = a. In bipolar coordinates zeta = tau + i sigma,
    # (z + a)/(z - a) = exp(zeta): grid rows are arcs of constant sigma through both poles, from |sigma| = pi/2
    # on the cap's edge to |sigma| = pi on the segment between the poles, which is the fold; columns are
    # curves of constant tau, matching the cap's edge at the longitude of each column.
    a = np.tan(np.deg2rad(90. - LAT0)/2)
    upper = lam < np.pi                          # columns 0 < i < NX/2 (and the pole at i = 0)
    with np.errstate(divide="ignore"):
        tau = np.log(np.abs(1/np.tan(lam/2)))
    frac = np.arange(1, 2*NYC+1)[:, None]/(2*NYC)
    sigma = np.where(upper, -1., 1.)[None, :]*(np.pi/2)*(1. + frac)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(tau[None, :] + 1j*sigma)
        z = a*(e + 1)/(e - 1)
    z[-1] = a*np.tanh(tau/2)                     # the fold row, exactly: the same point on both sides
    poles = np.isinf(tau)
    z[:, poles & (tau > 0)] = a                  # columns through the poles collapse onto the poles
    z[:, poles & (tau < 0)] = -a
    x_c = LON0 + np.rad2deg(np.angle(z))
    y_c = 90. - 2*np.rad2deg(np.arctan(np.abs(z)))

    x = np.concatenate([x_s, x_c], axis=0)
    y = np.concatenate([y_s, y_c], axis=0)
    return x, y


def streamfunction(shape, seed=0):
    """Random streamfunction on the vorticity points of the symmetric grid, single-valued across the fold."""
    psi = np.random.default_rng(seed).standard_normal(shape)
    psi[:, -1] = psi[:, 0]
    twin = (NX - np.arange(NX+1)) % NX
    psi[-1] = 0.5*(psi[-1] + psi[-1, twin])
    return psi


def tripolar_grid(symmetric):
    x, y = tripolar_supergrid()
    psi = streamfunction(x[::2, ::2].shape)
    u = psi[1:, :] - psi[:-1, :]                   # (yh, xq)
    v = -(psi[:, 1:] - psi[:, :-1])                # (yq, xh)
    ny = u.shape[0]
    ds = xr.Dataset(
        {"u": (("yh", "xq"), u), "v": (("yq", "xh"), v), "psi": (("yq", "xq"), psi)},
        coords={
            "xh": np.arange(NX) + 0.5, "xq": np.arange(NX+1) + 0.,
            "yh": np.arange(ny) + 0.5, "yq": np.arange(ny+1) + 0.,
            "geolon_c": (("yq", "xq"), x[::2, ::2]), "geolat_c": (("yq", "xq"), y[::2, ::2]),
            "geolon": (("yh", "xh"), x[1::2, 1::2]), "geolat": (("yh", "xh"), y[1::2, 1::2]),
            "geolon_u": (("yh", "xq"), x[1::2, ::2]), "geolat_u": (("yh", "xq"), y[1::2, ::2]),
            "geolon_v": (("yq", "xh"), x[::2, 1::2]), "geolat_v": (("yq", "xh"), y[::2, 1::2]),
        },
    )
    position = "outer"
    if not symmetric:
        # vorticity points to the north-east of tracer points only
        ds = ds.isel(xq=slice(1, None), yq=slice(1, None))
        position = "right"
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", position: "xq"}, "Y": {"center": "yh", position: "yq"}},
        boundary={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )


def section_transport(grid, lons, lats):
    i_c, j_c, _, _ = grid_section(grid, lons, lats, topology="MOM-tripolar")
    conv = convergent_transport(grid, i_c, j_c, utr="u", vtr="v", layer=None)
    return float(conv["conv_mass_transport"].sum()), i_c, j_c


def test_test_grid_is_a_tripolar_grid():
    grid = tripolar_grid(symmetric=True)
    lon, lat = grid._ds.geolon_c.values, grid._ds.geolat_c.values
    top = lat[-1, :NX]
    assert np.allclose(top, top[(NX - np.arange(NX)) % NX])        # the top row folds onto itself
    assert np.argmax(top) in (NX//4, 3*NX//4) and np.isclose(top.max(), 90.)
    assert np.allclose(lat[:, 0][lat[:, 0] >= LAT0], LAT0)            # column 0 collapses onto a pole
    v = grid._ds.v.values[-1]
    assert np.allclose(v, -v[::-1])                                     # flow across the fold is consistent


@pytest.mark.parametrize("symmetric, offset", [(True, 0), (False, -2)])
def test_fold_partner_offset(symmetric, offset):
    from sectionate.section import fold_partner_offset
    grid = tripolar_grid(symmetric)
    lon, lat = grid._ds.geolon_c.values, grid._ds.geolat_c.values
    if symmetric:
        lon, lat = lon[:, :-1], lat[:, :-1]     # as passed to the path finder: no repeated periodic column
    assert fold_partner_offset(lon, lat) == offset


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("lat", [61., 65., 70., 75., 80., 85., 88.])
def test_latitude_circle_across_fold_is_closed(symmetric, lat):
    """A latitude circle north of the cap's edge crosses the fold twice; zero net transport."""
    grid = tripolar_grid(symmetric)
    lons = np.linspace(3., 363., 37)
    transport, _, _ = section_transport(grid, lons, np.full(lons.size, lat))
    assert abs(transport) < 1.e-10


@pytest.mark.parametrize("symmetric", [True, False])
def test_closed_section_through_fold_points(symmetric):
    """Waypoints exactly on the fold, reached from both sides of it."""
    grid = tripolar_grid(symmetric)
    lon_c, lat_c = grid._ds.geolon_c.values, grid._ds.geolat_c.values
    lons = [lon_c[-1, 6], LON0 + 150., lon_c[-1, 30], LON0 + 250., lon_c[-1, 6]]
    lats = [lat_c[-1, 6], 72., lat_c[-1, 30], 70., lat_c[-1, 6]]
    transport, _, _ = section_transport(grid, lons, lats)
    assert abs(transport) < 1.e-10


@pytest.mark.parametrize("symmetric", [True, False])
def test_open_section_across_fold(symmetric):
    """Across the fold and the pole: the transport is the streamfunction difference between the ends."""
    grid = tripolar_grid(symmetric)
    lons, lats = [LON0 + 40., LON0 + 220.], [70., 70.]
    with pytest.warns(UserWarning, match="orientation of open sections"):
        transport, i_c, j_c = section_transport(grid, lons, lats)
    psi = grid._ds.psi.values
    dpsi = psi[j_c[-1], i_c[-1]] - psi[j_c[0], i_c[0]]
    assert abs(dpsi) > 0.1
    assert np.isclose(abs(transport), abs(dpsi), rtol=0., atol=1.e-10)
