"""Load the ECCOv4r4 native lat-lon-cap (LLC90) grid -- and the volume-transport
fields needed for the meridional overturning streamfunction -- as an ``xgcm.Grid``
that sectionate can consume directly.

The ECCO files are distributed by NASA PO.DAAC and require a (free) NASA Earthdata
Login. If a file is not already present under ``../data/`` it is downloaded with
``earthaccess`` (which reads credentials from ``~/.netrc``; run
``earthaccess.login(persist=True)`` once to set this up).

The native grid is MITgcm-staggered: vorticity points sit on the SW ('left')
corner and the coordinates are named ``XC/YC`` (centers) and ``XG/YG`` (corners).
We rename those to the ``geolon*/geolat*`` convention. sectionate consumes the
native 'left' staggering directly (see ``sectionate.gridutils.corner_position`` /
``corner_offset``); there is no need to convert to a symmetric ('outer') grid.
"""

import os
import glob
import numpy as np
import xarray as xr
import xgcm

ECCO_GEOMETRY_FILE = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
ECCO_GEOMETRY_SHORTNAME = "ECCO_L4_GEOMETRY_LLC0090GRID_V4R4"

ECCO_VOLUME_FLUX_SHORTNAME = "ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4"
ECCO_VOLUME_FLUX_GLOB = "OCEAN_3D_VOLUME_FLUX_mon_mean_*_ECCO_V4r4_native_llc0090.nc"

# Canonical xgcm face_connections for the 13-tile LLC90 grid
# (from the xgcm ECCOv4 example), keyed by the ECCO 'tile' dimension.
LLC90_FACE_CONNECTIONS = {"tile": {
    0:  {"X": ((12, "Y", False), (3, "X", False)), "Y": (None,            (1, "Y", False))},
    1:  {"X": ((11, "Y", False), (4, "X", False)), "Y": ((0, "Y", False), (2, "Y", False))},
    2:  {"X": ((10, "Y", False), (5, "X", False)), "Y": ((1, "Y", False), (6, "X", False))},
    3:  {"X": ((0,  "X", False), (9, "Y", False)), "Y": (None,            (4, "Y", False))},
    4:  {"X": ((1,  "X", False), (8, "Y", False)), "Y": ((3, "Y", False), (5, "Y", False))},
    5:  {"X": ((2,  "X", False), (7, "Y", False)), "Y": ((4, "Y", False), (6, "Y", False))},
    6:  {"X": ((2,  "Y", False), (7, "X", False)), "Y": ((5, "Y", False), (10, "X", False))},
    7:  {"X": ((6,  "X", False), (8, "X", False)), "Y": ((5, "X", False), (10, "Y", False))},
    8:  {"X": ((7,  "X", False), (9, "X", False)), "Y": ((4, "X", False), (11, "Y", False))},
    9:  {"X": ((8,  "X", False), None),            "Y": ((3, "X", False), (12, "Y", False))},
    10: {"X": ((6,  "Y", False), (11, "X", False)), "Y": ((7, "Y", False), (2, "X", False))},
    11: {"X": ((10, "X", False), (12, "X", False)), "Y": ((8, "Y", False), (1, "X", False))},
    12: {"X": ((11, "X", False), None),            "Y": ((9, "Y", False), (0, "X", False))},
}}


def download_ECCO_geometry(data_dir="../data"):
    """Return the local path to the ECCO geometry file, downloading it from
    PO.DAAC via earthaccess if it is not already present."""
    path = os.path.join(data_dir, ECCO_GEOMETRY_FILE)
    if os.path.exists(path):
        return path
    import earthaccess
    earthaccess.login()  # uses ~/.netrc
    results = earthaccess.search_data(short_name=ECCO_GEOMETRY_SHORTNAME, count=1)
    earthaccess.download(results, local_path=data_dir)
    return path


def download_ECCO_volume_flux(data_dir="../data", temporal=("2010-01-01", "2010-12-31")):
    """Return a sorted list of local ECCO monthly volume-flux files spanning
    `temporal`, downloading any that are missing from PO.DAAC via earthaccess.

    These files hold the native 'left'-staggered mass-weighted velocities
    ``UVELMASS`` (on the cell 'u'/west face, dims ``(k, tile, j, i_g)``) and
    ``VVELMASS`` (on the 'v'/south face, dims ``(k, tile, j_g, i)``), in m/s.
    """
    local = sorted(glob.glob(os.path.join(data_dir, ECCO_VOLUME_FLUX_GLOB)))
    if local:
        return local
    import earthaccess
    earthaccess.login()  # uses ~/.netrc
    results = earthaccess.search_data(
        short_name=ECCO_VOLUME_FLUX_SHORTNAME, temporal=temporal
    )
    earthaccess.download(results, local_path=data_dir)
    return sorted(glob.glob(os.path.join(data_dir, ECCO_VOLUME_FLUX_GLOB)))


def _ecco_dataset(data_dir="../data", temporal=("2010-01-01", "2010-12-31"),
                  with_transports=True):
    """Build the merged ECCO dataset: geometry (metrics, masks, coordinates) plus,
    optionally, the meridional/zonal volume transports across U/V cell faces."""
    geom = xr.open_dataset(download_ECCO_geometry(data_dir=data_dir))
    ds = geom.rename({"XC": "geolon", "YC": "geolat",
                      "XG": "geolon_c", "YG": "geolat_c"})

    if with_transports:
        vel_files = download_ECCO_volume_flux(data_dir=data_dir, temporal=temporal)
        vel = xr.open_mfdataset(vel_files, combine="by_coords")
        # Mass-weighted velocities already fold in the open-cell fraction (hFac), so
        # the volume transport across a face is simply velocity x face width x layer
        # thickness:  U-face (west)  uses dyG;  V-face (south) uses dxG;  drF is the
        # full layer thickness. Result is volume transport in m^3/s.
        utr = (vel["UVELMASS"] * geom["dyG"] * geom["drF"]).astype("float32")
        vtr = (vel["VVELMASS"] * geom["dxG"] * geom["drF"]).astype("float32")
        ds["utr"] = utr.transpose("time", "k", "tile", "j", "i_g")
        ds["vtr"] = vtr.transpose("time", "k", "tile", "j_g", "i")
        ds["utr"].attrs = {"units": "m3 s-1",
                           "long_name": "volume transport across cell west (u) face"}
        ds["vtr"].attrs = {"units": "m3 s-1",
                           "long_name": "volume transport across cell south (v) face"}
    return ds


def _ecco_grid(ds):
    """Wrap a merged ECCO dataset in a native ('left'-staggered) multi-tile
    ``xgcm.Grid`` with the LLC90 tile topology encoded via ``face_connections``."""
    return xgcm.Grid(
        ds, periodic=False, autoparse_metadata=False,
        coords={"X": {"center": "i", "left": "i_g"},
                "Y": {"center": "j", "left": "j_g"}},
        face_connections=LLC90_FACE_CONNECTIONS,
    )


def load_ECCO_LLC90_grid(data_dir="../data"):
    """Load the ECCOv4r4 LLC90 *geometry* as a native (MITgcm 'left'-staggered)
    multi-tile ``xgcm.Grid``.

    The returned grid has tracer-center coordinates ``geolon``/``geolat`` and
    cell-corner coordinates ``geolon_c``/``geolat_c`` (on the native 'left'
    vorticity position; sectionate supports this directly via
    ``corner_position``/``corner_offset``), carries ``Depth`` for land/ocean
    masking, and encodes the LLC90 tile topology via ``face_connections`` -- so
    sections can be traced across tile seams.
    """
    return _ecco_grid(_ecco_dataset(data_dir=data_dir, with_transports=False))


def load_ECCO_MOC_grid(data_dir="../data", temporal=("2010-01-01", "2010-12-31")):
    """Load the ECCOv4r4 LLC90 grid together with the volume transports needed to
    diagnose the meridional overturning streamfunction.

    Same native ('left') multi-tile grid as ``load_ECCO_LLC90_grid``, but the
    dataset additionally carries ``utr``/``vtr`` -- the volume transports (m^3/s)
    across the U (west) and V (south) cell faces, with vertical dimension ``k``
    (depth coordinate ``Z``) -- ready to pass to
    ``sectionate.transports.convergent_transport`` as ``utr="utr", vtr="vtr"``.
    """
    return _ecco_grid(_ecco_dataset(data_dir=data_dir, temporal=temporal,
                                    with_transports=True))
