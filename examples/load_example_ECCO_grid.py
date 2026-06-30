"""Load the ECCOv4r4 native lat-lon-cap (LLC90) grid -- and the volume-transport
fields needed for the meridional overturning streamfunction -- as an ``xgcm.Grid``
that sectionate can consume directly.

The required files are a small subset of NASA's ECCO V4r4 state estimate
(geometry plus the twelve monthly volume-flux files for 2010), redistributed on
Zenodo (concept DOI `10.5281/zenodo.21051424`, this version `10.5281/zenodo.21051920`)
purely to make this example reproducible without a NASA Earthdata Login. Any file
not already present under ``../data/`` is downloaded from Zenodo and checked against
its published MD5; the original PO.DAAC datasets and their DOIs are listed in the
Zenodo record's README. Please cite the original NASA sources (see that README).

The native grid is MITgcm-staggered: vorticity points sit on the SW ('left')
corner and the coordinates are named ``XC/YC`` (centers) and ``XG/YG`` (corners).
We rename those to the ``geolon*/geolat*`` convention. sectionate consumes the
native 'left' staggering directly (see ``sectionate.gridutils.corner_position`` /
``corner_offset``); there is no need to convert to a symmetric ('outer') grid.
"""

import os
import glob
import hashlib
import urllib.request
import numpy as np
import xarray as xr
import xgcm

# Zenodo record holding the redistributed ECCO V4r4 subset (version 2.0.0).
ZENODO_RECORD_ID = "21051920"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21051424"

ECCO_GEOMETRY_FILE = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
ECCO_VOLUME_FLUX_GLOB = "OCEAN_3D_VOLUME_FLUX_mon_mean_*_ECCO_V4r4_native_llc0090.nc"

# Published MD5 checksums (from the Zenodo record) for the files this example needs:
# the grid geometry and the twelve 2010 monthly volume-flux files.
ECCO_FILE_MD5 = {
    "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc": "2663a7e86d7a0e6f7ddf84124c8376a6",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-01_ECCO_V4r4_native_llc0090.nc": "9d4371c969b2887a6ec61bd32d8f94e9",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-02_ECCO_V4r4_native_llc0090.nc": "1cad72f3f1030995e6e2b07b41c7d01a",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-03_ECCO_V4r4_native_llc0090.nc": "2bd83ca7a7b4e96b91e6179e683d88e8",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-04_ECCO_V4r4_native_llc0090.nc": "dffc775c76c3418f7fa0aa919a3854a5",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-05_ECCO_V4r4_native_llc0090.nc": "b46303fe1ae85a31b57cd5213a0928b2",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-06_ECCO_V4r4_native_llc0090.nc": "1540a87364c33614af29f97b8bca0eec",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-07_ECCO_V4r4_native_llc0090.nc": "226f7bdc68bff510189f2a8242d7193e",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-08_ECCO_V4r4_native_llc0090.nc": "fbf24a53e650a9c07052aa5369472b04",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-09_ECCO_V4r4_native_llc0090.nc": "e1a4ac8847c8948317d018445d49556e",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-10_ECCO_V4r4_native_llc0090.nc": "17845a2cfeb1f53d4ea7f97150bb9b5d",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-11_ECCO_V4r4_native_llc0090.nc": "55086ea0467dd7bd3b21e17c5afb097d",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010-12_ECCO_V4r4_native_llc0090.nc": "1487f31096153b5dfe188755aceb6553",
}

ECCO_VOLUME_FLUX_FILES = sorted(
    f for f in ECCO_FILE_MD5 if f.startswith("OCEAN_3D_VOLUME_FLUX")
)

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


def _md5(path, chunk=1 << 20):
    """Return the hex MD5 digest of a file, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as fobj:
        for block in iter(lambda: fobj.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_from_zenodo(filename, data_dir="../data"):
    """Return the local path to `filename`, downloading it from the Zenodo record
    if it is missing or fails its published MD5 checksum.

    Existing files under `data_dir` (e.g. ones already obtained from PO.DAAC) are
    reused as-is once their checksum matches, so nothing is re-downloaded.
    """
    expected = ECCO_FILE_MD5[filename]
    path = os.path.join(data_dir, filename)
    if os.path.exists(path) and _md5(path) == expected:
        return path
    os.makedirs(data_dir, exist_ok=True)
    url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1"
    print(f"Downloading {filename} from Zenodo ...", flush=True)
    urllib.request.urlretrieve(url, path)
    got = _md5(path)
    if got != expected:
        raise ValueError(
            f"MD5 mismatch for {filename}: expected {expected}, got {got}"
        )
    return path


def download_ECCO_geometry(data_dir="../data"):
    """Return the local path to the ECCO geometry file, fetching it from Zenodo
    if it is not already present under `data_dir`."""
    return _fetch_from_zenodo(ECCO_GEOMETRY_FILE, data_dir=data_dir)


def download_ECCO_volume_flux(data_dir="../data"):
    """Return a sorted list of the twelve 2010 monthly volume-flux files, fetching
    any that are missing from Zenodo.

    These files hold the native 'left'-staggered mass-weighted velocities
    ``UVELMASS`` (on the cell 'u'/west face, dims ``(k, tile, j, i_g)``) and
    ``VVELMASS`` (on the 'v'/south face, dims ``(k, tile, j_g, i)``), in m/s.
    """
    return [_fetch_from_zenodo(f, data_dir=data_dir) for f in ECCO_VOLUME_FLUX_FILES]


def _ecco_dataset(data_dir="../data", with_transports=True):
    """Build the merged ECCO dataset: geometry (metrics, masks, coordinates) plus,
    optionally, the meridional/zonal volume transports across U/V cell faces."""
    geom = xr.open_dataset(download_ECCO_geometry(data_dir=data_dir))
    ds = geom.rename({"XC": "geolon", "YC": "geolat",
                      "XG": "geolon_c", "YG": "geolat_c"})

    if with_transports:
        vel_files = download_ECCO_volume_flux(data_dir=data_dir)
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


def load_ECCO_MOC_grid(data_dir="../data"):
    """Load the ECCOv4r4 LLC90 grid together with the volume transports needed to
    diagnose the meridional overturning streamfunction.

    Same native ('left') multi-tile grid as ``load_ECCO_LLC90_grid``, but the
    dataset additionally carries ``utr``/``vtr`` -- the volume transports (m^3/s)
    across the U (west) and V (south) cell faces, with vertical dimension ``k``
    (depth coordinate ``Z``), time-averaged over the twelve months of 2010 -- ready
    to pass to ``sectionate.transports.convergent_transport`` as
    ``utr="utr", vtr="vtr"``.
    """
    return _ecco_grid(_ecco_dataset(data_dir=data_dir, with_transports=True))
