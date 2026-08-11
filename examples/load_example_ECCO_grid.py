"""Load the ECCOv4r4 native lat-lon-cap (LLC90) grid -- and the volume-transport
fields needed for the meridional overturning streamfunction -- as an ``xgcm.Grid``
that sectionate can consume directly.

The required files are a small subset of NASA's ECCO V4r4 state estimate (the grid
geometry plus the 2010 monthly volume fluxes), redistributed on Zenodo purely to make
this example reproducible without a NASA Earthdata Login. Cite it by its concept DOI,
`10.5281/zenodo.21051424`, which is version-independent and always resolves to the
latest release; the particular version this module downloads is pinned by
``ZENODO_RECORD_ID`` below, which is the single place that version is recorded. Any
file not already present under ``../data/`` is downloaded from that record and checked
against its published MD5; the original PO.DAAC datasets and their DOIs are listed in
the Zenodo record's README. Please cite the original NASA sources (see that README).

The native grid is MITgcm-staggered: vorticity points sit on the SW ('left')
corner and the coordinates are named ``XC/YC`` (centers) and ``XG/YG`` (corners).
We rename those to the ``geolon*/geolat*`` convention. sectionate consumes the
native 'left' staggering directly (see ``sectionate.gridutils.corner_position`` /
``corner_offset``); there is no need to convert to a symmetric ('outer') grid.
"""

import os
import hashlib
import urllib.request
import xarray as xr
import xgcm

# Zenodo record holding the redistributed ECCO V4r4 subset. ZENODO_RECORD_ID pins the
# exact version that is downloaded, and is the only place that version is recorded;
# ZENODO_CONCEPT_DOI is the version-independent (citeable) DOI that always resolves to
# the latest version. Bumping ZENODO_RECORD_ID means re-checking ECCO_FILE_MD5 below,
# since each version publishes its own file list and checksums.
ZENODO_RECORD_ID = "21479854"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21051424"

ECCO_GEOMETRY_FILE = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"

# Published MD5 checksums (from the Zenodo record) for the files this example needs:
# the grid geometry, and a single file holding all twelve 2010 monthly volume fluxes
# along a `time` dimension.
ECCO_FILE_MD5 = {
    "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc": "2663a7e86d7a0e6f7ddf84124c8376a6",
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc": "a957bd1d133156b0ad6dd611bd39af7a",
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
    # Download to a temporary file and only move it into place once the checksum
    # verifies, so an interrupted download can't leave a truncated file masquerading
    # as the real one at `path`.
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        got = _md5(tmp)
        if got != expected:
            raise ValueError(
                f"MD5 mismatch for {filename}: expected {expected}, got {got}"
            )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def download_ECCO_geometry(data_dir="../data"):
    """Return the local path to the ECCO geometry file, fetching it from Zenodo
    if it is not already present under `data_dir`."""
    return _fetch_from_zenodo(ECCO_GEOMETRY_FILE, data_dir=data_dir)


def download_ECCO_volume_flux(data_dir="../data"):
    """Return the local paths of the 2010 monthly volume-flux data (currently a single
    file covering all twelve months), fetching from Zenodo any that are missing.

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
        ds, padding="fill", autoparse_metadata=False,
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
    (depth coordinate ``Z``) and a length-12 ``time`` dimension holding the twelve
    2010 monthly means -- ready to pass to
    ``sectionate.transports.convergent_transport`` as ``utr="utr", vtr="vtr"``
    (average over ``time`` first for an annual-mean overturning).
    """
    return _ecco_grid(_ecco_dataset(data_dir=data_dir, with_transports=True))
