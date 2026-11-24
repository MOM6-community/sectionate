import pytest

def test_load_section():
    from sectionate.utils import get_all_section_names, load_section
    section_names = get_all_section_names()
    load_section(section_names[0])

def test_get_geo_corners():
    import numpy as np
    import xarray as xr
    import xgcm
    coords = {
        "xh": np.arange(0, 10),
        "yh": np.arange(0, 10),
        "xq": np.arange(0, 11),
        "yq": np.arange(0, 11),
    }

    # Without coordinates
    ds = xr.Dataset(coords=coords)
    grid = xgcm.Grid(
        ds = ds,
        coords={
            "X":{"center":"xh", "outer":"xq"},
            "Y":{"center":"yh", "outer":"yq"}
        },
        padding={"X":"periodic", "Y":"extend"},
        autoparse_metadata=False
    )

    from sectionate.gridutils import get_geo_corners

    # Fail when (lon, lat) coordinates are missing
    with pytest.raises(ValueError):
        get_geo_corners(grid)

    # Pass when coordinates are there
    grid._ds = grid._ds.assign_coords({
        "lon_c": xr.DataArray(xr.broadcast(grid._ds.xq, grid._ds.yq)[0]),
        "lat_c": xr.DataArray(xr.broadcast(grid._ds.xq, grid._ds.yq)[1])
    })
    get_geo_corners(grid)