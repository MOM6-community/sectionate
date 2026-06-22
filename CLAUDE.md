# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sectionate` samples grid-consistent hydrographic sections from ocean model output. Given a list of geographic (lon, lat) waypoints, it traces a path along the model grid's velocity faces (the C-grid cell edges between vorticity/corner points) that best approximates the geodesic between waypoints, then computes transports of mass/tracers normal to that path. It currently only supports structured grids and is built around MOM6 conventions, using `xgcm.Grid` objects throughout.

## Commands

```bash
# Install dev environment (conda)
conda env create -f docs/environment.yml   # or: ci/environment.yml for the CI/test env
conda activate docs_env_sectionate
pip install -e .

# Run the full test suite
pytest

# Run a single test file / test
pytest sectionate/tests/test_section.py
pytest sectionate/tests/test_section.py::test_name -v
```

CI (`.github/workflows/ci.yml`) runs `pytest` against Python 3.11–3.14 in a conda-forge env. Publishing to PyPI happens on GitHub release via `publish-to-pypi.yml`. Version is single-sourced from `sectionate/version.py` (hatch reads it); bump it there for releases.

## Architecture

The pipeline has three stages, mapped to the three main modules. All operate on an `xgcm.Grid` whose `._ds` holds the model data and whose `.axes` metadata is used to detect grid symmetry and boundary topology.

1. **`section.py` — geometry / pathfinding.** Turns geographic waypoints into a sequence of grid corner indices `(i_c, j_c)` (vorticity/Q points) plus their coords `(lons_c, lats_c)`.
   - `Section` is the core data class: a named section defined by `coords` (accepts either a list of `(lon, lat)` 2-tuples or a `(lons, lats)` 2-tuple of arrays). `GriddedSection` subclasses it and binds the section to a specific `grid`, computing `(i_c, j_c)` on construction.
   - `grid_section(grid, lons, lats)` is the high-level entry point → delegates to `create_section_composite` → `create_section` → `infer_grid_path`/`infer_grid_path_from_geo`, the actual stepping algorithm that walks cell edges between consecutive grid points.
   - `join_sections(name, *sections)` concatenates child sections into a parent, optionally re-aligning each child's direction (via `align_coords`) so endpoints connect; children are retained on `.children`.
   - The `topology` argument (`"latlon"`, `"cartesian"`, `"MOM-tripolar"`) controls how paths wrap across grid boundaries. Spherical-distance helpers (`distance_on_unit_sphere`, `spherical_angle`) also live here.

2. **`transports.py` — transport calculation.** Converts the section's corner indices into the velocity points on the faces it crosses and integrates fluxes across them.
   - `uvindices_from_qindices` maps each pair of consecutive Q-point indices to the U or V velocity point on the face between them (this is where MOM6 symmetric vs. non-symmetric grid index offsets are handled). `uvcoords_from_uvindices` / `uvcoords_from_qindices` recover their coordinates.
   - `convergent_transport(grid, i_c, j_c, ...)` is the main result: lazily (dask) computes extensive transport normal to the section, sign convention "positive into" the polygon the section encloses, broadcasting over all dims except (X, Y). Orientation is determined by `is_section_counterclockwise` (uses `stereographic_projection`); `is_mask_inside` resolves the inside/outside sign.

3. **`tracers.py` — `extract_tracer`** samples a tracer field at the section's velocity faces (e.g. to plot a property along the section).

Supporting modules:
- **`gridutils.py`** — small helpers that read `xgcm.Grid` metadata: `get_geo_corners` (raises `ValueError` if the grid lacks lon/lat coords), `coord_dict`, `check_symmetric`. These are the seam between the abstract algorithms and a concrete grid's variable names.
- **`utils.py`** — section catalog I/O. `load_section`/`get_all_section_names` read predefined sections from `sectionate/catalog/*.json` (e.g. `Atlantic_transport_arrays.json`). `save_gridded_section`/`load_gridded_section` persist a `GriddedSection` to/from netCDF.

## Conventions

- "Corner" / "Q" / "vorticity" points are the `_c` (i_c, j_c, lons_c, lats_c) suffixed quantities; "U"/"V" velocity points live on the faces between them. The X axis is nominally zonal (U), Y nominally meridional (V), but the code does not assume true east/north — it works for curvilinear/tripolar grids as long as the grid is *locally* orthogonal.
- Grid symmetry (`check_symmetric`) and axis boundary types (`boundary={"X":"periodic","Y":"extend"}`) are read from `grid` metadata and threaded through pathfinding and index mapping; don't hardcode them.
- Public API is re-exported flat from `sectionate/__init__.py` (`from .section import *`, etc.), so functions are typically called as `sectionate.<name>`.
- Example notebooks in `examples/` and docs sources in `docs/` (Sphinx, built on Read the Docs) demonstrate end-to-end usage against the sample data in `data/`.
