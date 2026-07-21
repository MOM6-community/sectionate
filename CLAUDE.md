# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sectionate is a Python package for sampling grid-consistent hydrographic sections from structured ocean model outputs (e.g. MOM6). It traces paths along C-grid velocity faces between geographic waypoints and computes transports/tracer values along those sections. Only structured grids are supported.

## Development Setup

```bash
conda env create -f ci/environment.yml
conda activate test_env_sectionate
pip install -e .
```

**Local development environment:** use the `docs_env_sectionate` conda env
(`~/anaconda3/envs/docs_env_sectionate`) for developing this package — running the
test suite, executing the example notebooks, and any package work in this workspace.
Keep it updated as needed; in particular it must carry **xgcm >= 0.10.1**, which
introduces the bipolar north fold (`padding={"Y": {"fold": ...}}`, formerly #711),
the face-connection padding fix (#712), and the bare-`DataArray` `other_component`
vector-pad fix (#749). It is a plain PyPI release, so `pip install -e .` pulls it in
automatically (sectionate requires `xgcm>=0.10.1`). The ECCO example (notebook 5)
pulls its data subset from Zenodo (concept DOI `10.5281/zenodo.21051424`) via stdlib
`urllib` with MD5 checks — no `earthaccess`/Earthdata login needed. A correctly set up
env runs the full suite with **0 skips** (fold and multi-tile tests require this xgcm;
they `skip` on `xgcm < 0.10.1`).

## Commands

- **Run all tests:** `pytest`
- **Run a single test:** `pytest sectionate/tests/test_section.py::test_find_closest_grid_point`
- **Build docs:** `cd docs && make html` (requires `docs/environment.yml` environment)

## Definition of Done (always, before committing)

Before committing any change to this repository, always do all of the following — use
parallel agents where it is faster:

1. **Run the full test suite** (`pytest`) and confirm it passes.
2. **Re-execute the example notebooks** in `examples/` and confirm they run cleanly.
3. **Scan the repo for inconsistencies / outdated information** introduced by the change —
   including docstrings, `README.md`, files under `docs/`, the example notebooks, and this
   `CLAUDE.md` — and update them so docs and code stay in sync.

## Keeping pull requests in sync

When pushing a new commit to a branch that already has an open PR, also update that PR's
top-comment description so it stays consistent with the most recent commit:

- Revise the summary / "Changes" prose to reflect what the latest commit actually did.
- Update any task/status checklist — check off completed items (`- [ ]` → `- [x]`) and add
  new ones as needed.
- Edit it in place with `gh pr edit <number> --repo <owner>/<repo> --body-file <file>` (find
  the PR for the current branch with `gh pr view --json number,url`).

## AI Usage Policy

AI-assisted contributions to sectionate are welcome, but the person running the AI is
responsible for every change. This project follows the
[xgcm AI Usage Policy](https://github.com/hdrake/xgcm/blob/add-Claude.md/docs/contributor_guide.md),
which in turn adapts [xarray's](https://docs.xarray.dev/en/stable/contribute/ai-policy.html).
It applies to every change regardless of whether it was written by hand, with AI
assistance, or generated entirely by an AI tool. The essentials:

- **You own the diff.** Before opening or updating a PR, you must have read and understood
  every line and be able to explain why each change is correct — the same bar as a
  hand-written PR. Keep changes small, single-purpose, and free of unrelated edits.
- **Disclose AI assistance openly.** Add a `Co-Authored-By:` trailer to AI-assisted commits
  and note on any PR or comment that was drafted with AI.
- **Communicate in your own words.** PR descriptions, issue comments, and review responses
  must be your own; do not paste AI-generated text as a comment. Using AI to polish your own
  writing (grammar, phrasing) is fine as long as it introduces no inaccuracies.
- **Every code change ships with tests** and, per the *Definition of Done* above, a green
  suite and re-executed notebooks.
- **Discuss large AI-assisted contributions first.** For a substantial refactor, new
  subsystem, or migration, open an issue to agree scope before generating the diff — a large
  diff is fast to produce and slow to review.

## Architecture

The package is organized around a pipeline: define sections → map to grid → compute transports/tracers.

### Core Modules

- **`section.py`** — Section definition and grid path algorithms. `Section` holds named waypoint coordinates; `GriddedSection` extends it with grid index information. `grid_section()` is the main entry point that maps geographic waypoints to grid vorticity-point indices `(i_c, j_c)` by walking the grid between waypoints along the `curve` requested (`"great circle"`, the default geodesic, or `"latitude circle"`); each segment must span less than 180°. The walk is deterministic and direction-independent: it admits neighbors strictly closer to the endpoint plus any seam twin of the current cell (so periodic/fold-seam crossings do not depend on floating-point rounding), and breaks ties by index. Grid topology is inferred entirely from `xgcm.Grid` metadata — each axis' `boundary` (periodic wrap, fill/extend wall, or a single-tile bipolar north fold `{"Y": {"fold": ...}}`) and `face_connections` for multi-tile grids — so there is no `topology` keyword. The pathfinder consumes topology-aware neighbor maps built by `gridutils.build_neighbor_maps`.

- **`transports.py`** — Transport computation along sections. `uvindices_from_qindices()` converts vorticity-point indices to U/V velocity-point indices using a per-position corner offset (`gridutils.corner_offset`) covering all three C-grid staggerings: 'outer', 'right', and 'left' (see "Corner staggering" below). `convergent_transport()` is the main function: it lazily computes signed normal transports with configurable orientation (positive inward to the polygon defined by the section).

- **`tracers.py`** — `extract_tracer()` interpolates tracer data to U/V points along a section path for cross-section plotting.

- **`gridutils.py`** — Grid introspection utilities: `corner_position()` returns the shared vorticity corner position (`"outer"`, `"right"`, or `"left"`) and `corner_offset()` the corresponding velocity index shift used throughout transports/tracers; `check_outer()` is a thin wrapper (True iff `"outer"`). (Package source names positions only by these xgcm labels; their correspondence to MOM6/MITgcm/ECCO conventions is documented under "Corner staggering" below and in the example notebooks.) `coord_dict()` and `get_geo_corners()` extract coordinate/dimension names from `xgcm.Grid` metadata; `build_neighbor_maps()` builds the topology-aware neighbor maps the pathfinder consumes. For single-tile grids these come from xgcm halo padding; for every multi-tile grid they are projected from **`outer_topology(grid)`** (cached `_OuterTopology`), which reconstructs each face's full 'outer' (shared-corner) lattice, resolves every extended corner slot to the native corner storing that physical point (topological matching of the surrounding tracer cells — including a diagonal-cell recovery for normal-seam face corners and a 3-cell fingerprint for 3-tile junctions with native storage — with a small coordinate fallback only for the polar-cut corners no cell fingerprint can determine), and merges slots into physical corner *nodes* — so rotated/reversed seams, cube-vertex/tile junctions, boundary folds (e.g. LLC's south rows), and the polar grid cut are ordinary graph nodes, each resolved to its canonical native `(face, j, i)` (a physical corner may be stored once, on no face, or more than once). `_OuterTopology.padded_transports(u, v)` extends native transports to the outer lattice by reading each missing edge slot's *stored* twin through the node graph — a topology-exact, xgcm-independent alternative to `xgcm.pad(..., other_component=...)` (cell convergence built from `padded_transports` sums to exactly zero globally on ECCO LLC90). Even a correct vector pad (xgcm#749 fixes the bare-DataArray path; the dict form was always exact) cannot supply a halo for edges stored on *no* face, so `padded_transports` remains the exact path on grids with cuts/caps. Points stored on **no** face (a cubed-sphere stores `6N²` corners for `6N²+2` points; LLC90's 4th Arctic-cap vertex; the unstored lip of its 65°E/115°W polar grid cut under Antarctica) become edge-less nodes: sections and traced boundaries cannot pass through them and raise an error rather than return invalid indices.

- **`utils.py`** — Section catalog I/O. Sections can be loaded by name from JSON catalog files in `sectionate/catalog/`. Also provides `save_gridded_section()`/`load_gridded_section()` for persisting gridded sections.

### Key Concepts

- **Vorticity points (q-points):** Sections are defined as paths through vorticity-point indices `(i_c, j_c)`. Consecutive q-points define velocity faces (either U or V).
- **Corner staggering (three positions):** Vorticity sits at one of three xgcm positions: `"outer"` (MOM6 symmetric, M+1×N+1), `"right"` (MOM6 non-symmetric, M×N), or `"left"` (MITgcm/ECCO, incl. the lat-lon-cap, M×N). All three are native; they differ only by a per-position velocity index offset (`gridutils.corner_offset`: outer→0, right→+1, left→0). `"left"` indexes like `"outer"`; it differs only in array length and in that the *high* corner row/column is absent (so a section exactly on the north/east domain wall clips one row inside).
- **Sign conventions:** For a *closed* section, `convergent_transport()` determines orientation (clockwise/counterclockwise) using stereographic projection and signed polygon area, then applies sign corrections so positive transport means "inward" (toward the enclosed polygon). For an *open* section there is no enclosing polygon, so `positive_in` is undefined; it instead uses the **left-of-transect** convention — `positive_in=True` makes positive transport point to the left of the section as traversed from the first to the last waypoint — and emits a `UserWarning`. `is_section_counterclockwise()` is only consulted in the closed case.
- **xgcm.Grid dependency:** The package relies heavily on `xgcm.Grid` for grid metadata (axis boundaries, coordinate positions, dataset access via `grid._ds`).

## Version

Version is in `sectionate/version.py` and read by hatchling at build time.
