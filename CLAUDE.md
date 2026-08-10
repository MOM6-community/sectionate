# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sectionate is a Python package for sampling grid-consistent hydrographic sections from structured ocean model outputs. It traces paths along C-grid velocity faces between geographic waypoints and computes transports/tracer values along those sections. It supports any structured model whose grid can be described by an `xgcm.Grid` object.

## Development Setup

**One conda environment per branch/worktree, named `docs_env_sectionate_<branch-or-worktree-name>`.**
Create it from `docs/environment.yml`, overriding the baked-in `name:` with `-n`:

```bash
ENV="docs_env_sectionate_$(git rev-parse --abbrev-ref HEAD)"
conda env create -f docs/environment.yml -n "$ENV"
conda activate "$ENV"
pip install -e .    # resolves the branch's own runtime deps from pyproject.toml
python -m ipykernel install --user --name "$ENV" --display-name "$ENV"
```

`docs/environment.yml` pins none of the runtime dependencies; `pip install -e .` is
what resolves them, so install in that order and re-run it after any dependency
change in `pyproject.toml`. Reuse the env across runs on the same branch; remove it
with `conda env remove -n "$ENV"` once the branch or worktree is gone.
(`ci/environment.yml` is the minimal CI environment — pytest only, no plotting or
notebook stack. It is not sufficient for the notebooks.)

A correctly set up environment runs the full suite with **0 skips**. Skips mean one of
two things: fold and multi-tile tests `skip` on an xgcm older than the floor in
`pyproject.toml`, and the ECCO LLC90 and MOM6-fold tests `skip` when their example data
is absent from `data/`. In a fresh worktree the latter is the usual cause — `data/`
holds ~1.4 GB of downloaded input that each checkout otherwise re-fetches. Symlink it
from a checkout that already has it rather than downloading again (`data/*.nc` is
gitignored, but `MOM5_global_example_grid.nc` is tracked, so link the individual files,
not the directory):

```bash
cd data && for f in /path/to/other/checkout/data/*.nc; do
  b=$(basename "$f"); [ -e "$b" ] || ln -s "$f" "$b"
done
```

## Commands

- **Run all tests:** `pytest`
- **Run a single test:** `pytest sectionate/tests/test_section.py::test_find_closest_grid_point`
- **Build docs:** `cd docs && make html` (requires `docs/environment.yml` environment)

## Definition of Done (always, before committing or pushing)

Before committing **or pushing** any change to this repository, always do all of the
following — use parallel agents where it is faster. Both of the first two steps, every
time: not one or the other, and not only the notebooks that look related to the change.
A change that passes `pytest` but breaks a notebook is still a broken change, because
the notebooks are the rendered documentation.

Run them in this branch's own `docs_env_sectionate_<branch-or-worktree-name>` environment
(see *Development Setup*), not in whatever env happens to be active.

1. **Run the full test suite** (`pytest`) and confirm it passes with **0 skips**.
2. **Re-execute the example notebooks** in `examples/` and confirm they run cleanly:
   ```bash
   cd examples && jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=1800 \
     --ExecutePreprocessor.kernel_name=python3 *.ipynb
   ```
   Run from `examples/` — the notebooks resolve data paths relative to it. `--inplace`
   refreshes the committed outputs, which is intended; review the resulting diff.
   `kernel_name=python3` uses the active env's kernel, so the per-branch env name need
   not match the `kernelspec` recorded in each notebook.
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

- **`gridutils.py`** — Grid introspection utilities: `corner_position()` returns the shared vorticity corner position (`"outer"`, `"right"`, or `"left"`) and `corner_offset()` the corresponding velocity index shift used throughout transports/tracers; `check_outer()` is a thin wrapper (True iff `"outer"`). (Package source names positions only by these xgcm labels; their correspondence to MOM6/MITgcm/ECCO conventions is documented under "Corner staggering" below and in the example notebooks.) `coord_dict()` and `get_geo_corners()` extract coordinate/dimension names from `xgcm.Grid` metadata; `build_neighbor_maps()` builds the topology-aware neighbor maps the pathfinder consumes. For single-tile grids these come from xgcm halo padding; for every multi-tile grid they are projected from **`outer_topology(grid)`** (cached `_OuterTopology`), which reconstructs each face's full 'outer' (shared-corner) lattice, resolves every extended corner slot to the native corner storing that physical point (topological matching of the surrounding tracer cells — including a diagonal-cell recovery for normal-seam face corners and a 3-cell fingerprint for 3-tile junctions with native storage — with a small coordinate fallback only for the polar-cut corners no cell fingerprint can determine), and merges slots into physical corner *nodes* — so rotated/reversed seams, cube-vertex/tile junctions, boundary folds (e.g. LLC's south rows), and the polar grid cut are ordinary graph nodes, each resolved to its canonical native `(face, j, i)` (a physical corner may be stored once, on no face, or more than once). `_OuterTopology.padded_transports(u, v)` extends native transports to the outer lattice by reading each missing edge slot's *stored* twin through the node graph — a topology-exact, xgcm-independent alternative to `xgcm.pad(..., other_component=...)` (cell convergence built from `padded_transports` sums to exactly zero globally on ECCO LLC90, where it agrees bit-for-bit with the dict-form vector pad against the pinned xgcm). What it does not depend on is the face-connection halo being right (xgcm#712 was a general `_pad_face_connections` bug that could corrupt any face-connection pad, dict vector form included; the separate #749 fixed only the bare-`DataArray` dispatch), and it does not depend on the axis `fill_value`: edges stored on *no* face resolve to zero, a wall's true transport, rather than to whatever `fill_value` the axis declares — which is what makes it exact on grids whose `fill_value` is not 0 (on the cubed-sphere fixture, `fill_value=np.nan`, a vector pad leaves 23 of 128 cells' convergence NaN). Points stored on **no** face (a cubed-sphere stores `6N²` corners for `6N²+2` points; LLC90's 4th Arctic-cap vertex; the unstored lip of its 65°E/115°W polar grid cut under Antarctica) become edge-less nodes: sections and traced boundaries cannot pass through them and raise an error rather than return invalid indices.

- **`utils.py`** — Section catalog I/O. Sections can be loaded by name from JSON catalog files in `sectionate/catalog/`. Also provides `save_gridded_section()`/`load_gridded_section()` for persisting gridded sections.

### Key Concepts

- **Vorticity points (q-points):** Sections are defined as paths through vorticity-point indices `(i_c, j_c)`. Consecutive q-points define velocity faces (either U or V).
- **Corner staggering (three positions):** Vorticity sits at one of three xgcm positions: `"outer"` (MOM6 symmetric, M+1×N+1), `"right"` (MOM6 non-symmetric, M×N), or `"left"` (MITgcm/ECCO, incl. the lat-lon-cap, M×N). All three are native; they differ only by a per-position velocity index offset (`gridutils.corner_offset`: outer→0, right→+1, left→0). `"left"` indexes like `"outer"`; it differs only in array length and in that the *high* corner row/column is absent (so a section exactly on the north/east domain wall clips one row inside).
- **Sign conventions:** For a *closed* section, `convergent_transport()` determines orientation (clockwise/counterclockwise) using stereographic projection and signed polygon area, then applies sign corrections so positive transport means "inward" (toward the enclosed polygon). For an *open* section there is no enclosing polygon, so `positive_in` is undefined; it instead uses the **left-of-transect** convention — `positive_in=True` makes positive transport point to the left of the section as traversed from the first to the last waypoint — and emits a `UserWarning`. `is_section_counterclockwise()` is only consulted in the closed case.
- **xgcm.Grid dependency:** The package relies heavily on `xgcm.Grid` for grid metadata (axis boundaries, coordinate positions, dataset access via `grid._ds`).

## Version

**The git tag is the version.** There is no version string checked into the tree:
`hatch-vcs` derives it from the tag at build time and writes the gitignored
`sectionate/_version.py`, which ships inside the sdist and wheel. `sectionate/version.py`
is only a shim over that generated file — never add a version literal back to it. A
`0.0.0+unknown` from it means the package was imported without being built or installed,
not that a number is missing. Any checkout that installs the package needs its tags: a
shallow clone resolves a `.devN` version instead of the release line, which is why CI and
Read the Docs both fetch with full depth. See the Releasing section of `README.md`.
