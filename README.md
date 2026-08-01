# sectionate
A package to sample grid-consistent sections from ocean model outputs

(Currently only supports structured grids.)

[![PyPI](https://badge.fury.io/py/sectionate.svg)](https://badge.fury.io/py/sectionate)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sectionate)](https://anaconda.org/conda-forge/sectionate)
[![Docs](https://readthedocs.org/projects/sectionate/badge/?version=latest)](https://sectionate.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/MOM6-Community/sectionate)](https://github.com/MOM6-Community/sectionate)

Quick Start Guide
-----------------

**For users: minimal installation within an existing environment**
```bash
conda install -c conda-forge sectionate
```

**For developers: installing dependencies from scratch using `conda`**
```bash
git clone https://github.com/MOM6-community/sectionate.git
cd sectionate
conda env create -f docs/environment.yml
conda activate docs_env_sectionate
pip install -e .
python -m ipykernel install --user --name docs_env_sectionate --display-name "docs_env_sectionate"
jupyter-lab
```

Releasing
---------

**The git tag is the version.** `sectionate` has no version string checked into
the source tree: `hatch-vcs` derives it from the tag at build time and writes
`sectionate/_version.py` (gitignored, but shipped inside the sdist and wheel).
To cut a release you tag; there is no file to bump and nothing to keep in sync.

1. Make sure `master` is green and has everything you want in the release.
2. **Publish a GitHub Release** whose tag is `vX.Y.Z`, targeting the commit you
   want to ship:
   ```bash
   gh release create vX.Y.Z --target "$(git rev-parse origin/master)" \
     --title vX.Y.Z --generate-notes
   ```
   Publishing it (not merely pushing a tag) is what fires the workflow. Target
   the commit you actually want: a tag placed *before* the commit you meant to
   release builds the previous version, and PyPI rejects it as a duplicate.
3. The **Publish to PyPI** workflow builds from that tag and uploads. It checks
   out with `fetch-depth: 0` so the tag is visible to `hatch-vcs`, and asserts
   that the built version matches the tag before publishing.
4. Verify: <https://pypi.org/project/sectionate/>.
5. conda-forge builds from the PyPI sdist and lags by design; the autotick bot
   opens the version-bump PR. Its recipe must list `hatch-vcs` alongside
   `hatchling` in `host`, since it builds with `--no-build-isolation`.

Two things follow from the tag being the version. Never add a version literal
back to the tree — `sectionate/version.py` is a shim over the generated file, and
a `0.0.0+unknown` from it means the package was imported without being built or
installed, not that a number is missing. And any checkout that installs the
package needs its tags: a shallow clone, or a fork that never fetched them,
resolves a `.devN` version instead of the release line. That is why CI checks out
with `fetch-depth: 0` and Read the Docs unshallows in `post_checkout`.
