# sectionate
A package to sample grid-consistent sections from ocean model outputs

[![PyPI](https://badge.fury.io/py/sectionate.svg)](https://badge.fury.io/py/sectionate)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sectionate)](https://anaconda.org/conda-forge/sectionate)
[![Docs](https://readthedocs.org/projects/sectionate/badge/?version=latest)](https://sectionate.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/MOM6-Community/sectionate)](https://github.com/MOM6-Community/sectionate)

Quick Start Guide
-----------------

**For users: minimal installation within an existing environment**
```bash
conda install sectionate
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
