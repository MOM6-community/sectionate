# sectionate
A package to sample grid-consistent sections from ocean model outputs

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raphaeldussin/sectionate/master)

Quick Start Guide
-----------------

**For users: minimal installation within an existing environment**
```bash
pip install git+https://github.com/MOM6-community/sectionate.git@master
```

**For developers: installing from scratch using `conda`**
```bash
git clone https://github.com/MOM6-community/sectionate.git
cd sectionate
conda env create -f docs/environment.yml
conda activate docs_env_sectionate
pip install -e .
python -m ipykernel install --user --name docs_env_sectionate --display-name "docs_env_sectionate"
jupyter-lab
```
