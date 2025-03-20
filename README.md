# sectionate
A package to sample grid-consistent sections from ocean model outputs

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raphaeldussin/sectionate/master)

Quick Start Guide
-----------------

**For users: minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/sectionate.git@master
```

**For developers: installing from scratch using `conda`**
```bash
git clone git@github.com:hdrake/sectionate.git
cd sectionate
conda env create -f ci/environment.yml
conda activate test_env_sectionate
pip install -e .
python -m ipykernel install --user --name test_env_sectionate --display-name "test_env_sectionate"
jupyter-lab
```
