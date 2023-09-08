# sectionate
compute section in ocean models

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raphaeldussin/sectionate/master)

Quick Start Guide
-----------------

**Installing from scratch using `conda`**
```bash
git clone git@github.com:hdrake/sectionate.git
cd sectionate
conda env create -f ci/sectionate.yml
conda activate sectionate
pip install -e .
python -m ipykernel install --user --name sectionate --display-name "sectionate"
jupyter-lab
```

**Minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/sectionate.git@master
```
