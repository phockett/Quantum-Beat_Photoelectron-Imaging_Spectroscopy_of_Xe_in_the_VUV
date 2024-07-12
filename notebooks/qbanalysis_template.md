---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": ["remove-cell"]}

Notes (hidden cell)

+++

(page:PAGENAME)=
# TITLE

---

From prior work and data:

- Forbes, R. et al. (2018) ‘Quantum-beat photoelectron-imaging spectroscopy of Xe in the VUV’, Physical Review A, 97(6), p. 063417. Available at: https://doi.org/10.1103/PhysRevA.97.063417. arXiv: http://arxiv.org/abs/1803.01081, Authorea (original HTML version): https://doi.org/10.22541/au.156045380.07795038
- Data (OSF): https://osf.io/ds8mk/
- [Quantum Metrology with Photoelectrons (Github repo)](https://github.com/phockett/Quantum-Metrology-with-Photoelectrons), particularly the [Alignment 3 notebook](https://github.com/phockett/Quantum-Metrology-with-Photoelectrons/blob/master/Alignment/Alignment-3.ipynb). Functions from this notebook have been incorporated in the current project, under `qbanalysis.hyperfine`.

+++

### Imports

```{code-cell} ipython3
# Load packages
# Main functions used herein from qbanalysis.hyperfine
from qbanalysis.hyperfine import *
import numpy as np
from epsproc.sphCalc import setBLMs

from pathlib import Path

dataPath = Path('/tmp/xe_analysis')
# dataTypes = ['BLMall', 'BLMerr', 'BLMerrCycle']   # Read these types, should just do dir scan here.

# # Read from HDF5/NetCDF files
# # TO FIX: this should be identical to loadFinalDataset(dataPath), but gives slightly different plots - possibly complex/real/abs confusion?
# dataDict = {}
# for item in dataTypes:
#     dataDict[item] = IO.readXarray(fileName=f'Xe_dataset_{item}.nc', filePath=dataPath.as_posix()).real
#     dataDict[item].name = item

# Read from raw data files
from qbanalysis.dataset import loadFinalDataset
dataDict = loadFinalDataset(dataPath)

# Use Pandas and load Xe local data (ODS)
# These values were detemermined from the experimental data as detailed in ref. [4].
from qbanalysis.dataset import loadXeProps
xeProps = loadXeProps()
```

```{code-cell} ipython3
# v2 pkg
from qbanalysis.adv_fitting import * 
```

```{code-cell} ipython3
:tags: [hide-cell]

# Hide future warnings from Xarray concat for fitting on some platforms
import warnings
# warnings.filterwarnings('ignore')  # ALL WARNINGS
# warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
```

## Versions

```{code-cell} ipython3
import scooby
scooby.Report(additional=['qbanalysis','pemtk','epsproc', 'holoviews', 'hvplot', 'xarray', 'matplotlib', 'bokeh'])
```

```{code-cell} ipython3
# # Check current Git commit for local ePSproc version
# from pathlib import Path
# !git -C {Path(qbanalysis.__file__).parent} branch
# !git -C {Path(qbanalysis.__file__).parent} log --format="%H" -n 1
```

```{code-cell} ipython3
# # Check current remote commits
# !git ls-remote --heads https://github.com/phockett/qbanalysis
```

```{code-cell} ipython3
# Check current Git commit for local code version
import qbanalysis
!git -C {Path(qbanalysis.__file__).parent} branch
!git -C {Path(qbanalysis.__file__).parent} log --format="%H" -n 1
```

```{code-cell} ipython3
# Check current remote commits
!git ls-remote --heads https://github.com/phockett/Quantum-Beat_Photoelectron-Imaging_Spectroscopy_of_Xe_in_the_VUV
```
