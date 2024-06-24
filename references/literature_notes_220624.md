---
jupytext:
  formats: ipynb,md:myst
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

Added for ongoing notes and also bib testing.

For configuration see:

- https://jupyterbook.org/en/stable/tutorials/references.html#tutorials-references
- https://jupyterbook.org/en/stable/content/citations.html
- https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#roles-and-directives
- https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#referencing-style

Note styles configured in `_config.yml`, and also in {bibliography} directive, which can include filters and stuff too, e.g. https://jupyterbook.org/en/stable/content/citations.html#local-bibliographies

+++

# Literature notes
22/06/24

+++

{cite:ps}`gilb2006MappingTimedependentElectron`

- PADs and QB FT analysis.
    
- Basic wavepacket + ionization model, MQDT style.
    
- Brief discussion of entanglement and Bell states.
    
- Should relate to Blum stuff for more…?


+++

{cite:ps}`Greene1982`

- Universal alignment function, defines sign of alignment (with $\Delta J$), see eq. 16.

- Hyperfine beats (eq. 22), including depolarisation, and time-integrated case (eq. 24).

```{code-cell} ipython3
def A0(ji):
    """
    Define "universal alignment function" per Greene & Zare 1982.
        
    Greene, Chris H., and Richard N Zare. 1982. 
    “Photonization-Produced Alignment of Cd.” 
    Physical Review A 25 (4): 2031–37. 
    https://doi.org/10.1103/PhysRevA.25.2031.

    """
    
    # v1 - values DON'T match Fig 1 in manuscript?
    # return {1:-2/5 + 3/(5*(ji+1)),
    #         -1:-2/5 + 3/(5*ji),
    #         0:4/5 + 3/(5*ji*(ji+1))}
    
    # v2 - looks OK aside from J=0 terms, must be fixed in manuscript?
    #      Fig 1 shows 1, 0 terms = 0
    # MUST be incorrect for +1 case, since = -2/5+3/5 below, but other terms seem correct.
    return {1:-2/5 + 3/(5*(ji+1)),
        -1:-2/5 - 3/(5*ji),
        0:4/5 - 3/(5*ji*(ji+1))}

A0(1)
```

```{code-cell} ipython3
import pandas as pd

pd.DataFrame.from_dict({k:[v] for k,v in A0(1).items()})
```

```{code-cell} ipython3
import numpy as np
pd.DataFrame.from_dict(A0(np.arange(0,10)))
```

```{code-cell} ipython3
import hvplot.pandas 

A0df = pd.DataFrame.from_dict(A0(np.arange(0,10)))

# A0df = A0df.replace([np.inf, -np.inf], 0)  # Replace inf with 0?
A0df = A0df.replace([np.inf, -np.inf], np.nan)  # Replace inf with nan?
A0df.hvplot()
```

```{bibliography}
```

```{code-cell} ipython3

```
