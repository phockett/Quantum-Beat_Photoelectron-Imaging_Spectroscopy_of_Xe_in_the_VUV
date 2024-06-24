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

- Universal alignment function, defines sign of alignment (with $\Delta J$), see eq. 16. (Implemented in `qbanalysis.photoionization.A0`.)

$$
A_{0}^{\mathrm{col}}(j_{i};t)=\begin{cases}
-\frac{2}{5}+\frac{3}{5(j_{i}+1)} & t_{+1}=j_{i}+1\\
-\frac{2}{5}+\frac{3}{5j_{i}} & t_{-1}=j_{i}-1\\
\frac{4}{5}+\frac{3}{5j_{i}(j_{i}+1)} & t_{0}=j_{i}
\end{cases}
$$ (A0_alignment)

Where $t_{\pm1}$ are parity favoured, $t_0$ is parity unfavoured, and $t$ denotes angular momentum transfer $j_t = \Delta j$ (not "time"!).

- Hyperfine beats (eq. 22), including depolarisation, and time-integrated case (eq. 24).


```{code-cell} ipython3
# Calculate and plot Universal Alignment Function, per 
#     Greene, Chris H., and Richard N Zare. 1982. 
#     “Photonization-Produced Alignment of Cd.” 
#     Physical Review A 25 (4): 2031–37. 
#     https://doi.org/10.1103/PhysRevA.25.2031.

from qbanalysis.photoionization import *
import numpy as np

A0table = A0df(np.arange(0,10))
A0table
```

```{code-cell} ipython3
import hvplot.pandas 
import holoviews as hv

# Quick plot from PD data
A0plot = A0table.hvplot().opts(title="Universal alignment function vs. Ji, lines per dJ",show_grid=True)

# Load ref figure (image) and create layout
height = 600
width = 600
ref = hv.RGB.load_image('greene_zare_1982_fig1.png').opts(height=height,  width=width)
layout = A0plot.opts(height=height, width=width) + ref.opts(axiswise=True, xaxis=None, yaxis=None)
layout
```

Note discrepancy for Ji=0 case (seems to be an error or specific choice that inf == 0, in original plot), otherwise looks good.

+++

```{bibliography}
```
