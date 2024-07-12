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

## Mapping of time-dependent electron orbital alignment
{cite:ps}`gilb2006MappingTimedependentElectron`

- PADs and QB FT analysis.
    
- Basic wavepacket + ionization model, MQDT style.
   - E.g. Eq. 2 for the wavepacket: $|\Psi(t)〉=C_1|5d′[5/2]_3,M = 0〉 + C_2 e^{−i\Deltaωt} |8d[1/2]_1,M = 0〉$, "where C_{1,2} are complex amplitudes determined by the excitation step. $\Deltaω$ corresponds to the energy difference of 53 cm−1 between the states."
   - Photoionization to two channels: "The subsequent photoionization probe step leaves the ion in either of the 2P1/2 or 2P3/2 spin–orbit states, which are energetically separated by 5370 cm−1, allowing independent detection of photoelectrons produced in these two ionization channels by their different respective kinetic energies."
   - For MQDT discussion, see Sect. 3 therein, also Ref. 9 therein. Note similar treatment in {cite:ps}`Sato2016`, and mixing parameters are FITTED in that case.
    
    
- Brief discussion of entanglement and Bell states.
    
- Should relate to Blum stuff for more…?

+++

## Photonization-produced alignment of Cd
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

- Case with hyperfine levels (eq. 21)

$$
A_{0}^{\mathrm{col}}(j_{i};t)=A_{0}^{\mathrm{col}}(j_{i};0)g^{(2)}(t)
$$


- Hyperfine beats (eq. 22), including depolarisation, from Fano & Macek (cf. Blum version used herein):

$$
g^{(2)}(t)=\sum_{F',F}\frac{(2F'+1)(2F+1)}{2I+1}\left\{ \begin{array}{ccc}
F & F' & 2\\
j_i & j_i & I
\end{array}\right\} ^{2}\cos\left[\omega_{F',F}t\right]
$$

and time-integrated case (eq. 24), where $t$ is integrated over, and $\tau$ is the state lifetime (for fluorescence). 

$$
\bar{g}^{(2)}(t)=\sum_{F',F}\frac{(2F'+1)(2F+1)}{2I+1}\left\{ \begin{array}{ccc}
F & F' & 2\\
j_i & j_i & I
\end{array}\right\} ^{2}\frac{1}{1+\omega^2_{F',F}\tau^2}
$$

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

## Resonances in Photoelectron Angular Distributions
{cite:p}`Dill1973`

- Treatment in angular momentum transfer $j_t$ framework + MQDT.
- Xe 5p ionization and autoionization.

+++

## Angular distributions and quantum beats of photoelectrons from resonant two-photon ionization of lithium
{cite:p}`Chien1983`

HF coupling case with 9js from Chien, Ring-ling, Oliver Mullins, and R. Berry. 1983. “Angular Distributions and Quantum Beats of Photoelectrons from Resonant Two-Photon Ionization of Lithium.” Physical Review A 28 (4): 2078–84. https://doi.org/10.1103/PhysRevA.28.2078.

Liouville amplitude reduced transition matrix (eqn. 9 in {cite:p}`Chien1983`):

\begin{eqnarray}
\langle(l_{2}l_{2}')L_{2},(s_{2}s_{2}')0,(i_{2}i_{2}')0;L_{2}||\mathscr{S}||(i_{a}i_{a})0,(j_{a}j_{a})0,(11)P_{1},(11)P_{2};J\rangle_{L} & = & [P_{2},l_{2},l_{2}']^{1/2}[J_{1}]^{3/2}[j_{1}]^{2}\left(\begin{array}{ccc}
s & s & 0\\
l_{1} & l_{1} & J_{1}\\
j_{1} & j_{1} & J_{1}
\end{array}\right)\left(\begin{array}{ccc}
l_{1} & 1 & l_{2}\\
l_{1} & 1 & l_{2}'\\
J_{1} & P_{2} & L_{2}
\end{array}\right)\left(\begin{array}{ccc}
j_{a} & 1 & j_{1}\\
j_{a} & 1 & j_{1}\\
0 & P_{1} & J_{1}
\end{array}\right)\nonumber \\
 & \times & W(J_{1},t)\langle j_{1}||r_{1}||j_{a},1;j_{1}\rangle\langle j_{a},1;j_{2}||r_{2}^{\dagger}||j_{1}\rangle\langle l_{2}||r_{2}||l_{2},1;l_{2}\rangle\langle l_{1},1;l_{2}'||r_{2}^{\dagger}||l_{2}\rangle
\end{eqnarray}

NOTE: this assumes single uncoupled outgoing electron.

+++

### Application to $Xe(5p^{-1}6s^{1})\rightarrow Xe(5p^{-1}) + \epsilon p$

In this case we set the following:

- $l_1$ corresponds to the $6s$ ionizing electon, $l_1 = 0$
- $l_2$ corresponds to the free electron, $l_2 = l_1\pm1 = 1$
- $J_1 = l_1\pm S$ for the $6s$ electron, $J_1 = 0.5$ (NOTE this is distinct from $J=1$ for the total excited state...?)
    - In this case get only zeros...?  But for $J_1=0$ to get values, or for $J_1=1$ if $l_1,l_2$ include spin.
    - More to consider here...?
    - Note original derivation has $\delta_{J_1,F_1}$, so that may be the issue if not setting for F states...?
    - $F_1=1/2,3/2,5/2$ in practice.
    - **HOWEVER... note current 1e excited state component $nl^{2s+1}[K]_{J_e}=6s^{2}[1/2]_1$, where $K=J+l=1/2$, $J_e=K+s=1$ - may want to couple to $J_e$ rather than $l$?** In this case, $l_{2,max}=J_{2e,max}=2$, and $L_{max}=4$.
- For the ion, $L=1$ for the unpaired $5p$ electron, $J^+ = L\pm S$, hence the overall ionic term is $^{2s+1}L_{J}=^{2}P_{1/2,3/2}$. 
    - F states...? $F^+=J^+\pm I$
    - $F^+=0,1$ for $I=1/2, J^+=1/2$ (and 1/2 int steps?, 0.5)
    - $F^+=1,2$ for $I=3/2, J^+=1/2$ (and 1/2 int steps?, 1.5)
    - $F^+=1,2$ for $I=1/2, J^+=3/2$ (and 1/2 int steps?, 1.5)
    - $F^+=0,1.5,3$ for $I=3/2, J^+=3/2$ (in 3/2 steps)
    
Q: state-to-state (F) couplings required here...? Need to consider Hund's case/coupling schemes.

A: **see thesis (Eqn. 2.46)**, also recent example in {cite:p}`alarcon2023QuantumBeatsTwocolor`, Alarcón, M. A., A. Plunkett, J. K. Wood, D. Biswas, C. H. Greene, and A. Sandhu. 2023. “Quantum Beats in Two-Color Photoionization to the Spin-Orbit Split Continuum of Ar.” Physical Review A 108 (3): 033107. https://doi.org/10.1103/PhysRevA.108.033107. **Should couple in dJ, and allow ang mom transfer.** {cite:p}`Chien1983` is pure uncoupled case, i.e. e+photon > continuum only.

A: Also could couple $T_K$ here directly...? In that case $K=0,2$ and $l_1=K\pm J = 0,1,2,3$...?

```{code-cell} ipython3
# Test 9j couplings...
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j

# l1=0.5
# l2=1.5

# l1=1
# l2=2

# l1=0
# l2=1

# J1=0  # Allowed terms P=0,1,2, L=P
# J1=1.5   # Error on triangle conditions, but might be half-int issue? UPDATE: seems to be real, not numerical, issue.
         # SEE https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.wigner_9j
         # Set to force np.nan in these cases now.
# J1=1  # Only zeros, unless spin INCLUDED in l1,l2?

nineJ = []
for l1 in np.arange(0,4):
    for l2 in np.arange(l1-1,l1+2):
        for J1 in np.arange(0,4,0.5):  # 1/2 int terms maybe non-zero if l is 1/2 int.
            for P2 in np.arange(0,5,1):
                for L2 in np.arange(0,5,1):
                    # print(f'P2: {P2}, L2: {L2}')
                    # print(wigner_9j(l1,1,l2,l1,1,l2,J1,P2,L2))
                    try:
                        value = wigner_9j(l1,1,l2,l1,1,l2,J1,P2,L2, prec=8)  # Include prec=X to force numeric
                    except ValueError:
                        value = np.nan  # For invalid cases set np.nan

                    nineJ.append([l1,l2,J1,P2,L2,value])
                      
nineJDF = pd.DataFrame(nineJ,columns=['l1','l2','J1','P2','L2','9j'])

# Show non-zero terms only
nineJDF[np.abs(nineJDF['9j'])>1e-8]

# Show all terms
# nineJDF
```

```{code-cell} ipython3
# Quick test 1/2 int values OK...
# See https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.wigner_9j
wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=32)
```

## Photoionization from the Xe 4d orbitals of XeF2
{cite:p}`forbes2021PhotoionizationXe4d`

- Theory + spin-orbit effects.
- For SO code dev, see https://phockett.github.io/ePSdata/XeF2-preliminary/XeF2_ePS-expt_comp_271020_4d_v111120-dist.html
- For final results, see https://phockett.github.io/ePSdata/XeF2-preliminary/xe-xef2_plots-notes_220421.html

+++

## Quantum Beats in Two-Color Photoionization to the Spin-Orbit Split Continuum of Ar
{cite:p}`alarcon2023QuantumBeatsTwocolor`, Alarcón, M. A., A. Plunkett, J. K. Wood, D. Biswas, C. H. Greene, and A. Sandhu. 2023.
Physical Review A 108 (3): 033107. https://doi.org/10.1103/PhysRevA.108.033107.

- XUV + 2IR pump-probe.
- J-core coupled treatment of ionization (for SO case).
- MQDT + time-dependence model

“We choose the $J_{cs}$ coupling scheme, which couples the total angular momentum of the ion ($j_c$) with the spin of the electron to form the $J_{cs}$ quantum number, which then couples to the orbital angular momentum of the photoelectron to form the total angular momentum $J$. This simplifies the analysis of the photoelectron signal as it represents an incoherent sum over the $J_{cs}$ values of 0, 1, or 2, constrained by the dipole selection rules.” ([Alarcón et al., 2023, p. 4](zotero://select/library/items/8W8IIUX3)) ([pdf](zotero://open-pdf/library/items/CAUVVMJ8?page=4&annotation=PXC7CLMQ))

Application to current case:

- Excited state $J_e=1^o$
    - Excited state + photon > $J^\pi = 0^e,2^e$ symmetries.
    - Implies continuum $l=J=even$...? Or should also be coupled to $K$ and/or $\Delta J$?
    - Note current 1e excited state component $nl^{2s+1}[K]_{J_e}=6s^{2}[1/2]_1^o$, where $K=J+l=1/2$, $J_e=K+s=1$ - may want to couple to $J_e$ rather than $l$? In this case, $l_{2,max}=J_{2e,max}=2$, and $L_{max}=4$.
    - ALSO electronic wavepacket $T_K$, this component should purely be coupled to the outgoing e-, as distinct from the molecular axis alignment case in which we assume it is DECOUPLED.
- Ion $J_c=J^+=1/2,3/2$

Q: still unsure if $L=4$ from couplings as here, or *differential* for J+ states...?

+++

## TO TRY

- Rework normal AF ang mom coupling with $\Delta J/N$ terms included? Simply modulating by these may work, although will break current AF derivation which assumes decoupled.
- Use full, old-style, ang mom coupling scheme instead? This should be more appropriate for the current case, or general use where rotational coupling is allowed. Way to do this with $T_K$/wavepacket...? Ah, should just look similar to Eq. 2.49 in thesis, but with $T_K$ instead of $\rho_{M,M}$ (or can use time-dependent density matrix). Actually, just see [notes here](https://phockett.github.io/Quantum-Beat_Photoelectron-Imaging_Spectroscopy_of_Xe_in_the_VUV/4.01_hyperfine_beats_modelling_060624.html#equation-45d6ceb6-27b3-4c9d-9d4c-7b498c2b2183), think this is the same as old NO case. See also Eq. 8.1 for rotational spectator case, which should follow usual "atomic" rules.

+++

Old style, per thesis and QM1 Sect 3.1:

$$
\begin{eqnarray}
C(lm\lambda N_{t}M_{i}\mu_{\lambda}) & = & (2N_{t}+1)(-1)^{M_{+}+q}\left(\begin{array}{ccc}
N_{t} & 1 & l\\
M_{t} & p & m
\end{array}\right)\left(\begin{array}{ccc}
N_{+} & N_{i} & N_{t}\\
-M_{+} & M_{i} & M_{t}
\end{array}\right)\nonumber \\
 & \mathsf{x} & \left(\begin{array}{ccc}
N_{+} & N_{i} & N_{t}\\
-K_{+} & K_{i} & K_{t}
\end{array}\right)\left(\begin{array}{ccc}
N_{t} & 1 & l\\
-K_{t} & q & -\lambda
\end{array}\right)\nonumber \\
 & \mathsf{x} & \left(\begin{array}{ccc}
N_{+} & J_{+} & S_{+}\\
M_{+} & M_{J+} & M_{S+}
\end{array}\right)\left(\begin{array}{ccc}
N_{+} & J_{+} & S_{+}\\
K_{+} & P_{+} & \Sigma_{+}
\end{array}\right)\label{eq:geom-params-C}
\end{eqnarray}
$$

$$
\begin{eqnarray}
\gamma_{\alpha\alpha_{+}l\lambda ml'\lambda'm'} & = & (2N_{i}+1)(2N_{+}+1)(-i)^{l'-l}\sum_{M_{+}}\sum_{M_{i}M_{i}'}\sum_{N_{t}N_{t}'}\sum_{\mu_{\lambda}\mu_{\lambda}'}{}^{J_{i}K_{i}}\boldsymbol{\rho}_{M_{i}M_{i}'}\nonumber \\
 & \mathsf{x} & C(lm\lambda N_{t}M_{i}q)C(l'm'\lambda'N_{t}'M_{i}'q')\label{eq:gamma-state}
\end{eqnarray}
$$

+++

- Try normal case + differential isotope channels, should allow for high L in observables even if $l_{max}=1$, but may imply coherence between isotopes...? This will also surely affect the temporal traces? Unless incoherent addition also works here (maybe... just needs to look like L=4!).
- WAIT... QM1, sect. 3.1 and esp. 3.1.4.2, which is "full vibronic" case per Jon's derivations.
    - Should be able to adapt this... write den. mat./TKQ as TJtKt, and add 6j + 9j couplings to ion.
    - Oh, wait - has TKQ for excited state already, but need to add for ion state (or could, at least).
    - Quick 9j tester below...

+++

$$
\begin{eqnarray}
\beta_{L,M}(\epsilon,\,t) & = & -[L]^{\frac{1}{2}}\sum_{P,R}(-1)^{P}[P]^{\frac{1}{2}}E_{PR}(\hat{e})\sum_{K,Q}(-1)^{K+Q}\left(\begin{array}{ccc}
P & K & L\\
R & -Q & -M
\end{array}\right)\nonumber \\
 & \times & \sum_{n_{\alpha},n'_{\alpha'}}(-1)^{J_{\alpha}}[J_{\alpha},J'_{\alpha'}]^{\frac{1}{2}}\langle T((n_{\alpha},n'_{\alpha'};\,t)_{KQ}^{\dagger}\rangle\sum_{l,l'}(-1)^{l}[l,l']^{\frac{1}{2}}\left(\begin{array}{ccc}
l & l' & L\\
0 & 0 & 0
\end{array}\right)\nonumber \\
 & \times & \sum_{j_{t},j'_{t}}(-1)^{j_{t}}[j_{t},j'_{t}]\left\{ \begin{array}{ccc}
1 & 1 & P\\
j_{t} & j'_{t} & K\\
l & l' & L
\end{array}\right\} \sum_{q,q'}\sum_{\lambda,\lambda'}\sum_{k_{t},k'_{t}}(-1)^{q+q'}\left(\begin{array}{ccc}
l & 1 & j_{t}\\
\lambda & -q & k_{t}
\end{array}\right)\left(\begin{array}{ccc}
l' & 1 & j'_{t}\\
\lambda' & -q' & k'_{t}
\end{array}\right)\nonumber \\
 & \times & \sum_{J_{\alpha+},\tau_{\alpha+}}(-1)^{J_{\alpha+}}[J_{\alpha+}]\left\{ \begin{array}{ccc}
J_{\alpha} & j_{t} & J_{\alpha+}\\
j'_{t} & J'_{\alpha'} & K
\end{array}\right\} \sum_{K_{\alpha+}}|a_{K_{\alpha+}}^{J_{\alpha+}\tau_{\alpha+}}|^{2}\sum_{K_{\alpha},K'_{\alpha'}}a_{K_{\alpha}}^{J_{\alpha}\tau_{\alpha}}a_{K'_{\alpha'}}^{J'_{\alpha'}\tau'_{\alpha'}}\nonumber \\
 & \times & \left(\begin{array}{ccc}
J_{\alpha+} & J_{\alpha} & j_{t}\\
-K_{\alpha+} & K_{\alpha} & k_{t}
\end{array}\right)\left(\begin{array}{ccc}
J_{\alpha+} & J'_{\alpha'} & j'_{t}\\
-K_{\alpha+} & K'_{\alpha'} & k'_{t}
\end{array}\right)\sum_{\nu_{\alpha+},\alpha_{+}}\varepsilon(n_{\alpha+},n{}_{\alpha},\epsilon)\varepsilon^{*}(n_{\alpha+},n'{}_{\alpha'},\epsilon)\nonumber \\
 & \times & \sum_{\Gamma,\Gamma'}\sum_{\mu,\mu'}\sum_{h,h'}b_{hl\lambda}^{\Gamma\mu*}b_{h'l'\lambda'}^{\Gamma'\mu'}(-i)^{l-l'}e^{i(\sigma_{l}(\epsilon)-\sigma_{l'}(\epsilon))}\boldsymbol{D}_{\Gamma\mu hl}^{\alpha\nu_{\alpha}\alpha_{+}\nu_{\alpha+}}(q)\boldsymbol{D}_{\Gamma'\mu'h'l'}^{\alpha'\nu'_{\alpha'}\alpha_{+}\nu_{\alpha+}*}(q')\label{eq:AF-PAD-t-Asym-full-BO-case}
\end{eqnarray}
$$

```{code-cell} ipython3
# Test 9j couplings per QM1, Eqn. 3.32
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j

# l1=0.5
# l2=1.5

# l1=1
# l2=2

# l1=0
# l2=1

# J1=0  # Allowed terms P=0,1,2, L=P
# J1=1.5   # Error on triangle conditions, but might be half-int issue? UPDATE: seems to be real, not numerical, issue.
         # SEE https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.wigner_9j
         # Set to force np.nan in these cases now.
# J1=1  # Only zeros, unless spin INCLUDED in l1,l2?

nineJ = []
for l in np.arange(0,4):
    for lp in np.arange(l1-1,l1+2):
        for j in np.arange(0,4,0.5):  # 1/2 int terms maybe non-zero if l is 1/2 int.
            for jp in np.arange(0,4,0.5):  # 1/2 int terms maybe non-zero if l is 1/2 int.
                for P in np.arange(0,5,1):
                    for K in np.arange(0,5,1):
                        for L in np.arange(0,5,1):
                            # print(f'P2: {P2}, L2: {L2}')
                            # print(wigner_9j(l1,1,l2,l1,1,l2,J1,P2,L2))
                            try:
                                value = wigner_9j(1,1,P,j,jp,K,l,lp,L, prec=8)  # Include prec=X to force numeric
                            except ValueError:
                                value = np.nan  # For invalid cases set np.nan

                            nineJ.append([l,lp,j,jp,P,K,L,value])
                      
nineJDF = pd.DataFrame(nineJ,columns=['l','lp','j','jp','P','K','L','9j'])

# Show non-zero terms only
nineJDF[np.abs(nineJDF['9j'])>1e-8]

# Show all terms
# nineJDF
```

```{code-cell} ipython3
nineJDF[np.abs(nineJDF['9j'])>1e-8].drop_duplicates(ignore_index =True)
```

```{bibliography}
```
