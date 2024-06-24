"""
Photoionization functions for Quantum-beat photoelectron-imaging spectroscopy of Xe in the VUV

24/06/24

"""

import pandas as pd
import numpy as np


def A0(ji):
    """
    Define "universal alignment function" per Greene & Zare 1982.
        
    Greene, Chris H., and Richard N Zare. 1982. 
    “Photonization-Produced Alignment of Cd.” 
    Physical Review A 25 (4): 2031–37. 
    https://doi.org/10.1103/PhysRevA.25.2031.

    ji : single or array for calculation.
    
    Returns dictionary with items dJ = +1,0,-1

    """
    # NOTE: looks OK aside from J=0 terms, must be fixed in manuscript?
    #      Fig 1 shows 1, 0 terms = 0
    # MUST be incorrect for +1 case, since = -2/5+3/5 below, but other terms seem correct.
    return {1:-2/5 + 3/(5*(ji+1)),
        -1:-2/5 - 3/(5*ji),
        0:4/5 - 3/(5*ji*(ji+1))}


def A0df(ji):
    """
    Wrap A0 to PD Dataframe.
    
    Example
    
    >>> A0table = A0df(np.arange(0,10))
    >>> A0table.hvplot().opts(title="Universal alignment function vs. Ji, lines per dJ")
    
    """
    
    A0dfout = pd.DataFrame.from_dict(A0(np.arange(0,10)))
    A0dfout = A0dfout.replace([np.inf, -np.inf], np.nan)  # Replace inf with nan?
    
    return A0dfout