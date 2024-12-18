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



#***********  Compute photoionization full model


def blmCalc(calcDict, matE = None, isoKeys = None, JFlist = None,
            thres=1e-4):
    """
    Compute photoionziation ($\beta_{L,M}$ parameters) for state-selected case including spin.
    
    Application to Xe: compute ionization for isomers with given matrix elements, from electronic wavepacket model.
    
    Parameters
    ----------
    calcDict : dictionary
        Results from wavepacket model, as output by :py:func:`qbanalysis.adv_fitting.calcAdvFitModel()`.
        Also needs to have density matrices set, per :py:func:`qbanalysis.hyperfine.pmmFromQuantumBeat()`.
    
    matE : [Format notes to follow]
        Ionization matrix elements in Pandas format.
        
    isoKeys : list, optional
        Keys to use from calcDict.
        If None, use default case: isoKeys = ['129Xe','131Xe']
        
    JFlist : list, optional
        Final J states to loop over in calculations.
        If None, use default case: JFlist = [0.5,1.5]
        
    thres : float, optional, default = 1e-4
        Threshold used for gammaCalc routine.
        
        
    Returns
    -------
    betaJ : dict
        Contains results per state, indexed as betaJ[isoKey][J]
        
    """
    
    # Set default keys
    if isoKeys is None:
        isoKeys = ['129Xe','131Xe']
        
    # Set default Jf
    if JFlist is None:
        JFlist = [0.5,1.5]
    
    # from epsproc.geomFunc.gamma import gammaCalc
    # pd.set_option('display.max_rows', 1000)
    # %load_ext autoreload
    # %autoreload 2

    # from epsproc.geomFunc.gamma import gammaCalc

    # Set matE (from above/below)
    # matEXe = matEInputCleanM0 # matEinput, matEinputM0, matEInputClean, matEInputCleanM0
    betaJ = {}

    # Loop over states
    for isoKey in isoKeys:
        betaJ[isoKey] = {}
        
        # TODO: need to update Jf lists 
        # isoKey = '131Xe'
        Jilist = calcDict['modelDict'][isoKey].attrs['states']['JFlist']
        # Jf = JFlist[0][1]   # Final state J
        # JFlist = [0.5,1.5]  # Set manually for multiple Jf case!

        # Loop over keys and calculate...
        for Jf in JFlist:

            ## Spin weighting per Jf
            # Jf = 0.5

            # Compute spin weightings for given Jf case (==Jc in function).
            spinDict = gammaCalc.spinWeightings(selectors={'Jc':Jf})


            # Subselect and sum
            Ji = int(Jilist[0][0]) # Initial state J
            # Set terms for legacy code
            J = Ji
            Jp = Ji

            # pmmSub = pmmPkgDS[isoKey].sel({'J':J,'Jp':Jp})  #.sum(['K','Q'])
            # Version from calcDict - may want to check and call pmmFromQuantumBeat() if required.
            pmmSub = calcDict['pmmUn'][isoKey].sel({'J':J,'Jp':Jp})

            # Calc Cterms
            # 29/11/24 - allow multiple Jf/Nc for spin case. Set Jf=None for this case.
            #          - Add Nc to sumList
            # TODO: may want to allow for passing this?
            channel = [J,None,None,None]
            Cterms = gammaCalc.Ccalc(channel, spinWeightings=spinDict, thres=thres)

            # Calc gamma 
            gammaPmm, lPhase, Cpmm, Cterms = gammaCalc.gammaCalc(Cterms = Cterms, denMat=pmmSub,)
                                                                 # sumList = ['q','qp','Mi','Mip','Nt','Ntp','Mt','Mtp','Mc','Nc'])

            # Calc betas
            betaOut, betaOutNorm = gammaCalc.betaCalc(gammaPmm, matE=matE,
                                                  channel = channel) #, thres=1e-6) 
                                                  # cols=['l','m','matE1'])  # Cols currently hard-coded in betaCalc
                                                    # cols=['l','lam','matE1'])
            # BLMplot(betaXRremapped, backend='hv', xDim='t', thres=None,) 

            # VERSION with full var return for testing
            # betaCalcFull = gammaCalc.betaCalc(gammaPmm, matE=matEInputCleanMod,
            #                                           channel = [J,None,None,None], returnType='full') #, thres=1e-6) 

            # Assign results to dict
            betaJ[isoKey][Jf] = {'betaOut': betaOut, 
                                  'betaNorm':betaOutNorm,
                                  'Jf':Jf,
                                  'isoKey':isoKey,
                                  'spinW':spinDict,
                                  'gammaOut':(gammaPmm, lPhase, Cpmm, Cterms)}
            
    return betaJ

            
    