"""
Photoionization functions for Quantum-beat photoelectron-imaging spectroscopy of Xe in the VUV

24/06/24

"""

import pandas as pd
import numpy as np

import xarray as xr  # Currently only for type checking, so may want to skip

# For blm calcs
from qbanalysis.hyperfine import pmmFromQuantumBeat
from epsproc.geomFunc.gamma import gammaCalc
from epsproc import multiDimXrToPD

# For logging
from loguru import logger


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


def blmCalc(calcDict, matE = None, isoKeys = None, 
            JFlist = None, channel = None,
            thres=1e-4, forceCalc = False,
            trange = None,
            **kwargs):
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
        
    forceCalc : bool, default = False
        Force recalculation of gamma terms and override terms in calcDict if True.
        
    trange : list, optional, default = None
        Set t range limit if desired, trange = [tmin,tmax].
        This is used to sub-select on density matrix (pmm) which defines t-axis.
        
    **kwargs : unused
        Allow for arb arg passing for fitting wrapper.
        
        
    Returns
    -------
    betaJ : dict
        Contains results per state, indexed as betaJ[isoKey][J]
        
    
    May 2025 v2 revisiting and updating
    - Add optional calcs for missing terms in default case.
                
    Nov/Dec 2024 v1 in progress from demo notebook (4.05).
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

            # # Compute spin weightings for given Jf case (==Jc in function).
            # spinDict = gammaCalc.spinWeightings(selectors={'Jc':Jf})


            # Subselect and sum
            Ji = int(Jilist[0][0]) # Initial state J
            # Set terms for legacy code
            J = Ji
            Jp = Ji

            # pmmSub = pmmPkgDS[isoKey].sel({'J':J,'Jp':Jp})  #.sum(['K','Q'])
            if not 'pmmUn' in calcDict.keys():
                pmmFromQuantumBeat(calcDict)
                
            # Version from calcDict - may want to check and call pmmFromQuantumBeat() if required.
            # NOTE - for pmmFromQuantumBeat(calcDict) with uncertainties need to subselect from dataset too!
            pmmSub = calcDict['pmmUn'][isoKey][isoKey].sel({'J':J,'Jp':Jp})
            
            # Optionally set trange
            print(f"trange={trange}")
            if trange is not None:
                pmmSub = pmmSub.sel(t=slice(trange[0],trange[1]))

            # Calc Cterms
            # 29/11/24 - allow multiple Jf/Nc for spin case. Set Jf=None for this case.
            #          - Add Nc to sumList
            # TODO: may want to allow for passing this?
            # channel = [J,None,None,None]
            
            # 14/05/25 Can now pass (single) channel, for default set iwth J
            if channel is None:
                channel = [J,None,None,None]
               
            # 14/05/25 - add dict terms for gammas to avoid recalc for multiple (fitting) runs
            # Use tuple(channel) as key, check and create if not set
            if not 'gamma' in calcDict.keys():
                calcDict['gamma'] = {}
                
            # if 'gamma' in calcDict.keys():
            # Create dict if missing
            if not isoKey in calcDict['gamma'].keys():
                calcDict['gamma'][isoKey] = {}
                
            # Gamma key - index gamma params for reuse with channel + Jf
            gKey = channel.copy()
            gKey.append(Jf)
            gKey = tuple(gKey)

            # Compute or set Cterms
            if not forceCalc and (gKey in calcDict['gamma'][isoKey].keys()):
                logger.debug(f"Found gKey {gKey} in calcDict['gamma'][{isoKey}].")
                # print(f"Found gKey {gKey} in calcDict['gamma'][{isoKey}].")
                # print(f"Found channel {channel} in calcDict['gamma'][{isoKey}].")
                # Cterms = calcDict['gamma'][isoKey][tuple(channel)]['Cterms']  # Specific terms
                # locals().update(calcDict['gamma'][isoKey][tuple(channel)])   # Unpack to locals
                
                # locals unpacking seems to fail...? Test here... think I've used something similar elsewhere preivously...?
                # Var names appear in locals().keys(), but give NameError when trying to access?
                # CHECK PEMtk or ePSproc util functions...?
                # d = {'a': 1, 'b': 2}
                # locals().update(**d)
                
            else:
                logger.debug(f"Computing gammas for channel {channel}, {isoKey}, Jf={Jf}.")
                # print(f"Computing gammas for channel {channel}, {isoKey}, Jf={Jf}.")
                
                # Compute spin weightings for given Jf case (==Jc in function).
                spinDict = gammaCalc.spinWeightings(selectors={'Jc':Jf})
            
                Cterms = gammaCalc.Ccalc(channel, spinWeightings=spinDict, thres=thres)
                # Calc gamma 
                gammaPmm, lPhase, Cpmm, Cterms = gammaCalc.gammaCalc(Cterms = Cterms, denMat=pmmSub,)

                calcDict['gamma'][isoKey][gKey] = {'gammaPmm': gammaPmm,
                                                     'spinDict': spinDict,
                                                     'lPhase': lPhase, 
                                                     'Cpmm':Cpmm,
                                                     'Cterms': Cterms,}
                
                

            # locals().update(**d)
            # print(locals().keys())
            # print(a)
            
            # Just use dict in calcs below - much simpler than messing with unpacking etc.
            gammaDict = calcDict['gamma'][isoKey][gKey]
            
            # Original case - just calculate
            # Cterms = gammaCalc.Ccalc(channel, spinWeightings=spinDict, thres=thres)

            # Calc gamma 
            # gammaPmm, lPhase, Cpmm, Cterms = gammaCalc.gammaCalc(Cterms = Cterms, denMat=pmmSub,)
                                                                 # sumList = ['q','qp','Mi','Mip','Nt','Ntp','Mt','Mtp','Mc','Nc'])

            # Calc betas
            betaOut, betaOutNorm = gammaCalc.betaCalc(gammaDict['gammaPmm'], matE=matE,
                                                  channel = channel) #, thres=1e-6) 
                                                  # cols=['l','m','matE1'])  # Cols currently hard-coded in betaCalc
                                                    # cols=['l','lam','matE1'])
            # BLMplot(betaXRremapped, backend='hv', xDim='t', thres=None,) 

            # VERSION with full var return for testing
            # betaCalcFull = gammaCalc.betaCalc(gammaPmm, matE=matEInputCleanMod,
            #                                           channel = [J,None,None,None], returnType='full') #, thres=1e-6) 

            # Set to XR format...?
            # 
            # dataOutXR = betaOut.to_xarray().to_array()
            # dataOutXR = dataOutXR.rename({'variable':'t'})
            #
            
            # Assign results to dict
            betaJ[isoKey][Jf] = {'betaOut': betaOut, 
                                  'betaNorm':betaOutNorm,
                                  'Jf':Jf,
                                  'isoKey':isoKey,
                                  # 'spinW':spinDict,
                                  'channel':channel,
                                  'gKey':gKey}
                                  # 'gammaOut':(gammaPmm, lPhase, Cpmm, Cterms)}  # Updated case - this is now in calcDict['gamma']
            
    return betaJ


def matEReformat(matE, Eind = None, Eval = None):
    """
    Reformat matrix elements from Xarray in usual ePSproc style to Pandas DataFrame format for gammaCalc.
    
    TODO: generalise and move to gammaCalc code, this already has denMatReformat (see https://github.com/phockett/ePSproc/blob/56c01f0a1f3ba90c1409a32a276c241e04165638/epsproc/geomFunc/gamma/gammaCalc.py#L618C5-L618C19)
    
    18/05/25 v1 from test/demo 4.05 notebook.
    
    """
    
    # Set default E index
    if (Eind is None) and (Eval is None):
        Eind = 0
    
    # SHOULD ALSO test and set if value (not index) passed...
    # elif Eval is not None:
    #     pass
    
    # DEBUG...
    # print(f"MatE passed as type {type(matE)}")
    logger.debug(f"MatE passed as type {type(matE)}")
    
    # For ePSproc set matE, should already had PD set, just need to subselect and clean-up
    if isinstance(matE, xr.DataArray):
    # if type(matE).__name__   # Options for skipping Xarray import...?
        if hasattr(matE, 'pd'):
            # print(f"Converting from existing pd")
            logger.debug("Converting from existing pd")
            matEpd = matE.pd[Eind].to_frame()  # Subselect on E and ensure Frame
            
            # Clean up for gammaCalc
            matEpd.rename(columns={0:'matE1'}, inplace=True)
            matEpd = matEpd.droplevel(['Cont','Targ','Total','Type','it'])

            # DROP MU - optional
            matEpd = matEpd.droplevel('mu')
            matEpdClean = matEpd[~matEpd.index.duplicated()]  # Drop duplicated INDEX items
            
            return matEpdClean
            
        else:
            logger.debug("Setting matE with XR to PD routine.")
            # print(f"TODO: set to pd")
            # # TODO: use ep.multiDimXrToPD() here for flexibility if different format passed.
            # pass
        
            # From setMatE.py:
            # Set PD table - see also classes._IO.matEtoPD() method for wrapped version.
            # matE.attrs['pd'],_ = pdTest, _ = multiDimXrToPD(matE, colDims = eType if len(Evals)>1 else matE.attrs['harmonics']['keyDims'], thres=None, squeeze=False)
            # return multiDimXrToPD(matE, colDims='l', squeeze = False)  # TEST multiDimXrToPD - sort-of works, but will need to reformat!
                      
            # Version from PEMtk, needs non-default col dim if missing.
            # Q: minimal array requirement for gammaCalc?  Just (l,m)?
            # See also PEMtk fitClass.reconParams()
            # return data.pdConvSetFit(matErecon, colDim='m')  
            
            # TEST RECON - OK WITH HARD-CODED DIM NAME FOR SINGLETON E DIM!!!!
            # UGLY!
            # NOTE THIS ASSUMES FORMAT AS RETURNED BY fitting function via `reconParams()`, may not be general
            matErecon = matE.copy()
            matErecon.name = 'matE1'
            matEreconPD = matErecon.unstack().to_dataframe()
            
            return matEreconPD
            
    
    
#     # TODO: convert from params, as per main PEMtk fitClass method
#     # NOTE this needs to push back to PD in this case too.
#     if isinstance(matE, Parameters):
#         # Set matE from passed params object
#         # matE = self.reconParams(matE, lmmuList)
        
#         # PARAMS conv testing...
#         # Q: minimal array requirement for gammaCalc?  Just (l,m)?
#         matErecon = data.reconParams()
#         # matEreconPD = data.matEtoPD   # Class wrapped version, for data keys
        
#         return data.pdConvSetFit(matErecon, colDim='m')  # Version from PEMtk, needs non-default col dim if missing.

        
    # Return original matE if no conversion required
    return matE





#*********** Fitting (uses PEMtk functionality)

def blmCalcFit(matEinput, basisReturn = 'Full', renorm = True, betaType = 'betaOut',  #'betaNorm'
               betaRef = None,
               **kwargs):
    """
    Wrap blmCalc for use with PEMtk fitting routines, as backend function.
    
    For details see:
        - Backend notes, https://pemtk.readthedocs.io/en/latest/fitting/PEMtk_fitting_backends_demo_010922.html
        - Fitting routines source, https://pemtk.readthedocs.io/en/latest/_modules/pemtk/fit/fitClass.html#pemtkFit.afblmMatEfit
        - General fitting routine notes, https://pemtk.readthedocs.io/en/latest/fitting/PEMtk_fitting_demo_multi-fit_tests_130621-MFtests_120822-tidy-retest.html
        
    Note this need to function as a backend for :py:func:`afblmMatEfit`, generally called with:
    
        `BetaNormX = backend(matE, **basis, thres = thres, selDims = selDims, thresDims=thresDims, basisReturn = 'BLM', **kwargs)`
       
    14/05/25 v1 sketching.
    
    matEinput : matrix element for calculation
        - Standard ePSproc style Xarray or Pandas DataFrame format.
        - If Xarray, will be reformatted for gammaCalc, Pandas DataFrame at single E only, and drop additional indexers in current code (as of Dec. 2024).

    renorm : bool, optional, default = True
        Renormalised betas to B00 if true.

    basisReturn : optional, str, default = "BLM"
        - 'BLM' return Xarray of results only.
        - 'Full' return Xarray of results + basis set dictionary as set during the run.
        - Note other types just return Full, this provides compatibility with existing AF fitting routines, see options at https://github.com/phockett/ePSproc/blob/56c01f0a1f3ba90c1409a32a276c241e04165638/epsproc/geomFunc/afblmGeom.py#L634
        
    betaType : string, default = 'betaOut'
        Key for betas in output dataframe (as set in :py:func:`betaCalc()`).
        - 'betaOut' unnormalised betas.
        - 'betaOutNorm' normalised betas.
        
    betaRef : pd.DataFrame, optional
        If passed use to extend or chop model results (L,M) values to match data.
        NOT CURRENTLY IMPLEMENTED.
        To avoid NaNs in fitting, may need to run with `data.fit(nan_policy='omit')` in cases where (L,M) indexes diverge.
        See https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do
        Note this corresponds to a "missing data" case in the input, and should be consistent over a fitting run, BUT need to be careful that model is still physical.
        
    NOTE: currently blmCalc returns results in Pandas DataFrame, may want to convert to XR for use with PEMtk plotting routines. (But fitting OK.)
    
    """
    
    # Reformat matE if required
    matE = matEReformat(matEinput)
    
#     if JFlist is None:
#         JFlist = [0.5,1.5]
    
    # Compute BLMs
    # NOTE: currently need to set matE to kwarg, since this function expects matE as 1st arg.
    # ALSO: blmCalc currently expects calcDict as 1st arg, but seems to work OK with kwarg passing in testing?  Might be dubious...
    # betaJ = blmCalc(calcDict=kwargs['calcDict'], matE=matE, **kwargs)   # Fails - duplicate calcDict if kwarg
    # betaJ = blmCalc(matE=matE, **kwargs)   # Seems OK if calcDict in kwargs, although suspect dodgy?
    
    calcDict = kwargs.pop('calcDict')  # Better arg,kwarg passing - pop calcDict, then pass in correct order.
    betaJ = blmCalc(calcDict, matE=matE, **kwargs)
    
    # For fitting, sum over all channels (i.e. assume single dataset)
    # OR: if seperable, just calc for single channel above...?
    # for item in betaJ.keys():
        
    # Set params
    betaStack = []
    # JFsum = JFlist #  [JFlist[1]]   # Set JF cases to sum? Now just use all keys set below, but may want to allow for passing here too.

    # Get results (dataframes)
    for key in betaJ.keys():
        for Jf in betaJ[key].keys():
            betaStack.append(betaJ[key][Jf][betaType])

    # Add dataframes
    # For summing list of dataframes
    # From https://stackoverflow.com/a/45983359
    from functools import reduce
    betaSum = reduce(lambda x, y: x.add(y, fill_value=0), betaStack)

    if renorm:
        betaSum = betaSum/betaSum.loc[0,0]
    
    
    if basisReturn == 'Full':
        return betaSum, {'calcDict':calcDict,'betaJ':betaJ, 'isoKeys':kwargs['isoKeys'], 'JFlist':kwargs['JFlist']}  #**kwargs}  # NOTE: ensure only required kwargs passed here, otherwise can get duplicates 
        # return betaSum, {'calcDict':kwargs['calcDict'],'betaJ':betaJ} 
        # return {'calcDict':kwargs['calcDict']}
    
    # Return only calcDict as basis fns. for use in fitting routines
    elif basisReturn == "ProductBasis":
        # return betaSum, {'calcDict':locals()['calcDict']}
        return betaSum, {'calcDict':calcDict, 'isoKeys':kwargs['isoKeys'], 'JFlist':kwargs['JFlist']}  #**kwargs}  # NOTE: ensure only required kwargs passed here, otherwise can get duplicates
    
    # Default to only final BetaSum return for any other case
    else:
        return betaSum

    
    