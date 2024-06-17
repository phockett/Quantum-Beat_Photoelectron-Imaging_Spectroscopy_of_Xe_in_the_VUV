"""
Functions for modelling hyperfine quantum beats - basic fitting for hyperfine beats

17/06/24, PH

"""

from qbanalysis.hyperfine import computeModel, computeModelSum
import numpy as np
from epsproc.sphCalc import setBLMs
import scipy
import pandas as pd

from loguru import logger

# For uncertainties, alias some functions if used.
# Also set flag for use later.
try:
    from uncertainties import unumpy, ufloat_fromstr
    
    logger.info("Using uncertainties modules, Sympy maths functions will be forced to float outputs.")
    
    # Settings required for uncertainties to work in existing routines below...
    cosLocal = unumpy.cos
    
    from sympy.physics import wigner
    wigner_3j = lambda *args: float(wigner.wigner_3j(*args))
    wigner_6j = lambda *args: float(wigner.wigner_6j(*args))
    
    unFlag = True
        
except ImportError:
    # Use non-uncertainties funcs
    cosLocal = np.cos
    from sympy.physics.wigner import wigner_3j, wigner_6j
    
    unFlag = False
    
    
# UNITS
# TODO: make this nicer
cmToMHz = 29979.2458
    

#*** Fitting time-dependent signals using the hyperfine beat model

def residual(model,dataIn):
    """
    Calc least squares residual
    """
    res = (model - dataIn)**2  # Returning single value XR only in testing? Issue with dims?
                            # Ah, OK after fixing t-units
    # res = model.values - dataIn.values  # Force to NP, assumes matching size.

    return res

def setParams(xePropsIn, newVals, fitParamsCol = 'Splitting/cm−1'):
    """
    Replace single column in input dataframe with newVals.
    
    Note: no size checks here.
    """
    
    # Set splittings
    # fitParamsCol = 'Splitting/cm−1'
    xePropsUpdated = xePropsIn.copy()
    xePropsUpdated[fitParamsCol] = newVals
    
    return xePropsUpdated

# NOTE - setting trange here may be required.
# Fitting to full window tends to smooth out oscillations, may need to be more careful with residual func?
# trange=[0,200]  OK first part only
# trange=[0,500]  GOOD!
# trange=[0,800]  GOOD! Lower overall intensity than [0,500] case.
# trange=[0,1000]  GOOD! Lower overall intensity than [0,500] case.
# trange=None  OK, but t<0 data messes things up a bit.
def calcBasicFitModel(xData, xePropsFit = None, dataDict = None, fitFlag=True, trange=[0,1000]):
    """
    Calc model and residual for Scipy fitting.
    
    Set fitFlag=False to return all model results.
    """
    
    # Update fit params
    xePropsFit = setParams(xePropsFit,xData)
    
    # Compute model
    modelDict = computeModel(xePropsFit, tIn=dataDict['BLMall'].t*1e-12)  # Note t-units in s!
    # modelSum = computeModelSum(modelDict)['sum'] 
    modelDictSum, modelDA = computeModelSum(modelDict)
    modelSum = modelDictSum['sum']
    
    # Compute residual
    dataIn = dataDict['BLMall'].sel({'ROI':0,'l':4}).copy()
    modelIn = modelSum.sel({'K':2}).squeeze(drop=True)
    # modelIn.values = unumpy.nominal_values(modelIn)  # Use nominal values only?
    # modelIn['t'].values = modelIn['t'].values.astype(int) 
    modelIn = modelIn.assign_coords({'t':modelIn['t'].values.astype(int)})  # Force to int to match input data

    # Optionally set trange
    if trange is not None:
        modelIn = modelIn.sel(t=slice(trange[0],trange[1]))
        dataIn = dataIn.sel(t=slice(trange[0],trange[1]))
    
    res = residual(modelIn, dataIn.squeeze())
    
    if fitFlag:
        return unumpy.nominal_values(res.values)
    else:
        
        # Fix splitting value for 1.5,2.5 (derived case)
        # TODO: this is inconsistent with fitting later?
        iso=131
        dataCol = 'Splitting/cm−1'
        xePropsFit.loc[(iso,1.5,2.5,1.5), dataCol] = xePropsFit.loc[(iso,1.5,2.5,0.5), dataCol] - xePropsFit.loc[(iso,1.5,1.5,0.5), dataCol]
        
        return xePropsFit, modelDict, modelSum, modelIn, dataIn, res

    
def compareResults(xeProps, xePropsFit, fitParamsCol = 'Splitting/cm−1'):
    """
    Create comparison table of reference and fit results.
    """
    
    import pandas as pd
    
    diffData = pd.DataFrame([xeProps[fitParamsCol], xePropsFit[fitParamsCol], xeProps[fitParamsCol]-xePropsFit[fitParamsCol]]).T
    # diffData.columns.rename({n:item for n,item in enumerate(['original','fit','diff'])})
    diffData.columns = ['original','fit','diff']
    
    # Remove uncertainties? Makes sense if comparing diffs from different anlyses
    diffData['diff']= unumpy.nominal_values(diffData['diff'].values)
    
    return diffData



#******* Functions for A, B parameters

def extractABParams(xePropsFit):
    """
    Determine A & B parameters from hyperfine level splittings.
    
    This runs a quick fit with Scipy
    """
    
    #*** Set data to fit - NOTE 131 only!
    dataCols = xePropsFit.reset_index()   # Use reset here, xePropsFit.xs(131) drops index
    data131 = dataCols.loc[dataCols['Isotope']==131]

    #*** Extract A,B for 131Xe using Scipy fitting
    # x0in = np.random.rand(2)
    x0in = [2000,30]

    fitOut = scipy.optimize.least_squares(dEcalcWrapperScipy, x0in, bounds = ([0,-100],[2500,100]),
                                          kwargs = {'xeDataInPD':data131},
                                          verbose = 0,
                                          xtol=1e-12,ftol=1e-12,gtol=1e-18)
    
    # Set final results
    dataFit = dEcalc(data131, *fitOut.x)
    
    # Add fitted results to table
    dataFit['A/MHz'] = fitOut.x[0]
    dataFit['B/MHz'] = fitOut.x[1]
    
    # Add 129 case back in
    # Note phase = -1 by convention, since F<F' - now included in dE calc directly
    phase =-1
    
    # This works, but always throwing PD warning to use .loc. Not sure why.
    # data129 = dataCols.loc[dataCols['Isotope']==129]
    # data129.loc[data129['Isotope']==129,'A/MHz']= data129['Splitting/cm−1']/data129['F′'] * cmToMHz
    # cols = data129['Isotope']==129
    # data129.loc[cols,'A/MHz'] = data129.loc[cols,'Splitting/cm−1']/data129.loc[cols,'F′'] * cmToMHz * phase
    # data129.loc[cols,'dE'] = data129.loc[cols,'F′'] * data129.loc[cols,'A/MHz'] * 1/cmToMHz
    
    # v2, with dEcalc generalised...
    # Also reworked PD assignment, seems better using full DF? Something about single-value selection/series...? BUT THIS IS UGLY...
    Iind = dataCols['Isotope']
    data129 = dataCols.loc[Iind==129]
    dataCols.loc[Iind==129,'A/MHz']=data129['Splitting/cm−1']/data129['F′'] * cmToMHz * phase
    # data129.loc[cols,'A/MHz'] = data129.loc[cols,'Splitting/cm−1']/data129.loc[cols,'F′'] * cmToMHz * phase
    # data129 = dEcalc(data129, data129['A/MHz'], np.nan)
    # Update data129 and update dE
    data129 = dataCols.loc[Iind==129]
    data129 = dEcalc(data129, data129['A/MHz'], np.nan)
    
    # data129.loc[data129.columns['A/MHz']]
    # dF = xeData.F - xeData['F′']
    dataOut = pd.concat([data129,dataFit])
    
    return dataOut


# def extractABParams(xePropsFit):
#     """
#     Determine A & B parameters from hyperfine level splittings.
    
#     This runs a quick fit with Scipy
#     """
    
#     # Set data to fit - NOTE 131 only!
#     dataCols = xePropsFit.reset_index()   # Use reset here, xePropsFit.xs(131) drops index
#     data131 = dataCols.loc[dataCols['Isotope']==131]


#     # x0in = np.random.rand(2)
#     x0in = [2000,30]

#     fitOut = scipy.optimize.least_squares(dEcalcWrapperScipy, x0in, bounds = ([0,-100],[2500,100]),
#                                           kwargs = {'xeDataInPD':data131},
#                                           verbose = 2,
#                                           xtol=1e-12,ftol=1e-12,gtol=1e-18)
    
#     # Set final results
#     dataFit = dEcalc(data131, *fitOut.x)
    
#     # Add fitted results to table
#     dataFit['A/MHz'] = fitOut.x[0]
#     dataFit['B/MHz'] = fitOut.x[1]
    
#     # Add 129 case back in
#     data129 = dataCols.loc[dataCols['Isotope']==129]
#     data129.loc[data129['Isotope']==129,'A/MHz']= data129['Splitting/cm−1']/data129['F']
#     data129.append(dataFit)
    
#     return data129
    
    
def dEcalcWrapperScipy(x0,xeDataInPD=None):
    """
    Wrap dEcalc for Scipy least_squares...
    """
    
    dEOut = dEcalc(xeDataInPD, x0[0], x0[1])
    
    res = ((dEOut['Splitting/cm−1'] - dEOut['dE'])**2) #.squeeze()
    
    if unFlag:
        res = unumpy.nominal_values(res)
    
    return res


# v3, use Pandas dataframe for calcs.
# Simpler than Xarray in this case!
def dEcalc(dataInPD, A, B):
    """
    Calculate dE from A & B
    
    The hyperfine coupling constants can be determined by fitting to the usual form (see, e.g., ref. \cite{D_Amico_1999}):
    \begin{equation}
    \Delta E_{(F,F-1)}=AF+\frac{3}{2}BF\left(\frac{F^{2}+\frac{1}{2}-J(J+1)-I(I+1)}{IJ(2J-1)(2I-1)}\right)
    \end{equation}
    
    NOTE: units currently set for return values only.
    NOTE: this version assumes PD dataframe input. Multiindex including [Isotope,I,F], or columns OK.
    NOTE: for 129Xe case, selects on largest(F,F') AND applies phase as sign of dF.
    
    """
    # Set units
    units = 'cm-1'

    # Set J
    J=1
    
    # Set data - note reset index to just use col values
    dataPD = dataInPD.copy()
    if 'I' not in dataPD.columns:
        dataPD = dataPD.reset_index()
        
        
    I=dataPD['I']
    c1=0.5-J*(J+1)-I*(I+1)
    c2=I*J*(2*J-1)*(2*I-1)
    
    # F = dataPD['F']
    F = dataPD[['F','F′']].max(axis=1)  # Use max, allows for 129Xe case with values reversed
    dF = dataPD['F']- dataPD['F′']
    
    t1 = A*F
    t2 = (3/2)*B*((F**2 + c1)/c2)
    t2 = t2.fillna(0)

    dataOut = dataPD.copy()
    
    #*** Set outputs by isotope and dF
    # Default case, t1 only
    # Use sign dF as the phase
    dataOut['dE'] = t1 * np.sign(dF)
    
    # For 131Xe, use t1+t2
    dataOut.loc[dataOut['Isotope']==131,'dE'] = (t1+t2)
    
    # ...and replace cases with dF > 1
    dataOut.loc[np.abs(dataOut['F'] - dataOut['F′'])>1, 'dE'] = np.nan
    
    # Cheat here, and replace with sum for known case - should automate this!
    # Note use of .values otherwise data with Uncertainties may not propagate correctly.
    dataOut.loc[(dataOut['Isotope']==131) & (dataOut['F']==2.5) & (dataOut['F′']==0.5),'dE'] = dataOut[(dataOut['Isotope']==131) & (dataOut['F']==2.5) & (dataOut['F′']==1.5)]['dE'].values + dataOut[(dataOut['Isotope']==131) & (dataOut['F']==1.5) & (dataOut['F′']==0.5)]['dE'].values

    
    # Set units
    if units == 'cm-1':
        dataOut['dE'] = dataOut['dE']*(1/cmToMHz)
    

    return dataOut




# V2, testing subselected PD array > Xr
# Calculate dE for Xarray input - in this case all coords should match in size...
# A,B can be Xarray or scalar
# def dEv2(xeDataIn, A, B, units = 'cm-1'):
def dEv2(xeDataIn, A, B):
    """
    the hyperfine coupling constants can be determined by fitting to the usual form (see, e.g., ref. \cite{D_Amico_1999}):
    \begin{equation}
    \Delta E_{(F,F-1)}=AF+\frac{3}{2}BF\left(\frac{F^{2}+\frac{1}{2}-J(J+1)-I(I+1)}{IJ(2J-1)(2I-1)}\right)
    \end{equation}
    
    NOTE: units currently set for return values only.
    NOTE: Xarray input required for xeDataIn
    """
    # for iso in xeData.Isotope:
    #     print(item)
    units = 'cm-1'
    cmToMHz = 29979.2458
    
    # Isotope terms
    J=1
    I=xeDataIn.I
    c1=0.5-J*(J+1)-I*(I+1)
    c2=I*J*(2*J-1)*(2*I-1)
    
    # A/B terms
    F = xeDataIn.F
    # F = xeData['F′']  # TODO: fix 129 ordering, needs F,F' swapped! (Or enforce selection here...)
                        # Or swap on max value, or unique values... 
                        # Or check on dF, xeData.F - xeData['F′'] ...?
    # This works, but have some redundant values still
    # Ffixed = xr.where(xeData.F > xeData['F′'], xeData.F, xeData['F′'])  # Check greater
    # F = Ffixed[:,1]
    
    # Try unique vals only... Breaks 129 case...
    # F = xeData.F.where(xeData.F > 1)
    
    # Deltas... filter on these at return?
    # dF = xeData.F - xeData['F′']
    
    
    t1 = A*F  #* np.sign(dF)
    
    # For Xr case avoid Nan propagation
    if isinstance(B, xr.DataArray):
        if unFlag:
            B = xrUnFillna(B)
        else:
            B = B.fillna(0)
        
        
    t2 = (3/2)*B*((F**2 + c1)/c2)
    t2 = t2.fillna(0)
    
    # return t1,t2,t1+t2
    
    if units == 'MHz':
        dEout = t1+t2
    elif units == 'cm-1':
        dEout = (t1+t2)*(1/cmToMHz)

    # Check allowed terms...?
    # dEout = dEout.where(np.abs(dF)<2,np.nan)
        
        
    # TODO: general fix for F-F' > 1..?
    # dEXR.where(dEXR.F - dEXR['F′'] > 1)
    # Quick fix here for Xe131 case only
    # dEout.sel({'F':
    # dEout.loc[{'Isotope':131,'F':2.5,'F′':0.5, 'I':1.5}] = dEout.sel({'Isotope':131,'F':1.5,'F′':0.5, 'I':1.5}) + dEout.sel({'Isotope':131,'F':2.5,'F′':1.5, 'I':1.5})
    
    if unFlag:
        dEout.values = unumpy.nominal_values(dEout)
        
    return dEout


def dEv2Wrapper(xeDataInNP, A, B, xeDataIn):
    """
    Thin wrapper for xr.curvefit.
    
    Just swap NP data as passed for XR data to use existing function
    """
    
    dEOut = dEv2(xeDataIn, A, B)
    
    return dEout

    
def dEv2WrapperScipy(x0,xeDataIn=None):
    """
    ... and wrap for Scipy least_squares...
    """
    
    dEOut = dEv2(xeDataIn, x0[0], x0[1])
    
    res = ((xeDataIn - dEOut)**2).squeeze()
    
    return res.values

    
def xrUnFillna(xrData):
    """
    Implement xr.fillna for Uncertainties data types.
    """
    
    return xrData.where(~unumpy.isnan(xrData),0)


# V1, assumes full PD array > Xr
# Calculate dE for Xarray input - in this case all coords should match in size...
# A,B can be Xarray or scalar
def dE(xeDataIn, A, B, units = 'cm-1'):
    """
    the hyperfine coupling constants can be determined by fitting to the usual form (see, e.g., ref. \cite{D_Amico_1999}):
    \begin{equation}
    \Delta E_{(F,F-1)}=AF+\frac{3}{2}BF\left(\frac{F^{2}+\frac{1}{2}-J(J+1)-I(I+1)}{IJ(2J-1)(2I-1)}\right)
    \end{equation}
    
    NOTE: units currently set for return values only.
    """
    # for iso in xeData.Isotope:
    #     print(item)
    cmToMHz = 29979.2458
    
    # Isotope terms
    J=1
    I=xeDataIn.I
    c1=0.5-J*(J+1)-I*(I+1)
    c2=I*J*(2*J-1)*(2*I-1)
    
    # A/B terms
    # F = xeDataIn.F
    # F = xeData['F′']  # TODO: fix 129 ordering, needs F,F' swapped! (Or enforce selection here...)
                        # Or swap on max value, or unique values... 
                        # Or check on dF, xeData.F - xeData['F′'] ...?
    # This works, but have some redundant values still
    Ffixed = xr.where(xeData.F > xeData['F′'], xeData.F, xeData['F′'])  # Check greater
    F = Ffixed[:,1]
    
    # Try unique vals only... Breaks 129 case...
    # F = xeData.F.where(xeData.F > 1)
    
    # Deltas... filter on these at return?
    dF = xeData.F - xeData['F′']
    
    
    t1 = A*F* np.sign(dF)
    
    # For Xr case avoid Nan propagation
    if isinstance(B, xr.DataArray):
        if unFlag:
            B = xrUnFillna(B)
        else:
            B = B.fillna(0)
        
        
    t2 = (3/2)*B*((F**2 + c1)/c2)
    t2 = t2.fillna(0)
    
    # return t1,t2,t1+t2
    
    if units == 'MHz':
        dEout = t1+t2
    elif units == 'cm-1':
        dEout = (t1+t2)*(1/cmToMHz)

    # Check allowed terms...?
    dEout = dEout.where(np.abs(dF)<2,np.nan)
        
        
    # TODO: general fix for F-F' > 1..?
    # dEXR.where(dEXR.F - dEXR['F′'] > 1)
    # Quick fix here for Xe131 case only
    # dEout.sel({'F':
    # dEout.loc[{'Isotope':131,'F':2.5,'F′':0.5, 'I':1.5}] = dEout.sel({'Isotope':131,'F':1.5,'F′':0.5, 'I':1.5}) + dEout.sel({'Isotope':131,'F':2.5,'F′':1.5, 'I':1.5})
    
    
    return dEout
