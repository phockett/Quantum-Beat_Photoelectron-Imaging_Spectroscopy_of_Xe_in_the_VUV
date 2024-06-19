"""
Functions for modelling hyperfine quantum beats - advanced fitting for hyperfine beats

18/06/24, PH

"""


#*** Functions for calculation.
# NOTE: designed for use with lmfit, expect lmfit Parameters() or params dict.

import numpy as np
import xarray as xr

import lmfit
from lmfit import minimize, Parameters
from qbanalysis.basic_fitting import calcBasicFitModel
from qbanalysis.hyperfine import splitUncertaintiesToDataset

from loguru import logger

# For uncertainties, alias some functions if used.
# Also set flag for use later.
try:
    from uncertainties import unumpy, ufloat_fromstr
    
    logger.info("Using uncertainties modules, Sympy maths functions will be forced to float outputs.")
    
#     # Settings required for uncertainties to work in existing routines below...
#     cosLocal = unumpy.cos
    
#     from sympy.physics import wigner
#     wigner_3j = lambda *args: float(wigner.wigner_3j(*args))
#     wigner_6j = lambda *args: float(wigner.wigner_6j(*args))
    
    unFlag = True
        
except ImportError:
    # # Use non-uncertainties funcs
    # cosLocal = np.cos
    # from sympy.physics.wigner import wigner_3j, wigner_6j
    
    unFlag = False


def calcDecays(paramDict, isoDA):
    """
    Apply exponential decays, exp(-t/tau), per isotope.
    
    paramDict : dict
        Parameters dictionary with items 'tauZ' per isotope Z.
        
    isoDA : xr.dataarray
        Main data structure, with dims including "Isotope" and "t".
        Groupby & apply exponential decay per Isotope.
        
    """
    
    #*** Apply exponential decay (per isotope)
    # Easier way to do this...? Here set DA per isotope, then combine and multiply.
    # Quick test of multiply without all isotope dims also awkward, although maybe groupby would work...
    
    # Dict + loop version
    # decay = xr.zeros_like(isoDA)
#     decayDict = {}
#     for iso in [129,131]:
#         decayDict[iso] = np.exp(-isoDA.t/paramDict[f"tau{iso}"]).expand_dims({'Isotope':[f'{iso}Xe']})
#         # decay = decay + np.exp(-isoDA.t/paramDict[f"tau{iso}"]).expand_dims({'Isotope':[f'{iso}Xe']})
    
#     decayDA = isoDA*stackModelToDA(decayDict)
    
    # Groupby version - better...?
    # Test groupby...
    decay = isoDA.groupby("Isotope").map(lambda x: x*np.exp(-x.t/paramDict[f"tau{x.Isotope.values.item().rstrip('Xe')}"]))
    
    decaySum = decay.sum("Isotope").expand_dims({"Isotope":['sum']})
    
    decayOut = xr.concat([decay, decaySum], dim="Isotope")
    
    decayOut.name = 'decay'
    
    return decayOut


def ionizationPhenom(paramDict,modelDA):
    """
    Basic amplitude/phase + offset ionization channel model.

    """
    
    # Assume isotope independent...?
    # Also ignore total XS here
    if 'sum' in modelDA.Isotope:
        modelIn = modelDA.sel({'K':2, 'Isotope':'sum'})
    else:
        modelIn = modelDA.sum("Isotope").sel({'K':2})
    
    # ROI only version - assume isotope independent params
    modelOutComponents = []
    for ROI in [0,1]:
        lparams = [2,4]
        paramsAmp = [paramDict[f"l{l}_amp_{ROI}"] for l in lparams]
        paramsOffset = [paramDict[f"l{l}_offset_{ROI}"] for l in lparams]
        
        modelOutComponents.append((modelIn * xr.DataArray(paramsAmp,coords=[("l",lparams)]) + xr.DataArray(paramsOffset,coords=[("l",lparams)])).expand_dims({"ROI":[ROI]}))
        
        # for item in ['amp','offset']:
        #     [params.add(f"l{l}_{item}_{ROI}", value=1.0) for l in [2,4]]
    
    modelOut = xr.concat(modelOutComponents,dim="ROI")
    modelOut.name = "Ionization test"
    
    # K2t = decay.sel({'K':2,'Isotope':'129Xe'}).squeeze()
    # K2t = K2t * xr.DataArray([20,-10],coords=[("l",[0,2])]) + xr.DataArray([1,-1],coords=[("l",[0,2])])
    # K2t.name = "Ionization test"

    return modelOut
                                           

def calcAdvFitModel(params, **kwargs):
    """
    Wrap basic fit model, and add some features.
    
    Note passed params expect lmfit Parameters() object, or dictionary.
    
    """
    
    if isinstance(params, lmfit.parameter.Parameters):
        paramDict = params.valuesdict()
    else:
        paramDict = params
    
    #*** Run basic case as base
    # TODO: return modelDA here too
    # TODO: arg passing
    # xDataBasic = xData[0:4]  # For basic case only use first 4 (splitting) params
    # xDataBasic = [paramDict[k] for k,v in paramDict.items() if k.startswith('s')]  # For dict case, use s0...3
    xDataBasic = [paramDict[f"s{n}"] for n in range(0,4)]  # Assume number of params and enforce ordering. 
    calcBasicDict = calcBasicFitModel(xDataBasic, fitFlag=False, returnType='full', **kwargs)
    
    # Use original model results and apply additional params
    # modelDA = stackModelToDA(calcDict['modelDict'])  # Use original model?
    isoDA = calcBasicDict['modelDA'].sel({'Isotope':['129Xe','131Xe']})  # Use isotope-weighted results
    
    
    #*** Apply exponential decay (per isotope)
    # Easier way to do this...? Here set DA per isotope, then combine and multiply.
    # Quick test of multiply without all isotope dims also awkward, although maybe groupby would work...
    
    # Dict + loop version
    # decay = xr.zeros_like(isoDA)
#     decayDict = {}
#     for iso in [129,131]:
#         decayDict[iso] = np.exp(-isoDA.t/paramDict[f"tau{iso}"]).expand_dims({'Isotope':[f'{iso}Xe']})
#         # decay = decay + np.exp(-isoDA.t/paramDict[f"tau{iso}"]).expand_dims({'Isotope':[f'{iso}Xe']})
    
#     decayDA = isoDA*stackModelToDA(decayDict)
    
    # Groupby version - better...?
    # Test groupby...
    # decay = isoDA.groupby("Isotope").map(lambda x: x*np.exp(-x.t/paramDict[f"tau{x.Isotope.values.item().rstrip('Xe')}"]))
    # decay.name = 'decay'
    
    decay = calcDecays(paramDict, isoDA)

    #*** Apply ionization model
    # 
    ionization = ionizationPhenom(paramDict,decay)
    
    calcBasicDict.update({'decay':decay, 'ionization':ionization})
    
    return calcBasicDict
    
    
#*** Functions for lmfit parameters/fit setup

def initParams(xeProps):
    """
    Init lmfit Parameters() for Xe advanced hyperfine model
    
    """
    
    # Set labels for params
    pdMap = pdIndexMap(xeProps)

    # Setup parameters from df
    params = Parameters()

    # Iterate over PD rows and assign to params.
    # May be neater way to do this...?
    for item in xeProps.iterrows():
        itemVal = unumpy.nominal_values(item[1]['Splitting/cm−1'])
        params.add(item[1]['label'], value = itemVal, min = 0, max = 1)

    #*** Add additional params as required for the advanced model...
    # Lifetimes
    params.add("tau129", value = 5000, min=0, max=2e5)
    params.add("tau131", value = 5000, min=0, max=2e5)

    # *** Ionization model params 
    # Just set amplitude + offset for l=2,4 modelling
    # Will also need ROI (channel) here too...? Or just apply these to sum, if assumed iso independent
    # for iso in [129,131]:
    #     for item in ['amp','offset']:
    #         [params.add(f"l{l}_{item}_{iso}", value=1.0) for l in [2,4]]

    # ROI only version - assume isotope independent params
    # This will generate channels to match experimental data
    for ROI in [0,1]:
        for item in ['amp','offset']:
            [params.add(f"l{l}_{item}_{ROI}", value=np.random.uniform(-10,10), min=-10, max=10) for l in [2,4]]

    return params



def pdIndexMap(df):
    """
    Convert list of tuple labels to short str format from PD dataframe.
    
    Also append short names as column in df.
    
    Useful for setting up mappings to fitting params for lmfit.
    """
    
    indList = df.index
    
    # Parameter names for lmfit [a-z_][a-z0-9_]*, so replace '.' and '-' signs.
    nameList = ['_'.join(str(ele).replace('-','n').replace('.','') for ele in sub) for sub in indList]
    
    # shortNameList = [list(f"{str(ele)}_s{n}" for ele in sub) for n,sub in enumerate(indList)]
    # shortNameList = [f"I{str(sub[0])}_s{n}" for n,sub in enumerate(indList)]
    shortNameList = [f"s{n}" for n,sub in enumerate(indList)]
    
    # Generate map {full names : lables}
    indMapLong = dict(zip(indList, nameList))
    indMapShort = dict(zip(indList, shortNameList))
    
    # Append to original table
    df['label']=shortNameList
    
    return locals()


def pdParamsReplaceFromMap(df, pdMap, params, dataCol = 'Splitting/cm−1'):
    """
    Convert lmfit.params items back to original PD dataframe via labels.
    """

    dfOut = df.copy()

    for k,v in pdMap['indMapShort'].items():
        # Replace by keys
        # xeTest.loc[k][dataCol] = xeTest.loc[k][dataCol]*n

        # Replace by value (short name) lookup
        # This should ensure consistency in replacement ordering etc.
        dfOut.loc[dfOut['label']==v,dataCol] = params.valuesdict()[v]
        
        # print(params.valuesdict()[v])


    return dfOut
    
    
#*** Functions for fitting



def residualAdv(model,dataIn, dataUn = None):
    """
    Advanced model residual - use advanced model, and also uncertainties-weighted if present.
    
    """
    
    model.name = 'model'
    modelDS = splitUncertaintiesToDataset(model)
    modelDS = modelDS.assign_coords({'t':modelDS['t'].values.astype(int)})  # Force coords to avoid float comparison issues.
    
    # TODO: should check for Uncertainties data type here...?
    dataIn.name = 'data'
    dataDS = splitUncertaintiesToDataset(dataIn)
    dataDS = dataDS.assign_coords({'t':dataDS['t'].values.astype(int)})
    
    # Uncertainties - use data_std if present, skip if zero.
    # Use passed data if provided
    res = (dataDS['data'] - modelDS['model'])
    thres = 1e-10
    if (dataUn is None) and (dataDS['data_std'].max() < thres):
        pass
    elif dataDS['data_std'].max() > thres:
        res = res/dataDS['data_std']
    else:
        res = res/dataUn
        
    res.name = 'res'
    
    return res, dataDS, modelDS
    # return (dataDS-model) / uncertainty
    
    
# Test lmfit with new functions...

def calcAdvlmfit(params, trange=[0,1000], xePropsFit=None, dataDict=None):  #**kwargs):
    """
    Wrap advanced fitting model for lmfit use.
    
    """
    # print(locals().keys())
    # print(kwargs.keys())
    
    # Set dataIn...
    dataIn = dataDict['BLMall'].unstack().sel({'l':[2,4]}).copy()
    dataUn = dataDict['BLMerr'].unstack().sel({'l':[2,4]}).copy()  # Main data has uncertainties separately currently


    calcDict = calcAdvFitModel(params, xePropsFit=xePropsFit, dataDict=dataDict)
    # calcDict = calcAdvFitModel(params, **kwargs)
    
    # dataIn = dataDict['BLMall'].sel({'ROI':0,'l':4}).copy()
    # dataIn = dataDict['BLMall'].unstack().sel({'l':[2,4]}).copy()
    # dataUn = dataDict['BLMerr'].unstack().sel({'l':[2,4]}).copy()  # Main data has uncertainties separately currently

    # if trange is not None:
    #     modelIn = modelIn.sel(t=slice(trange[0],trange[1]))
    #     dataIn = dataIn.sel(t=slice(trange[0],trange[1]))

    # res = residual(calcDict['ionization'].squeeze(), dataIn.squeeze())
    # res.name = 'residual'

    res, dataDS, modelDS = residualAdv(calcDict['ionization'].squeeze(), dataIn.squeeze(), dataUn = dataUn)

    # Optionally set trange
    # NOTE: may also be set in calcBasicFitModel, so should be more careful here!
    if trange is not None:
        res = res.sel(t=slice(trange[0],trange[1]))
        # dataIn = dataIn.sel(t=slice(trange[0],trange[1]))
    
    return res.values