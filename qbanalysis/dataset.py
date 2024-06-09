"""
Load Xe datasets
06/06/24

Adapted from basic CCDS code templates, see https://cookiecutter-data-science.drivendata.org/all-options/#include-code-scaffold

"""

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from qbanalysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# For data IO
from scipy.io import loadmat

# Processing...
import numpy as np
import xarray as xr
import pandas as pd
from uncertainties import ufloat_fromstr
from epsproc.sphCalc import setBLMs

# For renorm
from epsproc.util.conversion import conv_BL_BLM

def loadXeProps(dataPath):
    """
    Load some atomic properties.
    
    """
    
    # Load hyperfine spectroscopy results.
    # As Table 1 in manuscript
    # Note - may also need to force dtypes here...?
    rawXeHyperfineResults = pd.read_excel(dataPath, sheet_name=1)

    # Tidy up
    # Lambda map...
    # Works for sub-selected cols
    # rawXeHyperfineResults[['A/MHz', 'B/MHz']].apply(lambda x: x.str.replace(' ',''))

    # Fails for full DF?
    # rawXeHyperfineResults.apply(lambda x: x.str.replace(' ','') if isinstance(x, str) else x, axis=1)

    # Applymap works overall - works elementwise.
    tidied = rawXeHyperfineResults.applymap(lambda x: x.replace(' ','') if isinstance(x, str) else x)
    tidied = tidied.replace('-','nan(nan)')  # For uncertainties defn.
    
    # Convert to Uncertainties type
    uList = ['A/MHz','B/MHz','Splitting/cm−1']
    tidied[uList] = tidied[uList].applymap(lambda x: ufloat_fromstr(x))  # OK
    
    # Set index
    tidied.set_index(['Isotope','I','F'], inplace=True)
    
    return tidied  #, rawXeHyperfineResults



def loadFinalDataset(dataPath):
    """
    Load final datasets only.
    
    Convert mat files to Xarrays.
    
    """
    
    # Load final dataset only
    filesIn =['cpBasex_results_cycleSummed_rot90_quad1_ROI_results_with_FT_NFFT1024_hanningWindow_270717.mat',
              'cpBasex_results_allCycles_ROIs_with_FTs_NFFT1024_hanningWindow_270717.mat']

    dataIn = loadDataset(dataPath, filesIn)
    
    # Process
    # Set BLM values...
    # Here from main dataset, n labels ROI
    #
    # TODO: neater stacking from dict with ep.util.conversion.datasetStack...?

    # Conversion options
    addXS = True
    renorm = True

    # Config data
    t=np.arange(-70,900,10) # t in ps

    dataROI = dataIn[0]['data']['dataROI']
    dataROIerr = dataIn[1]['data']['dataROI_cycles']

    # For errors...
    # err=[std(dataROIerr(n).beta(1).bCycle,0,2) std(dataROIerr(n).beta(2).bCycle,0,2)];

    BLMarr = []
    BLMdict = {}

    BLMerrArr = []
    BLMerrCycleArr = []

    # WEIRD - working in tests below for both n, but in loop only get 1 returned..?
    # Issue with data index or accidental overwrite...?
    # AHAH - issue was 'BLMX.expand_dims({'ROI':n})' for n=0, sets null dim.
    # SHOULD BE 'BLMX.expand_dims({'ROI':[n]})' for dim with labels.
    # for n in range(0,2):
    # #     norm=np.sum(dataROI[0,n]['specSum'])
    #     print(n)
    #     # Set B2,B4 from structure.
    #     BLM = np.array([dataROI[0,n]['beta']['bSum'][0][0], 
    #                     dataROI[0,n]['beta']['bSum'][0][1]])

    #     # Pass to ep...
    #     BLMX = setBLMs(BLM.squeeze(), t=t, LMLabels=np.array([[2,0],[4,0]]))
    #     BLMX.name = f'BLM input {n}'
    #     BLMX = BLMX.expand_dims({'ROI':n})

    #     BLMdict[n] = BLMX.copy()
    # #     BLMarr.append(BLMX.copy())

    #     BLMarr.append(BLMX.copy())

    for n in range(0,2):
        # ROI data
        BLM = np.array([dataROI[0,n]['beta']['bSum'][0][0], 
                        dataROI[0,n]['beta']['bSum'][0][1]])

        BLMX = setBLMs(BLM.squeeze(), t=t, LMLabels=np.array([[2,0],[4,0]]), addXS = addXS)

        BLMX = conv_BL_BLM(BLMX, to='sph', renorm=renorm)  # Convert BL > BLM

        BLMX.name = f'BLM input {n}'

        BLMX = BLMX.expand_dims({'ROI':[n]})

        BLMdict[n] = BLMX.copy()
        BLMarr.append(BLMX.copy())

        # Errors from per-cycle data
        # TODO: update setBLMs for arb dim handling? Would like to keep cycles here.
        BLMerr = np.array([dataROIerr[0,n]['beta']['bCycle'][0][0].std(axis=1),
                           dataROIerr[0,n]['beta']['bCycle'][0][1].std(axis=1)])

        BLMX = setBLMs(BLMerr.squeeze(), t=t, LMLabels=np.array([[2,0],[4,0]]), addXS = addXS)

        BLMX = conv_BL_BLM(BLMX, to='sph', renorm=renorm)  # Convert BL > BLM

        BLMX.name = f'BLM err {n}'

        BLMX = BLMX.expand_dims({'ROI':[n]})

    #     BLMdict[n] = BLMX.copy()
        BLMerrArr.append(BLMX.copy())

        #*** Try per cycle (ugly way)...
        BLMcycleArr = []
        for cycle in range(0,dataROIerr[0,n]['beta']['bCycle'][0][0].shape[1]):
            BLMcycle = np.array([dataROIerr[0,n]['beta']['bCycle'][0][0][:,cycle],
                           dataROIerr[0,n]['beta']['bCycle'][0][1][:,cycle]])

            BLMX = setBLMs(BLMcycle.squeeze(), t=t, LMLabels=np.array([[2,0],[4,0]]), addXS = addXS)
            BLMX = BLMX.expand_dims({'cycle':[cycle]})
            BLMX = conv_BL_BLM(BLMX, to='sph', renorm=renorm)  # Convert BL > BLM
            BLMcycleArr.append(BLMX.copy())

        BLMerrCycleArr.append(xr.concat(BLMcycleArr, dim='cycle').expand_dims({'ROI':[n]}))


    # Concat and plot...
    BLMall = xr.concat(BLMarr, dim='ROI')
    BLMerr = xr.concat(BLMerrArr, dim='ROI')
    BLMerrCycle = xr.concat(BLMerrCycleArr, dim='ROI')
    BLMerrCycle.name = 'BLM per cycle'
    
    logger.info(f"Processed data to Xarray OK.")
    
    return {'BLMall':BLMall,
            'BLMerr':BLMerr,
            'BLMerrCycle':BLMerrCycle,
           }
    
    
def loadDataset(dataPath, filesIn = None):
    """
    Load data via scipy.io.loadmap.
    
    Stack files to dictionary.
    
    """
    
    # Load
    dataIn = {}
    for n,item in enumerate(filesIn):
        filePath = Path(dataPath,item)
        # dataIn = loadmat(filePath.expanduser().as_posix())
        dataIn[n] = {'data':loadmat(filePath.expanduser().as_posix()),
                     'file':item}
        
        logger.info(f"Loaded data {item}.")

    return dataIn


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
