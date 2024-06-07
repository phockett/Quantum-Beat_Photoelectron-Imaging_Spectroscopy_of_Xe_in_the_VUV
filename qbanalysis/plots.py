from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from qbanalysis.config import FIGURES_DIR, PROCESSED_DATA_DIR

from epsproc.plot import hvPlotters
import numpy as np

app = typer.Typer()

def plotFinalDatasetBLMt(BLMall = None, BLMerrCycle = None, width = 800, **kwargs):
    """
    Plot final datasets with Holoviews/Bokeh.
    
    Output to match Fig. 5 in manuscript Forbes, R. et al. (2018) ‘Quantum-beat photoelectron-imaging spectroscopy of Xe in the VUV’, Physical Review A, 97(6), p. 063417. Available at: https://doi.org/10.1103/PhysRevA.97.063417. arXiv: http://arxiv.org/abs/1803.01081, Authorea (original HTML version): https://doi.org/10.22541/au.156045380.07795038
    
    Parameters
    ----------
    BLMall : Xarray
        Final dataset to plot
        
    BLMerrCycle : Xarray
        Per-cycle data, used to define uncertainties
        
    width : int, optional, default = 800
        Output plot width setting
        
    Returns
    -------
    Holoviews layout plot object.
    
    """
    
    # Quick plot - OK
    # BLMerr.unstack().squeeze().hvplot.line(x='t').overlay(['l','ROI'])

    xDim='t'
    # overlay=['l','ROI']
    overlay=['ROI']
    # Test as per methods in PEMtk.fit._plotters.BLMsetPlot
    # FAILS as is, with dims issues.
    # May need something like 'hvDS = hvDS.reduce(['Fit'], np.mean, spreadfn=np.std)'...?

    # # Convert to hv and use spread function - basic attempt
    # from epsproc.plot import hvPlotters
    # # self.hv.Dataset(daPlot.rename('BLM')) 
    # hvDS = hvPlotters.hv.Dataset(BLMerr.unstack()) 
    # hvPlot = hvDS.to(hvPlotters.hv.Spread, kdims = xDim) #.overlay(overlay)
    # hvPlot


    # Convert to hv and use spread function - using hvAgg
    # OK!
    # PLOTS, but need to fix abs values, currently offset - maybe due to use of mean here?
    # See https://holoviews.org/reference/elements/bokeh/Spread.html
    # Should pass final values here too? That should fix offset.
    # OR MAY JUST BE RENORM issue...?

    # self.hv.Dataset(daPlot.rename('BLM')) 
    hvDS = hvPlotters.hv.Dataset(BLMerrCycle.unstack().squeeze()) 
    hvDS = hvDS.reduce(['cycle'], np.mean, spreadfn=np.std)
    
    # Basic version, note mismatch with spread and values here
#     hvPlot = hvDS.to(hvPlotters.hv.Spread, kdims = xDim).overlay(overlay)
#     hvPlot * BLMall.unstack().squeeze().hvplot.line(x='t').overlay(['ROI'])

    # Ah, data is just Xarray, so can do additional renorm etc.
    # hvDS = hvPlotters.hv.Dataset(BLMerrCycle.unstack().squeeze()) 

    # hvDS.data = (hvDS.data/25) + BLMall.unstack().squeeze())
    hvDS.data['BLM per cycle'] = BLMall.unstack().squeeze()   # Replace BLM per cycle mean with BLMall data.

    hvPlot = hvDS.to(hvPlotters.hv.Spread, kdims = xDim).select(l=[2,4]).overlay(overlay)

    return (hvPlot * BLMall.unstack().squeeze().hvplot.line(x='t').overlay(['ROI'])).opts(width=width).layout('l').cols(1).opts(title='BLM(t), per Figure 5 in manuscript.')

    # This now matches Fig. 5 in published manuscript.


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
