"""
Download Xe dataset from OSF
06/06/24

Adapted from basic CCDS code templates, see https://cookiecutter-data-science.drivendata.org/all-options/#include-code-scaffold

"""

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

# For OSF downloads
import osfclient
import shutil
from qbanalysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def getProjectFile(project,dataPath,dataFile, overwrite=False):
    """
    Get project file from OSF repository
    
    Basic wrapper to implement osfclient fetch method (similar to https://osfclient.readthedocs.io/en/latest/_modules/osfclient/cli.html#fetch), but with dir creation + file unzipper.
    
    Inputs
    ------
    
    project : str
        OSF project ID.
        
    dataPath : str or Path object
        Dir for downloaded file.
        Will be created (with parents) if missing.
        
    dataFile : str or Path object
        Filename for file to download.
        Files in the OSF project will be parsed to find a match.
        
    overwrite : bool, optional, default = False
        If False, skip download if file exists.
    
    Returns
    -------
    dictionary
        Dictionary with parameters and filelist.
        
    """
    
    # Check Path wrappers OK
    dataPath = Path(dataPath)
    dataFile = Path(dataFile)
    
    # Set project
    # osfclient.cli.list_(project)
    osfProj = osfclient.OSF()
    projInstance = osfProj.project(project)
    projURL = f"https://osf.io/{project}/"
    logger.info(f"Found OSF project: {projInstance.title}, {projURL} .")

    # Get files
    projStore = projInstance.storage(provider='osfstorage')

    # Create destination dir if missing
    if not dataPath.exists():
        dataPath.mkdir(parents=True)
        logger.info(f"Created destination dir {dataPath}.")
    
    # Get single file from OSF
    # Adapted from https://osfclient.readthedocs.io/en/latest/_modules/osfclient/cli.html#fetch
    # Otherwise doesn't seem to be an option for this aside from CLI?
    localPath = dataPath.expanduser()/dataFile
    
    downloadFlag = True
    
    # Check if path exists, and skip download in some cases.
    if localPath.exists():
        logger.info(f"Found local file at {localPath}.")
        downloadFlag = False
        
        if overwrite:
            downloadFlag = True
        else:
            logger.info(f"Skipping download, pass `overwrite=True` to redownload.")
    
    if downloadFlag:
        logger.info(f"Scanning OSF project files...")
        for n,item in enumerate(projStore.files):
            if item.name == dataFile.as_posix():
                logger.info(f"Downloading {dataFile} (index n={n})...")

                with open(localPath, 'wb') as fp:
                    item.write_to(fp)

                logger.success(f"Downoaded data file to {localPath}.")
                break

        # Unzip dataFile
        if dataFile.suffix == '.zip':
            logger.info(f"Unzipping {dataFile}.")
            shutil.unpack_archive(localPath, dataPath.expanduser())

    # Check files
    fList = sorted(dataPath.expanduser().glob('*'))

    # Print file names only
    fNames = [item.name for item in fList]
    
    # Set dict for return
    projDict = {'project':project,
             'name':projInstance.title,
             'URL':projURL,
             'dataPath': dataPath,
             'dataFile': dataFile,
             'fullPath': localPath,
             'fileList': fList,
             'fileNames': fNames
            }

    return projDict


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    dataPath: Path = RAW_DATA_DIR,
    dataFile: Path = "Xe_hyperfine_VMI_processing_distro_211217.zip",
    project: str = 'ds8mk',
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    
    
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
    # -----------------------------------------
    
    fDict = getProjectFile(project,dataPath,dataFile)
    

    return fDict


if __name__ == "__main__":
    app()