"""
Functions for modelling hyperfine quantum beats

Adapted from the Quantum Metrology with Photoelectrons Alignment notebooks, https://github.com/phockett/Quantum-Metrology-with-Photoelectrons/blob/master/Alignment/

06/06/24, PH

"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sympy import *

from scipy.constants import hbar

from loguru import logger

# For hv plotting
from epsproc.plot import hvPlotters

# Pmm from TKQ, epsproc PKG version 23/07/24
from epsproc.calc.density import densityFromSphTensor

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


from epsproc.sphCalc import setBLMs

# Define Ylm(t,p) symbolically from Sympy
from sympy import Ynm
theta, phi = symbols("theta phi")
init_printing()

# Check np.int and replace if necessary, see https://stackoverflow.com/questions/5644836/in-python-how-does-one-catch-warnings-as-if-they-were-exceptions
# UPDATE - now just replaced np.int with int in code
# import warnings
# warnings.filterwarnings("error")

# try: 
#     np.int
# except (DeprecationWarning,AttributeError):
#     np.int = int

# # warnings.resetwarnings()
# warnings.filterwarnings("default")



#*** G parameters

def GJt(*args):
    """
    Send args to GJt() basic or list version, based on first arg type.
    """
    
    # For debug
    # print(locals()['args'])
    # print(isinstance(locals()['args'][0],list))
    
    if isinstance(locals()['args'][0],list) or isinstance(locals()['args'][0],np.ndarray):
        return GJtList(*locals()['args'])
    else:
        return GJtBasic(*locals()['args'])


# Define G(J;t)K, time-dependent part, single t or list t
# Set hbar = 1 to work in natural units
def GJtBasic(J,t,K,I,F,EF):
    """
    Compute GJt(J,t,K,I,F,EF).
    
    \begin{equation}
    G(J;t)_{K}=\frac{1}{2I+1}\sum_{F',F}(2F'+1)(2F+1)\left\{ \begin{array}{ccc}
    J & F' & I\\
    F & J & K
    \end{array}\right\} ^{2}\cos\left[\frac{(E_{F'}-E_{F})t}{\hbar}\right]
    \end{equation}
    
    """
    
    # Set for single or array t
    if type(t) is int:
        G = 0
    else: 
        G = np.zeros(t.shape[0])

    # Loop over pairs from a list of F states and energies
    for n1 in range(0,len(F),1):
        for n2 in range(0,len(F),1): 
            Gterm = (2*F[n2]+1)*(2*F[n1]+1)*(wigner_6j(J,F[n2],I,F[n1],J,K)**2)*cosLocal((EF[n2] - EF[n1])*t)
            
            G = np.add(G,Gterm)  # Allows for vector G addition
    
    return G*(1/(2*I+1))


# (re)Define G(J;t)K, time-dependent part.
# This version uses a list for the parameter set, [J, I, F, EF]
def GJtList(JFlist,K,t):
    """
    Compute GJt(J,t,K,I,F,EF) as GJtList(JFlist,K,t)
    This version uses a list for the parameter set, JFlist=[J, I, F, EF]
    
    \begin{equation}
    G(J;t)_{K}=\frac{1}{2I+1}\sum_{F',F}(2F'+1)(2F+1)\left\{ \begin{array}{ccc}
    J & F' & I\\
    F & J & K
    \end{array}\right\} ^{2}\cos\left[\frac{(E_{F'}-E_{F})t}{\hbar}\right]
    \end{equation}
    
    """
    
    if type(t) is int:
        G = 0
    else: 
        G = np.zeros(t.shape[0])
    
    # Set params assumed to be universal
    J = JFlist[0][0]
    I = JFlist[0][1]
    
    # Check if uncertainties are set
#     if isinstance(JFlist[0][3],uncertainties.core.AffineScalarFunc):
#         cosFunc = 

    for n1 in range(0,JFlist.shape[0],1):
        for n2 in range(0,JFlist.shape[0],1): 
            
            # Calculate (2*Fp+1)*(2*F+1)*(wigner_6j(J,Fp,I,F,J,K)**2)*cos((EFp - EF)*t/hbar) using terms from input list
            Gterm = (2*JFlist[n2][2]+1)*(2*JFlist[n1][2]+1)*(wigner_6j(J,JFlist[n2][2],I,JFlist[n1][2],J,K)**2)*cosLocal(((JFlist[n2][3] - JFlist[n1][3])*t)/hbar)
            
            G = np.add(G,Gterm)
                
    
    return G*(1/(2*I+1))




#*** State multipole definitions

# T(J)KQ and associated defns from Alignment 1 notebook
# https://github.com/phockett/Quantum-Metrology-with-Photoelectrons/blob/master/Alignment/Alignment-1.ipynb

# from sympy.physics.wigner import wigner_3j

# Define T(J,J')KQ matrix elements. 
# Eqn 4.9 in Blum (p118), note slight differences to eqn. 55, Zare, p236 - likely equivalent for J=Jp
# TODO: implement switch/dictionary for versions
def TjjpkqMatEle(Jp,J,K,Q,ver=0):
    """
    Compute TjjpkqMatEle(Jp,J,K,Q,ver=0)
    
    Define T(J,J')KQ matrix elements. 
    
    \begin{equation}
    \langle J'M'|\hat{T}(J',J)_{KQ}|JM\rangle=(-1)^{J'-M'}(2K+1)^{1/2}\left(\begin{array}{ccc}
    J' & J & K\\
    M' & -M & Q
    \end{array}\right)
    \end{equation}
    
    # Define T(J,J')KQ matrix elements. 
    # Eqn 4.9 in Blum (p118), note slight differences to eqn. 55, Zare, p236 - likely equivalent for J=Jp
    # TODO: implement switch/dictionary for versions
    
    """
    
    Jmax = max(J,Jp)
    TKQmm = np.zeros((2*Jmax+1,2*Jmax+1))
    
    for M in range(-J,J+1,1):
        for Mp in range (-Jp,Jp+1,1):
            
            # TKQmm[M+J][Mp+J] = (-1)**(J-M)*clebsch_gordan(J,J,K,Mp,-M,Q) # T(J)KQ, Zare eqn. 55
            # TKQmm[M+J][Mp+J] = (-1)**(J-Mp)*clebsch_gordan(J,J,K,Mp,-M,Q) # T(J)KQ dagger, Zare eqn. 62
            TKQmm[Mp+Jp][M+J] = (-1)**(Jp-Mp)*sqrt(2*K+1)*wigner_3j(Jp,J,K,Mp,-M,-Q)
            
            
    return TKQmm  

# TKQs state multipoles
# Eqn. 4.31 in Blum (p124) - cf. eqn. 62 in Zare, p237.
# NOTE - this is <T^dagger>
# Assume isotropic distribution, or state following a 1-photon transition if p is passed
def TKQpmm(Jp, J, Ji = 0, p = None):
    """
    Compute TKQpmm(Jp, J, Ji = 0, p = None)
    
    \begin{equation}
    \left\langle T(J',J)_{KQ}^{\dagger}\right\rangle =\sum_{M'M}\langle J'M'|\hat{\rho}|JM\rangle(-1)^{J'-M'}(2K+1)^{1/2}\left(\begin{array}{ccc}
    J' & J & K\\
    M' & -M & -Q
    \end{array}\right)
    \end{equation}
    
    # TKQs state multipoles
    # Eqn. 4.31 in Blum (p124) - cf. eqn. 62 in Zare, p237.
    # NOTE - this is <T^dagger>
    # Assume isotropic distribution, or state following a 1-photon transition if p is passed
    
    """
    
    # Determine density matrix - following 1-photon excitation to max(J,Jp)
    if p is not None:
        pmm = pmmCalcDiag(Ji,max(J,Jp),p) 
        # pass
    else:
    # Density matrix for isotropic ensemble
        pmm = np.eye(2*Jp+1,2*J+1)
    
    # Calculate T(Jf,K,Q) for pmm
    TKQ = []  # Native list to hold results
    thres = 1E-5
    Kmax = 2*max(J,Jp)+1
    
    for K in range(0,Kmax+1):
        for Q in range(0,K+1):
            KQmat = TjjpkqMatEle(Jp,J,K,Q) * pmm  # Array-wise multiplication of matrix elements
            TKQval = KQmat.sum()
            
            if np.abs(TKQval) > thres:
                TKQ.append([K,Q,KQmat.sum()])  # Store value if > threshold
            
    return np.array(TKQ)  # Convert to np array for later use


# Define T(J;t)KQ
def TJtKQ(JFlist,TKQ,t):
    """
    Compute TJtKQ(JFlist,TKQ,t)
    
    For the case of quantum beats from a manifold of (hyperfine) states,
    the state multipoles can be expressed as a product of an initial
    state, and time-dependent coefficients, as per Eqns. 4.131 and 4.134
    in Blum [1]:

    \begin{equation}
    \langle T(J;t)_{KQ}^{\dagger} \rangle =G(J;t)_{K}\langle T(J)_{KQ}^{\dagger}\rangle
    \end{equation}
    
    """
    
    Kref = -1
    TJt = []
    
    # Loop over initial TKQ values & calculate time-dependence
    for row in range(0,TKQ.shape[0],1):
        K = TKQ[row][0]
        Q = TKQ[row][1]
        
        # Check if GKvec already calculated for given K (independent of Q), assuming TKQ is ordered by K values
        if K != Kref:
            GKvec = GJt(JFlist,K,t)
            tempT = np.zeros(t.shape[0]) 
            Kref = K
        else:
            pass
        
        TJt.append(GKvec*TKQ[row][2])  # Set G*T value
        
    return np.array(TJt)


#**** Density matrix definitions

# Define 1-photon density matrix (final m-states), no frame rotation (diagonal)
# from sympy.physics.wigner import wigner_3j

def pmmCalcDiag(Ji,Jf,p):
    """
    Compute pmmCalcDiag(Ji,Jf,p)

    Defines a 1-photon density matrix (final m-states), no frame rotation (diagonal)
    
    The corresponding density matrix is proportional to the angular momentum coupling coefficient (see Sect. 7 in Blum [1]; also Sect. 3.1.1 and Eqn. 3.5 in Hockett [3], and Reid et. al. [4]):

    \begin{equation}
    \boldsymbol{\rho}(J_f)_{M',M}\propto\sum_{M_{g}}\left(\begin{array}{ccc}
    J_{i} & 1 & J_{f}\\
    -M_{i} & q & M_{f}
    \end{array}\right)^{2}
    \end{equation}
    
    Where it has been assumed that the initial state $J_g$ is isotropic.

    The properties of the final state $M$-level distribution will then depend on the transition ($\Delta J$) and the polarization of the light ($q$).
    
    """
    
    pmm = np.zeros((2*Jf+1,2*Jf+1))
    
    for Mf in range(-Jf,Jf+1,1):
        for Mi in range(-Ji,Ji+1,1):
            pmm[Mf+Jf][Mf+Jf] += wigner_3j(Ji,p[0],Jf,-Mi,p[1],Mf)**2
                
    return pmm   



# Define density matrix p(Jp,Np,J,N) from TKQ - general version, eqn. 4.34 in Blum (p125)
# Uses TKQ tensor values (list)
def pJpNpJN(Jp,J,TKQ):
    """
    Compute pJpNpJN(Jp,J,TKQ)
    
    \begin{equation}
    \langle J'N'|\hat{\rho}|JN\rangle=\sum_{N'N}(-1)^{J'-N'}(2K+1)^{1/2}\left(\begin{array}{ccc}
    J' & J & K\\
    N' & -N & -Q
    \end{array}\right)\left\langle T(J',J)_{KQ}^{\dagger}\right\rangle 
    \end{equation}
    
    # Define density matrix p(Jp,Np,J,N) from TKQ - general version, eqn. 4.34 in Blum (p125)
    # Uses TKQ tensor values (list)
    
    """
    
    Jmax = max(J,Jp)
    Pmm = np.zeros((2*Jmax+1,2*Jmax+1))
    
    for Mp in range(-Jp,Jp+1):
        for M in range(-J,J+1):
            for row in range(TKQ.shape[0]):
                Pmm[Mp+Jp][M+J] += (-1)**(Jp-Mp)*sqrt(2*TKQ[row][0]+1)*wigner_3j(Jp,J,TKQ[row][0],Mp,-M,-TKQ[row][1])*TKQ[row][2]
                
    return Pmm
 
    
# Define density matrix from existing TKQ values
# Uses existing epsproc routines
# 18/12/24 rough version from 4.05 draft notebook
def pmmFromQuantumBeat(calcDict, isoKeys = None):
    """
    Compute density matrix from TKQ tensor for quantum beat model.
    
    Set results to calcDict['pmm'] and calcDict['pmmUn'].
    
    If isoKeys = None, compute for both cases, i.e. isoKeys = ['129Xe','131Xe']
    
    """
    
    if isoKeys is None:
        isoKeys = ['129Xe','131Xe']
    
    # Loop over isotopes and compute...
    for isoKey in isoKeys:
        # Get TKQs, and convert to density matrix
        TKQ = calcDict['modelDict'][isoKey].copy()
        TKQ = TKQ.rename({'TKQ':'KQ'})
        pmm = densityFromSphTensor(TKQ)
        
        # Stash in calcDict
        if not 'pmm' in calcDict.keys():
            calcDict['pmm'] = {}
            
        calcDict['pmm'][isoKey] = pmm
    
        # May also want output from splitUncertaintiesToDataset(pmm) here...?
        # Also had this as plotter option? TBC.
        if not 'pmmUn' in calcDict.keys():
            calcDict['pmmUn'] = {}
            
        calcDict['pmmUn'][isoKey] = splitUncertaintiesToDataset(pmm)
        
    # Return calcDict for clarity, although note strictly necessary as no copy here.
    return calcDict

    
#***** Spherical harmonic functions
    
# Spatial distribution of J-vectors from T(J)KQ
# Use eqn. 101 in Blum (p148)

def calcW(TKQ, J = None, norm = 1.0,
          isoKey=None):
    """
    Compute real-space W(theta,phi) representation, spatial dist from TKQ. (Sum Ylm from a list, inc. ang mom coupling and optional normalisation.)
    
    The spatial representation of the ensemble can be defined in terms
    of the state multipoles - hence the name - by expanding in a suitable
    basis, usually the spherical harmonics. For example, for a single
    angular momentum state $J$, this is given by (Eqn. 4.101 in Blum):

    \begin{equation}
    W(\theta,\phi)=\left(\frac{1}{4\pi}\right)^{1/2}\sum_{KQ}(-1)^{J}(2J+1)^{1/2}\left(\begin{array}{ccc}
    J & J & K\\
    0 & 0 & 0
    \end{array}\right)\left\langle T(J)_{KQ}^{\dagger}\right\rangle Y_{KQ}(\theta,\phi)
    \end{equation}
    
    Parameters
    ----------
    
    TKQ : list, Numpy array, Xarray or dictionary
        $T_{K,Q}$ parameters to expand.
        Backend will use functions `sphSumTKQ` or `sphSumTKQX` depending on datatype.
        
        - For Xarray, assumes TKQ as set by `computeModel()` function, without uncertainties. 
        - If dictionary, should be `calcDict` format as set by `computeModel()` or `calcAdvFitModel()` functions. TKQ data from `calcDict['modelDict'][isoKey]`
        
    J : int or float, default = None
        J value to use.
        If None, will use `calcDict['modelDict'][isoKey].attrs['states']['Jf']` for calcDict case, or skip calc otherwise.
        
    Returns
    -------
    W : list or Xarray
        Datatype depends on input.
        list for legacy inputs.
        Xarray for updated inputs 
    
    """
    skipCalc = False
    
    # Set default isoKey, but only used for dict case.
    if isoKey is None:
        isoKey = '129Xe' #'131Xe'
        
    if isinstance(TKQ, xr.DataArray):
        try:
            # Check for uncertainties.
            # NOTE: should also be able to check TKQ.attrs['uncertainties']...? May not be updated?
            if TKQ.dtype != float:
                logger.info(f"TKQ datatype not recognised, please pass Xarray or list without uncertainites.")
                skipCalc=True

        except:
            logger.info(f"TKQ datatype can't be determined, please pass Xarray or list without uncertainites.")
            skipCalc=True
            
        if not skipCalc:
            if J is None:
                J = TKQ.attrs['states']['Jf']
                
            logger.info(f"TKQ Xarray data passed, processing TKQ from with J={J}.")
                
            return sphSumTKQX(TKQ, J, norm)
        
    # Dataset case - TODO, add option to simply pass isoKey here?
    # UPDATE: now set if passing dict datatype
    elif isinstance(TKQ, xr.Dataset):
        logger.info(f"TKQ xr.Dataset found, please pass xr.DataArray or list without uncertainites.")
        
    elif isinstance(TKQ,dict):
        if J is None:
            J = TKQ['modelDict'][isoKey].attrs['states']['Jf']
        
        logger.info(f"calcDict passed, processing TKQ from calcDict['modelDict'][{isoKey}], J={J}.")
        
        # Pull TKQ from calcDict
        TKQin = TKQ['modelDict'][isoKey].copy()

        # Split out uncertainties, not supported in calcW() currently
        TKQun = splitUncertaintiesToDataset(TKQin)

        # Calculate
        W = calcW(TKQun[isoKey],J)
        
        return W
        
    else:
        return sphSumTKQ(TKQ, J, norm)
    

# Function to sum Ylm from a list, with optional normalisation.
# Include additional 3j term to implement eqn. 101, for real-space W(theta,phi) representation.
def sphSumTKQ(A, J, norm = 1.0):
    """
    Compute real-space W(theta,phi) representation, spatial dist from TKQ. (Sum Ylm from a list, inc. ang mom coupling and optional normalisation.)
    
    The spatial representation of the ensemble can be defined in terms
    of the state multipoles - hence the name - by expanding in a suitable
    basis, usually the spherical harmonics. For example, for a single
    angular momentum state $J$, this is given by (Eqn. 4.101 in Blum):

    \begin{equation}
    W(\theta,\phi)=\left(\frac{1}{4\pi}\right)^{1/2}\sum_{KQ}(-1)^{J}(2J+1)^{1/2}\left(\begin{array}{ccc}
    J & J & K\\
    0 & 0 & 0
    \end{array}\right)\left\langle T(J)_{KQ}^{\dagger}\right\rangle Y_{KQ}(\theta,\phi)
    \end{equation}
    
    """
    
    Atp = 0
    thres = 1E-5
    
    # Loop over rows in input & add YKQ terms (should be able to convert to list comprehension for brevity)
    for row in range(A.shape[0]):  
        if np.absolute(A[row][2]) > thres:
            angMomTerm = (-1)**J * (2*J+1) * wigner_3j(J,J,A[row][0],0,0,0)
            Atp += angMomTerm*Ynm(int(A[row][0]),int(A[row][1]),theta,phi) * A[row][2]/norm # Add TKQ*Y(K,Q) term
            
    return Atp*sqrt(1/(4*pi))


# Function to sum Ylm from a list, with optional normalisation.
# Include additional 3j term to implement eqn. 101, for real-space W(theta,phi) representation.
# As sphSumTKQ(), but for Xarray formatted inputs. 
def sphSumTKQX(TKQ, J, norm = 1.0):
    """
    
    Compute real-space W(theta,phi) representation, spatial dist from TKQ. (Sum Ylm from a list, inc. ang mom coupling and optional normalisation.)

    NOTE: As sphSumTKQ(), but for Xarray formatted inputs. Assumes TKQ Xarray as set by `computeModel()` function, without uncertainties. 
    
    The spatial representation of the ensemble can be defined in terms
    of the state multipoles - hence the name - by expanding in a suitable
    basis, usually the spherical harmonics. For example, for a single
    angular momentum state $J$, this is given by (Eqn. 4.101 in Blum):

    \begin{equation}
    W(\theta,\phi)=\left(\frac{1}{4\pi}\right)^{1/2}\sum_{KQ}(-1)^{J}(2J+1)^{1/2}\left(\begin{array}{ccc}
    J & J & K\\
    0 & 0 & 0
    \end{array}\right)\left\langle T(J)_{KQ}^{\dagger}\right\rangle Y_{KQ}(\theta,\phi)
    \end{equation}
    
    """

    angMomTerm = []
    
    for K in TKQ.K.data:
        # angMomTerm[K] = (-1)**J * (2*J+1) * wigner_3j(J,J,K,0,0,0)
        angMomTerm.append((-1)**J * (2*J+1) * wigner_3j(J,J,K,0,0,0))

    # angMomTerm = 
    angMomXR = xr.DataArray(data = angMomTerm, coords = {"K":TKQ.K.data}, dims=["K"]) 

    W = angMomXR * TKQ.unstack() * sqrt(1/(4*np.pi))

    # Tidy up
    # FOr plotter need BLM data labels, and also ensure type(float) as above gives generic object result.
    W = W.rename({"K":"l","Q":"m"}).stack({'BLM':('l','m')}).astype(float)
    
    return W


# Define numerical sum over a list of harmonics defined symbolically.
def sphNList(Y, tList, pList=[0]):
    """
    Sum Sympy Ylm from list numerically.
    
    # Define numerical sum over a list of harmonics defined symbolically.
    
    """
    
    Ytp = []
    for t in tList:
        for p in pList:
            Ytp.append(Y.evalf(subs={theta:t,phi:p}))

    return np.array(Ytp)


# Compute states per demo notebook, https://phockett.github.io/Quantum-Beat_Photoelectron-Imaging_Spectroscopy_of_Xe_in_the_VUV/4.01_hyperfine_beats_modelling_060624.html
def computeModel(xeProps=None, tIn=None, tUn=None):
    """
    Calculate 1-photon abs. and hyperfine wavepacket evolution for 129 and 131 Xe, excitation at 133nm, per experiments in:
    
        Forbes, R. et al. (2018) ‘Quantum-beat photoelectron-imaging spectroscopy of Xe in the VUV’, Physical Review A, 97(6), p. 063417. Available at: https://doi.org/10.1103/PhysRevA.97.063417. arXiv: http://arxiv.org/abs/1803.01081, Authorea (original HTML version): https://doi.org/10.22541/au.156045380.07795038
    
    Method mainly follows the Alignment-3 notebook (https://github.com/phockett/Quantum-Metrology-with-Photoelectrons/blob/master/Alignment/Alignment-3.ipynb).
    
    Adapted to use either direct state settings for Xe (hard-coded below), or passed `xeProps` data (Pandas), as defined in qbanalysis.dataset.loadXeProps().
    
    `tUn` sets uncertainty for t-axis in calcs. If not set will default to FWHM = 170fs, sigma ~ 100fs, from experimental case.
    
    TODO: may want to implement TKQ data type in ePSproc and set via setBLMs()?
    
    """
    
    #*** Set list of states, see table 1 in ref. [4]

    # E values from cm-1 to J
    Jconv = 1.6021773E-19/8065.54429
    
    # For Xe case single J value only
    J = 1
    
    #*** Define intial & photon states
    Ji = 0  # Initial |J>
    p = (1,0)   # Coupling (photon) |1,q>

    #*** Set t-axis, in ps, if not passed
    if tIn is None:
        if unFlag:
            if tUn is None:
                tXC = 0.17   # Experimental cross-correlation = 170fs, should be FWHM... TBC...
                tUn = tXC/2*np.sqrt(2*np.log(2))  # sigma Txc - use as uncertainty on t?  ~0.1ps

            tIn = unumpy.uarray(np.arange(0,1000,5)*1e-12, tUn*np.ones(200)*1e-12)

        else:
            tIn = np.arange(0,1000,5)*1e-12
    
    #*** Direct state settings
    if xeProps is None:
        # Set states for Xe129 case
        JF129 = np.array([[1, 0.5, 0.5, 0*Jconv],[1, 0.5, 1.5, 0.2863*Jconv]])  # Differences in cm-1

        # Set states for Xe131 case
        JF131 = np.array([[1, 1.5, 0.5, 0*Jconv],[1, 1.5, 1.5, 0.0855*Jconv],[1, 1.5, 2.5, 0.2276*Jconv]])  # Differences in cm-1

        
    #*** Set states from xeProps (inc. uncertainties)
    else:
        
        # MESSY/UGLY!
        # From PD include uncertainties
        statesIn = xeProps.index[0]
        JF129 =  np.array([[J,*statesIn[1:-1], 0*Jconv],[J,*statesIn[2:], xeProps.loc[statesIn]['Splitting/cm−1']*Jconv]])


        # Set states for Xe131 case
        # JF131 = np.array([[1, 1.5, 0.5, 0*Jconv],[1, 1.5, 1.5, 0.0855*Jconv],[1, 1.5, 2.5, 0.2276*Jconv]])  # Differences in cm-1

        # From PD include uncertainties
        # TODO: fix state indexing here, need to subselect...
        JF131 = []
        for statesIn in xeProps.index[1:]:
            # JF131.append(np.array([[1,*statesIn[1:-1], 0*Jconv],[1,*statesIn[2:], xeProps.loc[statesIn]['Splitting/cm−1']*Jconv]]))

            # With unpack - works, but not quite correct for desired states
            # JF131.append([1,*statesIn[1:-1],xeProps.loc[statesIn]['Splitting/cm−1']*Jconv])

            # print(statesIn)
            I, F, Fp = statesIn[1:]  #[1:-1]
            if Fp == 1.5:
                pass
            else:
                JF131.append([J,I,F,xeProps.loc[statesIn]['Splitting/cm−1']*Jconv])

        # Tidy up...
        # Add F=1/2 as E=0
        JF131.append([J,I,0.5,0*Jconv])
        JF131 = np.array(JF131)
        # JF131 = np.array([[1, 1.5, 0.5, 0*Jconv],JF131[0:-1]])


    #*** 129Xe
    # Calculate 1-photon abs. and hyperfine wavepacket evolution

    # Set final state parameters by isotope
    JFlist = JF129
    Jf = int(JFlist[0][0]) # Final state J

    # Calculate T(J)KQ following 1-photon abs.
    TKQ = TKQpmm(Jf,Jf, Ji = Ji, p = p)
    # print(TKQ)

    # Calculate T(J;t)KQ
    TJt = TJtKQ(JFlist,TKQ,tIn)

    # Convert to Xarray
    if unFlag:
        basicXR129 = setBLMs(TJt, t=tIn/1e-12, LMLabels=TKQ[:,0:2].astype(int), dimNames=['TKQ', 't'])   # OK with uncertainties
    else:
        basicXR129 = setBLMs(TJt.astype(float), t=tIn/1e-12, LMLabels=TKQ[:,0:2].astype(int), dimNames=['TKQ', 't'])

    # Update some parameters for current case...
    basicXR129 = basicXR129.unstack('TKQ').rename({'l':'K','m':'Q'}).stack({'TKQ':('K','Q')})
    basicXR129.attrs['dataType']='TKQ'
    basicXR129.attrs['long_name']='Irreducible tensor parameters'  # Remove "long_name" attribs, can be an issue for multiple overlay plots.
    basicXR129.name = '129Xe'
    basicXR129.attrs['states'] = {'JFlist':JFlist, 'Ji':Ji, 'Jf':Jf, 'p':p}
    basicXR129.attrs['uncertainties'] = unFlag
    
    
    #*** 131Xe
    # Calculate 1-photon abs. and hyperfine wavepacket evolution

    # Set final state parameters by isotope
    JFlist = JF131
    Jf = int(JFlist[0][0]) # Final state J

    # Calculate T(J)KQ following 1-photon abs.
    TKQ = TKQpmm(Jf,Jf, Ji = Ji, p = p)

    # print(TKQ)

    # Calculate T(J;t)KQ
    TJt = TJtKQ(JFlist,TKQ,tIn)

    # Convert to Xarray
    if unFlag:
        basicXR131 = setBLMs(TJt, t=tIn/1e-12, LMLabels=TKQ[:,0:2].astype(int), dimNames=['TKQ', 't'])
    else:
        basicXR131 = setBLMs(TJt.astype(float), t=tIn/1e-12, LMLabels=TKQ[:,0:2].astype(int), dimNames=['TKQ', 't'])

    # Update some parameters for current case...
    basicXR131 = basicXR131.unstack('TKQ').rename({'l':'K','m':'Q'}).stack({'TKQ':('K','Q')})
    basicXR131.attrs['dataType']='TKQ'
    basicXR131.attrs['long_name']='Irreducible tensor parameters' # Remove "long_name" attribs, can be an issue for multiple overlay plots.
    basicXR131.name = '131Xe'
    basicXR131.attrs['abundance'] = 0.212324  # (30)
    basicXR131.attrs['states'] = {'JFlist':JFlist, 'Ji':Ji, 'Jf':Jf, 'p':p}
    basicXR131.attrs['uncertainties'] = unFlag
    
    # Set natural abundances
    # Source: https://en.wikipedia.org/wiki/Isotopes_of_xenon#List_of_isotopes
    if unFlag:
        basicXR129.attrs['abundance'] = ufloat_fromstr('0.264006(82)')
        basicXR131.attrs['abundance'] = ufloat_fromstr('0.212324(30)')
    else:
        basicXR129.attrs['abundance'] = 0.264006  # (82)
        basicXR131.attrs['abundance'] = 0.212324  # (30)
    
    return {'129Xe':basicXR129, '131Xe':basicXR131}


def computeModelSum(modelDict, renormFlag = True):
    """
    Compute sum over items in modelDict, weighted by abundances.
    
    Return components dict and Xarray versions.
    """
    
    n=0
    renorm = 0
    
    for k,v in modelDict.items():
        if n==0:
            components = {'sum':xr.zeros_like(v)}
        
        components[k]=(v * v.attrs['abundance'])
        components['sum'] = components['sum'] + components[k]
        
        # Renorm by total pop
        renorm = renorm + v.attrs['abundance']
        
        n=n+1
    
    if renormFlag:
        components['sum'] = components['sum']/renorm
    
    components['sum'].name = 'sum'
    components['sum'].attrs = {'data':'sum', 'renormFlag':renormFlag, 'renorm':renorm}
    
    return components, stackModelToDA(components)


def stackModelToDA(modelDict, stackDim='Isotope'):
    """
    Stack dictionary of Xarrays to new dataarray or dataset.
    
    If stackDim is not in input arrays, it will be added.
    """
    
    # Concat from keys
    if stackDim in next(iter(modelDict.values())).dims:
        xrDA = xr.concat([modelDict[k] for k in modelDict.keys()], stackDim)
    
    # Create stackDim if missing & concat
    # NOTE: assume stackDim = key in this case.
    else:
        xrDA = xr.concat([modelDict[k].expand_dims({stackDim:[k]}) for k in modelDict.keys()], stackDim)
    
    return xrDA
        
    
# For uncertainties case, function to split XR data
def splitUncertaintiesToDataset(dataIn, setTNominal = True):
    """
    For Xarray with Uncertainties, build dataset and split on nominal and uncertainty values.
    Useful for plotting.
    
    If `setTNominal=True`, then also replace t-coordinate with `unumpy.nominal_values(t)`
    
    """
    
    # Set nominal values
    dataNom = dataIn.copy()
    dataNom.values = unumpy.nominal_values(dataIn)
    # dataNom.name = f"{dataIn.name}_nom"
    
    # Remove "long_name" attribs if present, can be an issue for multiple overlay plots.
    if 'long_name' in dataNom.attrs.keys():
        dataNom.attrs.pop('long_name')
    
    # Set uncertainties/std. devs.
    dataUn = dataIn.copy()
    dataUn.values = unumpy.std_devs(dataIn)
    dataUn.name = f"{dataIn.name}_std"
    
    # Remove "long_name" attribs if present, can be an issue for multiple overlay plots.
    if 'long_name' in dataUn.attrs.keys():
        dataUn.attrs.pop('long_name')
    
    DS = dataNom.to_dataset()
    DS = DS.assign(dataUn.to_dataset())
    
    # Replace t coords?
    if setTNominal:
        DS = DS.assign_coords({"t":unumpy.nominal_values(DS.t)})
           
    return DS


def plotHyperfineModel(dataIn, plotSpread = True,
                       overlay = None,
                       **kwargs):
    """
    Holoviews plot from model data.
    
    If data has uncertainties, extract nominal values and plot with spread (or pass `plotSpread = False` to skip).
    
    If overlay = None, overlay(['K','Q','Isotope']) will be applied to plot. Pass dims to override.
    
    kwargs are passed to hv.opts()
    """

    if overlay is None:
        overlay = ['K','Q']  # Default case, note no dim checks here.
        # if 'TKQ' in dataIn.dims:
        #     overlay = ['K','Q']
        
        # Also overlay isotope dim if present
        if 'Isotope' in dataIn.dims:
            overlay.append('Isotope')
        
    
    if unFlag:
        DS = splitUncertaintiesToDataset(dataIn)
        hvDS = hvPlotters.hv.Dataset(DS.unstack())
    # hvDS = hvDS.reduce(['component'], np.mean, spreadfn=np.std)
    # hv.Curve(errors) * hv.ErrorBars(errors)
    else:
        hvDS = hvPlotters.hv.Dataset(dataIn.unstack())
    
    
    if plotSpread:
        return hvDS.to(hvPlotters.hv.Spread, kdims = ['t']).overlay(overlay).opts(title = dataIn.name, **kwargs) * hvDS.to(hvPlotters.hv.Curve, kdims = ['t']).overlay(overlay).opts(**kwargs)
    
    else:
        # hvDS = hvPlotters.hv.Dataset(dataIn.unstack())
        return hvDS.to(hvPlotters.hv.Curve, kdims = ['t']).overlay(overlay).opts(title = dataIn.name, **kwargs)
    

    
# def plotHyperfineModelComparison(