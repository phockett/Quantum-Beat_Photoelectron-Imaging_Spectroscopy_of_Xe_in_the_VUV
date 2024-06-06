"""
Functions for modelling hyperfine quantum beats

Adapted from the Quantum Metrology with Photoelectrons Alignment notebooks, https://github.com/phockett/Quantum-Metrology-with-Photoelectrons/blob/master/Alignment/

06/06/24, PH

"""


import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.physics.wigner import wigner_3j, wigner_6j
from scipy.constants import hbar

# Define Ylm(t,p) symbolically from Sympy
from sympy import Ynm
theta, phi = symbols("theta phi")
init_printing()


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
            Gterm = (2*F[n2]+1)*(2*F[n1]+1)*(wigner_6j(J,F[n2],I,F[n1],J,K)**2)*np.cos((EF[n2] - EF[n1])*t)
            
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

    for n1 in range(0,JFlist.shape[0],1):
        for n2 in range(0,JFlist.shape[0],1): 
            
            # Calculate (2*Fp+1)*(2*F+1)*(wigner_6j(J,Fp,I,F,J,K)**2)*np.cos((EFp - EF)*t/hbar) using terms from input list
            Gterm = (2*JFlist[n2][2]+1)*(2*JFlist[n1][2]+1)*(wigner_6j(J,JFlist[n2][2],I,JFlist[n1][2],J,K)**2)*np.cos(((JFlist[n2][3] - JFlist[n1][3])*t)/hbar)
            
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
 
    
#***** Spherical harmonic functions
    
# Spatial distribution of J-vectors from T(J)KQ
# Use eqn. 101 in Blum (p148)

# Function to sum Ylm from a list, with optional normalisation.
# Include additional 3j term to implement eqn. 101, for real-space W(theta,phi) representation.
def sphSumTKQ(A, J, norm = 1.0):
    """
    Compute sphSumTKQ(A, J, norm = 1.0)
    
    # Function to sum Ylm from a list, with optional normalisation.
    # Include additional 3j term to implement eqn. 101, for real-space W(theta,phi) representation.
    
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
            Atp += angMomTerm*Ynm(np.int(A[row][0]),np.int(A[row][1]),theta,phi) * A[row][2]/norm # Add TKQ*Y(K,Q) term
            
    return Atp*sqrt(1/(4*pi))


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