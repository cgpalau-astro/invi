"""Functions to compute the rotation angles to determine the stream reference frame (principal axes)."""

import termcolor as _tc
import numpy as _np
import scipy as _scipy

import fnc as _fnc
import invi as _invi

__all__ = ["eigenvalues", "likelihood", "print_results"]

#-----------------------------------------------------------------------------

def eigenvalues(st_dgc, varphi_x, varphi_y ,varphi_z):
    #Rotation
    st_AAF = _invi.coordinates.rotation.aaf(st_dgc, varphi_x, varphi_y, varphi_z)

    #Eigenvaules: eig_i = F_i/J_i [rad/kpc^2]
    eig_1 = st_AAF[6]/st_AAF[3]
    eig_2 = st_AAF[7]/st_AAF[4]
    eig_3 = st_AAF[8]/st_AAF[5]

    return eig_1, eig_2, eig_3


def _differential_entropy(x, method="ebrahimi"):
    """Estimate differential entropy from a uni-dimensional random sample.
    Note
    ----
    1)method: 'ebrahimi', 'correa'"""
    return _scipy.stats.differential_entropy(x, method=method)


def likelihood(x, st_dgc):
    #Rotation angles assuming a phase-space in cartesian coordinates
    varphi_x = x[0] #[rad]
    varphi_y = x[1]
    varphi_z = x[2]

    #Compute eigenvalues assuming a diagonal Hessian
    eig_1, eig_2, eig_3 = eigenvalues(st_dgc, varphi_x, varphi_y, varphi_z)

    #Evaluate numerically the entropy of the distributions of eigenvalues
    lk = _differential_entropy(eig_1) + _differential_entropy(eig_2) + _differential_entropy(eig_3)

    return lk


def print_results(st_dgc, varphi_0, results, eig):
    #---------------------------------------------------------------------
    def print_eig(eig, n):
        print(f"{n}: {_np.median(eig)*1_000.0:0.3f} ± {_fnc.stats.mad(eig)*1_000.0:0.3f}")
    #---------------------------------------------------------------------
    #Result angles
    varphi_x = results.x[0] #[rad]
    varphi_y = results.x[1]
    varphi_z = results.x[2]

    _tc.cprint("Algorithm", 'blue')
    print(f"Likelihood [lower better]: {likelihood([varphi_x, varphi_y, varphi_z], st_dgc)}")
    print(f"Evaluations: {results.nfev:_}")

    print("-"*51)
    _tc.cprint("Angles [rad]", 'blue')
    print(f"varphi_0: {varphi_0[0]:0.10f}, {varphi_0[1]:0.10f}, {varphi_0[2]:0.10f}")
    _tc.cprint(f"Result  : {varphi_x:0.10f}, {varphi_y:0.10f}, {varphi_z:0.10f}", 'green')

    print("-"*51)
    _tc.cprint("Eigenvalues: median ± mad [mrad/kpc^2]", 'blue')
    print_eig(eig[0], 1)
    print_eig(eig[1], 2)
    print_eig(eig[2], 3)

    print("-"*51)
    _tc.cprint("Eigenvalues ratio: |median|", 'blue')
    print(f"eig_1/eig_2 = {_np.median(_np.abs(eig[0]/eig[1])):0.3f}")
    print(f"eig_1/eig_3 = {_np.median(_np.abs(eig[0]/eig[2])):0.3f}")
    print("-"*51)

#-----------------------------------------------------------------------------
