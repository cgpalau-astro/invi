"""Photometric system conversion, magnitudes, and Hertzsprung-Russell Diagram
functions.

Note
----
1)  Absolute magnitude: M
    Apparent magnitude: m

2)  When the band is specified, e.g. Gaia G magnitude:
    Absolute magnitude: M_G, G
    Apparent magnitude: m_G, g

3)  Colour indices, e.g. Gaia BP and RP magnitudes:
    Colour index: G_BP-G_RP, BPRP, bprp

4)  Useful functions related to magnitudes:
    https://pyastronomy.readthedocs.io/en/latest/index.html"""

import warnings as _warnings
import numpy as _np

import fnc.utils.decorators as _decorators

__all__ = ["m_to_M", "M_to_m", "m_to_m_obs",
           "m_M_to_d", "mM_to_d", "d_to_mM",
           "BPRP_to_Teff",
           "Teff_to_BPRP",
           "BV_to_BPRP",
           "BV_to_BPRP_approx",
           "BPRP_to_BV",
           "VI_to_BPRP",
           "BPRP_to_VI",
           "spectral_type"]

#-----------------------------------------------------------------------------

def m_to_M(m, d):
    """Absolute magnitude M of a star of apparent magnitude m located at a
    distance d from the observer.

    Parameters
    ----------
    m:  float
        Apparent magnitude [mag]

    d:  float
        Distance [kpc]
        Range: d > 0

    Returns
    -------
    M:  float
        Absolute magnitude [mag]"""

    M = m - 5.0 * _np.log10(d) - 10.0
    return M

def M_to_m(M, d):
    """Apparent magnitude m of a star of absolute magnitude M located at a
    distance d from the observer.

    Parameters
    ----------
    M: float
        Absolute magnitude [mag]

    d:  float
        Distance [kpc]
        Range: d > 0

    Returns
    -------
    m:  float
        Apparent magnitude [mag]"""

    m = M + 5.0 * _np.log10(d) + 10.0
    return m

def m_to_m_obs(m, d_star, d_obs):
    """Given a star with apparent magnitude m located at a distance d_star,
    m_obs is its apparent magnitude when the star is located at a distance
    d_obs.

    Parameters
    ----------
    m:  float
        Apparent magnitude at d_star [mag]

    d_star: float
        Distance star [kpc]
        Range: d_star > 0

    d_obs: float
        Distance observer [kpc]
        Range: d_obs > 0

    Returns
    -------
    m_obs: float
        Apparent magnitude at d_obs [mag]"""

    m_obs = m - 5.0 * _np.log10(d_star / d_obs)
    return m_obs

#-----------------------------------------------------------------------------

def m_M_to_d(m, M):
    """Distance modulus m-M [mag] to distance [kpc]."""
    return 10.0**((m - M - 10.0)/5.0)

def mM_to_d(mM):
    """Distance modulus mM = m - M [mag] to distance [kpc]."""
    return 10.0**((mM - 10.0)/5.0)

def d_to_mM(d):
    """Distance [kpc] to distance modulus mM = m - M [mag]."""
    return 5.0*_np.log10(d) + 10.0

#-----------------------------------------------------------------------------

def _corresponding_root(a, b, c, d):
    """Corresponding root to the polynomials used in the conversion functions
    for:
    a == 0:        bx^2 + cx + d = 0
    a != 0: ax^3 + bx^2 + cx + d = 0"""
    if _np.isclose(a, 0.0):
        root = (-c + (c**2 - 4*b*d)**(1/2)) / (2*b)
    else:
        d0 = b**2 - 3*a*c
        d1 = 2*b**3 - 9*a*b*c + 27*d*a**2
        k = ((d1 + (d1**2 - 4*d0**3)**(1/2)) / 2)**(1/3)
        root = -(b + k + d0 / k) / (3*a)
    return root

#-----------------------------------------------------------------------------

@_decorators.vectorize
def BPRP_to_Teff(BPRP):
    """Gaia BPRP colour index to effective temperature in K.

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2019hsax.conf..548C/abstract
    Journal: https://www.sea-astronomia.es/sites/default/files/archivos/proceedings13/Instrumentacion/oral/carrascojm.pdf

    Parameters
    ----------
    BPRP: float
        Gaia-DR2 BPRP magnitude [mag]
        Range: -0.578763 <= BPRP <= 2.5

    Returns
    -------
    Teff: float
        Effective temperature [K]
        Range: 3_300.0 <= Teff <= 50_400.0"""

    Teff = 5040.0 / _np.polyval([0.0, -0.046397, 0.55278, 0.43547], BPRP)

    if not -0.578763 <= BPRP <= 2.5:
        _warnings.warn(f"BPRP = {BPRP} out of range: -0.578763 <= BPRP <= 2.5 [mag]")

    return Teff

@_decorators.vectorize
def Teff_to_BPRP(Teff):
    """Effective temperature in K to Gaia BPRP colour index

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2019hsax.conf..548C/abstract
    Journal: https://www.sea-astronomia.es/sites/default/files/archivos/proceedings13/Instrumentacion/oral/carrascojm.pdf

    Parameters
    ----------
    Teff: float
        Effective temperature [K]
        Range: 3_300.0 <= Teff <= 50_400.0
    Returns
    -------
    BPRP: float
        Gaia-DR2 BPRP magnitude [mag]
        Range: -0.578763 <= BPRP <= 2.5"""

    BPRP = _corresponding_root(0.0, -0.046397, 0.55278, 0.43547 - 5040 / Teff)

    if not 3_300.0 <= Teff <= 50_400.0:
        _warnings.warn(f"Teff = {Teff} out of range: 3299.6413 <= Teff <= 50_400.0 [K]")

    return BPRP

#-----------------------------------------------------------------------------

@_decorators.vectorize
def BV_to_BPRP(BV):
    """Johnson-Cousins (UBVRI) to Gaia (G, BP, RP) photometric system.

    Note
    ----
    1)  The range of validity does not appear in the reference.

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract
    arXiv: https://arxiv.org/pdf/1008.0815.pdf

    Parameters
    ----------
    BV: float
        BV magnitude [mag]
        Range: -0.3914 <= BV <= 2.0835

    Returns
    -------
    BPRP: float
        BPRP magnitude [mag]
        Range: -0.4657 <= BPRP <= 3.0138"""

    BPRP = _np.polyval([0.0061, -0.0269, 1.429, 0.0981], BV)

    if not -0.3914 <= BV <= 2.0835:
        _warnings.warn(f"BV = {BV} out of range: -0.3914 <= BV <= 2.0835 [mag]")

    return BPRP

@_decorators.vectorize
def BV_to_BPRP_approx(BV):
    """Approximation of BV_to_BPRP. This function is used in the reddening
    correction method."""
    BPRP = 1.429 * BV
    return BPRP

@_decorators.vectorize
def BPRP_to_BV(BPRP):
    """Gaia (G, BP, RP) to Johnson-Cousins (UBVRI) photometric system.

    Note
    ----
    1)  The range of validity does not appear in the reference.

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract
    arXiv: https://arxiv.org/pdf/1008.0815.pdf

    Parameters
    ----------
    BPRP: float
        BPRP magnitude [mag]
        Range: -0.4657 <= BPRP <= 3.0138

    Returns
    -------
    B:  float
        BV magnitude [mag]
        Range: -0.3914 <= BV <= 2.0835"""

    BV = _corresponding_root(0.0061, -0.0269, 1.429, 0.0981 - BPRP)

    if not -0.4657 <= BPRP <= 3.0138:
        _warnings.warn(f"BPRP = {BPRP} out of range: -0.4657 <= BPRP <= 3.0138 [mag]")

    return BV

#-----------------------------------------------------------------------------

@_decorators.vectorize
def VI_to_BPRP(VI):
    """Johnson-Cousins (UBVRI) to Gaia (G, BP, RP) photometric system.

    Note
    ----
    1)  The range of validity does not appear in the reference.

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract
    ArXiv: https://arxiv.org/pdf/1008.0815.pdf

    Parameters
    ----------
    VI: float
        VI magnitude [mag]
        Range: -0.4 <= VI <= 6.0

    Returns
    -------
    BPRP: float
        BPRP magnitude [mag]
        Range: 0.5585 <= BPRP <= 5.8458"""

    BPRP = _np.polyval([0.0041, -0.0614, 1.2061, -0.0660], VI)

    if not -0.4 <= VI <= 6.0:
        _warnings.warn(f"VI = {VI} out of range: -0.4 <= VI <= 6.0 [mag]")

    return BPRP

@_decorators.vectorize
def BPRP_to_VI(BPRP):
    """Johnson-Cousins (UBVRI) to Gaia (G, BP, RP) photometric system.

    Note
    ----
    1)  The range of validity does not appear in the reference.

    Reference
    ---------
    NASA/ADS: https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract
    ArXiv: https://arxiv.org/pdf/1008.0815.pdf

    Parameters
    ----------
    BPRP: foat
        BPRP magnitude [mag]
        Range: 0.5585 <= BPRP <= 5.8458

    Returns
    -------
    VI: float
        VI magnitude [mag]
        Range: -0.4 <= VI <= 6.0"""

    VI = _corresponding_root(0.0041, -0.0614, 1.2061, -0.0660 - BPRP)

    if not 0.5585 <= BPRP <= 5.8458:
        _warnings.warn(f"BPRP = {BPRP} out of range: 0.5585 <= BPRP <= 5.8458 [mag]")

    return VI

#-----------------------------------------------------------------------------

@_decorators.vectorize
def spectral_type(Teff):
    """Spectral types in function of the effective temperature (Teff) in K.

    Note
    ----
    1)  Spectral types K have not been checked!
    2)  The python package pyastronomy includes several functions related to
        spectral type.
        Web: https://pyastronomy.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    Teff: float
        Effective temperature [K]
        Range: 3_500.0 <= Teff <= 30_000.0

    Returns
    -------
    spts: str
        Spectral type"""

    if 30_000.0 >= Teff >= 10_000.0:
        if Teff <= 20_000.0:
            spts = 'B5V'
        else:
            spts = 'B0V'

    elif 10_000.0 >= Teff >= 7_500.0:
        if Teff <= 8_750.0:
            spts = 'A5V'
        else:
            spts = 'A0V'

    elif 7_500.0 >= Teff >= 6_000.0:
        spts = 'F0V'

    elif 6_000.0 >= Teff >= 5_200.0:
        if Teff <= 5_600.0:
            spts = 'G5V'
        else:
            spts = 'G0V'

    elif 5_200.0 >= Teff >= 3_500.0:
        if Teff <= 3_700.0 + 150.0:
            spts = 'K1IIIMP' #'K1III'
        elif 3_700.0 + 150.0 <= Teff <= 3_700.0 + 150.0 * 40:
            spts = 'K4V'
        else:
            spts = 'K0V'

    else:
        _warnings.warn(f"Teff = {Teff:_} out of range: 3_500 <= Teff <= 30_000 [K]")
        spts = _np.nan

    return spts

#-----------------------------------------------------------------------------
