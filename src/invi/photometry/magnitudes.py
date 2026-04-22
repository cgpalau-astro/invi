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

import fnc as _fnc

import fnc.utils.decorators as _decorators

__all__ = ["m_to_M", "M_to_m", "m_to_m_obs",
           "m_M_to_d", "mM_to_d", "d_to_mM", "std_d",
           #--------------------------------
           "BPRP_to_Teff",
           "Teff_to_BPRP",
           "BV_to_BPRP",
           "BV_to_BPRP_approx",
           "BPRP_to_BV",
           "VI_to_BPRP",
           "BPRP_to_VI",
           #--------------------------------
           "spectral_type",
           #--------------------------------
           #Gaia <-> DESI
           "g_to_r","BPRP_to_GR",
           "flux_to_mag", "flux_inv_var_to_mag_std",
           #--------------------------------
           #Gaia fluxes to magnitudes
           "flux_to_m", "flux_g_to_m_g", "flux_bp_to_m_bp", "flux_rp_to_m_rp",
           "m_to_flux", "m_g_to_flux_g", "m_bp_to_flux_bp", "m_rp_to_flux_rp"]

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


def std_d(loc_m, loc_M, scale_m, scale_M):
    """Standard deviation of the distance [kpc] given the apparent m and
    absolute magnitude and their uncertainties [mag]."""
    mM_loc = loc_m - loc_M
    mM_scale = _np.sqrt(scale_m**2.0 + scale_M**2.0)

    loc = (mM_loc - 10.0)/5.0
    scale = mM_scale/5.0
    t = _np.log(10.0)
    var = _fnc.stats.norm.moment(t*2.0, loc, scale) - _fnc.stats.norm.moment(t, loc, scale)**2.0
    return _np.sqrt(var)

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

@_fnc.utils.decorators.vectorize
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


@_fnc.utils.decorators.vectorize
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

@_fnc.utils.decorators.vectorize
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


@_fnc.utils.decorators.vectorize
def BV_to_BPRP_approx(BV):
    """Approximation of BV_to_BPRP. This function is used in the reddening
    correction method."""
    BPRP = 1.429 * BV
    return BPRP


@_fnc.utils.decorators.vectorize
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

@_fnc.utils.decorators.vectorize
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


@_fnc.utils.decorators.vectorize
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

@_fnc.utils.decorators.vectorize
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

def g_to_r(x):
    """From Gaia g magnitude to DESI r magnitude.

    Note
    ----
    1)  g beyond [16.0, 21.2] is a projection."""

    #coef = _np.array([0.08028422 0.99380828]) #Polynomial fit

    coef = _np.array([-0.12301272,  1.00256271]) #Huber fit
    y = _np.polynomial.Polynomial(coef)(x)
    #sel = _np.logical_not(_fnc.numeric.within_equal(x, 16.0, 21.2))
    #y[sel] = _np.nan
    return y


def BPRP_to_GR(x):
    """From Gaia BPRP in [0.0, 2.0] colour index to DESI GR colour index."""

    #coef = np.array([-2.28804006e-01,  5.90129794e-01,  5.92467196e-01, -1.53551109e-01,
    #                 -2.34415789e+00,  5.03995862e+00, -4.59143279e+00,  2.20519961e+00,
    #                 -5.85023196e-01,  8.12996428e-02, -4.63153019e-03]) #Interval BPRP [0.0, 3.6]]

    coef = _np.array([-0.23516911,  0.75328378, -0.13990304,  0.37342144, -0.14382782]) #Interval BPRP [0.0, 2.0]
    y = _np.polynomial.Polynomial(coef)(x)
    sel = _np.logical_not(_fnc.numeric.within_equal(x, 0.0, 2.0))
    y[sel] = _np.nan
    return y


def flux_to_mag(f):
    """From flux [nano-maggies] to magnitude [mag]."""
    return 22.5 - 2.5*_np.log10(f)


def flux_inv_var_to_mag_std(f_mean, f_inv_var):
    """Flux inverse variance [nanomaggy^-2] to magnitude standard deviation [mag]."""
    f_std = _np.sqrt(1.0/f_inv_var)
    return 2.5*f_std/f_mean/_np.log(10.0)

#-----------------------------------------------------------------------------

def flux_to_m(flux, zero_point):
    """Flux [electrons/s] to magnitude [mag].

    Note
    ----
    1)  https://dc.g-vo.org/tableinfo/gaia.dr2epochflux
    2)  https://www.cosmos.esa.int/web/gaia/dr3-passbands#"""
    return -2.5*_np.log10(flux) + zero_point


def flux_g_to_m_g(flux_g):
    """Gaia G Flux [electrons/s] to Gaia G magnitude [mag]."""
    #zero-point G mag: 25.6873668671 \pm 0.0027553202 mag
    return flux_to_m(flux_g, 25.6873668671)


def flux_bp_to_m_bp(flux_bp):
    """Gaia BP Flux [electrons/s] to Gaia BP magnitude [mag]."""
    #zero-point BP mag: 25.3385422158 \pm 0.0027901700 mag
    return flux_to_m(flux_bp, 25.3385422158)


def flux_rp_to_m_rp(flux_rp):
    """Gaia RP Flux [electrons/s] to Gaia RP magnitude [mag]."""
    #zero-point RP mag: 24.7478955012 \pm 0.0037793818 mag
    return flux_to_m(flux_rp, 24.7478955012)

#-----------------------------------------------------------------------------

def m_to_flux(m, zero_point):
    """Magnitude [mag] to flux [electrons/s]."""
    return 10.0**((zero_point - m)/2.5)


def m_g_to_flux_g(m_g):
    """Gaia G magnitude [mag] to Gaia G Flux [electrons/s]."""
    #zero-point G mag: 25.6873668671 \pm 0.0027553202 mag
    return m_to_flux(m_g, 25.6873668671)


def m_bp_to_flux_bp(m_bp):
    """Gaia BP magnitude [mag] to Gaia BP Flux [electrons/s]."""
    #zero-point BP mag: 25.3385422158 \pm 0.0027901700 mag
    return m_to_flux(m_bp, 25.3385422158)


def m_rp_to_flux_rp(m_rp):
    """Gaia RP magnitude [mag] to Gaia RP Flux [electrons/s]."""
    #zero-point RP mag: 24.7478955012 \pm 0.0037793818 mag
    return m_to_flux(m_rp, 24.7478955012)

#-----------------------------------------------------------------------------
