"""Functions to select DESI stars."""

import numpy as _np

import invi as _invi
import fnc as _fnc

_scipy = _fnc.utils.lazy.Import("scipy")

__all__ = ["linear_regression", "determination_eps", "linear_intv_cut", "metallicity_cut",
           "final_selection"]

#-----------------------------------------------------------------------------

def linear_regression(x, y, x0, f_scale):

    def pol(coef, x, y):
        return coef[0] + coef[1]*x - y

    not_nan = _np.isfinite(x) & _np.isfinite(y)
    res = _scipy.optimize.least_squares(pol, x0=x0, args=(x[not_nan], y[not_nan]), loss='huber', f_scale=f_scale)
    coef = res.x

    return coef


def determination_eps(x, y, coef, x0, sigma=2):

    def fun(eps, x, y, coef, sigma):
        sel = linear_intv_cut(x, y, coef, eps[0])
        n = len(sel[sel])
        n_tot = len(sel)
        return (n/n_tot - _fnc.stats.norm.sigma_level(sigma))**2.0 #sigma=2 : 95 per cent level

    res = _scipy.optimize.minimize(fun, x0, method='Nelder-Mead', args=(x, y, coef, sigma), tol=1E-12)
    eps = res.x[0]

    return eps


def linear_intv_cut(x, y, coef, eps):

    y_sup = _np.polynomial.Polynomial(coef)(x) + eps
    y_inf = _np.polynomial.Polynomial(coef)(x) - eps

    return _fnc.numeric.within_equal(y, y_inf, y_sup)

#-----------------------------------------------------------------------------

def metallicity_cut(FeH, AlphaFe, coef, eps):
    return linear_intv_cut(FeH, AlphaFe, coef, eps) & _fnc.numeric.within(AlphaFe, 0.0, 1.2)

#-----------------------------------------------------------------------------

def final_selection(cm, prm):
    #Phase-space
    alpha = _np.asarray(cm['ICRS']['alpha'])
    delta = _np.asarray(cm['ICRS']['delta'])
    phi_1 = _np.asarray(cm['phi']['1'])
    phi_2 = _np.asarray(cm['phi']['2'])
    mu_r = _np.asarray(cm['ICRS']['mu_r_desi'])

    #Magnitudes and reddening correction
    GR_red = _np.asarray(cm['photometry']['GR_red_desi_orb_est'])
    GR = _invi.photometry.reddening.correction_DESI(delta, alpha, GR_red, correction=prm['M68']['mock']['cmd37_correction'])
    R = _np.asarray(cm['photometry']['R_desi_orb_est'])

    #Metallicities
    FeH = _np.asarray(cm['FeH'])
    AlphaFe = _np.asarray(cm['AlphaFe'])

    #Metallicity cut
    coef, eps = prm['M68']['selection']['desi']['metallicity'].values()
    sel_met = _invi.selection.desi.metallicity_cut(FeH, AlphaFe, coef, eps)

    #Phi_1 and radial velocity cut
    coef, eps = prm['M68']['selection']['desi']['phi_vr'].values()
    sel_phi_1_vr = _invi.selection.desi.linear_intv_cut(phi_1, mu_r, coef, eps)

    #Phi_2 cut
    sel_phi_2 = _fnc.numeric.within_equal(phi_2, -13.0, -5.0) #[deg]

    #CMD cut
    points_ms = _np.array(prm['M68']['selection']['desi']['cmd']['main_seq']).T
    #points_hb = _np.array(prm['M68']['selection']['desi']['cmd']['horizontal_branch']).T
    #points_rg = _np.array(prm['M68']['selection']['desi']['cmd']['red_giants']).T

    sel_cmd = _invi.misc.polygon_selection(GR, R, points_ms) #| _invi.misc.polygon_selection(GR, R, points_hb) | _invi.misc.polygon_selection(GR, R, points_rg)

    #Total selection
    sel = sel_met & sel_cmd & sel_phi_1_vr & sel_phi_2

    return sel

#-----------------------------------------------------------------------------

def final_selection_gaia_photometry(cm, prm):
    #Phase-space
    alpha = _np.asarray(cm['ICRS']['alpha'])
    delta = _np.asarray(cm['ICRS']['delta'])
    phi_1 = _np.asarray(cm['phi']['1'])
    phi_2 = _np.asarray(cm['phi']['2'])
    mu_r = _np.asarray(cm['ICRS']['mu_r_desi'])

    #Magnitudes and reddening correction
    BPRP = _np.asarray(cm['photometry']['BPRP_orb_est'])
    G = _np.asarray(cm['photometry']['G_orb_est'])

    #Metallicities
    FeH = _np.asarray(cm['FeH'])
    AlphaFe = _np.asarray(cm['AlphaFe'])

    #Metallicity cut
    coef, eps = prm['M68']['selection']['desi']['metallicity'].values()
    sel_met = _invi.selection.desi.metallicity_cut(FeH, AlphaFe, coef, eps)

    #Phi_1 and radial velocity cut
    coef, eps = prm['M68']['selection']['desi']['phi_vr'].values()
    sel_phi_1_vr = _invi.selection.desi.linear_intv_cut(phi_1, mu_r, coef, eps)

    #Phi_2 cut
    sel_phi_2 = _fnc.numeric.within_equal(phi_2, -13.0, -5.0) #[deg]

    #CMD cut
    points_ms = _np.array(prm['M68']['selection']['polygon']['main_seq']).T
    #points_hb = _np.array(prm['M68']['selection']['polygon']['horizontal_branch']).T
    #points_rg = _np.array(prm['M68']['selection']['polygon']['red_giants']).T

    sel_cmd = _invi.misc.polygon_selection(BPRP, G, points_ms)# | _invi.misc.polygon_selection(BPRP, G, points_hb) | _invi.misc.polygon_selection(BPRP, G, points_rg)

    #Total selection
    sel = sel_met & sel_cmd & sel_phi_1_vr & sel_phi_2

    return sel

#-----------------------------------------------------------------------------
