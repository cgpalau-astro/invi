"""Plots for Jupyter notebooks in code/14-main_component folder."""

import numpy as _np

import fnc as _fnc
import invi as _invi

__all__ = ["number_stars", "plot_phi", "plot_radec", "plot_proper_motion"]

#-----------------------------------------------------------------------------

def number_stars(final_sel, main_comp):
    cacoon = _np.logical_not(main_comp)
    n_main_comp = len(final_sel['ICRS']['alpha'][main_comp])
    n_cacoon = len(final_sel['ICRS']['alpha'][cacoon])
    print("Number stars:")
    print(f"Main comp. = {n_main_comp:_}")
    print(f"Cacoon     = {n_cacoon:_}")

#-----------------------------------------------------------------------------

def plot_phi(limits, final_sel_phi, nbody_phi=None, aspect='auto'):

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))

    ax.scatter(final_sel_phi[2], final_sel_phi[1], s=2.0, c="k")

    if nbody_phi is not None:
        ax.scatter(nbody_phi[2], nbody_phi[1], s=2.0, c="r")

    x_range = [-5.0, 120.0]
    y_range = [-13.0, -5.0]

    ax.plot(x_range, [limits[1][0], limits[1][0]], c='orange')
    ax.plot(x_range, [limits[1][1], limits[1][1]], c='orange')

    ax.plot([limits[0][0], limits[0][0]], y_range, c='orange')
    ax.plot([limits[0][1], limits[0][1]], y_range, c='orange')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect(aspect)
    ax.set_xlabel("$\\phi_1$ [deg]")
    ax.set_ylabel("$\\phi_2$ [deg]")

#-----------------------------------------------------------------------------

def plot_radec(final_sel, prm_gc, stream, cacoon, M68=None):

    _fig, ax = _fnc.plot.figure(fc=2)

    ax.scatter(final_sel['ICRS']['alpha'][stream], final_sel['ICRS']['delta'][stream], s=1.0, c="k")
    ax.scatter(final_sel['ICRS']['alpha'][cacoon], final_sel['ICRS']['delta'][cacoon], s=1.0, c="r")

    ax.scatter(prm_gc['ICRS']['alpha'], prm_gc['ICRS']['delta'], s=5.0, c="b")

    if M68 is not None:
        alpha = _np.asarray(M68['stars']['phase_space']['ICRS']['alpha'])
        delta = _np.asarray(M68['stars']['phase_space']['ICRS']['delta'])
        ax.scatter(alpha, delta, s=1.0, c="g")

    _invi.notebook.selection.set_axis_sky_coord(ax, prm_gc)


def plot_proper_motion(final_sel, prm_gc, stream, cacoon, M68=None):

    _fig, ax = _fnc.plot.figure(fc=2)

    ax.scatter(final_sel['ICRS']['mu_alpha_str'][stream], final_sel['ICRS']['mu_delta'][stream], s=1.0, c="k")
    ax.scatter(final_sel['ICRS']['mu_alpha_str'][cacoon], final_sel['ICRS']['mu_delta'][cacoon], s=1.0, c="r")

    ax.scatter(prm_gc['ICRS']['mu_alpha_str'], prm_gc['ICRS']['mu_delta'], s=5.0, c="b")

    if M68 is not None:
        mu_alpha_str = _np.asarray(M68['stars']['phase_space']['ICRS']['mu_alpha_str'])
        mu_delta = _np.asarray(M68['stars']['phase_space']['ICRS']['mu_delta'])
        ax.scatter(mu_alpha_str, mu_delta, s=1.0, c="g")

    _invi.notebook.selection.set_axis_pm(ax)

#-----------------------------------------------------------------------------
