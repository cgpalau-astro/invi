"""Plots for Jupyter notebooks in code/10-selection folder."""

import numpy as _np

import fnc as _fnc
import invi as _invi

__all__ = ["hr_gaia", "HR_gaia", "set_axis_sky_coord", "set_axis_pm", "hr", "HR"]

#-----------------------------------------------------------------------------
#6-selection

def _hr_set_axis(ax, x_limit, y_limit):
    #-----------------------------------
    #Lower x axis
    ax.set_xlabel("$(g_{\\rm BP}-g_{\\rm RP})$ [mag]")
    ax.set_xlim(x_limit)
    #-----------------------------------
    #Upper x axis
    ax2 = ax.twiny()
    ax2.set_xlabel("$T_{\\rm eff}$ [K]")
    ax2.set_xlim(x_limit)

    xticks = []
    xticklabels = []
    for item in [30_000, 10_000, 6_500, 5_000, 4_000, 3_500]:
        xticks.append(_invi.photometry.magnitudes.Teff_to_BPRP(item))
        xticklabels.append(str(item))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    #-----------------------------------
    #Left y axis
    ax.set_ylabel("$g$ [mag]")
    ax.set_ylim(y_limit)
    #-----------------------------------


def hr_gaia(gc, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=2)
    kwargs = {'s':s, 'linewidth':0.5}

    ax.scatter(gc['photometry']['bprp'],
               gc['photometry']['g'],
               label="Reddening corrected", c="r", **kwargs)
    ax.legend(markerscale=3)

    x_limit = [-1.0, 3.0]
    y_limit = [22.0, 11.0]
    _hr_set_axis(ax, x_limit, y_limit)
    return ax


def _HR_set_axis(ax, x_limit, y_limit):
    #-----------------------------------
    #Lower x axis
    ax.set_xlabel("$(G_{\\rm BP}-G_{\\rm RP})$ [mag]")
    ax.set_xlim(x_limit)
    #-----------------------------------
    #Upper x axis
    ax2 = ax.twiny()
    ax2.set_xlabel("$T_{\\rm eff}$ [K]")
    ax2.set_xlim(x_limit)

    xticks = []
    xticklabels = []
    for item in [30_000, 10_000, 6_500, 5_000, 4_000, 3_500]:
        xticks.append(_invi.photometry.magnitudes.Teff_to_BPRP(item))
        xticklabels.append(str(item))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    #-----------------------------------
    #Left y axis
    ax.set_ylabel("$G$ [mag]")
    ax.set_ylim(y_limit)
    #-----------------------------------


def HR_gaia(gc, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=2)
    kwargs = {'s':s, 'linewidth':0.5}

    ax.scatter(gc['photometry']['BPRP'],
               gc['photometry']['G'],
               label="Reddening corrected", c="r", **kwargs)
    ax.legend(markerscale=3)

    x_limit = [-1.0, 3.0]
    y_limit = [8.0, -4.0]
    _HR_set_axis(ax, x_limit, y_limit)
    return ax


def set_axis_sky_coord(ax, prm_gc):
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")
    ax.set_xlim(170.0, 310.0)
    ax.set_ylim(-30.0, 90.0)
    ax.set_aspect(1.0)

    _invi.plot.lines_b15(ax)

    dec_cut = prm_gc['selection']['dec_cut']
    ra_cut = prm_gc['selection']['ra_cut']
    ax.plot([170.0, 310.0], [dec_cut, dec_cut], linestyle="--", color="r")
    ax.plot([ra_cut, ra_cut], [-90.0, 90.0], linestyle="--", color="r")


def set_axis_pm(ax):
    ax.set_aspect(1)
    ax.set_xlabel("mu_alpha_str [mas/yr]")
    ax.set_ylabel("mu_delta [mas/yr]")
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(-10.0, 10.0)

#-----------------------------------------------------------------------------

def hr(prm, s=0.2):
    fig, ax = _fnc.plot.figure(1, 1, fc=(2.2,2.2))
    x_limit = [-1.0, 3.0]
    y_limit = [30.0, 10.0]
    _hr_set_axis(ax, x_limit, y_limit)
    return fig, ax


def HR(prm, s=0.2):
    fig, ax = _fnc.plot.figure(1, 1, fc=(2.2,2.2))
    x_limit = [-1.0, 3.0]
    y_limit = [13.0, -4.0]
    _HR_set_axis(ax, x_limit, y_limit)
    return fig, ax

#-----------------------------------------------------------------------------
