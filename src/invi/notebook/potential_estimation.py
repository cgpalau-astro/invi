"""Plots for Jupyter notebooks in code/20-potential_estimation folder."""

import fnc as _fnc
import invi as _invi

__all__ = ["plot_alpha"]

#-----------------------------------------------------------------------------

def _set_axes(ax, xlabel, ylabel):
    #Globular cluster position
    ax.scatter(0.0, 0.0, s=10.0, c="r")

    li = 0.025*1000.0
    ax.set_xlim(-li, li)
    ax.set_ylim(-li, li)
    ax.set_aspect(1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_alpha(sample_alpha, prm_gc):
    alpha_dict = _invi.dicts.alpha(sample_alpha, prm_gc['stream']['varphi'])
    ALPHA = alpha_dict['ALPHA']

    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5, 1.5))

    ax[0].scatter(ALPHA['A_1']*1000.0, ALPHA['A_3']*1000.0, s=0.1, c="k")
    _set_axes(ax[0], "A_1 [mrad]", "A_3 [mrad]")

    ax[1].scatter(ALPHA['A_2']*1000.0, ALPHA['A_3']*1000.0, s=0.1, c="k")
    _set_axes(ax[1], "A_2 [mrad]", "A_3 [mrad]")

#-----------------------------------------------------------------------------
