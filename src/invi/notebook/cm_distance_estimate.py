"""Plots for Jupyter notebooks in code/19-cm_distance_estimate folder."""

import numpy as _np
import scipy as _scipy

import fnc as _fnc
import invi as _invi

__all__ = ["set_axes", "plot_isochrone", "plot_distance_estimate"]

#-----------------------------------------------------------------------------

def set_axes(ax):
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(13.0, -5.0)
    ax.legend(markerscale=5)
    ax.set_xlabel("BPRP [mag]")
    ax.set_ylabel("G [mag]")


def plot_isochrone(ax, isochrone):
    x = _np.linspace(-1.0, 3.0, 100_000)
    color = ['b', 'g', 'r']
    for i in range(3):
        y = isochrone[i](x)
        ax.plot(x, y, linestyle='-', color=color[i])

#-----------------------------------------------------------------------------

def plot_distance_estimate(bprp, d, D, r, sel):
    error = (D-r)*1_000.0 #[pc]
    #-----------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=(3.0, 1.25))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    #-----------------------------------------------
    color = ['b', 'g', 'r']
    for i in range(3):
        ax1.scatter(bprp, (d.T[i]-r)*1_000.0, s=0.1, color=color[i])

    lim = 1E4
    ax1.set_ylim(-lim, lim)
    ax1.set_xlim(-1.0, 3.0)
    ax1.set_xlabel('BPRP')
    ax1.set_ylabel('Error d [pc]')
    #-----------------------------------------------
    not_nan = _np.logical_not(_np.isnan(D))
    loc = _np.mean(error[not_nan])
    scale = _np.std(error[not_nan])
    ax2.scatter(bprp, error, s=0.1, label=f'Mean = {loc:0.3f}±{scale:0.3f} pc', c='0.75')

    loc = _np.mean(error[sel])
    scale = _np.std(error[sel])
    ax2.scatter(bprp[sel], error[sel], s=0.1, label=f'Mean = {loc:0.3f}±{scale:0.3f} pc')

    lim = 500.0
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-1.0, 3.0)
    ax2.set_xlabel('BPRP')
    ax2.set_ylabel('Error D [pc]')
    ax2.legend(markerscale=10.0, loc='upper center', bbox_to_anchor=(0.5, 1.3))
    #-----------------------------------------------
    h = ax3.hist(error[sel], bins=50, density=True)

    lim = 2_000.0
    x = _np.linspace(-lim, lim, 1_000)

    loc = _np.mean(error[sel])
    scale = _np.std(error[sel])
    y = _scipy.stats.norm(loc=loc, scale=scale).pdf(x)
    ax3.plot(x, y, c='r', label=f'Mean = {loc:0.3f}±{scale:0.3f} pc')

    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    #ax3.set_xlim(-lim, lim)
    ax3.set_ylim(0.0, _np.max(h[0])*1.5)
    ax3.set_xlabel('Error D [pc]')
    ax3.set_ylabel('PDF')

#-----------------------------------------------------------------------------
