"""Plots for Jupyter notebooks in code/17-stream_simulation."""

import numpy as _np
import scipy as _scipy
import matplotlib.pyplot as _plt

import fnc as _fnc

import invi as _invi
import invi.stream.simulation._model_3 as _model_3
import invi.stream.simulation._core as _ssc

__all__ = [#1-distribution
           "peaks_definition",
           "plot_A1_F1", "plot_Ai_Fi", "plot_Ji",
           "normalise_selection", "plot_hist",
           "plot_time", "plot_time_internal_comp",
           "plot_A1", "plot_A1_internal_comp",
           "plot_time_peaks", "plot_A1_peaks", "plot_A1_uni",
           "plot_ALPHAi",
           #Model 1 and Model 2
           "observable_limits", "plot_A1_long", "plot_A1_long_internal_comp",
           "plot_func_radius", "plot_mean_F1", "plot_std_F1", "plot_stars_stripped",
           #Model 3
           "plot_Ar_Fi", "plot_Ar_wrap_Fi", "plot_Ar_wrap_centre_Fi", "plot_Ar_model",
           "fit_parameters", "determine_limit", "plot_fit", "plot_Ar_stripping_Ar"
           ]

#-----------------------------------------------------------------------------

def peaks_definition(time, st, prm_gc, prm, T=1_500.0, intv=70.0):

    def sel(time, st, intv, peri):
        return (-time > -peri-intv) & (-time < -peri+intv) & st

    #Determination time pericentres passages from Galactocentric spherical radius
    #t_peris = _invi.stream.simulation._model_1.time_pericentres_r(prm_gc, prm, T, N=np.int64(T*10)+1)
    #Determination time pericentres passages from angle Ar of the globular cluster
    t_peris = _invi.stream.simulation._model_1.time_pericentres_Ar(prm_gc, prm, T)

    #Delay between the maximum mass loss and the pericentre passage
    t_peris = t_peris - prm_gc['stream']['simulation']['delay']

    p1 = sel(time, st, intv, t_peris[0])
    p2 = sel(time, st, intv, t_peris[1])
    p3 = sel(time, st, intv, t_peris[2])

    uni = st & (_np.logical_not(p1) & _np.logical_not(p2) & _np.logical_not(p3))

    return {'p1': p1, 'p2': p2, 'p3': p3, 'uni': uni}

#-----------------------------------------------------------------------------

def plot_A1_F1(data, data_ref=None, tips=False, x_limit=1.2, s=0.2, gray='0.8', zorder=0):
    """Example: data = {'A1': A1, 'F1': F1, 'leading': l, 'trailing': t}"""

    #Define figure
    _fig, ax = _fnc.plot.figure(fc=1.75)
    ax.set_xlabel("A1 [rad]")
    ax.set_ylabel("F1 [rad/Gyr]")
    y_limit = 1.1
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)

    if tips:
        ax.set_xlim(0.2, 0.9)
        ax.set_ylim(0.3, 0.9)

    #Plot reference data
    if data_ref is not None:
        l = data_ref['leading']
        t = data_ref['trailing']
        ax.scatter(data_ref['A1'][l], data_ref['F1'][l]*1_000.0, s=s, c=gray, zorder=zorder)
        ax.scatter(data_ref['A1'][t], data_ref['F1'][t]*1_000.0, s=s, c=gray, zorder=zorder)

    #Plot data
    l = data['leading']
    t = data['trailing']
    ax.scatter(data['A1'][l], data['F1'][l]*1_000.0, s=s, c="r", label='leading')
    ax.scatter(data['A1'][t], data['F1'][t]*1_000.0, s=s, c="k", label='trailing')
    ax.legend(markerscale=35.0*s, loc='upper left')


def plot_Ai_Fi(data, data_ref=None, gray='0.8', zorder=0):
    """Example: data = {'Ai': A2, 'Fi': F2, 'leading': l, 'trailing': t}"""

    #Define figure
    _fig, ax = _fnc.plot.figure(fc=1.75)
    ax.set_xlabel("Ai [mrad]")
    ax.set_ylabel("Fi [mrad/Gyr]")
    x_lim = 30.0
    y_lim = x_lim
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    #Plot reference data
    if data_ref is not None:
        l = data_ref['leading']
        t = data_ref['trailing']
        ax.scatter(data_ref['Ai'][l]*1_000.0, data_ref['Fi'][l]*1_000.0**2, s=0.2, c=gray, zorder=zorder)
        ax.scatter(data_ref['Ai'][t]*1_000.0, data_ref['Fi'][t]*1_000.0**2, s=0.2, c=gray, zorder=zorder)

    #Plot data
    l = data['leading']
    t = data['trailing']
    ax.scatter(data['Ai'][l]*1_000.0, data['Fi'][l]*1_000.0**2, s=0.2, c="r", label='leading')
    ax.scatter(data['Ai'][t]*1_000.0, data['Fi'][t]*1_000.0**2, s=0.2, c="k", label='trailing')
    ax.legend(markerscale=7)


def plot_Ji(data, data_ref=None, gray='0.8', zorder=0):
    """Example: data = {'Ja': J1, 'Jb': J2, 'leading': l, 'trailing': t}"""

    #Define figure
    _fig, ax = _fnc.plot.figure(fc=1.75)
    ax.set_xlabel("Ji [kpc^2/Myr]")
    ax.set_ylabel("Ji [kpc^2/Myr]")
    x_lim = 1.0E-1
    y_lim = x_lim
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    #Plot reference data
    if data_ref is not None:
        l = data_ref['leading']
        t = data_ref['trailing']
        ax.scatter(data_ref['Ja'][l], data_ref['Jb'][l], s=0.2, c=gray, zorder=zorder)
        ax.scatter(data_ref['Ja'][t], data_ref['Jb'][t], s=0.2, c=gray, zorder=zorder)

    #Plot data
    l = data['leading']
    t = data['trailing']
    ax.scatter(data['Ja'][l], data['Jb'][l], s=0.2, c="r", label='leading')
    ax.scatter(data['Ja'][t], data['Jb'][t], s=0.2, c="k", label='trailing')
    ax.legend(markerscale=7)

#-----------------------------------------------------------------------------

def normalise_selection(x, sel):
    y = _np.copy(x)
    y[sel] = -y[sel]
    return y


def plot_hist(dist, data, data_ref=None, bins=60, rng=None, gray='0.8'):
    if rng is None:
        std = _np.std(data)
        med = _np.median(data)
        rng = _np.array([med-std*5.0, med+std*5.0])

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': True}

    _fig, ax = _fnc.plot.figure(fc=1.5)
    ax.set_xlim(rng)
    ax.set_xlabel("F_i [mrad/Gyr = micro_rad/Myr]")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(data_ref, color=gray, **kwargs, label="data_ref")

        #Fit and plot dist
        var = dist.fit(data_ref)
        x = _np.linspace(rng[0], rng[1], 1_000)
        y = dist.pdf(x, *var)
        ax.plot(x, y, c=gray)

    #Plot data
    _h = ax.hist(data, color="k", **kwargs, label="data")

    #Fit and plot dist
    var = dist.fit(data)
    x = _np.linspace(rng[0], rng[1], 1_000)
    y = dist.pdf(x, *var)
    ax.plot(x, y, c="r", label=f"data fit: {dist.name}")
    line = _fnc.numeric.print_array(_np.asarray(var))
    ax.set_title(f"{line}", fontsize=8)

    ax.legend()

    return ax

#-----------------------------------------------------------------------------

def plot_time(time, time_ref=None, bins=200, rng=None, density=False, gray='0.8'):
    if rng is None:
        rng = _np.array([-1.5, 0.0]) #[Gyr]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    _fig, ax = _fnc.plot.figure(fc=(1.8, 1.2))
    ax.set_xlim(rng)
    ax.set_xlabel("$t_s$ [Gyr]")
    ax.set_ylabel("counts")

    #Plot reference data
    if time_ref is not None:
        _h = ax.hist(-time_ref/1_000.0, color=gray, **kwargs)

    #Plot data
    _h = ax.hist(-time/1_000.0, color="k", **kwargs)


def plot_time_internal_comp(data, data_ref=None, bins=200, rng=None, density=False, alpha=0.25):

    def get_var(data, si):
        sel = data['internal']['leading'][si] | data['internal']['trailing'][si]
        return -data['time'][sel]/1_000.0

    if rng is None:
        rng = _np.array([-1.5, 0.0]) #[Gyr]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    _fig, ax = _fnc.plot.figure(fc=(1.8, 1.2))
    ax.set_xlim(rng)
    ax.set_xlabel("$t_s$ [Gyr]")
    ax.set_ylabel("counts")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(get_var(data_ref, 's1'), color="r", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 's2'), color="g", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 's3'), color="b", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'sn'), color="k", **kwargs, alpha=alpha)

    #Plot data
    _h = ax.hist(get_var(data, 's1'), color="r", **kwargs)
    _h = ax.hist(get_var(data, 's2'), color="g", **kwargs)
    _h = ax.hist(get_var(data, 's3'), color="b", **kwargs)
    _h = ax.hist(get_var(data, 'sn'), color="k", **kwargs)

#-----------------------------------------------------------------------------

def plot_A1(A1, A1_ref=None, bins=200, rng=None, y_limit=125, density=False, gray='0.8'):

    if rng is None:
        rng = _np.array([-1.2, 1.2]) #[rad]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_ylim([0.0, y_limit])
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")

    #Plot reference data
    if A1_ref is not None:
        _h = ax.hist(A1_ref, color=gray, **kwargs)

    #Plot data
    _h = ax.hist(A1, color="k", **kwargs)


def plot_A1_internal_comp(data, data_ref=None, bins=200, rng=None, density=False, alpha=0.25):
    def get_var(data, si):
        sel = data['internal']['leading'][si] | data['internal']['trailing'][si]
        return -data['A1'][sel]

    if rng is None:
        rng = _np.array([-1.2, 1.2]) #[rad]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(get_var(data_ref, 's1'), color="r", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 's2'), color="g", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 's3'), color="b", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'sn'), color="k", **kwargs, alpha=alpha)

    #Plot data
    _h = ax.hist(get_var(data, 's1'), color="r", **kwargs)
    _h = ax.hist(get_var(data, 's2'), color="g", **kwargs)
    _h = ax.hist(get_var(data, 's3'), color="b", **kwargs)
    _h = ax.hist(get_var(data, 'sn'), color="k", **kwargs)

#-----------------------------------------------------------------------------

def plot_time_peaks(data, data_ref=None, bins=200, rng=None, alpha=0.25):

    def get_var(data, pi):
        sel = data['structure'][pi] | data['structure'][pi]
        return -data['time'][sel]/1_000

    if rng is None:
        rng = _np.array([-1.5, 0.0]) #[Gyr]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': False}

    _fig, ax = _fnc.plot.figure(fc=(1.8, 1.2))
    ax.set_xlim(rng)
    ax.set_xlabel("$t_s$ [Gyr]")
    ax.set_ylabel("counts")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(get_var(data_ref, 'p1'), color="r", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'p2'), color="g", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'p3'), color="b", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'uni'), color="k", **kwargs, alpha=alpha)

    #Plot data
    _h = ax.hist(get_var(data, 'p1'), color="r", **kwargs)
    _h = ax.hist(get_var(data, 'p2'), color="g", **kwargs)
    _h = ax.hist(get_var(data, 'p3'), color="b", **kwargs)
    _h = ax.hist(get_var(data, 'uni'), color="k", **kwargs)


def plot_A1_peaks(data, data_ref=None, bins=200, rng=None, alpha=0.25):
    def get_var(data, pi):
        sel = data['structure'][pi] | data['structure'][pi]
        return -data['A1'][sel]

    if rng is None:
        rng = _np.array([-1.2, 1.2]) #[rad]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': False}

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(get_var(data_ref, 'p1'), color="r", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'p2'), color="g", **kwargs, alpha=alpha)
        _h = ax.hist(get_var(data_ref, 'p3'), color="b", **kwargs, alpha=alpha)
        #_h = ax.hist(get_var(data_ref, 'uni'), color="k", **kwargs, alpha=alpha)

    #Plot data
    _h = ax.hist(get_var(data, 'p1'), color="r", **kwargs)
    _h = ax.hist(get_var(data, 'p2'), color="g", **kwargs)
    _h = ax.hist(get_var(data, 'p3'), color="b", **kwargs)
    #_h = ax.hist(get_var(data, 'uni'), color="k", **kwargs)


def plot_A1_uni(data, data_ref=None, bins=200, rng=None, alpha=0.25):
    def get_var(data, pi):
        sel = data['structure'][pi] | data['structure'][pi]
        return -data['A1'][sel]

    if rng is None:
        rng = _np.array([-1.2, 1.2]) #[rad]

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': False}

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")

    #Plot reference data
    if data_ref is not None:
        _h = ax.hist(get_var(data_ref, 'uni'), color="k", **kwargs, alpha=alpha)

    #Plot data
    _h = ax.hist(get_var(data, 'uni'), color="k", **kwargs)

#-----------------------------------------------------------------------------

def plot_ALPHAi(data, data_ref=None, gray='0.8'):

    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5, 1.5))
    ax1, ax2 = ax[0], ax[1]

    if data_ref is not None:
        l = data_ref['l']
        t = data_ref['t']
        ax1.scatter(data_ref['A_1'][l]*1000.0, data_ref['A_3'][l]*1000.0, s=0.1, c=gray)
        ax1.scatter(data_ref['A_1'][t]*1000.0, data_ref['A_3'][t]*1000.0, s=0.1, c=gray)

    l = data['l']
    t = data['t']
    ax1.scatter(data['A_1'][l]*1000.0, data['A_3'][l]*1000.0, s=0.1, c="r", label='leading')
    ax1.scatter(data['A_1'][t]*1000.0, data['A_3'][t]*1000.0, s=0.1, c="k", label='trailing')

    ax1.set_aspect(1.0)
    li = 0.025*1000.0
    ax1.set_xlim(-li, li)
    ax1.set_ylim(-li, li)
    ax1.set_xlabel("A_1 [mrad]")
    ax1.set_ylabel("A_3 [mrad]")
    ax1.legend(markerscale=10)

    if data_ref is not None:
        l = data_ref['l']
        t = data_ref['t']
        ax2.scatter(data_ref['A_2'][l]*1000.0, data_ref['A_3'][l]*1000.0, s=0.1, c=gray)
        ax2.scatter(data_ref['A_2'][t]*1000.0, data_ref['A_3'][t]*1000.0, s=0.1, c=gray)

    l = data['l']
    t = data['t']
    ax2.scatter(data['A_2'][l]*1000.0, data['A_3'][l]*1000.0, s=0.1, c="r")
    ax2.scatter(data['A_2'][t]*1000.0, data['A_3'][t]*1000.0, s=0.1, c="k")

    ax2.set_aspect(1.0)
    li = 0.025*1000.0
    ax2.set_xlim(-li, li)
    ax2.set_ylim(-li, li)
    ax2.set_xlabel("A_2 [mrad]")
    ax2.set_ylabel("A_3 [mrad]")

#-----------------------------------------------------------------------------
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#-----------------------------------------------------------------------------

def observable_limits(ax, prm_gc):
    A1_inf, A1_sup = prm_gc['age_estimate']['A1_rng']
    peak = 0.47 #[rad]

    ax.axvspan(A1_inf, A1_inf, color='orange', linestyle='--')
    ax.axvspan(A1_sup, A1_sup, color='orange', linestyle='--')
    ax.axvspan(peak, peak, color='orange', linestyle='--', linewidth=0.75)


def plot_A1_long(T, A1, prm_gc, bins_per_rad=40, rng=None, y_limit=0.6, density=False, savefig=False):

    if rng is None: #[rad]
        rng = [_np.min(A1), _np.max(A1)]

    bins = _np.int64((rng[1] - rng[0])*bins_per_rad)

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_ylim([0.0, y_limit])
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")
    observable_limits(ax, prm_gc)
    ax.set_title(f"T = {T:_} Myr, n_stars={len(A1):_}")

    #Plot data
    ax.hist(A1, color="k", **kwargs)

    if savefig:
        fig.savefig(f'../../data/misc/sel_func/{_np.int64(T)}.png', dpi=600)
        _plt.close(fig)


def plot_A1_long_internal_comp(A1, prm_gc, structure, bins_per_rad=40, x_limit=None, y_limit=0.6, gray='0.8', density=False):

    if x_limit is None:
        x_limit = _np.max(_np.abs(A1))

    rng = [-x_limit, x_limit] #[rad]

    bins = _np.int64(2.0*x_limit*bins_per_rad)

    kwargs = {'bins': bins, 'range': rng, 'histtype': 'step', 'density': density}

    _fig, ax = _fnc.plot.figure(fc=(2.5, 1.0))
    ax.set_xlim(rng)
    ax.set_ylim([0.0, y_limit])
    ax.set_xlabel("$A1$ [rad]")
    ax.set_ylabel("counts")
    observable_limits(ax, prm_gc)

    ax.hist(A1, color=gray, **kwargs)

    n_peris = len(structure.keys()) - 1
    cmap = _plt.get_cmap('rainbow_r', n_peris)
    for i in range(n_peris):
        si = structure[f'p{i+1}']
        ax.hist(A1[si], color=cmap(i), **kwargs)

    uni = structure['uni']
    ax.hist(A1[uni], color='k', **kwargs)

#-----------------------------------------------------------------------------

def _min_f(m, x, y):
    corr = _np.abs(_fnc.stats.correlation([x**m, y])[0][1])
    return _np.abs(corr - 1.0)


def _fit(x, y):
    #Remove nan and y=0.0
    sel = _np.logical_not(_np.isnan(x) | _np.isnan(y) | (y==0.0))

    x_sel = x[sel]
    y_sel = y[sel]

    #Determine exponent 'm' that optimises corr(x,y)=1.0
    res = _scipy.optimize.minimize(_min_f, x0=1.0, args=(x_sel, y_sel), method='Nelder-Mead', options={'xatol': 1E-12})
    m = res.x[0]

    print(f"Correlation = {_fnc.stats.correlation([x_sel**m, y_sel])[0][1]:0.6f}\n")

    #Linear fit to x**m
    a, b = _np.polyfit(x_sel**m, y_sel, 1)

    return m, a, b


def _print_res_F(m, a, b):
    """y = a*x**m + b"""
    print(f"m = {m:0.4f}")
    print(f"a = {a*1_000:0.4f} #[mrad/Myr * kpc^-m]")
    print(f"b = {b*1_000:0.4f} #[mrad/Myr]")


def _print_res_stars(m, a, b):
    """y = a*x**m + b"""
    print(f"m = {m:0.4f}")
    print(f"a = {a:0.4f} #[stars * kpc^-m]")
    print(f"b = {b:0.4f} #[stars]")


def _plot_linear_fit(ax, x, a, b, fc=0.15):
    x = _np.linspace(_np.min(x)*(1.0-fc), _np.max(x)*(1.0+fc), 1_000)
    y = a*x + b
    ax.plot(x, y, c='r')


def plot_func_radius(x, y, x_label, y_label):

    m, a, b = _fit(x, y)

    if y_label == 'stars_stripped [stars]':
        _print_res_stars(m, a, b)
    else:
        _print_res_F(m, a, b)

    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.75, 1.25))
    ax1, ax2 = ax[0], ax[1]

    ax1.scatter(x, y, s=3.0, c='k')
    ax1.set_xlabel(f"{x_label} [kpc]")
    ax1.set_ylabel(y_label)

    ax2.scatter(x**m, y, s=3.0, c='k')
    _plot_linear_fit(ax2, x**m, a, b)
    ax2.set_xlabel(f"{x_label}^{m:0.4f} [-]")
    ax2.set_ylabel(y_label)

    return ax1


def plot_mean_F1(y, data):

    time, F1, st, time_F1, mean_F1, t, _std_F1, _stars_stripped = data.values()

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("F1 [rad/Gyr]")
    ax.set_xlim(-1.5, 0.0)
    ax.set_ylim(0.0, 0.9)

    ax.scatter(-time[st]/1_000.0, _np.abs(F1[st])*1_000.0, s=0.015, c="k")

    ax.scatter(-time_F1/1_000.0, mean_F1*1_000.0, s=8.0, c='r')
    #ax.errorbar(-time_F1/1_000.0, mean_F1*1_000.0, std_F1*1_000.0, c='r')

    ax.plot(t/1_000.0, y*1_000.0, c='k')


def plot_std_F1(y, data):

    _time, _F1, _st, time_F1, _mean_F1, t, std_F1, _stars_stripped = data.values()

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("std(F1) [rad/Gyr]")
    ax.set_xlim(-1.5, 0.0)
    ax.set_ylim(0.0, 0.15)

    ax.scatter(-time_F1/1_000.0, std_F1*1_000.0, s=8.0, c='r')

    ax.plot(t/1_000.0, y*1_000.0, c='k')


def plot_stars_stripped(y, data):

    time, _F1, st, time_F1, _mean_F1, t, _std_F1, stars_stripped = data.values()

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("mass loss [stars]")
    ax.set_xlim(-1.5, 0.0)
    ax.set_ylim(0.0, 350)

    ax.hist(-time[st]/1_000.0, bins=101, histtype='step', density=False, range=[-1_400.0/1_000.0, -50.0/1_000.0])

    ax.scatter(-time_F1/1_000.0, stars_stripped, s=8.0, c='r')

    ax.plot(t/1_000.0, y, c='k')

#-----------------------------------------------------------------------------
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#-----------------------------------------------------------------------------
#Model 3

def _set_x_axes(ax, data):
    gc_Ar_T = data['gc_Ar_T']
    gc_Ar_0 = data['gc_Ar_0']

    #X_inf axis
    x = _np.array([gc_Ar_T/2.0/_np.pi, -2.0, -1.0, 0.0, gc_Ar_0/2.0/_np.pi, 1.0])*2.0*_np.pi
    ax.set_xticks(x)
    ax.set_xticklabels(['$\\theta_{-T}$', '$-4\\pi$', '$-2\\pi$', '$0$', '$\\theta_0$', '$2\\pi$'])
    ax.set_xlim(gc_Ar_T, gc_Ar_0)
    ax.set_xlabel("Ar [rad] (Including delay)")

    #X_sup axis
    ax_sup = ax.twiny()
    ax_sup.set_xlim(gc_Ar_T, gc_Ar_0)
    ax_sup.set_xlabel("Ar [rad] (Including delay)")


def plot_Ar_Fi(i, data):

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    match i:
        case 1:
            ax.set_ylabel("F1 [rad/Gyr]")
            ax.scatter(data['st_stripping_Ar'], _np.abs(data['st_F1'])*1_000.0, s=0.015, c="k")
            ax.scatter(data['Ar_mean_intv'], data['mean_F1']*1_000.0, s=8.0, c='r')
            ax.errorbar(data['Ar_mean_intv'], data['mean_F1']*1_000.0, data['std_F1']*1_000.0, c='r')
            y_limit = 0.9
        case 2:
            ax.set_ylabel("F2 [mrad/Gyr]")
            ax.scatter(data['st_stripping_Ar'], _np.abs(data['st_F2'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0
        case 3:
            ax.set_ylabel("F3 [mrad/Gyr]")
            ax.scatter(data['st_stripping_Ar'], _np.abs(data['st_F3'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0

    ax.set_ylim(0.0, y_limit)
    _set_x_axes(ax, data)


def plot_Ar_wrap_Fi(i, data):

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    match i:
        case 1:
            ax.set_ylabel("F1 [rad/Gyr]")
            ax.scatter(_model_3.wrap(data['st_stripping_Ar']), _np.abs(data['st_F1'])*1_000.0, s=0.015, c="k")
            ax.scatter(_model_3.wrap(data['Ar_mean_intv']), data['mean_F1']*1_000.0, s=8.0, c='r')
            ax.errorbar(_model_3.wrap(data['Ar_mean_intv']), data['mean_F1']*1_000.0, data['std_F1']*1_000.0, c='r', ls='none')
            y_limit = 0.9
        case 2:
            ax.set_ylabel("F2 [mrad/Gyr]")
            ax.scatter(_model_3.wrap(data['st_stripping_Ar']), _np.abs(data['st_F2'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0
        case 3:
            ax.set_ylabel("F3 [mrad/Gyr]")
            ax.scatter(_model_3.wrap(data['st_stripping_Ar']), _np.abs(data['st_F3'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0

    ax.set_ylim(0.0, y_limit)
    x = _np.array([0.0, 1.0, 2.0])*_np.pi
    ax.set_xticks(x)
    ax.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$'])


def plot_Ar_wrap_centre_Fi(i, data):

    def transform(x):
        return _model_3.centre(_model_3.wrap(x))

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    match i:
        case 1:
            ax.set_ylabel("F1 [rad/Gyr]")
            ax.scatter(transform(data['st_stripping_Ar']), _np.abs(data['st_F1'])*1_000.0, s=0.015, c="k")
            ax.scatter(transform(data['Ar_mean_intv']), data['mean_F1']*1_000.0, s=8.0, c='r')
            ax.errorbar(transform(data['Ar_mean_intv']), data['mean_F1']*1_000.0, data['std_F1']*1_000.0, c='r', ls='none')
            y_limit = 0.9
        case 2:
            ax.set_ylabel("F2 [mrad/Gyr]")
            ax.scatter(transform(data['st_stripping_Ar']), _np.abs(data['st_F2'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0
        case 3:
            ax.set_ylabel("F3 [mrad/Gyr]")
            ax.scatter(transform(data['st_stripping_Ar']), _np.abs(data['st_F3'])*1_000.0**2.0, s=0.015, c="k")
            y_limit = 20.0

    ax.set_ylim(0.0, y_limit)
    x = _np.array([-1.0, 0.0, 1.0])*_np.pi
    ax.set_xticks(x)
    ax.set_xticklabels(['$-\\pi$', '$0$', '$\\pi$'])


def plot_Ar_model(case, data, model):

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    match case:
        case 'mean':
            ax.set_ylabel("mean(F1) [rad/Gyr]")
            ax.scatter(data['st_stripping_Ar'], _np.abs(data['st_F1'])*1_000.0, s=0.015, c="k")
            ax.scatter(data['Ar_mean_intv'], data['mean_F1']*1_000.0, s=8.0, c='r')

            ax.plot(data['Ar_intv_model'], model*1_000.0, c='k')

            y_limit = 0.9

        case 'std':
            ax.set_ylabel("std(F1) [rad/Gyr]")
            ax.scatter(data['Ar_mean_intv'], data['std_F1']*1_000.0, s=8.0, c='r')

            ax.plot(data['Ar_intv_model'], model*1_000.0, c='k')

            y_limit = 0.15

        case 'mass':
            ax.set_ylabel("number stars stripped")

            ax.hist(data['st_stripping_Ar'],
                    bins=data['N']-1,
                    histtype='step',
                    density=False,
                    range=[data['Ar_intv'][0], data['Ar_intv'][-1]],
                    color='r')

            ax.scatter(data['Ar_mean_intv'], data['n_stars_stripped'], s=8.0, c='r')

            ax.plot(data['Ar_intv_model'], model, c='k')

            y_limit = 175.0

    ax.set_ylim(0.0, y_limit)
    _set_x_axes(ax, data)


def fit_parameters(x, y, limit):
    kwargs = {'method': 'trf', 'nan_policy': 'omit', 'maxfev': 10_000}
    #---------------------------------------
    def double_exp_1(Ar, A, C, s0):
        return A*_np.exp( -Ar/(s0*_np.pi) ) + C

    sel = x <= limit*_np.pi
    coef, _ = _scipy.optimize.curve_fit(double_exp_1, x[sel], y[sel], **kwargs)
    A, C, s0 = coef
    #---------------------------------------
    def double_exp_2(Ar, s1):
        return A*_np.exp( (Ar-_np.pi*2.0)/(s1*_np.pi) ) + C

    bounds = (0.1/_np.pi, 1.0/_np.pi)
    sel = x > limit*_np.pi
    coef, _ = _scipy.optimize.curve_fit(double_exp_2, x[sel], y[sel], bounds=bounds, **kwargs)
    s1 = coef[0]

    #Check whether the result of the optimization is within specified bounds
    assert _fnc.numeric.within_equal(s1, *bounds)
    #---------------------------------------
    return _np.array([A, C, s0, s1])


def determine_limit(x, y):

    def optimal_limit(limit, x, y):
        """Evaluate the optimal limit for the fit."""
        coef = fit_parameters(x, y, limit)

        y_fit = _model_3.double_exp(x, coef)
        return _np.sum( _np.abs(y_fit-y)**2.0 )

    res = _scipy.optimize.minimize(optimal_limit, x0=1.5, method='nelder-mead', args=(x, y))
    limit = res.x[0]
    print(f"limit = {limit:0.4f}π")

    return limit


def plot_fit(x, y, y_label, limit):
    _fig, ax = _fnc.plot.figure(fc=(1.5, 1.2))

    x_ticks = _np.array([0.0, 1.0, 2.0])*_np.pi
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$'])

    ax.set_xlabel('Ar [rad]')
    ax.set_ylabel(y_label)
    #--------------------------------------------------------
    ax.scatter(_model_3.desp(x, limit), y, c='0.75', s=2.0)
    ax.scatter(x, y, s=2.0, c='k')
    #--------------------------------------------------------
    ax.axvline(x=limit*_np.pi, color="gray", linestyle="--")
    #--------------------------------------------------------
    coef = fit_parameters(x, y, limit)
    _fnc.numeric.print_array(coef)

    eps = 1E-12
    x_fit = _np.linspace(limit*_np.pi+eps, 2.0*_np.pi, 1_000)
    y_fit = _model_3.double_exp(x_fit, coef)
    ax.plot(_model_3.desp(x_fit, limit), y_fit, c='0.75')

    x_fit = _np.linspace(0.0, 2.0*_np.pi, 1_000)
    y_fit = _model_3.double_exp(x_fit, coef)
    ax.plot(x_fit, y_fit, c='r')
    #--------------------------------------------------------
    return coef


def plot_Ar_stripping_Ar(data, sample, norm_model_mass):

    _fig, ax = _fnc.plot.figure(fc=(2.0, 1.5))

    kwargs = {'bins': data['N']-1,
              'histtype': 'step',
              'density': True,
              'range': [data['Ar_intv'][0], data['Ar_intv'][-1]]}

    #Histogram N-body
    ax.hist(data['st_stripping_Ar'], color='r', **kwargs, label='N-body')

    #Normalised model
    ax.plot(data['Ar_intv_model'], norm_model_mass, c='b')

    #Histogram sample
    ax.hist(sample, color='k', **kwargs, label='model')

    ax.legend()
    ax.set_ylabel("PDF")
    ax.set_ylim(0.0, 0.35)
    _set_x_axes(ax, data)

#-----------------------------------------------------------------------------
