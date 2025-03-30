"""Plots for Jupyter notebooks in code/18-age_estimate."""

import numpy as _np
import scipy as _scipy
import matplotlib.pyplot as _plt

import fnc as _fnc
import invi as _invi

__all__ = ["hist_counts",
           "plot_radec", "hist_density", "hist_area",
           "plot_time_pericentres" ]

#-----------------------------------------------------------------------------

def hist_counts(A1, A1_mock, prm_gc):
    _fig, ax = _fnc.plot.figure(fc=(2.5,1.0))

    kwargs = {'bins': prm_gc['age_estimate']['bins'],
              'range': prm_gc['age_estimate']['A1_rng'],
              'histtype': 'step',
              'density': False}

    ax.hist(A1, color='k', **kwargs, label='GDR3')
    ax.hist(A1_mock, color='b', **kwargs, label='mock')

    ax.legend()
    ax.set_xlim(0.1, 1.1)
    ax.set_ylim(0.0, 12.0)
    ax.set_xlabel('A1_orb_est [rad]')
    ax.set_ylabel('counts')
    _invi.notebook.stream_simulation.observable_limits(ax, prm_gc)

#-----------------------------------------------------------------------------

def plot_radec(name_folder, T, prm_gc, s=0.5):
    #Load data
    res = _fnc.utils.store.load(f"../../data/misc/surface_density/{name_folder}/res.opy")

    n_samples = len(res)

    fig, ax = _fnc.plot.figure(fc=1.75)

    for i in range(n_samples):
        data = res[i]
        sim_stripping_time = -_invi.inverse.integration_time(data['sim_AAF'])
        sel = sim_stripping_time > -T

        ax.scatter(data['ra'][sel], data['dec'][sel], s=s, c='darkblue')

    for i in range(n_samples):
        data = res[i]
        sim_stripping_time = -_invi.inverse.integration_time(data['sim_AAF'])
        sel = data['sel'] & (sim_stripping_time > -T)

        ax.scatter(data['ra'][sel], data['dec'][sel], s=s, c='r')

    ax.scatter(10_000.0, 0.0, s=s, c='darkblue', label='sim')
    ax.scatter(10_000.0, 0.0, s=s, c='r', label='sim sel')
    ax.legend(loc='lower right', markerscale=5)
    ax.set_title(f"T = {T:_} Myr, n_samples={n_samples}")

    _invi.notebook.selection.set_axis_sky_coord(ax, prm_gc)

#-----------------------------------------------------------------------------

def _set_axes_hist(ax, T, prm_gc, n_samples, y_limit, y_label):
    ax.legend()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0.0, y_limit)
    ax.set_xlabel('A1_orb_est [rad]')
    ax.set_ylabel(y_label)
    ax.set_title(f"T = {T} Myr, n_samples={n_samples}")

    _invi.notebook.stream_simulation.observable_limits(ax, prm_gc)


def hist_density(name_folder, T, A1, prm_gc, color='r', y_limit=6.0):

    #Load distribution properties
    bins = prm_gc['age_estimate']['bins']
    rng = prm_gc['age_estimate']['A1_rng']
    bw = prm_gc['age_estimate']['bw']

    #Load data for time T
    n_samples, A1_sim, _mean_stars, _std_stars = _invi.stream.age_estimation.load_surface_density_data(name_folder, T)

    fig, ax = _fnc.plot.figure(fc=(2.5,1.0))

    #Histogram observed GDR3 data
    ax.hist(A1,
            color='k',
            label="GDR3",
            bins=bins,
            range=rng,
            histtype='step',
            density=True)

    x = _np.linspace(rng[0], rng[1], 1_000)

    #PDF using a histogram
    hist = _invi.stream.age_estimation.DistributionHist(A1_sim, bins, rng)
    y = hist.pdf(x)
    ax.plot(x, y, color=color, alpha=0.5, label=f'sim hist bins = {hist.bins:_}')

    #PDF using a KDE
    kde = _invi.stream.age_estimation.DistributionKDE(A1_sim, bw, rng)
    y = kde.pdf(x)
    ax.plot(x, y, color=color, label=f'sim KDE bw = {kde.bw:0.3}')

    _set_axes_hist(ax, T, prm_gc, n_samples, y_limit, 'PDF')


def hist_area(name_folder, T, A1, prm_gc, color='r'):

    #Load distribution properties
    bins = prm_gc['age_estimate']['bins']
    rng = prm_gc['age_estimate']['A1_rng']

    #Load data for time T
    n_samples, A1_sim, mean_stars, std_stars = _invi.stream.age_estimation.load_surface_density_data(name_folder, T)

    fig, ax = _fnc.plot.figure(fc=(2.5,1.0))

    kwargs = {'bins': bins,
              'range': rng}

    n_stars = len(A1)
    area = n_stars*(rng[1] - rng[0])/bins
    _invi.plot.hist_area(ax,
                         A1,
                         area=area,
                         c='k',
                         label=f"GDR3 = {n_stars}",
                         **kwargs)

    n_stars_sim = len(A1_sim)
    _invi.plot.hist_area(ax,
                         A1_sim,
                         area=area*n_stars_sim/n_stars/n_samples,
                         c=color,
                         label=f"sim = {mean_stars:0.1f} ± {std_stars:0.1f}",
                         **kwargs)

    _set_axes_hist(ax, T, prm_gc, n_samples, 12.0, 'counts')

#-----------------------------------------------------------------------------

def plot_time_pericentres(ax, prm_gc, prm, T, peak_intv=70.0):
    """Plot time pericentres from 0.0 to T in Gyr."""

    T = _np.int64(T) #[Myr]
    peak_intv /= 1_000.0 #[Gyr]

    t_peris = _invi.stream.simulation._model_1.time_pericentres_Ar(prm_gc, prm, T) #[Myr]
    t_peris = -t_peris/1_000.0 #[Gyr]

    delay = prm_gc['stream']['simulation']['delay']/1_000.0 #[Gyr]

    for t_peri in t_peris:
        ax.axvspan(t_peri, t_peri, alpha=1.0, color='orange')
        ax.axvspan(t_peri-peak_intv, t_peri+peak_intv, alpha=0.1, color='orange')
        ax.axvspan(t_peri+delay, t_peri+delay, alpha=1.0, color='purple')

#-----------------------------------------------------------------------------
