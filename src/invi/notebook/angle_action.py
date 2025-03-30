"""Plots for Jupyter notebooks in code/4-angle_action folder."""

import time as _time
import numpy as _np

import fnc as _fnc
import invi as _invi

__all__ = ["cyl_sph", "car", "t_cyl_sph",
           "aaf", "parameter", "b",
           "aaf_gc"]

#-----------------------------------------------------------------------------

def _angle_relative_freq(t, aaf, index):
    if index == 'r':
        A = _invi.coordinates.unwrap_section_corrected(aaf['A'+index], Ar=True)
    else:
        A = _invi.coordinates.unwrap_section_corrected(aaf['A'+index], Ar=False)

    F = aaf['F'+index]

    return A - A[0] - F*t

#-----------------------------------------------------------------------------
#1-isochrone_approx_Fr_or_FR

def cyl_sph(orb):
    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5,1.5))
    ax1, ax2 = ax[0], ax[1]
    kwargs = {'alpha': 1.0, 'linewidth': 0.25}
    lim_x = [0.0, 10.0]
    lim_y = [-10.0, 10.0]

    ax1.plot(orb['orbit']['FSR']['cyl']['R'], orb['orbit']['FSR']['cyl']['z'], **kwargs)
    ax1.set_aspect(1.0)
    ax1.set_xlabel("R [kpc]")
    ax1.set_ylabel("z [kpc]")
    ax1.set_xlim(lim_x)
    ax1.set_ylim(lim_y)

    ax2.plot(orb['orbit']['FSR']['sph']['r'], orb['orbit']['FSR']['cyl']['z'], **kwargs)
    ax2.set_aspect(1.0)
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel("z [kpc]")
    ax2.set_xlim(lim_x)
    ax2.set_ylim(lim_y)

def car(orb):
    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5,1.5))
    kwargs = {'alpha': 1.0, 'linewidth': 0.25}
    ax1, ax2 = ax[0], ax[1]
    lim = [-10.0, 10.0]

    ax1.plot(orb['orbit']['FSR']['car']['x'], orb['orbit']['FSR']['car']['y'], **kwargs)
    ax1.set_aspect(1.0)
    ax1.set_xlabel("x [kpc]")
    ax1.set_ylabel("y [kpc]")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)

    ax2.plot(orb['orbit']['FSR']['car']['x'], orb['orbit']['FSR']['car']['z'], **kwargs)
    ax2.set_aspect(1.0)
    ax2.set_xlabel("x [kpc]")
    ax2.set_ylabel("z [kpc]")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)

def t_cyl_sph(orb):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.0,1.25))
    ax.plot(orb['t'], orb['orbit']['FSR']['cyl']['R'], c="r", label="R")
    ax.plot(orb['t'], orb['orbit']['FSR']['sph']['r'], c="k", label="r")
    ax.set_xlim(0.0, 1_000.0)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel("r , R [kpc]")
    ax.legend()

#-----------------------------------------------------------------------------
#2-isochrone_approx_parameters

def aaf(aaf, fft, t):
    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.75, 1.25))
    ax1, ax2 = ax[0], ax[1]

    ax1.plot(t, aaf['Jr'])
    ax1.plot(t, aaf['Jphi'])
    ax1.plot(t, aaf['Jz'])
    ax1.set_xlabel("t [Myr]")
    ax1.set_ylabel("J_i [kpc^2 / Myr]")

    ax2.plot(t, aaf['Fr']*1_000.0, label="J_r")
    ax2.plot(t, aaf['Fphi']*1_000.0, label="J_phi")
    ax2.plot(t, aaf['Fz']*1_000.0, label="J_z")

    kwargs = {'linestyle': "--", 'color': "grey"}
    ax2.plot([t[0], t[-1]], [fft['Fr'], fft['Fr']], label="fft F_i", **kwargs)
    ax2.plot([t[0], t[-1]], [fft['Fphi'], fft['Fphi']], **kwargs)
    ax2.plot([t[0], t[-1]], [fft['Fz'], fft['Fz']], **kwargs)

    ax2.set_xlabel("t [Myr]")
    ax2.set_ylabel("F_i [rad / Gyr]")
    ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

def parameter(t, w_fsr, potential_galpy, fft, par_ref, index, parameter, values):
    #------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=(4.0, 1.25))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    #------------------------------------------------
    b, maxn, tintJ, ntintJ = par_ref.values()
    #------------------------------------------------
    J = 'J' + index
    F = 'F' + index
    #------------------------------------------------
    time = _np.array([])
    for par in values:

        match parameter:
            case 'b':
                b = par
            case 'maxn':
                maxn = par
            case 'tintJ':
                tintJ = par
            case 'ntintJ':
                ntintJ = par

        start_time = _time.perf_counter_ns()
        w_aaf = _invi.coordinates.FSR_to_aaf(w_fsr, potential_galpy, b, maxn, tintJ, ntintJ, n_cpu=None, progress=True)
        end_time = _time.perf_counter_ns()
        elapsed_time = end_time - start_time
        time = _np.append(time, elapsed_time)

        aaf = _invi.dicts.array_to_dict(w_aaf, 'aaf')

        label = f"{parameter} = {par} : {elapsed_time/_np.min(time):0.2f}"
        ax1.plot(t, _angle_relative_freq(t, aaf, index))
        ax2.plot(t, aaf[J])
        ax3.plot(t, aaf[F]*1_000.0, label=label)
    #------------------------------------------------
    ax1.set_xlabel("t [Myr]")
    ax1.set_ylabel(f"A_{index} [rad]")
    #ax1.legend()

    ax2.set_xlabel("t [Myr]")
    ax2.set_ylabel(f"J_{index} [kpc^2 / Myr]")
    #ax2.legend()

    kwargs = {'linestyle': "--", 'color': "grey"}
    ax3.plot([t[0], t[-1]], [fft[F], fft[F]], **kwargs)

    ax3.set_xlabel("t [Myr]")
    ax3.set_ylabel(f"F_{index} [rad / Gyr]")
    ax3.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


def b(w_fsr, potential_galpy, par_ref, rng, n):
    #------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.0, 1.25))
    #------------------------------------------------
    b, maxn, tintJ, ntintJ = par_ref.values()
    #------------------------------------------------
    b = _np.linspace(rng[0], rng[1], n)
    cv = _np.zeros(n)
    for i in range(n):
        w_aaf = _invi.coordinates.FSR_to_aaf(w_fsr, potential_galpy, b[i], maxn, tintJ, ntintJ, n_cpu=None, progress=True)
        aaf = _invi.dicts.array_to_dict(w_aaf, 'aaf')
        cv[i] = _fnc.stats.cv(aaf['Jr'])
    #------------------------------------------------
    ax.plot(b, cv, label="b")
    i = _np.argmin(cv)
    print(f"b min = {b[i]}")
    ax.scatter(b[i], cv[i])
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel("cv(J_r) [kpc^2 / Myr]")
    ax.set_xlim(rng)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

#-----------------------------------------------------------------------------
#4-angles_actions_gc

def aaf_gc(t, w_aaf_dict, w_aaf_acc_dict):
    #---------------------------------------------------------
    fc = (4.0, 1.25)
    index = ['r', 'phi', 'z']
    #---------------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=fc)
    for j in range(3):
        ax[j].plot(t, _angle_relative_freq(t, w_aaf_dict, index[j]), label='invi')
        ax[j].plot(t, _angle_relative_freq(t, w_aaf_acc_dict, index[j]), label='accuracy')
        ax[j].set_xlabel("t [Myr]")
        ax[j].set_ylabel(f"Delta A_{index[j]} [rad]")
    ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #---------------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=fc)
    for j in range(3):
        ax[j].plot(t, w_aaf_dict['J'+index[j]], label='invi')
        ax[j].plot(t, w_aaf_acc_dict['J'+index[j]], label='accuracy')
        ax[j].set_xlabel("t [Myr]")
        ax[j].set_ylabel(f"J_{index[j]} [kpc^2 / Myr]")
    ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #---------------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=fc)
    for j in range(3):
        ax[j].plot(t, w_aaf_dict['F'+index[j]]*1_000.0, label='invi')
        ax[j].plot(t, w_aaf_acc_dict['F'+index[j]]*1_000.0, label='accuracy')
        ax[j].set_xlabel("t [Myr]")
        ax[j].set_ylabel(f"F_{index[j]} [rad / Gyr]")
    ax[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

#-----------------------------------------------------------------------------
