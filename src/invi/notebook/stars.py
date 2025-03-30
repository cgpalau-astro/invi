"""Plots for Jupyter notebooks in code/5-stars folder."""

import numpy as _np
import tqdm as _tqdm

import fnc as _fnc
import invi as _invi

__all__ = ["components",
           "likelihood", "eigenvalues", "axes",
           "aaf", "ratio",
           "freq_comparison"]

#-----------------------------------------------------------------------------
#1-classification

def components(s_aaf_dict, components, size=0.1):

    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.0, 2.0))

    x = s_aaf_dict['AAF']['A1']
    y = s_aaf_dict['AAF']['A2']

    ax.scatter(x[components['stream']], y[components['stream']], s=size, color="k")
    ax.scatter(x[components['gc']], y[components['gc']], s=size, color="r")
    ax.scatter(x[components['escapees']], y[components['escapees']], s=size, color="b")

    ax.scatter(x[components['leading']], y[components['leading']]+0.50, s=size, color="k")
    ax.scatter(x[components['trailing']], y[components['trailing']]+0.75, s=size, color="k")

    desp = 0.5
    for item in ['s1', 's2', 's3', 'sn']:
        sel = components['internal']['leading'][item]
        ax.scatter(x[sel], y[sel]-desp, s=size, color="g")

        sel = components['internal']['trailing'][item]
        ax.scatter(x[sel], y[sel]-desp, s=size, color="g")

        desp += 0.1

    ax.set_aspect(1.0)
    limit = 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('A1 [rad]')
    ax.set_ylabel('A2 [rad]')

#-----------------------------------------------------------------------------
#2-principal_axes

def likelihood(variable, varphi_0, st_dgc, n=200, progress=True):
    #Initial condition angle [rad]
    varphi_x, varphi_y, varphi_z = varphi_0
    #-------------------------------------------------------------
    varphi = _np.linspace(-_np.pi/2.0, _np.pi/2.0, n)
    lk = _np.zeros(n)
    for i in _tqdm.tqdm(range(n), ncols=78, disable=not progress):
        match variable:
            case "x":
                x = [varphi[i], varphi_y, varphi_z]
            case "y":
                x = [varphi_x, varphi[i], varphi_z]
            case "z":
                x = [varphi_x, varphi_y, varphi[i]]

        lk[i] = _invi.stream.principal_axes.likelihood(x, st_dgc)
    #-------------------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 1, fc=1.4)

    #Plot likelihood
    ax.plot(varphi, lk)

    #Plot Minimum likelihood
    varphi_min = varphi[_np.argmin(lk)]
    ax.plot([varphi_min, varphi_min],
            [_np.min(lk), _np.max(lk)],
            linestyle="--",
            c="r",
            label=f"Min. Likelihood: varphi_{variable}={varphi_min:0.5f} [rad]")

    ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1)
    ax.set_xlabel(f"alpha_{variable} [rad]")
    ax.set_ylabel("Likelihood")
    ax.set_xlim(-_np.pi/2.0, _np.pi/2.0)
    #-------------------------------------------------------------


def eigenvalues(st_AAF):
    #-----------------------------------------------
    def plot_median_mad(ax, eig):
        median = _np.median(eig)
        mad = _fnc.stats.mad(eig)
        x = [median - mad, median + mad]
        ax.fill_between(x, 0, 250.0, color='red', alpha=0.15)
        ax.plot([median, median], [0.0, 250.0], linestyle="--", color="r")
    #-----------------------------------------------
    eig_1 = st_AAF[6]/st_AAF[3] * 1_000.0 #[mrad/kpc^2]
    eig_2 = st_AAF[7]/st_AAF[4] * 1_000.0
    eig_3 = st_AAF[8]/st_AAF[5] * 1_000.0
    #-----------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=(3.5, 1))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]

    kwargs = {'bins': 1_000, 'color': 'k', 'histtype': 'step'}

    ax1.hist(eig_1, range=[-12.0, -8.5], **kwargs)
    ax1.set_xlim(-12.0, -8.5)
    ax1.set_ylim(0.0, 150.0)

    ax2.hist(eig_2, range=[-0.7, 0.0], **kwargs)
    ax2.set_xlim(-0.7, 0.0)
    ax2.set_ylim(0.0, 250.0)

    ax3.hist(eig_3, range=[0.18, 0.3], **kwargs)
    ax3.set_xlim(0.18, 0.3)
    ax3.set_ylim(0.0, 250.0)

    plot_median_mad(ax1, eig_1)
    plot_median_mad(ax2, eig_2)
    plot_median_mad(ax3, eig_3)

    ax1.set_ylabel("Number")
    ax1.set_xlabel("eig_1 [mrad/kpc^2]")
    ax2.set_xlabel("eig_2 [mrad/kpc^2]")
    ax3.set_xlabel("eig_3 [mrad/kpc^2]")


def axes(i, st_AAF, eig, limit=1.0):
    #i in {1, 2, 3}
    i -= 1

    _fig, ax = _fnc.plot.figure(1, 1, 1.75)

    #Select stream stars
    ax.scatter(st_AAF[i], st_AAF[i+6]*1_000.0, s=0.1, color="k")
    ax.scatter(st_AAF[i], st_AAF[i+3]*_np.median(eig[i])*1_000.0, s=0.1, color="r")

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("A" + str(i+1) + " [rad]")
    ax.set_ylabel("F" + str(i+1) + " [rad/Gyr]")

#-----------------------------------------------------------------------------
#3-plot

def aaf(x, y, s_aaf_dict, components, limit=(1.5, 1.5), units=('-', '-'), fc=(1.0, 1.0), aspect=1.0):
    _fig, ax = _fnc.plot.figure(1, 1, fc=1.75)

    a = s_aaf_dict['AAF'][x] * fc[0]
    b = s_aaf_dict['AAF'][y] * fc[1]

    c = components['gc']
    st = components['stream']

    ax.scatter(a[c], b[c], s=0.1, color="r")
    ax.scatter(a[st], b[st], s=0.1, color="k")

    if aspect != None:
        ax.set_aspect(aspect)

    ax.set_xlim(-limit[0], limit[0])
    ax.set_ylim(-limit[1], limit[1])
    ax.set_xlabel(x + f" [{units[0]}]")
    ax.set_ylabel(y + f" [{units[1]}]")


def ratio(x, y, z, s_aaf_dict, gc_aaf_dict, components, limit=(0.005, 0.005)):
    _fig, ax = _fnc.plot.figure(1, 1, fc=1.75)

    a = s_aaf_dict['aaf'][x] / s_aaf_dict['aaf'][y]
    b = s_aaf_dict['aaf'][z] / s_aaf_dict['aaf'][y]

    a -= gc_aaf_dict[x] / gc_aaf_dict[y]
    b -= gc_aaf_dict[z] / gc_aaf_dict[y]

    c = components['gc']
    st = components['stream']

    ax.scatter(a[c], b[c], s=0.1, color="r")
    ax.scatter(a[st], b[st], s=0.1, color="k")
    ax.scatter(0.0, 0.0, s=15.0, color="g")

    ax.set_aspect(1.0)

    ax.set_xlim(-limit[0], limit[0])
    ax.set_ylim(-limit[1], limit[1])
    ax.set_xlabel(x + '/' + y + " [-]")
    ax.set_ylabel(z + '/' + y + " [-]")

#-----------------------------------------------------------------------------
#7-eig_hessian

def freq_comparison(M68, eig_torus, eig_rot, s=0.5):
    def set_axis(ax, limit, i):
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel(f"A{i} [rad]")
        ax.set_ylabel(f"F{i} [rad/Gyr]")
    #----------------------------------------------------
    st = M68['stars']['components']['stream']

    A1 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['A1'])
    A2 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['A2'])
    A3 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['A3'])

    J1 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['J1'])
    J2 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['J2'])
    J3 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['J3'])

    F1 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['F1'])
    F2 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['F2'])
    F3 = _np.asarray(M68['stars']['ang_act_freq']['AAF']['F3'])
    #----------------------------------------------------
    _fig, ax = _fnc.plot.figure(1, 3, fc=(3.0, 1.25))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    #----------------------------------------------------
    ax1.scatter(A1[st], F1[st]*1_000.0, s=s, color="k", label='N-body')
    ax1.scatter(A1[st], J1[st]*eig_rot[0]*1_000.0, s=s, color="b", label='rot')
    ax1.scatter(A1[st], J1[st]*eig_torus[0]*1_000.0, s=s, color="r", label='torus')

    limit = 1.2
    set_axis(ax1, limit, '1')
    ax1.legend(markerscale=5.0)
    #----------------------------------------------------
    ax2.scatter(A2[st], F2[st]*1_000.0, s=s, color="k")
    ax2.scatter(A2[st], J2[st]*eig_rot[1]*1_000.0, s=s, color="b")
    ax2.scatter(A2[st], J2[st]*eig_torus[1]*1_000.0, s=s, color="r")

    limit = 3.0E-2
    set_axis(ax2, limit, '2')
    #----------------------------------------------------
    ax3.scatter(A3[st], F3[st]*1_000.0, s=s, color="k")
    ax3.scatter(A3[st], J3[st]*eig_rot[2]*1_000.0, s=s, color="b")
    ax3.scatter(A3[st], J3[st]*eig_torus[2]*1_000.0, s=s, color="r")

    limit = 3.6E-2
    set_axis(ax3, limit, '3')
    #----------------------------------------------------

#-----------------------------------------------------------------------------
