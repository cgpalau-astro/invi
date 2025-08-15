"""Spheroidal to cartesian coordinates map, and plot spheroidal grid."""

import numpy as _np

__all__ = ["spheroidal_to_car", "car_to_spheroidal", "plot_spheroidal_grid"]

#-----------------------------------------------------------------------------

def _oblate_to_car(delta, mu, nu, psi):
    """Oblate spheroidal to cartesian coordinates.

    Note
    ----
    1)  Delta on the x axis.
    2)  mu in [0, inf)
    3)  nu in [-pi/2, pi/2)
    4)  phi in [-pi, pi)

    Web
    ---
    1)  https://www.wikiwand.com/en/articles/Oblate_spheroidal_coordinates"""
    #-----------------------------------------------
    if delta < 0.0:
        raise ValueError("'delta' must be positive")
    #-----------------------------------------------
    x = delta * _np.cosh(mu) * _np.cos(nu) * _np.cos(psi)
    y = delta * _np.cosh(mu) * _np.cos(nu) * _np.sin(psi)
    z = delta * _np.sinh(mu) * _np.sin(nu)

    return x, y, z

def _car_to_oblate(delta, x, y, z):
    """Cartesian to oblate spheroidal coordinates.

    Note
    ----
    1)  Delta on the x axis.

    Web
    ---
    1)  https://www.wikiwand.com/en/articles/Oblate_spheroidal_coordinates"""
    #-----------------------------------------------
    if delta < 0.0:
        raise ValueError("'delta' must be positive")
    #-----------------------------------------------
    R = _np.sqrt(x**2.0 + y**2.0)
    d1 = _np.sqrt((R + delta)**2.0 + z**2.0)
    d2 = _np.sqrt((R - delta)**2.0 + z**2.0)

    mu = _np.abs( _np.arccosh((d1 + d2) / (2.0*delta)) )
    nu = _np.arccos((d1 - d2) / (2.0*delta)) * _np.sign(z)
    psi = _np.arctan2(y, x)

    return mu, nu, psi

#-----------------------------------------------------------------------------

def _prolate_to_car(delta, mu, nu, psi):
    """Prolate spheroidal to cartesian coordinates.

    Note
    ----
    1)  Delta on the x axis.
    2)  mu in [0, inf)
    3)  nu in [0, pi)
    4)  phi in [0, 2*pi)

    Web
    ---
    1)  https://www.wikiwand.com/en/articles/Prolate_spheroidal_coordinates"""
    #-----------------------------------------------
    if delta < 0.0:
        raise ValueError("'delta' must be positive")
    #-----------------------------------------------
    x = delta * _np.sinh(mu) * _np.sin(nu) * _np.cos(psi)
    y = delta * _np.sinh(mu) * _np.sin(nu) * _np.sin(psi)
    z = delta * _np.cosh(mu) * _np.cos(nu)

    return x, y, z

def _car_to_prolate(delta, x, y, z):
    """Cartesian to prolate spheroidal coordinates.

    Note
    ----
    1)  Delta on the z axis.

    Web
    ---
    1)  https://www.wikiwand.com/en/articles/Prolate_spheroidal_coordinates"""
    #-----------------------------------------------
    if delta < 0.0:
        raise ValueError("'delta' must be positive")
    #-----------------------------------------------
    R = _np.sqrt(x**2.0 + y**2.0)
    d1 = _np.sqrt(R**2.0 + (z + delta)**2.0)
    d2 = _np.sqrt(R**2.0 + (z - delta)**2.0)

    mu = _np.arccosh((d1 + d2) / (2.0*delta))
    nu = _np.arccos((d1 - d2) / (2.0*delta))
    psi = _np.arctan2(y, x)

    return mu, nu, psi

#-----------------------------------------------------------------------------

def _sph_to_car(mu, nu, psi):
    """Spherical to cartesian coordinates."""
    x = mu * _np.sin(nu) * _np.cos(psi)
    y = mu * _np.sin(nu) * _np.sin(psi)
    z = mu * _np.cos(nu)

    return x, y, z

def _car_to_sph(x, y, z):
    """Cartesian to spherical coordinates."""
    mu = _np.sqrt(x**2.0 + y**2.0 + z**2.0)
    nu = _np.arccos(z / mu)
    psi = _np.arctan2(y, x)

    return mu, nu, psi

#-----------------------------------------------------------------------------

def spheroidal_to_car(delta, mu, nu, psi):
    """Spheroidal to cartesian coordinates."""
    if delta > 0.0:
        x, y, z = _prolate_to_car(delta, mu, nu, psi)
    elif delta == 0.0:
        x, y, z = _sph_to_car(mu, nu, psi)
    elif delta < 0.0:
        x, y, z = _oblate_to_car(_np.abs(delta), mu, nu, psi)

    return x, y, z

def car_to_spheroidal(delta, x, y, z):
    """Cartesian to spheroidal coordinates."""
    if delta > 0.0:
        mu, nu, psi = _car_to_prolate(delta, x, y, z)
    elif delta == 0.0:
        mu, nu, psi = _car_to_sph(x, y, z)
    elif delta < 0.0:
        mu, nu, psi = _car_to_oblate(_np.abs(delta), x, y, z)

    return mu, nu, psi

#-----------------------------------------------------------------------------

def plot_spheroidal_grid(ax, delta=1.0, n_mu=12, n_nu=10, xz_limit=1.0, size_focus=10.0, line_style=None):
    """Plot spheroidal coordinates on the x-z cartesian plane.

    Parameters
    ----------
    ax: Axes
    delta: float
    n_mu: int
        Number of mu lines (radial coordinate for delta=0).
    n_nu: int
        Number of nu lines (angular coordinate for delta=0).
    xz_limit: float
        x_limit for delta > 0. z_limit for delta < 0.
    size_focus: float
        Size dot indicating foci position.
    line_style: dict
        Style coordinate lines: {'alpha':1.0, 'color':'0.75', 'linestyle':'--', 'linewidth':0.5}

    Returns
    -------
    ax: Axes"""
    if line_style is None:
        line_style = {'alpha': 1.0, 'color': '0.75', 'linestyle': '--', 'linewidth': 0.5}
    #------------------------------------------------------
    #Number of points used to draw the lines.
    n = 10_000
    #Plot on plane x-z
    psi = 0.0
    #------------------------------------------------------
    #Define limits
    if delta > 0.0:
        mu_limit = _np.arcsinh(xz_limit / delta)
    elif delta == 0.0:
        mu_limit = xz_limit
    elif delta < 0.0:
        mu_limit = _np.arcsinh(xz_limit / _np.abs(delta))
    #------------------------------------------------------
    #Plot nu
    nu = _np.linspace(0.0, 2.0*_np.pi, n)
    for mu in _np.linspace(0.0, mu_limit, n_nu):
        x, _y, z = spheroidal_to_car(delta, mu, nu, psi)
        ax.plot(x, z, **line_style, zorder=0)
    #------------------------------------------------------
    #Plot mu
    mu = _np.linspace(0.0, mu_limit, n)
    for nu in _np.arange(0.0, 2.0*_np.pi, _np.pi / n_mu):
        x, _y, z = spheroidal_to_car(delta, mu, nu, psi)
        ax.plot(x, z, **line_style, zorder=0)
    #------------------------------------------------------
    #Plot foci
    if delta > 0.0:
        ax.scatter(0.0, delta, alpha=1.0, color="k", s=size_focus, zorder=1)
        ax.scatter(0.0, -delta, alpha=1.0, color="k", s=size_focus, zorder=1)
    elif delta == 0.0:
        ax.scatter(0.0, 0.0, alpha=1.0, color="k", s=size_focus, zorder=1)
    elif delta < 0.0:
        ax.scatter(delta, 0.0, alpha=1.0, color="k", s=size_focus, zorder=1)
        ax.scatter(-delta, 0.0, alpha=1.0, color="k", s=size_focus, zorder=1)
    #------------------------------------------------------
    return ax

#-----------------------------------------------------------------------------
