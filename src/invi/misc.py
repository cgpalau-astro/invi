"""Miscellaneous functions.

Example clicker
---------------

import mpl_point_clicker

#Definition clicker on plot:
klicker = mpl_point_clicker.clicker(ax, ["points"], markers=["x"])

#Get points from klicker:
points_polygon = klicker.get_positions()["points"]

#Plot polygon
invi.plot.general.polygon(ax, points_polygon)

#Select points within polygon:
sel = invi.misc.polygon_selection(x, y, points_polygon)"""

import numpy as _np
import invi as _invi

import fnc as _fnc
_shapely = _fnc.utils.lazy.Import('shapely')

__all__ = ["angular_distance", "euclidean_distance",
           "polygon_selection", "print_array",
           "Sphere",
           "seed"]

#-----------------------------------------------------------------------------

def angular_distance(ra1, dec1, ra2, dec2):
    """Angular distance [deg]."""

    ra1 = _invi.units.deg_to_rad(ra1)
    dec1 = _invi.units.deg_to_rad(dec1)
    ra2  = _invi.units.deg_to_rad(ra2)
    dec2 = _invi.units.deg_to_rad(dec2)

    X = _np.cos(dec2)*_np.cos(ra2) - _np.cos(dec1)*_np.cos(ra1)
    Y = _np.cos(dec2)*_np.sin(ra2) - _np.cos(dec1)*_np.sin(ra1)
    Z = _np.sin(dec2)              - _np.sin(dec1)

    C = _np.sqrt(X**2.0 + Y**2.0 + Z**2.0)
    rho = 2.0*_np.arcsin(C/2.0)

    return _invi.units.rad_to_deg(rho)


def euclidean_distance(x0, y0, x1, y1):
    return _np.sqrt((x0-x1)**2.0 + (y0-y1)**2.0)

#-----------------------------------------------------------------------------

def polygon_selection(x, y, points_polygon):
    """Select points ('x', 'y')  within a polygon defined by 'points_polygon'.

    Note
    ----
    1)  The points at the perimeter of the polygon are not included."""

    #Polygon definition
    polygon = _shapely.geometry.Polygon(shell=points_polygon)

    #Polygon selection
    n = len(x)
    sel = [[]]*n
    for i in range(n):
        point = _shapely.geometry.Point(x[i], y[i])
        sel[i] = point.within(polygon)

    return _np.array(sel)

#-----------------------------------------------------------------------------

class Sphere:
    @staticmethod
    def rvs(size, random_state):
        """Random sample on the surface of a sphere in ra, dec.

        Note
        ----
        1)  To cartesian:
        alpha = invi.units.deg_to_rad(ra)
        delta = invi.units.deg_to_rad(dec)

        r = 1.0
        x = r * np.cos(-delta) * np.cos(alpha)
        y = r * np.cos(-delta) * np.sin(alpha)
        z = r * np.sin(delta)"""

        rng = _np.random.default_rng(random_state)

        costheta = rng.uniform(low=-1.0, high=1.0, size=size)
        dec = _invi.units.rad_to_deg(_np.arccos(costheta) - _np.pi/2.0)
        ra = rng.uniform(low=0.0, high=360.0, size=size)

        return ra, dec

#-----------------------------------------------------------------------------

def seed(rng):
    """It generates a different random integer from a generator 'rng' every time
    that this function is executed.

    Note
    ----
    1) Definition random generator: rng = np.random.default_rng(random_state)"""

    return rng.integers(low=0, high=10_000_000, dtype=_np.int64, endpoint=True)

#-----------------------------------------------------------------------------
