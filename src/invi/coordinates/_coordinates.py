"""Coordinate transformations."""

#Note: Errors should be raised when coordinates are out of range.

import ctypes as _c
from pathlib import Path as _Path

import numpy as _np

import fnc as _fnc
import invi as _invi

__all__ = ["ICRS_to_FSR", "FSR_to_ICRS",

           "car_to_cyl", "cyl_to_car",
           "car_to_cyl_galpy", "cyl_galpy_to_car",
           "car_to_sph", "sph_to_car",

           "ICRS_to_sph", "sph_to_ICRS",
           "ICRS_to_phi", "phi_to_ICRS",

           "FSR_to_aaf", "aaf_to_FSR",

           "unwrap_section_corrected", "inv_mod2pi",
           "aaf_to_dgc", "dgc_to_AAF", "aaf_to_AAF",
           "dgc_to_aaf", "AAF_to_dgc", "AAF_to_aaf",

           "parallax_to_r", "r_to_parallax",

           "radec_to_equal_area", "equal_area_to_radec",

           "alpha_to_ALPHA", "ALPHA_to_alpha"]

#-----------------------------------------------------------------------------
#C Implementations

#Absolute path to master.so:
_base_path = _Path(__file__).parent
_file_path = (_base_path / "c/master.so").resolve()

_C = _c.CDLL(_file_path)


_C.ICRS_esf_to_FSR_dex_car.argtypes = [_c.c_double*6, _c.c_double*6]
_C.ICRS_esf_to_FSR_dex_car.restype = _c.POINTER(_c.c_double*6)

def _ICRS_esf_to_FSR_dex_car(x, prm_sun, v_lsr):
    y = [prm_sun['R'], prm_sun['z'],
         prm_sun['U'], prm_sun['V'], prm_sun['W'],
         v_lsr]

    a = (_c.c_double * len(x))(*x)
    b = (_c.c_double * len(y))(*y)
    return _np.array(_C.ICRS_esf_to_FSR_dex_car(a, b).contents)


_C.FSR_dex_car_to_ICRS_esf.argtypes = [_c.c_double*6, _c.c_double*6]
_C.FSR_dex_car_to_ICRS_esf.restype = _c.POINTER(_c.c_double*6)

def _FSR_dex_car_to_ICRS_esf(x, prm_sun, v_lsr):
    y = [prm_sun['R'], prm_sun['z'],
         prm_sun['U'], prm_sun['V'], prm_sun['W'],
         v_lsr]

    a = (_c.c_double * len(x))(*x)
    b = (_c.c_double * len(y))(*y)
    return _np.array(_C.FSR_dex_car_to_ICRS_esf(a, b).contents)

#-----------------------------------------------------------------------------

def ICRS_to_FSR(w_icrs, prm_sun, v_lsr):
    """Coordinate change: ICRS Spherical to FSR dex Cartesian.

    Parameters
    ----------
    w_icrs : np.array
        Coordinates in ICRS Spherical: np.shape(ic_fsr) = (6,n)
        [r, delta, alpha, vr, mu_dela, mu_alpha_str]
        [kpc, deg, deg, km/s, mas/yr, mas/yr]
    prm_sun : dict
        Parameters of the Sun
    v_lsr : float
        Velocity Local Standard of Rest [kpc/Myr].

    Returns
    -------
    np.array
        Coordinates in FSR dex cartesian
        np.shape(ic_fsr) = (6,n)
        [x, y, z, vx, vy, vz] - [kpc, kpc/Myr]"""
    #----------------------------------------------------------------------
    shape = _np.shape(w_icrs)

    if len(shape) == 1:
        if shape[0] != 6:
            raise ValueError(f"ERROR: np.shape(w_icrs) = {shape} != (6,)")
    else:
        if shape[0] != 6:
            raise ValueError(f"ERROR: np.shape(w_icrs) = {shape} != (6,n)")
    #----------------------------------------------------------------------
    v_lsr = _invi.units.kpcMyr_to_kms(v_lsr)
    #----------------------------------------------------------------------
    if len(shape) == 1:
        w_fsr = _ICRS_esf_to_FSR_dex_car(w_icrs.T, prm_sun, v_lsr)
    else:
        n = len(w_icrs.T)
        w_fsr = [[]]*n
        for i in range(n):
            w_fsr[i] = _ICRS_esf_to_FSR_dex_car(w_icrs.T[i], prm_sun, v_lsr)
        w_fsr = _np.array(w_fsr).T
    #----------------------------------------------------------------------
    w_fsr = _invi.units.astronomic_to_galactic(w_fsr)
    #----------------------------------------------------------------------
    return w_fsr

#-----------------------------------------------------------------------------

def FSR_to_ICRS(w_fsr, prm_sun, v_lsr):
    """Coordinate change: FSR dex Cartesian to ICRS Spherical.

    Parameters
    ----------
    w_fsr : np.array
        Coordinates in FSR dex cartesian
        np.shape(ic_fsr) = (6,n)
        [x, y, z, vx, vy, vz] - [kpc, kpc/Myr]
    prm_sun : dict
        Parameters of the Sun
    v_lsr : float
        Velocity Local Standard of Rest [km/s].

    Returns
    -------
    np.array
        Coordinates in ICRS spherical
        np.shape(ic_fsr) = (6,n)
        [r, delta, alpha, vr, mu_dela, mu_alpha_str]
        [kpc, deg, deg, km/s, mas/yr, mas/yr]"""
    #----------------------------------------------------------------------
    shape = _np.shape(w_fsr)

    if len(shape) == 1:
        if shape[0] != 6:
            raise ValueError(f"ERROR: np.shape(w_icrs) = {shape} != (6,)")
    else:
        if shape[0] != 6:
            raise ValueError(f"ERROR: np.shape(w_icrs) = {shape} != (6,n)")
    #----------------------------------------------------------------------
    v_lsr = _invi.units.kpcMyr_to_kms(v_lsr)
    w_fsr = _invi.units.galactic_to_astronomic(w_fsr)
    #----------------------------------------------------------------------
    if len(shape) == 1:
        w_icrs = _FSR_dex_car_to_ICRS_esf(w_fsr.T, prm_sun, v_lsr)
    else:
        n = len(w_fsr.T)
        w_icrs = [[]]*n
        for i in range(n):
            w_icrs[i] = _FSR_dex_car_to_ICRS_esf(w_fsr.T[i], prm_sun, v_lsr)
        w_icrs = _np.array(w_icrs).T
    #----------------------------------------------------------------------
    return w_icrs

#-----------------------------------------------------------------------------

def car_to_cyl(w_car):
    x, y, z, vx, vy, vz = w_car

    R = _np.sqrt(x**2.0 + y**2.0)
    phi = _np.arctan2(y, x) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! and write test

    cos_phi = _np.cos(phi)
    sin_phi = _np.sin(phi)

    vR   =  cos_phi  *vx + sin_phi  *vy
    vphi = -sin_phi/R*vx + cos_phi/R*vy

    w_cyl = _np.array([R, phi, z, vR, vphi, vz])

    return w_cyl


def cyl_to_car(w_cyl):
    R, phi, z, vR, vphi, vz = w_cyl

    x = R*_np.cos(phi)
    y = R*_np.sin(phi)

    cos_phi = _np.cos(phi)
    sin_phi = _np.sin(phi)

    vx = cos_phi*vR - R*sin_phi*vphi
    vy = sin_phi*vR + R*cos_phi*vphi

    w_car = _np.array([x, y, z, vx, vy, vz])

    return w_car

#-----------------------------------------------------------------------------

#Cylindrical coordinates in galpy format and units:

def car_to_cyl_galpy(w_car):
    w_car_galpy = _invi.units.galactic_to_galpy(w_car)
    R, phi, z, vR, vphi, vz = car_to_cyl(w_car_galpy)
    w_cyl_galpy = _np.array([R, vR, vphi*R, z, vz, phi])
    return w_cyl_galpy


def cyl_galpy_to_car(w_cyl_galpy):
    R, vR, R_vphi, z, vz, phi = w_cyl_galpy
    w_cyl_galpy = _np.array([R, phi, z, vR, R_vphi/R, vz])
    w_car_galpy = cyl_to_car(w_cyl_galpy)
    w_car = _invi.units.galpy_to_galactic(w_car_galpy)
    return w_car

#-----------------------------------------------------------------------------

def car_to_sph(w_car):
    x, y, z, vx, vy, vz = w_car

    r = _np.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta = _np.arccos(z/r)
    #phi = _np.arctan2(y, x)
    phi = _fnc.numeric.arctan2(y, x)

    cos_phi = _np.cos(phi)
    sin_phi = _np.sin(phi)
    cos_theta = _np.cos(theta)
    sin_theta = _np.sin(theta)

    vr     =  sin_theta*cos_phi*vx   + sin_theta*sin_phi*vy   + cos_theta*vz
    vtheta =  cos_phi*cos_theta/r*vx   + sin_phi*cos_theta/r*vy   - sin_theta/r*vz
    vphi   = -sin_phi/sin_theta/r*vx + cos_phi/sin_theta/r*vy

    w_sph = _np.array([r, theta, phi, vr, vtheta, vphi])

    return w_sph


def sph_to_car(w_sph):
    r, theta, phi, vr, vtheta, vphi = w_sph

    cos_phi = _np.cos(phi)
    sin_phi = _np.sin(phi)
    cos_theta = _np.cos(theta)
    sin_theta = _np.sin(theta)

    x = r*sin_theta*cos_phi
    y = r*sin_theta*sin_phi
    z = r*cos_theta

    vx = sin_theta*cos_phi*vr + r*cos_theta*cos_phi*vtheta - r*sin_theta*sin_phi*vphi
    vy = sin_theta*sin_phi*vr + r*cos_theta*sin_phi*vtheta + r*sin_theta*cos_phi*vphi
    vz = cos_theta*vr         - r*sin_theta*vtheta

    w_car = _np.array([x, y, z, vx, vy, vz])

    return w_car

#-----------------------------------------------------------------------------

def ICRS_to_sph(w_icrs):
    r, delta, alpha, mu_r, mu_delta, mu_alpha_str = w_icrs

    cosd = _np.cos(_invi.units.deg_to_rad(delta))

    theta = _invi.units.deg_to_rad(delta) + _np.pi/2.0
    phi = _invi.units.deg_to_rad(alpha)
    vr = _invi.units.kms_to_kpcMyr(mu_r)
    vtheta = _invi.units.masyr_to_radMyr(mu_delta)
    vphi = _invi.units.masyr_to_radMyr(mu_alpha_str/cosd)

    return _np.array([r, theta, phi, vr, vtheta, vphi])


def sph_to_ICRS(w_sph):
    r, theta, phi, vr, vtheta, vphi = w_sph

    delta = _invi.units.rad_to_deg(theta - _np.pi/2.0)
    alpha = _invi.units.rad_to_deg(phi)

    cosd = _np.cos(_invi.units.deg_to_rad(delta))

    mu_r = _invi.units.kpcMyr_to_kms(vr)
    mu_delta = _invi.units.radMyr_to_masyr(vtheta)
    mu_alpha_str = _invi.units.radMyr_to_masyr(vphi)*cosd

    return _np.array([r, delta, alpha, mu_r, mu_delta, mu_alpha_str])

#-----------------------------------------------------------------------------
#Stream coordinates phi

def _linear(w_car, J):
    """Linear transformation"""
    x, y, z, vx, vy, vz = w_car
    a, b, c = _np.dot(J, _np.array([x, y, z]).T)
    va, vb, vc = _np.dot(J, _np.array([vx, vy, vz]).T)
    return _np.array([a, b, c, va, vb, vc])


def _dot(w, J):
    """Dot product for each element of the array."""
    if _np.shape(w) == (6,):
        return _linear(w, J)

    n = len(w[0])
    w_rot = [[]]*n
    for i in range(n):
        w_rot[i] = _linear(w.T[i], J)
    return _np.asarray(w_rot).T


class ICRS_to_phi:
    @staticmethod
    def angles(w_icrs, alpha_x, alpha_y, alpha_z):
        w_sph = ICRS_to_sph(w_icrs)
        w_car = sph_to_car(w_sph)
        w_car_rot = _invi.coordinates.rotation.w(w_car, alpha_x, alpha_y, alpha_z)
        w_sph_rot = car_to_sph(w_car_rot)
        w_phi = sph_to_ICRS(w_sph_rot) #w_icrs_rot
        return w_phi

    @staticmethod
    def matrix(w_icrs, J):
        w_sph = ICRS_to_sph(w_icrs)
        w_car = sph_to_car(w_sph)
        w_car_rot = _dot(w_car, J)
        w_sph_rot = car_to_sph(w_car_rot)
        w_phi = sph_to_ICRS(w_sph_rot) #w_icrs_rot
        return w_phi


class phi_to_ICRS:
    @staticmethod
    def angles(w_phi, alpha_x, alpha_y, alpha_z):
        w_sph_rot = ICRS_to_sph(w_phi) #w_icrs_rot
        w_car_rot = sph_to_car(w_sph_rot)
        w_car = _invi.coordinates.rotation.w_inv(w_car_rot, alpha_x, alpha_y, alpha_z)
        w_sph = car_to_sph(w_car)
        w_icrs = sph_to_ICRS(w_sph)
        return w_icrs

    @staticmethod
    def matrix(w_phi, J):
        w_sph_rot = ICRS_to_sph(w_phi) #w_icrs_rot
        w_car_rot = sph_to_car(w_sph_rot)
        w_car = _dot(w_car_rot, J.T)
        w_sph = car_to_sph(w_car)
        w_icrs = sph_to_ICRS(w_sph)
        return w_icrs

#-----------------------------------------------------------------------------
#Angles, actions, and frequencies

def FSR_to_aaf(*args, **kwargs):
    return _invi.galpy.aaf.isochrone_approx(*args, **kwargs)


def aaf_to_FSR(*args, **kwargs):
    return _invi.galpy.aaf.torus_mapper(*args, **kwargs)

#-----------------------------------------------------------------------------

def unwrap_section_corrected(x, Ar=False):
    """Corrects unwrap when only a section of the leading arm is selected."""
    x_unwrap = _np.unwrap([x], period=2.0*_np.pi)[0]
    if (Ar == True) & (_np.median(x_unwrap) < _np.pi):
        x_unwrap += 2.0*_np.pi
    return x_unwrap


def inv_mod2pi(s_aaf):
    """Unwrap angles (set for M68)."""
    s_aaf_unwrap = _np.copy(s_aaf)
    s_aaf_unwrap[0] = unwrap_section_corrected(s_aaf[0], Ar=True)
    s_aaf_unwrap[1] = unwrap_section_corrected(s_aaf[1])
    s_aaf_unwrap[2] = unwrap_section_corrected(s_aaf[2])
    return s_aaf_unwrap

#-----------------------------------------------------------------------------

def aaf_to_dgc(s_aaf, gc_aaf):
    """Angle, action, frequencies i=(r, phi, z) -> Relative to globular cluster i=(r, phi, z)."""
    s_aaf_unwrap = inv_mod2pi(s_aaf)
    return _np.array([s_aaf_unwrap[i] - gc_aaf[i] for i in range(9)])


def dgc_to_AAF(aaf_dgc, varphi):
    """Angle, action, frequencies relative to globular cluster i=(r, phi, z) ->
    Principal axes of the stream i=(1, 2, 3)."""
    return _invi.coordinates.rotation.aaf(aaf_dgc, varphi['x'], varphi['y'], varphi['z'])


def aaf_to_AAF(s_aaf, gc_aaf, varphi):
    """Angle, action, frequencies i=(r, phi, z) -> Principal axes of the stream i=(1, 2, 3)."""
    s_dgc = aaf_to_dgc(s_aaf, gc_aaf)
    s_AAF = dgc_to_AAF(s_dgc, varphi)
    return s_AAF

#-----------------------------------------------------------------------------

def _mod2pi(s_aaf_unwrap):
    """Wrap angles (mod 2*pi)."""
    s_aaf = _np.copy(s_aaf_unwrap)
    s_aaf[0] = s_aaf_unwrap[0] % (2.0*_np.pi)
    s_aaf[1] = s_aaf_unwrap[1] % (2.0*_np.pi)
    s_aaf[2] = s_aaf_unwrap[2] % (2.0*_np.pi)
    return s_aaf


def dgc_to_aaf(s_dgc, gc_aaf):
    """Angle, action, frequencies relative to globular cluster i=(r, phi, z) -> i=(r, phi, z)"""
    s_aaf_unwrap = _np.array([s_dgc[i] + gc_aaf[i] for i in range(9)])
    s_aaf = _mod2pi(s_aaf_unwrap)
    return s_aaf


def AAF_to_dgc(s_AAF, varphi):
    """Angle, action, frequencies in the principal axes of the stream i=(1, 2, 3) -> Relative to globular cluster i=(r, phi, z)"""
    s_dgc = _invi.coordinates.rotation.aaf_inv(s_AAF, varphi['x'], varphi['y'], varphi['z'])
    return s_dgc


def AAF_to_aaf(s_AAF, gc_aaf, varphi):
    """Angle, action, frequencies in the principal axes of the stream i=(1, 2, 3) -> i=(r, phi, z)"""
    s_dgc = AAF_to_dgc(s_AAF, varphi)
    s_aaf = dgc_to_aaf(s_dgc, gc_aaf)
    return s_aaf

#-----------------------------------------------------------------------------

def parallax_to_r(parallax):
    """Parallax [mas] to r [kpc]."""
    return 1.0/parallax

def r_to_parallax(r):
    """r [kpc] to parallax [mas]."""
    return 1.0/r

#-----------------------------------------------------------------------------
#Equal area sky coordinates

def radec_to_equal_area(ra, dec):
    """From ra, dec to equal area coordinates.

    Note
    ----
    1)  ea_1 in [0, 4]
        ea_2 in [-1, 1]"""

    ea_1 = _invi.units.deg_to_rad(ra)/_np.pi*2.0

    #_np.cos(_np.deg2rad(dec) - _np.pi/2)
    ea_2 = _np.sin(_invi.units.deg_to_rad(dec))

    return ea_1, ea_2

def equal_area_to_radec(ea_1, ea_2):
    """From equal area coordinates to ra, dec."""

    ra = _invi.units.rad_to_deg(ea_1*_np.pi/2.0)
    dec = _invi.units.rad_to_deg(_np.arcsin(ea_2))

    return ra, dec

#-----------------------------------------------------------------------------
#Stripping points

def alpha_to_ALPHA(s_alpha, varphi):
    """Stripping points relative to the progenitor alpha_i i={r, theta, phi} ->
    Principal axes of the stream ALPHA_i i={1, 2, 3}"""
    s_ALPHA = _invi.coordinates.rotation.xyz(s_alpha[0], s_alpha[1], s_alpha[2],
                                             varphi['x'], varphi['y'], varphi['z'])
    return _np.array(s_ALPHA)


def ALPHA_to_alpha(s_ALPHA, varphi):
    """Principal axes of the stream ALPHA_i i={1, 2, 3} ->
    stripping points relative to the progenitor alpha_i i={r, theta, phi}"""
    s_alpha = _invi.coordinates.rotation.xyz_inv(s_ALPHA[0], s_ALPHA[1], s_ALPHA[2],
                                                 -varphi['x'], -varphi['y'], -varphi['z'])
    return _np.array(s_alpha)

#-----------------------------------------------------------------------------
