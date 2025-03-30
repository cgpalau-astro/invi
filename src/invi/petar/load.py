"""Load data from petar snapshots.

Note
----
1)  Files compressed with: gzip --best data.x"""

import numpy as _np

import invi as _invi

__all__ = ["delta_gc", "FSR", "mass"]

#-----------------------------------------------------------------------------

def delta_gc(path_file, snapshot):
    """Phase-space position of the stars relative to the globular cluster from
    petar snapshot file 'data.snapshot.gz'."""

    file_name = f"{path_file}/data.{snapshot}.gz"

    #Phase-space stars relative to globular cluster [pc, pc/Myr]
    w_delta_petar = _np.loadtxt(file_name, skiprows=1, usecols=(1,2,3,4,5,6))

    #Sort stars
    n = _np.loadtxt(file_name, skiprows=1, usecols=(9))
    w_delta_petar = w_delta_petar[_np.argsort(n)]

    w_delta = _invi.units.petar_to_galactic(w_delta_petar.T) #[kpc, kpc/Myr]

    return w_delta

#-----------------------------------------------------------------------------

def FSR(path_file, snapshot):
    """Phase-space position of the stars in FSR from petar snapshot file
    'data.snapshot.gz'."""

    file_name = f"{path_file}/data.{snapshot}.gz"

    #Phase-space globular cluster centre
    w_gc = _np.loadtxt(file_name, skiprows=0, usecols=(3,4,5,6,7,8), max_rows=1)

    #Phase-space stars relative to globular cluster
    w_delta = _np.loadtxt(file_name, skiprows=1, usecols=(1,2,3,4,5,6))

    #Sort stars
    n = _np.loadtxt(file_name, skiprows=1, usecols=(9))
    w_delta = w_delta[_np.argsort(n)]

    #FSR
    w_fsr_petar = w_gc + w_delta #[pc, pc/Myr]
    w_fsr = _invi.units.petar_to_galactic(w_fsr_petar.T) #[kpc, kpc/Myr]

    return w_fsr

#-----------------------------------------------------------------------------

def mass(path_file, snapshot):
    """Mass of the stars in FSR from petar snapshot file
    'data.snapshot.gz'."""

    file_name = f"{path_file}/data.{snapshot}.gz"

    #Load mass and number stars
    mass, n = _np.loadtxt(file_name, skiprows=1, usecols=(0, 9), unpack=True)

    #Sort stars
    return mass[_np.argsort(n)]

#-----------------------------------------------------------------------------
