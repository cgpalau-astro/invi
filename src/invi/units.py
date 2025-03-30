"""Definition and conversion of units.

Note
----
1)  Units definition: [length, velocity, mass]
    Galactic:         [kpc, kpc/Myr, M_sun]
    Astronomic:       [kpc, km/s, M_sun]
    Galpy:            [uL, uV, uM] = [8.0 kpc, 220 km/s, 90031153175.68097 M_sun]
    PeTar:            [pc, pc/Myr, M_sun]

2)  In the invi package all functions are defined in galactic units.
3)  Galpy and agama are defined in galpy units.
4)  Constants:
    G = 0.0044983099795944 #[pc^3/Msun/Myr^2]
    GMscale = 2.4692087520131E-09  #[galpy GM unit]/[pc^3/Myr^2]
5)  The functions to change units assume cartesian coordinates."""

from dataclasses import dataclass as _dataclass
import numpy as _np

#-----------------------------------------------------------------------------
#Galactic <-> astronomic

def kpc_to_km(x): return x*(1_000.0*30856775813057.0)
def km_to_kpc(x): return x/(1_000.0*30856775813057.0)

def kpc_to_pc(x): return x*1_000.0
def pc_to_kpc(x): return x/1_000.0

def inv_yr_to_inv_s(x): return x/(60*60*365.25*24)
def inv_s_to_inv_yr(x): return x*(60*60*365.25*24)

def inv_Myr_to_inv_s(x): return x/(1.0E6*60*60*365.25*24)
def inv_s_to_inv_Myr(x): return x*(1.0E6*60*60*365.25*24)

def inv_Myr2_to_inv_s2(x): return x/((1.0E6*60*60*365.25*24)**2.0)
def inv_s2_to_inv_Myr2(x): return x*((1.0E6*60*60*365.25*24)**2.0)

def kpcMyr_to_kms(x): return inv_Myr_to_inv_s(kpc_to_km(x))
def kms_to_kpcMyr(x): return inv_s_to_inv_Myr(km_to_kpc(x))

def kms_to_pcMyr(x): return inv_s_to_inv_Myr(km_to_kpc(kpc_to_pc(x)))
def pcMyr_to_kms(x): return inv_Myr_to_inv_s(kpc_to_km(pc_to_kpc(x)))

def kpcMyr_to_pcMyr(x): return kpc_to_pc(x)
def pcMyr_to_kpcMyr(x): return pc_to_kpc(x)

def galactic_to_astronomic(w):
    x = w[0]
    y = w[1]
    z = w[2]
    vx = kpcMyr_to_kms(w[3])
    vy = kpcMyr_to_kms(w[4])
    vz = kpcMyr_to_kms(w[5])
    return _np.array([x, y, z, vx, vy, vz])

def astronomic_to_galactic(w):
    x = w[0]
    y = w[1]
    z = w[2]
    vx = kms_to_kpcMyr(w[3])
    vy = kms_to_kpcMyr(w[4])
    vz = kms_to_kpcMyr(w[5])
    return _np.array([x, y, z, vx, vy, vz])

#-----------------------------------------------------------------------------
#Galactic and astronomic <-> galpy

@_dataclass(frozen=True)
class u:
    L: float = 8.0 #[kpc]
    V: float = 220.0 #[km/s]
    M: float = 90031153175.68097 #[Msun]

def kpc_to_uL(x): return x/u.L
def uL_to_kpc(x): return x*u.L

def pc_to_uL(x): return x/u.L/1_000.0
def uL_to_pc(x): return x*u.L*1_000.0

def kms_to_uV(x): return x/u.V
def uV_to_kms(x): return x*u.V

def kpcMyr_to_uV(x): return kms_to_uV(kpcMyr_to_kms(x))
def uV_to_kpcMyr(x): return kms_to_kpcMyr(uV_to_kms(x))

def Msun_to_uM(x): return x/u.M
def uM_to_Msun(x): return x*u.M

#M_sun/kpc^3 to uM/uL^3
def Msuninvkpc3_to_uMinvuL3(x): return x*u.L**3.0/u.M

def Myr_to_uT(x): return x/u.L*kms_to_kpcMyr(u.V)
def uT_to_Myr(x): return x*u.L/kms_to_kpcMyr(u.V)

#uL^2/uT to kpc^2/Myr
def uL2invuT_to_kpc2invMyr(x): return uV_to_kms(kms_to_kpcMyr(uL_to_kpc(x)))
def kpc2invMyr_to_uL2invuT(x): return kms_to_uV(kpcMyr_to_kms(kpc_to_uL(x)))

#1/uT to 1/Myr
def invuT_to_invMyr(x): return 1.0/uT_to_Myr(1.0/x)
def invMyr_to_invuT(x): return 1.0/Myr_to_uT(1.0/x)

#1/uL to 1/kpc
def invuL_to_invkpc(x): return 1.0/uL_to_kpc(1.0/x)
def invkpc_to_invuL(x): return 1.0/kpc_to_uL(1.0/x)


def astronomic_to_galpy(w):
    x = kpc_to_uL(w[0])
    y = kpc_to_uL(w[1])
    z = kpc_to_uL(w[2])
    vx = kms_to_uV(w[3])
    vy = kms_to_uV(w[4])
    vz = kms_to_uV(w[5])
    return _np.array([x, y, z, vx, vy, vz])


def galpy_to_astronomic(w):
    x = uL_to_kpc(w[0])
    y = uL_to_kpc(w[1])
    z = uL_to_kpc(w[2])
    vx = uV_to_kms(w[3])
    vy = uV_to_kms(w[4])
    vz = uV_to_kms(w[5])
    return _np.array([x, y, z, vx, vy, vz])


def galactic_to_galpy(w):
    x = kpc_to_uL(w[0])
    y = kpc_to_uL(w[1])
    z = kpc_to_uL(w[2])
    vx = kpcMyr_to_uV(w[3])
    vy = kpcMyr_to_uV(w[4])
    vz = kpcMyr_to_uV(w[5])
    return _np.array([x, y, z, vx, vy, vz])


def galpy_to_galactic(w):
    x = uL_to_kpc(w[0])
    y = uL_to_kpc(w[1])
    z = uL_to_kpc(w[2])
    vx = uV_to_kpcMyr(w[3])
    vy = uV_to_kpcMyr(w[4])
    vz = uV_to_kpcMyr(w[5])
    return _np.array([x, y, z, vx, vy, vz])


def aaf_to_afa_galpy(aaf):
    """From galactic [rad, kpc^2/Myr, rad/Myr] to galpy units and format."""
    Ar = _np.array([aaf[0]])
    Aphi = _np.array([aaf[1]])
    Az = _np.array([aaf[2]])

    Jr = _np.array([kpc2invMyr_to_uL2invuT(aaf[3])])
    Jphi = _np.array([kpc2invMyr_to_uL2invuT(aaf[4])])
    Jz = _np.array([kpc2invMyr_to_uL2invuT(aaf[5])])

    Fr = _np.array([invMyr_to_invuT(aaf[6])])
    Fphi = _np.array([invMyr_to_invuT(aaf[7])])
    Fz = _np.array([invMyr_to_invuT(aaf[8])])

    return (Jr, Jphi, Jz, Fr, Fphi, Fz, Ar, Aphi, Az)


def afa_galpy_to_aaf(afa_galpy):
    """From galpy to galactic [rad, kpc^2/Myr, rad/Myr] units and format."""
    A = _np.array([afa_galpy[6][0], afa_galpy[7][0], afa_galpy[8][0]])
    J = uL2invuT_to_kpc2invMyr( _np.array([afa_galpy[0][0], afa_galpy[1][0], afa_galpy[2][0]]) )
    F = invuT_to_invMyr( _np.array([afa_galpy[3][0], afa_galpy[4][0], afa_galpy[5][0]]) )
    return (A[0], A[1], A[2], J[0], J[1], J[2], F[0], F[1], F[2])

#-----------------------------------------------------------------------------
#Galactic <-> petar

def galactic_to_petar(w):
    x = kpc_to_pc(w[0])
    y = kpc_to_pc(w[1])
    z = kpc_to_pc(w[2])
    vx = kpcMyr_to_pcMyr(w[3])
    vy = kpcMyr_to_pcMyr(w[4])
    vz = kpcMyr_to_pcMyr(w[5])
    return _np.array([x, y, z, vx, vy, vz])

def petar_to_galactic(w):
    x = pc_to_kpc(w[0])
    y = pc_to_kpc(w[1])
    z = pc_to_kpc(w[2])
    vx = pcMyr_to_kpcMyr(w[3])
    vy = pcMyr_to_kpcMyr(w[4])
    vz = pcMyr_to_kpcMyr(w[5])
    return _np.array([x, y, z, vx, vy, vz])

#-----------------------------------------------------------------------------
#Angles

def deg_to_rad(x): return x*(2.0*_np.pi)/360.0
def rad_to_deg(x): return x/(2.0*_np.pi)*360.0

def mas_to_rad(x): return x*(1.0/60.0/60.0/360.0*2.0*_np.pi/1_000.0)
def rad_to_mas(x): return x/(1.0/60.0/60.0/360.0*2.0*_np.pi/1_000.0)

def mas_to_deg(x): return rad_to_deg(mas_to_rad(x))
def deg_to_mas(x): return rad_to_mas(deg_to_rad(x))

def masyr_to_radMyr(x): return mas_to_rad(x*1.0E6)

def radMyr_to_masyr(x): return rad_to_mas(x/1.0E6)

#-----------------------------------------------------------------------------
#Scale

def micro_to_milli(x): return x/1_000.0
def milli_to_micro(x): return x*1_000.0

def milli_to_unit(x): return x/1_000.0
def unit_to_milli(x): return x*1_000.0

def micro_to_unit(x): return x/1.0E6
def unit_to_micro(x): return x*1.0E6

#-----------------------------------------------------------------------------
