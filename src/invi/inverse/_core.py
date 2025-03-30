"""Core function used for the inverse time integration method."""

import numpy as _np

__all__ = ["integration_time", "integration"]

#-----------------------------------------------------------------------------

def integration_time(s_dgc):
    #Modulus and integration time: s_dgc = [A_i, J_i, F_i] i=(r, phi, z)
    mod_A = _np.sqrt( s_dgc[0]**2.0 + s_dgc[1]**2.0 + s_dgc[2]**2.0 ) #[rad]
    mod_F = _np.sqrt( s_dgc[6]**2.0 + s_dgc[7]**2.0 + s_dgc[8]**2.0 ) #[rad/Myr]
    time = mod_A / mod_F #[Myr]
    return time


def integration(s_dgc):
    #Integration time
    time = integration_time(s_dgc) #[Myr]

    #Inverse time integration: alpha = A - F*time
    #s_dgc = [A_i, J_i, F_i] i=(r, phi, z)
    alpha_r   = s_dgc[0] - s_dgc[6] * time #[rad]
    alpha_phi = s_dgc[1] - s_dgc[7] * time
    alpha_z   = s_dgc[2] - s_dgc[8] * time

    return _np.array([alpha_r, alpha_phi, alpha_z])

#-----------------------------------------------------------------------------
