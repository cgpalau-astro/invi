"""Functions related to astro_limepy package."""

import termcolor as _tc
import invi.astro_limepy as _limepy

__all__ = ["model_parameters"]

#-----------------------------------------------------------------------------

def model_parameters(W0, g, r_half, mass, verbose=True):
    """Calculate parameters of the generalised king model model using the
    astro_limepy package."""
    model = _limepy.limepy(W0, g, M=mass, rh=r_half, verbose=False)
    r_core = model.r0 #[pc]
    r_truncation = model.rt #[pc]
    r_v = model.rv #[?]
    #----------------------------------------------------
    if verbose:
        _tc.cprint("Information:", 'light_blue')
        print("      arXiv : 1901.08072")
        print("         W0 : Dimensionless central potential")
        print("          g : Truncation parameter: 0=Woolley, 1=Michie-King, 2=Wilson")
        print("r_trucation : Truncation radius")
        print("     r_core : Core radius")
        print("     r_half : Half mass radius")
        print("        r_v : ?")
        _tc.cprint("\nInput parameters:", 'light_blue')
        print(f"    W0 = {W0} [-]")
        print(f"     g = {g} [-]")
        print(f"r_half = {r_half} [pc]")
        print(f"  mass = {mass} [M_sun]")
        _tc.cprint("\nOutput parameters:", 'light_blue')
        print(f"      r_core = {r_core:0.3f} [pc]")
        print(f"r_truncation = {r_truncation:0.3f} [pc]")
        print(f"         r_v = {r_v:0.3f} [?]")
    #----------------------------------------------------
    return r_core, r_truncation, r_v

#-----------------------------------------------------------------------------
