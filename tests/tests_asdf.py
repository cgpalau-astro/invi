"""Tests involving asdf files.

Note
----
1)  Run pytest from 'invi' folder: 'python3.11 -m pytest -Wall -vs --durations=0 lib/invi/tests/tests_asdf.py'

2)  pylint tests_asdf.py --disable={C0301,C0103,C0116,C0415}"""

#-----------------------------------------------------------------------------

def test_inverse_time_integration():
    import asdf
    import tomllib
    import numpy as np

    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Load asdf file
    M68 = asdf.open("data/M68.asdf", lazy_load=True)

    #Simulated stars in ICRS
    s_icrs = invi.dicts.dict_to_array(M68['stars']['phase_space']['ICRS'])

    #Selection stream stars
    st = M68['stars']['components']['stream']
    st_icrs = s_icrs.T[st].T

    #Star sample
    N = 8
    sample_icrs = st_icrs.T[0:N].T

    #The aaf in the asdf file has been computed with the 'accuracy' options
    prm['isochrone_approx']['invi'] = prm['isochrone_approx']['accuracy']

    #Inverse time integration
    _, sample_alpha = invi.inverse.integration_general_potential([1.0, 1.0, 1.0, 1.0], sample_icrs, prm['M68'], prm)
    alpha_dict = invi.dicts.alpha(sample_alpha, prm['M68']['stream']['varphi'])

    for item in ['a_r', 'a_phi', 'a_z']:
        a = alpha_dict['alpha'][item]
        a_ref = M68['stars']['inv_int']['alpha'][item][st][0:N]
        assert np.all(np.isclose(a/a_ref, 1.0, rtol=2E-4))

#-----------------------------------------------------------------------------
