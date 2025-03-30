"""Tests involving petar input/output files.

Note
----
1)  Run pytest from 'invi' folder: 'python3.13 -m pytest -Wall -vs --durations=0 lib/invi/tests/tests_petar.py'

2)  pylint tests_petar.py --disable={C0301,C0103,C0116,C0415}"""

#-----------------------------------------------------------------------------

def test_calssification():
    import tomllib
    import numpy as np
    import invi

    def assert_consistency(n):
        assert n['total'] == n['gc'] + n['st'] + n['e']
        assert n['st'] == n['t'] + n['l']
        assert n['l'] == n['ls1'] + n['ls2'] + n['ls3'] + n['lsn']
        assert n['t'] == n['ts1'] + n['ts2'] + n['ts3'] + n['tsn']

    def assert_numbers(n):
        assert n['total'] == 432_384
        assert n['st'] == 5_471
        assert n['l'] == 2_720
        assert n['t'] == 2_751

        assert n['ls1'] == 825
        assert n['ls2'] == 1_008
        assert n['ls3'] == 800
        assert n['lsn'] == 87

        assert n['ts1'] == 846
        assert n['ts2'] == 1_028
        assert n['ts3'] == 794
        assert n['tsn'] == 83

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Load petar stars
    s_fsr = invi.petar.load.FSR("data/petar/M68_run_0", snapshot=1_500)

    #Globular cluster position in FSR and aaf
    gc_fsr, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm['M68'], prm)

    #Load aaf stars
    s_aaf = np.load("data/misc/s_aaf.npz")['s_aaf']

    #Star classification
    components = invi.stars.components.classify(prm['M68'], s_fsr, gc_fsr, gc_aaf, s_aaf)

    n = invi.stars.components.number(components)
    assert_consistency(n)
    assert_numbers(n)

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Position stars in ICRS
    s_icrs = invi.coordinates.FSR_to_ICRS(s_fsr, prm['sun'], mw.v_lsr)
    s_icrs_dict = {'ICRS': invi.dicts.array_to_dict(s_icrs, 'ICRS')}

    #Stream main component
    main_comp = invi.stars.components.main_component(s_icrs_dict, prm['M68'])

    mc = components['stream'] & main_comp
    assert np.count_nonzero(mc) == 1_475

#-----------------------------------------------------------------------------

def test_eig_stream():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Load petar stars
    s_fsr = invi.petar.load.FSR("data/petar/M68_run_0", snapshot=1_500)

    #Globular cluster position in FSR and aaf
    gc_fsr, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm['M68'], prm)

    #Load aaf stars
    s_aaf = np.load("data/misc/s_aaf.npz")['s_aaf']

    #Star classification
    components = invi.stars.components.classify(prm['M68'], s_fsr, gc_fsr, gc_aaf, s_aaf)

    st_dgc = invi.coordinates.aaf_to_dgc(s_aaf, gc_aaf).T[components['stream']].T

    #Compute AAF
    varphi_x, varphi_y, varphi_z = prm['M68']['stream']['varphi'].values()

    #Compute eigenvalues
    eig = invi.stream.principal_axes.eigenvalues(st_dgc, varphi_x, varphi_y, varphi_z)
    eig = np.median(eig, 1)*1_000.0 #[mrad/kpc^2]

    eig_ref = np.array([-10.08179896, -0.30346407, 0.23946733])

    assert np.all(np.isclose(eig/eig_ref, 1.0))

#-----------------------------------------------------------------------------

def test_selection_function():
    import tomllib
    import numpy as np
    import invi

    #Load parameters
    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Load petar stars in FSR [kpc, kpc/Myr]
    s_fsr = invi.petar.load.FSR("data/petar/M68_run_0", snapshot=1_500)

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Position stars in ICRS
    s_icrs = invi.coordinates.FSR_to_ICRS(s_fsr, prm['sun'], mw.v_lsr)
    s_icrs_dict = invi.dicts.array_to_dict(s_icrs, 'ICRS')

    #Generate synthetic population from CMD37
    mass = prm['M68']['M_initial']
    cmd_file = "data/synthetic_population/M68/population.dat.zip"
    synthetic_population = invi.globular_cluster.synthetic_population.generate(mass, cmd_file, verbose=False)

    #Stars photometry
    s_phot = invi.stars.photometry(s_icrs_dict, synthetic_population)

    #Mock reddening
    mock_red = invi.mock.reddening(s_icrs_dict,
                                   s_phot,
                                   prm['M68']['mock']['cmd37_correction'])

    #Mock selection function
    mock_sf = invi.mock.selection_function(s_icrs_dict,
                                           s_phot,
                                           mock_red,
                                           random_state=prm['M68']['mock']['random_state_sel_func'])

    #Stars in the observable section of the stream (not obscured by foreground)
    observable_section = s_icrs_dict['delta'] > -8.0 #[deg]

    #Bright stars
    under_20 = s_phot['g'] < 20.0 #[mag]
    under_18 = s_phot['g'] < 18.0 #[mag]

    #Approximation final observed stars
    psf_obs_sec = mock_sf['pass_sel_func'] & observable_section
    under_20 = psf_obs_sec & under_20
    under_18 = psf_obs_sec & under_18

    assert np.count_nonzero(observable_section) == 1752
    assert np.count_nonzero(under_20) == 116
    assert np.count_nonzero(under_18) == 19

#-----------------------------------------------------------------------------

def test_stripping_time():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Globular cluster position in FSR and aaf
    _gc_fsr, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm['M68'], prm)

    #Load stars
    s_aaf = np.load("data/misc/s_aaf.npz")['s_aaf']

    #Action, angle and frequency relative to the globular cluster
    s_dgc = invi.coordinates.aaf_to_dgc(s_aaf, gc_aaf)

    #Action, angle and frequency relative in principal axis reference frame
    s_AAF = invi.coordinates.aaf_to_AAF(s_aaf, gc_aaf, prm['M68']['stream']['varphi'])

    #Stripping times
    time_dgc = invi.inverse.integration_time(s_dgc)
    time_AAF = invi.inverse.integration_time(s_AAF)

    assert np.all(np.isclose(time_dgc/time_AAF, 1.0))

#-----------------------------------------------------------------------------
