"""Definition of tests.

Note
----
1)  Run pytest from 'invi' folder: 'python3.13 -m pytest -Wall -vs --durations=0 lib/invi/tests/tests.py'

    -Wall : show all warnings
    -x : stops after first failure
    -v : verbose output
    -s : prints output from print()
    --durations=0 : Print elapsed time for each test

    -k 'name_test_1 and name_test_2' : Run name_test_1 and name_test_2 only.
    -k 'not name_test_1 and not name_test_2' : Exclude name_test_1 and name_test_2.

2)  Name of the test functions has to start with 'test_'.

3)  Decorator to skip test: import pytest ; @pytest.mark.skip(reason="")

4)  Test notebooks from 'invi' folder: 'pytest --nbmake -Wall -vs --durations=0 -k 'not 2-isochrone_approx_parameters_invi and not 3-isochrone_approx_parameters_accuracy and not 4-torus_mapper' code/'

5)  pylint lib/invi/tests/tests.py --disable={C0301,C0103,C0116,C0415}"""

#import pytest

#-----------------------------------------------------------------------------

def test_magnitudes():
    import numpy as np
    import invi.photometry.magnitudes as mag

    #Gaia BPRP colour index
    BPRP_ref = np.array([-0.1, 2.0])

    BPRP = mag.Teff_to_BPRP(mag.BPRP_to_Teff(BPRP_ref))
    assert np.isclose(BPRP/BPRP_ref, 1.0).all()

    BPRP = mag.BV_to_BPRP(mag.BPRP_to_BV(BPRP_ref))
    assert np.isclose(BPRP/BPRP_ref, 1.0).all()

    BPRP = mag.VI_to_BPRP(mag.BPRP_to_VI(BPRP_ref))
    assert np.isclose(BPRP/BPRP_ref, 1.0).all()

    #Distance modulus
    d_ref = mag.m_M_to_d(11.37, 4.75)
    d = mag.mM_to_d(mag.d_to_mM(d_ref))
    assert np.isclose(d, d_ref)


def test_reddening():
    import numpy as np
    import invi

    #M68 globular cluster
    dec = -26.744
    ra = 189.867
    BPRP_ref = 0.8
    G_ref = 2.5

    kwargs = {'dec': dec, 'ra':ra, 'cmd37_correction': 0.0}

    #Simulation reddening
    BPRP_red, G_red = invi.photometry.reddening.simulation(BPRP=BPRP_ref, G=G_ref, **kwargs)

    #Correction simulated reddening
    BPRP, G = invi.photometry.reddening.correction(BPRP_red=BPRP_red, G_red=G_red, **kwargs)

    assert np.isclose(BPRP, BPRP_ref)
    assert np.isclose(G, G_ref)

#-----------------------------------------------------------------------------

def test_galactic_coordinate_change():
    import numpy as np
    import invi

    #Position M68 ICRS
    w_icrs_ref = np.array([10.404, -26.744, 189.867, -92.07, 1.779, -2.739])

    #Sun parameters
    sun = {'R': 8.275, 'z': 0.0208, 'U': 11.1, 'V': 12.24, 'W': 7.25}

    #Velocity LSR
    v_lsr = 0.2334 #[kpc/Myr]

    #Position M68 in FSR [kpc, kpc/Myr]
    w_fsr = invi.coordinates.ICRS_to_FSR(w_icrs_ref, sun, v_lsr)

    #Position M68 in ICRS
    w_icrs = invi.coordinates.FSR_to_ICRS(w_fsr, sun, v_lsr)

    assert np.isclose(w_icrs/w_icrs_ref, 1.0).all()


def test_coordinate_change():
    import invi.coordinates as co
    import numpy as np

    random_state = 123
    rng = np.random.default_rng(random_state)

    mean = np.zeros(6)
    cov = np.diag(np.ones(6))
    xv_ref = rng.multivariate_normal(mean, cov, size=100).T
    x_ref = xv_ref[0:3]

    xv = co.cyl_to_car(co.car_to_cyl(xv_ref))
    assert np.isclose(xv, xv_ref).all()

    xv = co.cyl_galpy_to_car(co.car_to_cyl_galpy(xv_ref))
    assert np.isclose(xv, xv_ref).all()

    xv = co.sph_to_car(co.car_to_sph(xv_ref))
    assert np.isclose(xv, xv_ref).all()

    for delta in [-1.25, 0.0, 1.25]:
        x = co.spheroidal_to_car(delta, *co.car_to_spheroidal(delta, *x_ref))
        assert np.isclose(x, x_ref).all()

    #Position M68 ICRS
    w_icrs_ref = np.array([10.404, -26.744, 189.867, -92.07, 1.779, -2.739])

    w_icrs = co.sph_to_ICRS(co.ICRS_to_sph(w_icrs_ref))
    assert np.isclose(w_icrs, w_icrs_ref).all()

    #Coordinates aligned with the stream
    alpha = [0.23, 0.78, -1.23] #[rad]
    w_icrs = co.phi_to_ICRS.angles(co.ICRS_to_phi.angles(w_icrs_ref, *alpha), *alpha)
    assert np.isclose(w_icrs, w_icrs_ref).all()

    J = co.rotation.angles_to_matrix(*alpha, order='zxy', verbose=False)
    w_icrs = co.phi_to_ICRS.matrix(co.ICRS_to_phi.matrix(w_icrs_ref, J), J)
    assert np.isclose(w_icrs, w_icrs_ref).all()

    #Stripping points
    s_alpha_ref = np.array([1, 1, 1])
    varphi = {'x': -1.3957, 'y': 0.5708, 'z': 0.6112}
    s_alpha = co.ALPHA_to_alpha(co.alpha_to_ALPHA(s_alpha_ref, varphi), varphi)
    assert np.isclose(s_alpha, s_alpha_ref).all()


def test_coordinate_change_aaf():
    import tomllib
    import numpy as np
    import invi.coordinates as co

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    gc_aaf = np.array([6.04261082,  0.50492507, 1.57969712,
                       0.93539294, -2.44089168, 0.81370949,
                       0.01375105, -0.00964741, 0.01008775])

    s_aaf_ref = np.array([6.02959401,  0.48499608, 1.59605351,
                          0.93113756, -2.41328845, 0.80913206,
                          0.01388605, -0.00973711, 0.01018720])

    s_dgc_ref = np.array([-1.301681e-02, -1.992899e-02,  1.635639e-02,
                          -4.255380e-03,  2.760323e-02, -4.577430e-03,
                           1.350000e-04, -8.970000e-05,  9.945000e-05])

    varphi = prm['M68']['stream']['varphi']

    s_aaf = co.dgc_to_aaf(co.aaf_to_dgc(s_aaf_ref, gc_aaf), gc_aaf)
    assert np.isclose(s_aaf/s_aaf_ref, 1.0).all()

    s_dgc = co.AAF_to_dgc(co.dgc_to_AAF(s_dgc_ref, varphi), varphi)
    assert np.isclose(s_dgc/s_dgc_ref, 1.0).all()

    s_aaf = co.AAF_to_aaf(co.aaf_to_AAF(s_aaf_ref, gc_aaf, varphi), gc_aaf, varphi)
    assert np.isclose(s_aaf/s_aaf_ref, 1.0).all()


def test_equal_area():
    import scipy
    import invi
    import fnc.stats.kolmogorov_smirnov as k_s

    ra, dec = invi.misc.Sphere.rvs(size=20_000, random_state=1234)
    ea_1, ea_2 = invi.coordinates.radec_to_equal_area(ra, dec)

    u = scipy.stats.uniform(loc=0.0, scale=4.0)
    ks = k_s.test(ea_1, u.cdf)
    assert k_s.result(ea_1, ks)

    u = scipy.stats.uniform(loc=-1.0, scale=2.0)
    ks = k_s.test(ea_2, u.cdf)
    assert k_s.result(ea_2, ks)

    u = scipy.stats.uniform(loc=0.0, scale=360.0)
    ks = k_s.test(ra, u.cdf)
    assert k_s.result(ra, ks)

    u = scipy.stats.uniform(loc=-90.0, scale=180.0)
    ks = k_s.test(dec, u.cdf)
    assert k_s.result(dec, ks) is False

#-----------------------------------------------------------------------------

def test_dict_to_array():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    w_icrs_ref = np.array([prm['M68']['ICRS']['r'],
                           prm['M68']['ICRS']['delta'],
                           prm['M68']['ICRS']['alpha'],
                           prm['M68']['ICRS']['mu_r'],
                           prm['M68']['ICRS']['mu_delta'],
                           prm['M68']['ICRS']['mu_alpha_str']])

    w_icrs = invi.dicts.dict_to_array(prm['M68']['ICRS'])

    assert (w_icrs == w_icrs_ref).all()


def test_dict_array_conversion():
    import numpy as np
    import invi

    for item in ['ICRS', 'car', 'cyl', 'sph']:
        w_ref = np.array([(0.0 + i) for i in range(6)])
        xdict = invi.dicts.array_to_dict(w_ref, item)
        w = invi.dicts.dict_to_array(xdict)
        assert (w == w_ref).all()

    w_ref = np.array([(0.0 + i) for i in range(9)])
    xdict = invi.dicts.array_to_dict(w_ref, 'aaf')
    w = invi.dicts.dict_to_array(xdict)
    assert (w == w_ref).all()

#-----------------------------------------------------------------------------

def test_units_aaf():
    import numpy as np
    import invi

    aaf_ref = np.arange(1, 10, 1)
    aaf = invi.units.afa_galpy_to_aaf( invi.units.aaf_to_afa_galpy(aaf_ref) )
    assert np.isclose(aaf, aaf_ref).all()

#-----------------------------------------------------------------------------

def test_galpy_agama_df():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Parameters king model
    W0 = prm['M68']['king']['W0']
    g = prm['M68']['king']['g']
    r_half = prm['M68']['king']['r_half'] #[pc]
    mass = prm['M68']['M_initial'] #[M_sun]

    #Use astro_limepy to determine r_core
    r_core, _r_truncation, _r_v = invi.astro_limepy.utils.model_parameters(W0, g, r_half, mass, verbose=False)

    #Random samples
    kwargs = {'mass': mass, 'r_core': r_core, 'W0': W0, 'g': g, 'size': 100_000}
    nbody_limepy = invi.globular_cluster.phase_space.King.rvs.limepy(**kwargs, random_state=12378)
    nbody_agama = invi.globular_cluster.phase_space.King.rvs.agama(**kwargs)

    def characteristic_time(nbody):
        d = np.sqrt( nbody[0]**2.0 + nbody[1]**2.0 + nbody[2]**2.0 )
        v = np.sqrt( nbody[3]**2.0 + nbody[4]**2.0 + nbody[5]**2.0 )
        return np.median(d/v)

    t0 = characteristic_time(nbody_limepy)
    t1 = characteristic_time(nbody_agama)

    assert np.isclose(t0/t1, 1.0, rtol=2.7E-3)

#-----------------------------------------------------------------------------

def test_orbit_bck_frw():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    orb = invi.globular_cluster.orbit.bck_frw(prm['M68'], prm)
    w_ref = np.array([10.404, -26.744, 189.867, -92.07, 1.779, -2.739])
    w = invi.dicts.dict_to_array(orb['orbit']['ICRS']).T[-1]

    assert np.isclose(w, w_ref).all()


def test_orbit_frw():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    T = 1_500.0
    N = 1_500
    orb = invi.globular_cluster.orbit.frw(prm['M68'], prm, -T, N)

    w_fsr = invi.dicts.dict_to_array(orb['orbit']['FSR']['car']).T[-1]

    w = invi.units.galactic_to_petar(w_fsr)

    w_ref = np.array([-21240.35039168949, 5134.770398414408, -18373.336514263352,
                       85.00972143483914, 94.36691209913434, 57.42603596137717])

    assert np.isclose(w, w_ref).all()

#-----------------------------------------------------------------------------

def test_galpy_potential():
    import tomllib
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Milky Way potential (galpy)
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    bulge = mw.potential[0]
    disc = mw.potential[1]
    halo = mw.potential[2]

    bulge_n, _bulge_args = invi.galpy.potential.n_args(bulge)[1:3]
    disc_n, _disc_args = invi.galpy.potential.n_args(disc)[1:3]
    halo_n, _halo_args = invi.galpy.potential.n_args(halo)[1:3]

    assert bulge_n == 15
    assert disc_n == 5
    assert halo_n == 22

#-----------------------------------------------------------------------------

def test_frequencies():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Orbit M68
    T = 50_000.0 #[Myr]
    N = 50_001
    orb = invi.globular_cluster.orbit.frw(prm['M68'], prm, T, N)

    w_fsr = invi.dicts.dict_to_array(orb['ic']['FSR']['car'])

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])
    potential_galpy = mw.potential

    #Computation angles, actions, and frequencies
    b, maxn, tintJ, ntintJ = prm['isochrone_approx']['accuracy'].values()
    aaf = invi.coordinates.FSR_to_aaf(w_fsr, potential_galpy, b, maxn, tintJ, ntintJ)
    F_aaf = aaf[6:] * 1_000 #[rad/Gyr]

    #Frequencies from NAFF-naif
    t = orb['t']
    Fr = invi.naif.dominant_peak(orb['orbit']['FSR']['sph']['r'], t) #[rad/Myr]
    Fphi = -invi.naif.dominant_peak(orb['orbit']['FSR']['cyl']['phi'], t)
    Fz = invi.naif.dominant_peak(orb['orbit']['FSR']['cyl']['z'], t)
    F_fft = np.array([Fr, Fphi, Fz])*1_000.0 #[rad/Gyr]

    assert np.isclose(F_aaf/F_fft, 1.0).all()

#-----------------------------------------------------------------------------

def test_staeckel_isochrone_approx():
    import tomllib
    import numpy as np
    import invi

    from galpy.actionAngle import estimateDeltaStaeckel

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Orbit parameters
    T = 1_000.0 #[Myr]
    N = 1_001

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Position M68 ICRS
    w_icrs = invi.dicts.dict_to_array(prm['M68']['ICRS'])

    #Position M68 in FSR [kpc, kpc/Myr]
    w_fsr = invi.coordinates.ICRS_to_FSR(w_icrs, prm['sun'], mw.v_lsr)

    #Determination delta factor Staeckel Fudge
    delta = invi.galpy.aaf.scale_factor(w_fsr, mw.potential, T, N, estimateDeltaStaeckel)

    #Angle, actions, and frequencies using the Staeckel Fudge
    aaf_sf = invi.galpy.aaf.staeckel_fudge(w_fsr, mw.potential, delta)

    #Angle, actions, and frequencies using the Isochrone Approximation
    b, maxn, tintJ, ntintJ = prm['isochrone_approx']['invi'].values()

    aaf_ia = invi.galpy.aaf.isochrone_approx(w_fsr, mw.potential, b, maxn, tintJ, ntintJ)

    assert np.isclose(aaf_sf/aaf_ia, 1.0, rtol=1.16E-2).all()


def test_isochrone_approx_gc():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Globular cluster position in FSR and aaf
    gc_fsr, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm['M68'], prm)

    #M68 angles, actions, frequencies
    gc_aaf_ref = np.array([6.04261080, 0.50492508, 1.57969711,
                           0.93538887,-2.44089168, 0.81370994,
                           0.01375105,-0.00964741, 0.01008775])

    assert np.isclose(gc_aaf/gc_aaf_ref, 1.0).all()

    #Lz = x*px - y*px = J_phi
    Lz = gc_fsr[0]*gc_fsr[4] - gc_fsr[1]*gc_fsr[3]

    assert np.isclose(gc_aaf[4], Lz)

#-----------------------------------------------------------------------------

def test_galpy_agama_potential():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Definition galpy Milky Way potential
    mw_galpy = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Definition agama Milky Way potential
    mw_agama = invi.agama.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    assert np.isclose(mw_galpy.v_lsr/mw_agama.v_lsr, 1.0, rtol=2.2E-5)


def test_staeckel_galpy_agama():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Potential galpy
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])
    potential_galpy = mw.potential

    #Potential agama
    mw = invi.agama.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])
    potential_agama = mw.potential

    #Position M68 ICRS
    w_icrs = invi.dicts.dict_to_array(prm['M68']['ICRS'])

    #Position M68 in FSR [kpc, kpc/Myr]
    v_lsr = 0.23340332767515876 #[kpc/Myr]
    w_fsr = invi.coordinates.ICRS_to_FSR(w_icrs, prm['sun'], v_lsr)

    #Angle, actions, and frequencies using the Staeckel Fudge galpy
    delta = 0.23285242439935322
    aaf_galpy = invi.galpy.aaf.staeckel_fudge(w_fsr, potential_galpy, delta)

    #Angle, actions, and frequencies using the Staeckel Fudge agama
    aaf_agama = invi.agama.aaf.staeckel_fudge(w_fsr, potential_agama)

    assert np.isclose(aaf_galpy/aaf_agama, 1.0, rtol=1.2E-2).all()


def test_orbit_galpy_agama():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    ic_fsr = np.array([4.11682242, 7.31191959, 6.14359406,
                       0.1744117 ,-0.28313277, 0.01801012]) #[kpc, kpc/Myr]
    T = 1_500.0 #[Myr]
    N = 1_501

    #Milky Way potential galpy
    mw_galpy = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Orbit galpy
    _t, w_fsr_galpy = invi.galpy.orbit.integrate(ic_fsr, mw_galpy.potential, T, N)

    #Milky Way potential agama
    mw_agama = invi.agama.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Orbit galpy
    _t, w_fsr_agama = invi.agama.orbit.integrate(ic_fsr, mw_agama.potential, T, N)

    assert np.isclose(w_fsr_galpy.T[-1] / w_fsr_agama.T[-1], 1.0, rtol=0.8E-2).all()

#-----------------------------------------------------------------------------

def test_torus_mapper():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Isochrone Approximation
    gc_fsr_ref, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm['M68'], prm)

    #Torus Mapper
    gc_fsr = invi.coordinates.aaf_to_FSR(gc_aaf, mw.potential, tol=prm['torus_mapper']['tol'])

    assert np.isclose(gc_fsr/gc_fsr_ref, 1.0, rtol=5.4E-3).all()


def test_eig_hessian():
    import tomllib
    import numpy as np

    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Definition Milky Way potential
    mw = invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Globular cluster M68
    gc_icrs = invi.dicts.dict_to_array(prm['M68']['ICRS'])
    gc_fsr = invi.coordinates.ICRS_to_FSR(gc_icrs, prm['sun'], mw.v_lsr)

    #Parameters accuracy
    b, maxn, tintJ, ntintJ = prm['isochrone_approx']['accuracy'].values()

    #Angle, actions, and frequencies globular cluster
    gc_aaf = invi.coordinates.FSR_to_aaf(gc_fsr, mw.potential, b, maxn, tintJ, ntintJ)

    #Compute eigenvalues with torus mapper
    eig_torus = invi.galpy.aaf.eig_hessian(gc_aaf, mw.potential, tol=prm['torus_mapper']['tol'])

    #Eigenvalues from rotation N-body simulation
    eig_rot = prm['M68']['stream']['simulation']['eig']

    assert np.isclose(eig_torus/eig_rot, 1.0, rtol=9E-2).all()

#-----------------------------------------------------------------------------

def test_misc():
    import numpy as np
    import invi

    #Distance between Vega and Altair [deg]
    d_ang = invi.misc.angular_distance(279.23473479, +38.78368896, 297.69582730, +08.86832120)
    assert np.isclose(d_ang, 34.19518431050961)

    #Points within polygon
    points_polygon = np.array([[0, 0], [1, 0], [1, 1]])
    x = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0.8, 0.2, 0.8, 0.2])
    assert (invi.misc.polygon_selection(x, y, points_polygon) == [False, False, False, True]).all()

#-----------------------------------------------------------------------------

def test_cross_match():
    import numpy as np
    import invi

    #Data
    x0 = np.array([1.2, 1.2, 2.3, 1.5, 2.0])
    y0 = np.array([1.1, 1.1, 1.3, 1.7, 2.0])
    id_0 = np.array(['a', 'a', 'b', 'c', 'd'])

    x1 = np.array([0.3, 1.2, 1.5, 1.5, 1.5, 1.2, 2.0])
    y1 = np.array([0.2, 1.1, 1.7, 1.7, 1.7, 1.1, 2.0])
    z1 = np.array([0, 1, 2, 3, 4, 5, 6])
    id_1 = np.array(['e', 'a', 'c', 'c', 'c', 'a', 'd'])

    #Cross match by euclidean distance
    matches_ecl = invi.cross_match.distance(x0, y0, x1, y1, eps=1.0E-5, metric='euclidean', verbose=False)
    z0_ecl = invi.cross_match.get_data(z1, matches_ecl)

    #Cross match by id
    matches_id = invi.cross_match.id(id_0, id_1, verbose=False)
    z0_id = invi.cross_match.get_data(z1, matches_id)

    assert np.array_equal(z0_ecl, z0_id, equal_nan=True)

#-----------------------------------------------------------------------------

def assert_consistency(n):
    assert n['total'] == n['gc'] + n['st'] + n['e']
    assert n['st'] == n['t'] + n['l']
    assert n['l'] == n['ls1'] + n['ls2'] + n['ls3'] + n['lsn']
    assert n['t'] == n['ts1'] + n['ts2'] + n['ts3'] + n['tsn']


def test_time_pericentres():
    import tomllib
    import numpy as np
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    T = 5_500.0 #[Myr]
    N = np.int64(T*10) + 1
    t_peris_Ar = invi.stream.simulation._model_1.time_pericentres_Ar(prm['M68'], prm, T)
    t_peris_r = invi.stream.simulation._model_1.time_pericentres_r(prm['M68'], prm, T, N=N)

    assert np.all(np.isclose(t_peris_Ar/t_peris_r, 1.0, rtol=4E-4))


def test_simulation_model_1():
    import tomllib
    import invi

    def assert_numbers(n):
        assert n['total'] == 5_402
        assert n['st'] == 5_402
        assert n['l'] == 2_701
        assert n['t'] == 2_701

        assert n['ls1'] == 886
        assert n['ls2'] == 874
        assert n['ls3'] == 795
        assert n['lsn'] == 146

        assert n['ts1'] == n['ls1']
        assert n['ts2'] == n['ls2']
        assert n['ts3'] == n['ls3']
        assert n['tsn'] == n['lsn']

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    T = 1_500.0 #[Myr]
    random_state = 10

    #Simulated stream
    sim_AAF_arr, _structure = invi.stream.simulation.model_1(prm['M68'], prm, T, random_state)
    components = invi.stream.simulation.classify(prm['M68'], sim_AAF_arr)

    n = invi.stars.components.number(components)
    assert_consistency(n)
    assert_numbers(n)


def test_simulation_model_2():
    import tomllib
    import invi

    def assert_numbers(n):
        assert n['total'] == 6_000
        assert n['st'] == 6_000
        assert n['l'] == 3_000
        assert n['t'] == 3_000

        assert n['ls1'] == 900
        assert n['ls2'] == 910
        assert n['ls3'] == 911
        assert n['lsn'] == 279

        assert n['ts1'] == n['ls1']
        assert n['ts2'] == n['ls2']
        assert n['ts3'] == n['ls3']
        assert n['tsn'] == n['lsn']

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Mass-loss that reproduces the simulation
    prm['M68']['stream']['simulation']['model_2']['mass_loss'] = 2.0 #[stars/Myr]

    T = 1_500.0 #[Myr]
    random_state = 133

    #Simulated stream
    sim_AAF_arr, _structure = invi.stream.simulation.model_2(prm['M68'], prm, T, random_state)
    components = invi.stream.simulation.classify(prm['M68'], sim_AAF_arr)

    n = invi.stars.components.number(components)
    assert_consistency(n)
    assert_numbers(n)


def test_double_exp_wrap_stripp_Ar():
    import tomllib
    import numpy as np
    import scipy

    import fnc
    import invi.stream.simulation._model_3 as model_3

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Parameters
    prm_mod3 = prm['M68']['stream']['simulation']['model_3']
    coef = prm_mod3['stripped_stars_coef']

    #Initialisation data dict
    data = {'T': 4_500.0} #[Myr]

    #Determination Ar and Fr of the globular cluster
    data = model_3.globular_cluster_radial_coordinates(data, prm['M68'], prm)

    #Test normalisation constant 'pdf_double_exp_wrap'
    nc_num = model_3.NormalisationConstant.numeric(data['gc_Ar_T'], data['gc_Ar_0'], coef, limit=200)
    nc_exact = model_3.NormalisationConstant.exact(data['gc_Ar_T'], data['gc_Ar_0'], coef)
    assert np.isclose(nc_num, nc_exact)

    #Test normalisation 'pdf_double_exp_wrap'
    area, _ = scipy.integrate.quad(model_3.pdf_double_exp_wrap,
                                   a=data['gc_Ar_T'],
                                   b=data['gc_Ar_0'],
                                   args=(data, coef),
                                   limit=200)
    assert np.isclose(area, 1.0)

    #Random sample of stripping Ar
    n_stars = 100_000
    random_state = 10
    rng = np.random.default_rng(random_state)
    sample = model_3.stripping_Ar(data, coef, n_stars, rng)

    #Test that the random sample follows the PDF with the Kolmogorov–Smirnov test
    dist_spline = fnc.stats.DistSpline(model_3.pdf_double_exp_wrap,
                                       x_inf=data['gc_Ar_T'],
                                       x_sup=data['gc_Ar_0'],
                                       n_points=np.int64(data['T']))

    ks = fnc.stats.kolmogorov_smirnov.test(sample,
                                           dist_spline.cdf,
                                           args=(data, coef))
    assert fnc.stats.kolmogorov_smirnov.result(sample, ks)


def test_simulation_model_3():
    import tomllib
    import invi

    def assert_numbers(n):
        assert n['total'] == 6_000
        assert n['st'] == 6_000
        assert n['l'] == 3_000
        assert n['t'] == 3_000

        assert n['ls1'] == 934
        assert n['ls2'] == 942
        assert n['ls3'] == 917
        assert n['lsn'] == 207

        assert n['ts1'] == n['ls1']
        assert n['ts2'] == n['ls2']
        assert n['ts3'] == n['ls3']
        assert n['tsn'] == n['lsn']

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Mass-loss that reproduces the simulation
    prm['M68']['stream']['simulation']['model_3']['mass_loss'] = 2.0 #[stars/Myr]

    T = 1_500.0 #[Myr]
    random_state = 10

    #Simulated stream
    sim_AAF_arr, _structure = invi.stream.simulation.model_3(prm['M68'], prm, T, random_state)
    components = invi.stream.simulation.classify(prm['M68'], sim_AAF_arr)

    n = invi.stars.components.number(components)
    assert_consistency(n)
    assert_numbers(n)

#-----------------------------------------------------------------------------

def test_pm_uncertainties():
    import numpy as np
    import pygaia.errors

    import invi

    #Apparent magnitude [mag]
    g = np.linspace(10.0, 20.5, 1_000)

    mu_alpha_cosdelta_unc, mu_delta_unc = pygaia.errors.astrometric.proper_motion_uncertainty(g, release="dr3") #[micro-arcseconds/yr]
    mu_alpha_str_unc = invi.units.micro_to_milli(mu_alpha_cosdelta_unc) #[mas/yr]
    mu_delta_unc = invi.units.micro_to_milli(mu_delta_unc) #[mas/yr]

    #Interpolation
    intpl_unc = np.exp(g*0.725 - 15.0) #[mas/yr]

    #Errors
    error_d = intpl_unc - mu_delta_unc
    error_a = intpl_unc - mu_alpha_str_unc

    assert (np.abs(error_d) < 0.06).all()
    assert (np.abs(error_a) < 0.1).all()

#-----------------------------------------------------------------------------

def test_cm_distance_estimate():
    import tomllib
    import numpy as np

    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Load isochrone
    name_file = "data/synthetic_population/M68/isochrone.dat.zip"
    isochrone = invi.stars.cm_estimate.load_data_isochrone(name_file)

    #Star in M68 globular cluster
    bprp = 0.756 #[mag]
    g = 20.56739632 #[mag]

    #Estimate distance
    d = invi.stars.cm_estimate.distance(bprp, g, isochrone['splines']) #[kpc]

    #Closest distance to the reference
    d_ref = prm['M68']['ICRS']['r'] #[kpc]
    D = invi.stars.cm_estimate.select([d], ref=[d_ref])

    assert D == d[0]
    assert np.isclose(D/d_ref, 1.0, rtol=7E-4)

#-----------------------------------------------------------------------------

