"""Tests loading data files.

Note
----
1)  Run pytest from 'invi' folder: 'python3.13 -m pytest -Wall -vs --durations=0 lib/invi/tests/tests_data.py'

2)  pylint tests_data.py --disable={C0301,C0103,C0116,C0415}"""

#-----------------------------------------------------------------------------

def test_gaia():
    import tomllib
    import invi

    with open("data/parameters.toml", "rb") as f:
        prm = tomllib.load(f)

    #Globular cluster
    file_name = "data/gaia_dr3/M68/globular_cluster.zip"
    _gc = invi.data.globular_cluster.load(file_name, prm['M68'], verbose=False)

    #Stream catalogue
    file_name = "data/gaia_dr3/M68/stream.csv"
    _st = invi.data.gaia.load(file_name, cmd37_correction=0.0, verbose=False)

    #Stream mock catalogue
    file_name = "data/gaia_dr3/M68/mock_stream.csv"
    _mock_st = invi.data.gaia.load(file_name, cmd37_correction=0.0, verbose=False)


def test_ibata():
    import invi
    #Ibata streamfinder catalogue
    file_name = "data/ibata/data.zip"
    _ibata = invi.data.ibata.load(file_name, cmd37_correction=0.0, verbose=False)


def test_sos():
    import invi
    #Survey of surveys DR1 radial velocities catalogue
    file_name = "data/sos_dr1/sosdr1.zip"
    _sos = invi.data.sos.load(file_name, verbose=False)


def test_asdf():
    import asdf
    #Desi and M68 simulation
    file_list = ["data/desi/iron.asdf",
                 "data/desi/loa.asdf",
                 "data/M68.asdf"]

    for file_name in file_list:
        f = asdf.open(file_name, lazy_load=False)
        f.validate()
        f.close()


def test_cmd():
    import invi
    #CMD37 synthetic population
    cmd_file = "data/synthetic_population/M68/population.dat.zip"
    _cmd = invi.globular_cluster.synthetic_population._load_CMD_file(cmd_file, verbose=False)

def test_csv():
    import invi
    #Stream additional data
    file_name = "data/stream_additional_data.csv"
    _add_data = invi.data.csv.load(file_name, verbose=False)

#-----------------------------------------------------------------------------
