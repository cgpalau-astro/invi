"""Utils for petar and Gravity Server (SJTU)."""

import os as _os
import numpy as _np

import fnc as _fnc

__all__ = ["upload_zip_folder",
           "last_snapshot", "download_snapshot",
           "download_compressed_folder",
           "gzip_data"]

def _get_data():
    user = _os.getlogin()
    key = f"/home/{user}/.ssh/id_rsa_Gravity"
    return user, key

#-----------------------------------------------------------------------------

def upload_zip_folder(file_name):
    """Upload zipped petar folder."""
    user, key = _get_data()
    print(f"scp -i {key} -C /home/{user}/projects/invi/data/petar/{file_name}.zip cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/")

#-----------------------------------------------------------------------------

def last_snapshot(file_name):
    """Print and return last snapshot."""

    #Download data.status
    _, key = _get_data()
    line = f"scp -i {key} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{file_name}/data.status /tmp/"
    _fnc.utils.bash.cmd(line)

    #Check status
    data_status = _np.loadtxt("/tmp/data.status", usecols=0)
    snapshot = _np.int64(data_status[-1])

    #Print last snapshot
    print(f"Last snapshot = {snapshot}")

    return snapshot


def download_snapshot(file_name, snapshot):
    """Download snapshot at /tmp/ folder."""

    _, key = _get_data()

    #Download file
    line = f"scp -i {key} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{file_name}/data.{snapshot} /tmp/"
    _fnc.utils.bash.cmd(line)

    #Compress file
    _fnc.utils.bash.cmd(f"gzip --fast /tmp/data.{snapshot}")

#-----------------------------------------------------------------------------

def download_compressed_folder(file_name):
    """Download compressed petar folder with the results of the simulation."""
    user, key = _get_data()
    print(f"scp -i {key} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/{file_name}.tar.xz /home/{user}/projects/invi/data/petar/")

#-----------------------------------------------------------------------------

def _gzip(init):
    """Compress using gzip."""
    fnc.utils.bash.cmd(f"gzip --best {init[1]}/data.{init[0]}")


def gzip_data(folder_name, progress=True):
    """Compress with gzip all data files."""

    #Get last snapshot
    data_status = np.loadtxt(f"{folder_name}/data.status", usecols=0)
    last_snapshot = np.int64(data_status[-1])

    #Initialization input parameter
    init = [(i, folder_name) for i in range(0, last_snapshot+1, 1)]

    #Count number CPUs
    n_cpu = os.cpu_count()

    #Compress data.i files in destination
    output = fnc.utils.pool.run(_gzip, init, n_cpu, progress)

#-----------------------------------------------------------------------------
