"""Utils for petar and Gravity Server (SJTU).

    Note
    ----
    1)  Gravity server: https://gravity-doc.github.io/#/"""

import os as _os
import numpy as _np

import fnc as _fnc

__all__ = ["print_upload_zip_folder",
           "last_snapshot", "download_snapshot",
           "print_download_compressed_folder",
           "gzip_data"]


def _get_data():
    user = _os.getlogin()
    key = f"/home/{user}/.ssh/id_rsa_Gravity"
    return user, key

#-----------------------------------------------------------------------------

def print_upload_zip_folder(file_name):
    """Upload zipped petar folder."""
    user, key = _get_data()
    print(f"scp -i {key} -C /home/{user}/projects/invi/data/petar/{file_name}.zip cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/")

#-----------------------------------------------------------------------------

def last_snapshot(file_name, verbose=True):
    """Print and return last snapshot."""

    #Check internet connection
    if not _fnc.info.internet_connection():
        print("[Error] No internet connection")
        return None

    _, key = _get_data()

    #Compress file
    #Modern versions of gzip allow --keep option to keep the compressed file
    line_0 = f"ssh -i {key} cgarcia@login01.gravity.sjtu.edu.cn 'gzip --best --to-stdout /home/cgarcia/NBody/Runs/{file_name}/data.status > /home/cgarcia/trash/data.status.gz'"

    #Download file
    line_1 = f"scp -i {key} cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/trash/data.status.gz /tmp/"

    #Remove file
    line_2 = f"ssh -i {key} cgarcia@login01.gravity.sjtu.edu.cn 'rm /home/cgarcia/trash/data.status.gz'"

    #Run lines
    line = line = ' ; '.join([line_0, line_1, line_2])
    _fnc.utils.bash.cmd(line)

    #Check status
    data_status = _np.loadtxt("/tmp/data.status.gz", usecols=0)
    snapshot = _np.int64(data_status[-1])

    #Print last snapshot
    if verbose:
        print(f"Last snapshot = {snapshot}")

    return snapshot


def download_snapshot(file_name, snapshot):
    """Download snapshot at /tmp/ folder."""

    #Check internet connection
    if not _fnc.info.internet_connection():
        print("[Error] No internet connection")
        return

    _, key = _get_data()

    #Compress file
    #Modern versions of gzip allow --keep option to keep the compressed file
    line_0 = f"ssh -i {key} cgarcia@login01.gravity.sjtu.edu.cn 'gzip --best --to-stdout /home/cgarcia/NBody/Runs/{file_name}/data.{snapshot} > /home/cgarcia/trash/data.{snapshot}.gz'"

    #Download file
    line_1 = f"scp -i {key} cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/trash/data.{snapshot}.gz /tmp/"

    #Remove file
    line_2 = f"ssh -i {key} cgarcia@login01.gravity.sjtu.edu.cn 'rm /home/cgarcia/trash/data.{snapshot}.gz'"

    #Run lines
    line = line = ' ; '.join([line_0, line_1, line_2])
    _fnc.utils.bash.cmd(line)

#-----------------------------------------------------------------------------

def print_download_compressed_folder(file_name):
    """Download compressed petar folder with the results of the simulation."""
    user, key = _get_data()
    print(f"scp -i {key} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/{file_name}.tar.xz /home/{user}/projects/invi/data/petar/")

#-----------------------------------------------------------------------------

def _gzip(init):
    """Compress using gzip."""
    _fnc.utils.bash.cmd(f"gzip --best {init[1]}/data.{init[0]}")


def gzip_data(folder_name, snapshot_range, progress=True):
    """Compress all data files within snapshot_range using gzip."""

    #Initialization input parameter
    init = [(i, folder_name) for i in snapshot_range]

    #Count number CPUs
    n_cpu = _os.cpu_count()

    #Compress data.i files in destination
    _ = _fnc.utils.pool.run(_gzip, init, n_cpu, progress)

#-----------------------------------------------------------------------------
