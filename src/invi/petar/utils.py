"""Utils for petar and Gravity Server (SJTU)."""

import os as _os

__all__ = ["upload_zip_folder", "download_tar_folder"]

def _get_key(user):
    return f"/home/{user}/.ssh/id_rsa_Gravity"

#-----------------------------------------------------------------------------

def upload_zip_folder(name_file):
    """Upload zipped petar folder."""
    user = _os.getlogin()
    key = _get_key(user)
    print(f"scp -i {key} -C /home/{user}/Projects/invi/data/petar/{name_file}.zip cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/")

def download_tar_folder(name_file):
    """Download tar petar folder."""
    user = _os.getlogin()
    key = _get_key(user)
    print(f"scp -i {key} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/{name_file}.tar /home/{user}/Projects/invi/data/petar/")

#-----------------------------------------------------------------------------
"""
def last_snapshot(name, verbose=True):
    curr_directory = _os.getcwd()
    data_status = _np.loadtxt(f"{curr_directory}/runs/{name}/data.status", usecols=0)
    last_snap = int(data_status[-1])
    if verbose == True: cprint.ok(f"\nLast snapshot = {last_snap}")
    return last_snap

def download_last_snapshot(name):
    curr_directory = _os.getcwd()#.replace(" ","\ ")
    print(f"scp -i {KEY} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{name}/data.status {curr_directory}/runs/{name}/")
    last_snap = last_snapshot_petar(name, verbose=False)
    print(f"scp -i {KEY} -C cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{name}/data.{last_snap} {curr_directory}/runs/{name}/")
"""
#-----------------------------------------------------------------------------
"""
def download_files(name):
    curr_directory = _os.getcwd()
    down_load = ""
    for file in ["data.status", "data.0", "data.1500", "output.log",
                 "ic.input", "pot.conf", "run.qsub", "input.par",
                 "input.par.galpy", "input.par.hard"]:
        down_load += f"scp -i {KEY} cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{name}/{file} {curr_directory}/runs/{name}/;"
    print(down_load)

def compress_download_files(name):
    zip = f"zip -q -7 -r {name}.zip {name}/"
    print(zip)
    print("")
    curr_directory = _os.getcwd()
    down_zip = f"scp -i {KEY} cgarcia@login01.gravity.sjtu.edu.cn:/home/cgarcia/NBody/Runs/{name}.zip {curr_directory}/runs/"
    print(down_zip)
"""
#-----------------------------------------------------------------------------
