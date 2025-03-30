"""PeTar compress output.

Run: qsub run.qsub

run.qsub
--------

#!/bin/bash
#PBS -N compress
#PBS -m abe
#PBS -M cgarcia@sjtu.edu.cn
#PBS -l nodes=1:ppn=72
#PBS -l mem=360gb
#PBS -l walltime=72:00:00
#PBS -q small
#PBS -j oe
#PBS -o /tmp/output_compress.log

module load python/python-3.8.5
python3.8 /home/cgarcia/compress.py"""

import os
import fnc

N = 1_500
ORIGIN = "/home/cgarcia/NBody/Runs/limepy_cmd_rei_2/"
DESTINATION = "/home/cgarcia/M68_run_0"

#-----------------------------------------------------------------------------

def gzip(i):
    """Compress using gzip."""
    fnc.utils.bash.cmd(f"gzip --best {DESTINATION}/data.{i}")

#-----------------------------------------------------------------------------

#Make destination folder
os.makedirs(DESTINATION, exist_ok=True)

#Copy all files to destination
fnc.utils.bash.cmd(f"cp {ORIGIN}/* {DESTINATION}")

#Initialization input parameter
i = list(range(0, N+1, 1))

#Count number CPUs
n_cpu = os.cpu_count()

#Compress data.i files in destination
output = fnc.utils.pool.run(gzip, i, n_cpu=n_cpu, progress=True)

#Tar destination
fnc.utils.bash.cmd(f"tar -cf {DESTINATION}.tar {DESTINATION}")

#Remove destination
fnc.utils.bash.cmd(f"rm -rf {DESTINATION}")

#-----------------------------------------------------------------------------
