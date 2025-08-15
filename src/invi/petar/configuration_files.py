"""Petar configuration files."""

import numpy as _np

import invi as _invi

__all__ = ["ic_input", "pot_conf", "run_qsub", "resume_qsub", "compress_qsub"]

#-----------------------------------------------------------------------------

def ic_input(prm_job):
    """Generation ic.input for petar code."""

    #Initial condition globular cluster from galpy orbit
    x_gc, y_gc, z_gc, vx_gc, vy_gc, vz_gc = prm_job['ic']

    #Initial conditions stars
    x, y, z, vx, vy, vz = prm_job['phase_space']

    #Mass stars
    mass = prm_job['mass']

    #Write petar configuration file
    file_path = prm_job['file_path'] + prm_job['name']
    with open(f"{file_path}/ic.input", "w", encoding="ASCII") as file:

        file.write(f"0 {len(x)} 0 {x_gc} {y_gc} {z_gc} {vx_gc} {vy_gc} {vz_gc}\n")

        for i in range(len(x)):
            line = f"{mass[i]:+0.15E} {x[i]:+0.15E} {y[i]:+0.15E} {z[i]:+0.15E} {vx[i]:+0.15E} {vy[i]:+0.15E} {vz[i]:+0.15E} 0 0 {i+1} 0 0 0 0 0 0 0 0 0 0 0\n"
            file.write(line)

#-----------------------------------------------------------------------------

def pot_conf(prm_job):
    """Generation pot.conf for petar code."""

    #Turn a single component into a list
    potential = prm_job['potential']
    if not isinstance(potential, list):
        potential = [potential]

    #Number potential components
    n_comp = len(potential)

    #Get number and arguments characterising each potential component
    num = ''
    args = ''
    for item in potential:
        item_num, item_args = _invi.galpy.potential.n_args(item)[1:3]
        num += str(item_num[0]) + ' '
        args += ' '.join(map(str, item_args)) + ' '
    num += '\n'

    #Write configuration file
    file_path = prm_job['file_path'] + prm_job['name']
    with open(f"{file_path}/pot.conf", "w", encoding="ASCII") as file:
        file.write("0.0 1\n")
        file.write(f"{n_comp} 0 0.0 0.0 0.0 0.0 0.0 0.0\n")
        file.write(num)
        file.write(args)

#-----------------------------------------------------------------------------

def run_qsub(prm_job, verbose=True):
    """Generation run.conf for petar code."""

    #Execution time
    match prm_job['type_job']:
        case 'debug':
            walltime = '3:00:00'
        case 'small':
            walltime = '72:00:00'
        case _:
            raise ValueError("type_job = {'small', 'debug'}")

    #Time as an integer
    T = _np.int64(prm_job['T'])

    #Definition file lines
    lines = f"""#!/bin/bash
#PBS -N petar_{prm_job['name']}
#PBS -m abe
#PBS -M cgarcia@sjtu.edu.cn
#PBS -l nodes=1:ppn=72
#PBS -l mem={prm_job['memory']}gb
#PBS -l walltime={walltime}
#PBS -q {prm_job['type_job']}
#PBS -j oe
#PBS -o /home/cgarcia/NBody/Runs/{prm_job['name']}/output.log

module load python/python-3.8.5
module load compiler/gcc-6.5.0
module load mpi/mpich-3.2.1-gcc
module load gsl/2.5
cd /home/cgarcia/NBody/Runs/{prm_job['name']}/
OMP_NUM_THREADS=1 mpiexec -n {prm_job['cores']} petar -u 1 -t {T} -o 1.0 -s {0.5**prm_job['s']} -r {prm_job['r']} --galpy-conf-file pot.conf ic.input"""

    #Write configuration file
    file_path = prm_job['file_path'] + prm_job['name']
    with open(f"{file_path}/run.qsub", "w", encoding="ASCII") as file:
        file.write(lines)

    #Print configuration file
    if verbose:
        print(lines)


def resume_qsub(prm_job):
    """Resume petar run.

    Note
    ----
    1)  Substitute 'x' in data.x for the number of the last snapshot."""

    #Resume command
    line = f"OMP_NUM_THREADS=1 mpiexec -n {prm_job['cores']} petar -p input.par data.x"

    file_path = prm_job['file_path'] + prm_job['name']

    #Read file line by line and substitute the last line
    with open(f"{file_path}/run.qsub", "r", encoding="ASCII") as file:
        lines = file.readlines()
    lines[-1] = line

    #Write resume file
    with open(f"{file_path}/resume.qsub", "w", encoding="ASCII") as file:
        for line in lines:
            file.write(line)


def compress_qsub(prm_job):
    """Compress output petar."""

    #Definition file lines
    lines = f"""#!/bin/bash
#PBS -N compress_{prm_job['name']}
#PBS -m abe
#PBS -M cgarcia@sjtu.edu.cn
#PBS -l nodes=1:ppn=72
#PBS -l mem=360gb
#PBS -l walltime=72:00:00
#PBS -q small
#PBS -j oe
#PBS -o /home/cgarcia/trash/compress_{prm_job['name']}_output.log

tar --create --file=/home/cgarcia/{prm_job['name']}.tar /home/cgarcia/NBody/Runs/{prm_job['name']}/
xz --compress -4 --threads=0 /home/cgarcia/{prm_job['name']}.tar
"""

    #Write configuration file
    file_path = prm_job['file_path'] + prm_job['name']
    with open(f"{file_path}/compress.qsub", "w", encoding="ASCII") as file:
        file.write(lines)

#-----------------------------------------------------------------------------
