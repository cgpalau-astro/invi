"""Petar configuration files."""

import invi as _invi

__all__ = ["ic_input", "run_qsub", "pot_conf"]

#-----------------------------------------------------------------------------

def ic_input(job_prm):
    """Generation ic.input for petar code."""
    #----------------------------------------------------
    #Initial condition globular cluster from galpy orbit
    x_gc, y_gc, z_gc, vx_gc, vy_gc, vz_gc = job_prm['ic']
    #----------------------------------------------------
    #Initial conditions stars
    x, y, z, vx, vy, vz = job_prm['phase_space']
    #Mass stars
    mass = job_prm['mass']
    #----------------------------------------------------
    #Write petar configuration file
    file_path = job_prm['file_path'] + job_prm['name']
    with open(f"{file_path}/ic.input", "w", encoding="utf-8") as file:

        file.write(f"0 {len(x)} 0 {x_gc} {y_gc} {z_gc} {vx_gc} {vy_gc} {vz_gc}\n")

        for i in range(len(x)):
            #line = "{0:+0.15E} {1:+0.15E} {2:+0.15E} {3:+0.15E} {4:+0.15E} {5:+0.15E} {6:+0.15E} 0 0 {7} 0 0 0 0 0 0 0 0 0 0 0\n".format(mass[i], x[i], y[i], z[i], vx[i], vy[i], vz[i], i+1)
            line = f"{mass[i]:+0.15E} {x[i]:+0.15E} {y[i]:+0.15E} {z[i]:+0.15E} {vx[i]:+0.15E} {vy[i]:+0.15E} {vz[i]:+0.15E} 0 0 {i+1} 0 0 0 0 0 0 0 0 0 0 0\n"
            file.write(line)
    #----------------------------------------------------

#-----------------------------------------------------------------------------

def run_qsub(job_prm, verbose=True):
    """Generation run.conf for petar code."""
    #--------------------------------------------------------
    match job_prm['type_job']:
        case 'debug':
            walltime = '3:00:00'
        case 'small':
            walltime = '72:00:00'
        case _:
            raise ValueError("type_job = {'small', 'debug'}")
    #--------------------------------------------------------
    line = f"""#!/bin/bash
#PBS -N petar_{job_prm['name']}
#PBS -m abe
#PBS -M cgarcia@sjtu.edu.cn
#PBS -l nodes=1:ppn=72
#PBS -l mem={job_prm['memory']}gb
#PBS -l walltime={walltime}
#PBS -q {job_prm['type_job']}
#PBS -j oe
#PBS -o /home/cgarcia/NBody/Runs/{job_prm['name']}/output.log

module load python/python-3.8.5
module load compiler/gcc-6.5.0
module load mpi/mpich-3.2.1-gcc
module load gsl/2.5
cd /home/cgarcia/NBody/Runs/{job_prm['name']}/
OMP_NUM_THREADS=1 mpiexec -n {job_prm['cores']} petar -u 1 -t {job_prm['T']} -o 1.0 -s {0.5**job_prm['s']} -r {job_prm['r']} --galpy-conf-file pot.conf ic.input"""
    #--------------------------------------------------------
    #Write configuration file
    file_path = job_prm['file_path'] + job_prm['name']
    with open(f"{file_path}/run.qsub", "w", encoding="utf-8") as file:
        file.write(line)
    #--------------------------------------------------------
    #Print configuration file
    if verbose:
        print(line)
    #--------------------------------------------------------

#-----------------------------------------------------------------------------

def pot_conf(job_prm):
    """Generation pot.conf for petar code."""
    #-------------------------------------------------
    bulge = job_prm['potential'][0]
    disc = job_prm['potential'][1]
    halo = job_prm['potential'][2]

    #Get number characterising each potential and its arguments
    bulge_n, bulge_args = _invi.galpy.potential.n_args(bulge)[1:3]
    disc_n, disc_args = _invi.galpy.potential.n_args(disc)[1:3]
    halo_n, halo_args = _invi.galpy.potential.n_args(halo)[1:3]

    args = ' '.join(map(str, bulge_args)) + ' ' + ' '.join(map(str, disc_args)) + ' ' + ' '.join(map(str, halo_args))
    #-------------------------------------------------
    #Write configuration file
    file_path = job_prm['file_path'] + job_prm['name']
    with open(f"{file_path}/pot.conf", "w", encoding="utf-8") as file:
        file.write("0.0 1\n")
        file.write("3 0 0.0 0.0 0.0 0.0 0.0 0.0\n")
        file.write(f"{bulge_n[0]} {disc_n[0]} {halo_n[0]}\n")
        file.write(args)
    #-------------------------------------------------

#-----------------------------------------------------------------------------
