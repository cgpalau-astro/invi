"""Information about petar execution.

Note
----
1)  File has to be located at: /home/cgarcia/.local/bin"""

#!/usr/bin/env python3
import os
import sys
import numpy as np
import termcolor as tc

import fnc

#-----------------------------------------------------------------------------

def print_line():
    print("-"*79)

def print_time_stats(path, T):
    file_0 = path + "data.0"
    file_1 = path + f"data.{T}"
    file_2 = path + f"data.{T-1}"

    #Determine elapsed time
    first_snapshot_time = elapsed_time(file_0, path + "data.1") #[s]
    total_time = elapsed_time(file_0, file_1) #[s]
    mean_snapshot_time = total_time/T #[s]
    last_snapshot_time = elapsed_time(file_1, file_2) #[s]

    print(f"Elapsed Time:        {total_time/3600.0/24.0:0.2} [d] | {total_time/3600.0:0.4} [h]")
    print(f"First Snapshot Time: {first_snapshot_time/60.0:0.3} [m] | {first_snapshot_time:0.5} [s]")
    print(f"Mean Snapshot Time:  {mean_snapshot_time/60.0:0.3} [m] | {mean_snapshot_time:0.5} [s]")
    print(f"Last Snapshot Time:  {last_snapshot_time/60.0:0.3} [m] | {last_snapshot_time:0.5} [s]")
    print_line()

def print_eta(path, N, T):
    file_1 = path + f"data.{T}"
    file_2 = path + f"data.{T-1}"

    last_snapshot_time = elapsed_time(file_1, file_2) #[s]

    eta = (N-T) * last_snapshot_time

    print_line()
    if eta > 72*3600.0 :
        tc.cprint(f"WARNING ETA: {eta/3600.0/24.0:0.2} [d] | {eta/3600.0:0.4} [h] > 72 [h]", "red")
    else:
        tc.cprint(f"ETA:                {eta/3600.0/24.0:0.2} [d] | {eta/3600.0:0.4} [h]", "green")

#-----------------------------------------------------------------------------

def last_snapshot(file_path):
    t = np.loadtxt(file_path, unpack=True, usecols=0)
    if t.size == 0:
        return -1

    T = np.int64(t[-1])
    return T

def elapsed_time(file_0, file_1):
    t_0 = os.stat(file_0).st_ctime #[s]
    t_1 = os.stat(file_1).st_ctime #[s]
    elapsed_time = np.abs(t_1 - t_0)
    return elapsed_time

#-----------------------------------------------------------------------------

def is_executing(argv):
    output_0 = fnc.utils.bash.cmd(f"qstat | grep cgarcia | grep {argv}")
    output_1 = fnc.utils.bash.cmd("showq | grep JOBNAME | grep REMAINING")
    output_2 = fnc.utils.bash.cmd("showq | grep cgarcia")
    if output_0 == "":
        tc.cprint("No execution", "red")
    else:
        tc.cprint("Executing:", "green")
        print(output_0[0:-1])
        print(output_1[0:-2])
        print(output_2[0:-1])
    print_line()

#-----------------------------------------------------------------------------
#Warnings

def print_hle_warning():
    hard = len([filename for filename in os.listdir(path) if filename.startswith("hard_large_energy")])
    if hard > 0:
        tc.cprint("WARNING: Hard Large Energy", "red")
    else:
        tc.cprint("No Hard Large Energy", "green")

    large_cluster = fnc.utils.bash.cmd(f"cat {path}output.log | grep \"Large cluster\" | tail -n 1")[:-1]
    if large_cluster != "":
        tc.cprint(f"WARNING: {large_cluster}", "red")
    else:
        tc.cprint("No Large Cluster", "green")

    print_line()

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    #Number snapshots
    N = 1_500

    #Directory
    path = f"/home/cgarcia/NBody/Runs/{sys.argv[1]}/"
    print_line()
    print(f"Directory: {path}")
    print_line()

    #Get last snapshot
    T = last_snapshot(path + "data.status")

    if T < 2:
        tc.cprint("ERROR: Number snapshots < 2", "red")
    elif T == N:
        tc.cprint(f"Last Snapshot: {T}/{N}", "green")
        print_line()
        print_time_stats(path, T)
    else:
        is_executing(sys.argv[1])
        tc.cprint(f"Last Snapshot: {T}/{N}", "yellow")

        print_eta(path, N, T)
        print_time_stats(path, T)

    if T >= 2:
        print_hle_warning()

#-----------------------------------------------------------------------------
