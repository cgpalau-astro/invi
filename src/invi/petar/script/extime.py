#!/usr/bin/env python3

"""Information about petar execution.

Note
----
1)  File has to be located at: /home/cgarcia/.local/bin"""

import os
import sys
import numpy as np
import termcolor as tc

import fnc

#-----------------------------------------------------------------------------

def print_time_stats(path, T):
    file_0 = path + "data.0"
    file_1 = path + f"data.{T}"
    file_2 = path + f"data.{T-1}"

    #Determine elapsed time
    first_snapshot_time = elapsed_time(file_0, path + "data.1") #[s]
    total_time = elapsed_time(file_0, file_1) #[s]
    mean_snapshot_time = total_time/T #[s]
    last_snapshot_time = elapsed_time(file_1, file_2) #[s]

    tc.cprint("Time:", 'light_blue')
    print(f"Total elapsed:  {fnc.utils.human_readable.time(total_time)}")
    print(f"First snapshot: {fnc.utils.human_readable.time(first_snapshot_time)}")
    print(f"Mean snapshot:  {fnc.utils.human_readable.time(mean_snapshot_time)}")
    print(f"Last snapshot:  {fnc.utils.human_readable.time(last_snapshot_time)}\n")


def print_eta(path, N, T):
    file_1 = path + f"data.{T}"
    file_2 = path + f"data.{T-1}"

    last_snapshot_time = elapsed_time(file_1, file_2) #[s]

    eta = (N-T) * last_snapshot_time

    if eta > 72*3600.0 :
        weta = tc.colored("[Warning]: ETA", 'yellow')
        tc.cprint(f"{weta} {fnc.utils.human_readable.time(eta)} > 72 [h]\n")
    else:
        ceta = tc.colored("ETA:", 'light_blue')
        tc.cprint(f"{ceta} {fnc.utils.human_readable.time(eta)}\n")

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

def print_status(argv):
    output_0 = fnc.utils.bash.cmd(f"qstat | grep cgarcia | grep {argv}")
    output_1 = fnc.utils.bash.cmd("showq | grep JOBNAME | grep REMAINING")
    output_2 = fnc.utils.bash.cmd("showq | grep cgarcia")
    if output_0 == "":
        tc.cprint("No execution", 'red')
    else:
        tc.cprint("Status:", 'light_blue')
        print(output_0[0:-1])
        print()
        print(output_1[0:-2])
        print(output_2[0:-1])
        print()

#-----------------------------------------------------------------------------

def print_petar_warnings():
    tc.cprint("PeTar warnings:", 'light_blue')

    hard = len([filename for filename in os.listdir(path) if filename.startswith("hard_large_energy")])
    if hard > 0:
        tc.cprint("[Warning]: Hard Large Energy", "yellow")
    else:
        print("No hard large energy")

    large_cluster = fnc.utils.bash.cmd(f"cat {path}output.log | grep \"Large cluster\" | tail -n 1")[:-1]
    if large_cluster != "":
        tc.cprint(f"[Warning]: {large_cluster}", "yellow")
    else:
        print("No large cluster")

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    #Number snapshots
    N = 1_500

    #Directory
    path = f"/home/cgarcia/NBody/Runs/{sys.argv[1]}/"
    cdir = tc.colored("Directory:", 'light_blue')
    tc.cprint(f"{cdir} {path}\n")

    #Get last snapshot
    T = last_snapshot(path + "data.status")

    if T < 2:
        tc.cprint("[Error]: Number of snapshots < 2", 'red')
    elif T == N:
        tc.cprint(f"Last Snapshot: {T}/{N}", 'green')
        print_time_stats(path, T)
    else:
        print_status(sys.argv[1])
        csnap = tc.colored("Last Snapshot:", 'light_blue')
        tc.cprint(f"{csnap} {T}/{N}\n")

        print_eta(path, N, T)
        print_time_stats(path, T)

    #Print large energy and large cluster warnings
    if T >= 2:
        print_petar_warnings()
