#! /usr/bin/env python3

# dependencies
import argparse
import glob
import os
import sys
import shutil
import subprocess
import tempfile

import numpy as np

import measures
import dnam_tools
import h5tools

""" PARAMETERS - forcing generator """
PATH_FORCING_GEN = "/home/100/dn8909/source/forcing_generator/"

ST_POWER_LAW_EXP = "-2.0"
VELOCITY = "1.0e5"
#ST_ENERGY = "1.03e-2 * velocity**3/L" # for beta=1 (0.82e-2 for UG)
ST_ENERGY = "1.7e-2 * velocity**3/L" # for beta=2
ST_STIRMAX = "(256+eps) * twopi/L"

""" PARAMETERS - FLASH """
PATH_FLASH = "/home/100/dn8909/source/flash4.0.1-rsaa_fork/"
PATH_DATA = "/scratch/ek9/dn8909/data/"
FLASH_EXE = "flash4"

CHKFILE_PREFIX = "Turb_hdf5_chk"
PLOTFILE_PREFIX = "Turb_hdf5_plt_cnt"
PARTFILE_PREFIX = "Turb_hdf5_part"

RHO_0 = 4.23e-21
RES_DENS_FACTOR = 1.0
LREFINE_MAX = 3

KEEP_FILES_AT_SFE = np.linspace(0.01,0.20,20)

""" PARAMETERS - QSUB """
PROJECT = "ek9"
QUEUE = "normal"
WALLTIME = "24:00:00"
NCPUS = 528
MEM = f"{NCPUS * 4}GB"
NAME_JOB = "b2tA"

"""
================================================================================
TASK FUNCTIONS
================================================================================
"""
def replace_file(filename, begin_at=0, comment_symbol='#', **kwargs):
    """
   DESCRIPTION
        finds the parameters in kwargs and replace its value as specified in kwargs
    """

    # open files
    print(f"opening {filename} for replacing...")
    temp_file = tempfile.NamedTemporaryFile(mode='w+t')
    existing_file = open(filename, 'r')

    line_no = 1
    for line in existing_file:

        # replacement begins only when we arrive at the specified line number
        if line_no >= begin_at:
            # check whether the parameters exist
            for key, value in kwargs.items():
                # if the key is found in the line
                if line.strip().startswith(key):
                    print(f"(line {line_no}) found: {line.rstrip()}")
                    i_start = line.find("=")  # replace from one space after the equal sign
                    i_end = line.find(comment_symbol)  # until the comment starts (if found)
                    # if there are no comments found
                    if i_end == -1 :
                        line = f"{line[:i_start+1]} {value}\n"
                    else :
                        len = i_end-i_start-2
                        line = f"{line[:i_start+1]} {value:<{len}}{line[i_end:]}"
                    print(f"(line {line_no}) =>     {line.rstrip()}")
                    break
            # endfor kwargs.items()
        # endif line_no >= begin_at

        temp_file.writelines(line)
        line_no += 1
    # endfor existing_file

    # replace the file
    temp_file.seek(0)
    shutil.copy2(temp_file.name, filename)
    temp_file.close()
    os.chmod(filename, 0o644)
    print("finished writing!")
    print(f"replaced {filename}!")


def generate_forcingfile(seed=140281):
    """
    DESCRIPTION
        runs the forcing generator with the given random seed number
        the other default parameters are defined as global variables (see top)
    """
    print(f"making the forcingfile with seed : {seed}...")
    # update the Fortran code
    replace_file(PATH_FORCING_GEN + "forcing_generator.F90", begin_at=623, comment_symbol='!',
                 st_power_law_exp=ST_POWER_LAW_EXP,
                 velocity=VELOCITY,
                 st_energy=ST_ENERGY,
                 st_stirmax=ST_STIRMAX,
                 st_seed=seed)

    # make the forcing generator and run
    print("making and running the forcing generator...")
    cp = subprocess.run(["make", "-C", PATH_FORCING_GEN], capture_output=True, check=True)
    fg_name = os.path.join(PATH_FORCING_GEN, "forcing_generator")
    cp = subprocess.run(fg_name, capture_output=True, check=True, text=True)

    # the last line of the stdout contains the name of the forcing file
    last_stdout = cp.stdout.split("\n")[-2]

    forcingfile_name = last_stdout[last_stdout.find('"') + 1:last_stdout.rfind('"')]
    print(f"forcing file created: {forcingfile_name}!")
    return forcingfile_name


def submit_job(project=PROJECT, queue=QUEUE, walltime=WALLTIME, ncpus=NCPUS, mem=MEM, dir="",
               action="", script_name="job.sh", job_name="test", prev_job_id=None):
    """
    DESCRIPTION
        writes a pbs script and submit the job and returns the job name
    """
    path_jobscript = os.path.join(dir, script_name)
    print(f"writing jobscript to {path_jobscript}...")
    job = open(path_jobscript, 'w')

    # to request more than more than one node one must request full nodes (multiples of 48)
    if (ncpus > 48 and ncpus%48 != 0) :
        ncpus_gadi = 48*(ncpus//48 + 1)
        mem_gadi = f"{ncpus_gadi * 4}GB"
    else :
        ncpus_gadi = ncpus
        mem_gadi = mem

    job.write("#!/bin/bash \n")
    job.write(f"#PBS -P {project} \n")
    job.write(f"#PBS -q {queue} \n")
    job.write(f"#PBS -l walltime={walltime} \n")
    job.write(f"#PBS -l ncpus={ncpus_gadi} \n")
    job.write(f"#PBS -l mem={mem_gadi} \n")
    job.write(f"#PBS -l storage=scratch/ek9 \n")
    job.write("#PBS -l wd \n")
    job.write(f"#PBS -N {job_name} \n")
    job.write("#PBS -j oe \n")
    if prev_job_id is not None:
        job.write(f"#PBS -W depend=afterany:{prev_job_id} \n")

    for line in action.split("\n"):
        job.write(line + "\n")
    job.close()

    print("submitting...")
    cp = subprocess.run(["qsub", path_jobscript], capture_output=True, check=True, text=True)
    job_id = cp.stdout.split()[-1]

    print(f"completed! job id: {job_id}")
    return job_id

"""
================================================================================
MACRO FUNCTIONS
================================================================================
"""
def first_sim(original_dir, seed=140281, depend=None):
    """
    generate the forcing file and submit the driving phase simulation
    """
    # full path to flash4 file and flash.par
    path_flash_exe = os.path.join(original_dir, "flash4")
    path_flash_par = os.path.join(original_dir, "flash.par")

    # check whether all necessary files exist
    flash_exists = os.path.exists(path_flash_exe)
    flash_par_exists = os.path.exists(path_flash_par)
    if (not flash_exists) or (not flash_par_exists):
        sys.exit("flash files not found! check the directory again.")

    # generate the forcing file
    forcingfile_name = generate_forcingfile(seed)

    # create the directory and copy the flash files
    new_dir = f"{original_dir}_{seed:06d}/"
    os.mkdir(new_dir)
    shutil.copy2(path_flash_exe, new_dir)
    shutil.copy2(path_flash_par, new_dir)

    # modify the flash.par file
    flash_par_new = os.path.join(new_dir, "flash.par")
    replace_file(flash_par_new,
                 st_infilename=f'"../{forcingfile_name}"',
                 useGravity='.false.',
                 useParticles='.false.',
                 restart='.false.',
                 checkpointFileNumber="0",
                 plotFileNumber="0",
                 particleFileNumber="0",
                 plotFileIntervalTime="3.086e12",
                 particleFileIntervalTime="3.086e12",
                 tmax="6.173e13",
                 usePolytrope=".false.",
                 lrefine_max="1"
                 )

    # submit the simulation
    stdout = "shell.out"
    action = f"cd {new_dir} \nmpirun -np {NCPUS} flash4 1>{stdout} 2>&1"
    job_id = submit_job(project=PROJECT, queue=QUEUE, walltime=WALLTIME,
                        ncpus=NCPUS, mem=MEM, dir=new_dir, action=action,
                        script_name="job.sh", job_name=NAME_JOB+f"_{seed}", prev_job_id=depend)
    return job_id


def restart_grav1(original_dir, seed=140281, depend=None, wo_plt=False):
    """
    create a sink simulation and submit a short job
    for the re-initialisation required for turning on the gravity
    """
    # full path to flash4 file and flash.par
    path_flash_exe = os.path.join(original_dir, "flash4_grav")

    # full path to the turbulence-only simulation
    path_turb_sim = f"{original_dir}_{seed:06d}/"
    path_flash_par = os.path.join(path_turb_sim, "flash.par")

    # check whether all necessary files and directories exist
    flash_exists = os.path.exists(path_flash_exe)
    flash_par_exists = os.path.exists(path_flash_par)
    turb_sim_exists = os.path.exists(path_turb_sim)
    if (not flash_exists):
        sys.exit(f"{path_flash_exe} does not exist")
    elif not turb_sim_exists:
        sys.exit(f"{path_turb_sim} does not exist")
    elif not flash_par_exists:
        sys.exit(f"{path_flash_par} does not exist")

    # create the directory for the sink simulation
    new_dir = f"{original_dir}_{seed:06d}_sink/"
    os.mkdir(new_dir)
    # copy the flash executable and flash.par
    shutil.copy2(path_flash_exe, new_dir)
    shutil.copy2(path_flash_par, new_dir)

    # find the last checkfile and create a symbolic link
    chkfiles = os.path.join(os.getcwd(), path_turb_sim, f"{CHKFILE_PREFIX}_????")
    chkfile_last = dnam_tools.get_file(chkfiles, loc='last')
    link_chkfile_last = os.path.join(new_dir, os.path.split(chkfile_last)[1])
    cp = subprocess.run(["ln", "-s", chkfile_last, link_chkfile_last], capture_output=True, check=True, text=True)
    print(f"checkfile link created at: {link_chkfile_last}")

    # read the last checkfile
    h5t = h5tools.H5File(chkfile_last)

    # whether or not to generate the plotfiles
    particle_interval = "1.543e11"
    if wo_plt :
        plot_interval = "1.000e99"
    else :
        plot_interval = particle_interval

    # modify the flash.par file
    flash_par_new = os.path.join(new_dir, "flash.par")
    replace_file(flash_par_new,
                 useGravity=".false.",
                 useParticles=".false.",
                 restart=".true.",
                 nend="2",
                 checkpointFileNumber=f"{h5t.params['checkpointfilenumber']}",
                 plotFileNumber=f"{h5t.params['plotfilenumber']}",
                 particleFileNumber=f"{h5t.params['plotfilenumber']}",
                 plotFileIntervalTime=plot_interval,
                 particleFileIntervalTime=particle_interval,
                 tmax="3.086e14",
                 usePolytrope=".true.",
                 res_rescale_dens=".true.",
                 res_dens_factor=RES_DENS_FACTOR,
                 rho_ambient=RHO_0*RES_DENS_FACTOR
                 )

    # submit the simulation
    stdout = "shell.out.init"
    action = f"cd {new_dir} \nmpirun -np {NCPUS} flash4_grav 1>{stdout} 2>&1"
    job_id = submit_job(project=PROJECT, queue=QUEUE, walltime="02:00:00",
                        ncpus=NCPUS, mem=MEM, dir=new_dir, action=action,
                        script_name="job.sh.init", job_name=f"gr1_{seed}", prev_job_id=depend)
    return job_id


def restart_grav2(original_dir, seed=140281, depend=None):
    """
    submit a second restart for gravity simulation
    """
    # full path to the current simulation
    new_dir = f"{original_dir}_{seed:06d}_sink/"

    # exit if the folder is not found
    new_dir_exists = os.path.exists(new_dir)
    if not new_dir_exists:
        sys.exit(f"{new_dir} does not exist")

    # find the last checkfile
    chkfiles = os.path.join(new_dir, f"{CHKFILE_PREFIX}_????")
    chkfile_last = dnam_tools.get_file(chkfiles, loc='last')

    # read the last checkfile
    h5t = h5tools.H5File(chkfile_last)

    # modify the flash.par file
    flash_par_new = os.path.join(new_dir, "flash.par")
    replace_file(flash_par_new,
                 useGravity=".true.",
                 useParticles=".true.",
                 restart=".true.",
                 nend="1000000",
                 checkpointFileNumber=f"{h5t.params['checkpointfilenumber']}",
                 tmax="3.086e14",
                 usePolytrope=".true.",
                 res_rescale_dens=".false.",
                 res_dens_factor="1.0"
                 )

    # submit the simulation
    stdout = "shell.out.init2"
    action = f"cd {new_dir} \nmpirun -np {NCPUS} flash4_grav 1>{stdout} 2>&1"
    job_id = submit_job(project=PROJECT, queue=QUEUE, walltime=WALLTIME,
                        ncpus=NCPUS, mem=MEM, dir=new_dir, action=action,
                        script_name="job.sh.init2", job_name=f"{NAME_JOB}_{seed}", prev_job_id=depend)
    return job_id

def restart_gravamr(original_dir, seed=140281, fork=True, depend=None) :
    """
    submit the gravity simulation for the AMR simulation
    """
    # full path to the turbulence-only simulation
    path_turb_sim = f"{original_dir}_{seed:06d}/"
    path_flash_exe = os.path.join(path_turb_sim, "flash4")
    path_flash_par = os.path.join(path_turb_sim, "flash.par")

    # check whether all necessary files and directories exist
    flash_par_exists = os.path.exists(path_flash_par)
    turb_sim_exists = os.path.exists(path_turb_sim)
    if not turb_sim_exists:
        sys.exit(f"{path_turb_sim} does not exist")
    elif not flash_par_exists:
        sys.exit(f"{path_flash_par} does not exist")

    # find the last checkfile
    chkfiles = os.path.join(os.getcwd(), path_turb_sim, f"{CHKFILE_PREFIX}_????")
    chkfile_last = dnam_tools.get_file(chkfiles, loc='last')

    # if fork is turned on, create a new directory
    if fork :
        # create the directory for the sink simulation
        new_dir = f"{original_dir}_{seed:06d}_sink/"
        os.mkdir(new_dir)
        print(f"{new_dir} created!")
        # copy the flash executable and flash.par
        shutil.copy2(path_flash_exe, new_dir)
        shutil.copy2(path_flash_par, new_dir)
        print("flash files copied!")

        # create a symbolic link of the last checkfile
        link_chkfile_last = os.path.join(new_dir, os.path.split(chkfile_last)[1])
        cp = subprocess.run(["ln", "-s", chkfile_last, link_chkfile_last],
                            capture_output=True, check=True, text=True)
        print(f"checkfile link created at: {link_chkfile_last}")
    # if fork is not turned on, just work on the turbulence directory
    else :
        new_dir = path_turb_sim

    # read the last checkfile
    h5t = h5tools.H5File(chkfile_last)

    # modify the flash.par file
    flash_par_new = os.path.join(new_dir, "flash.par")
    replace_file(flash_par_new,
                 useGravity=".true.",
                 useParticles=".true.",
                 restart=".true.",
                 checkpointFileNumber=f"{h5t.params['checkpointfilenumber']}",
                 plotFileNumber=f"{h5t.params['plotfilenumber']}",
                 particleFileNumber=f"{h5t.params['plotfilenumber']}",
                 plotFileIntervalTime="1.543e11",
                 particleFileIntervalTime="1.543e11",
                 tmax="3.086e14",
                 res_rescale_dens=".true.",
                 res_dens_factor=RES_DENS_FACTOR,
                 rho_ambient=RHO_0*RES_DENS_FACTOR,
                 lrefine_max=LREFINE_MAX
                 )

    # submit the simulation
    stdout = "shell.out.gravinit"
    action = f"cd {new_dir} \nmpirun -np {NCPUS} flash4_grav 1>{stdout} 2>&1"
    job_id = submit_job(
        project=PROJECT, queue=QUEUE, walltime=WALLTIME, ncpus=NCPUS, mem=MEM,
        dir=new_dir, action=action, script_name="job.sh.gravinit",
        job_name=f"{NAME_JOB}_gr_{seed}", prev_job_id=depend
        )
    return job_id

def restart(original_dir, n_jobs, depend=None, flash_name=FLASH_EXE) :
    """
    submit the jobs multiple times for easier restart
    """
    print(f"initialising {n_jobs} restarts at {original_dir}...")
    # check if job.sh exists already
    jobshs = os.path.join(original_dir, "job[0-9]*.sh")
    jobsh_last = dnam_tools.get_file(jobshs,loc='last')
    if (jobsh_last is not None) and (jobsh_last != "job.sh") :
        # extract the number in the filename
        jobsh_filename = os.path.split(jobsh_last)[-1]
        i_last = int( ''.join([c for c in jobsh_filename if c.isdigit()]) )
        print(f"{jobsh_last} found!")
    else :
        i_last = 0

    # repeat for the number of jobs
    prev_job_id = depend
    for i in range(i_last+1, n_jobs+i_last+1) :
        script_name = f"job{i}.sh"
        stdout = f"shell.out{i:02d}"
        action = ( f"cd {original_dir}\n"
                    "prep_restart.py -auto\n"
                   f"mpirun -np {NCPUS} {FLASH_EXE} 1>{stdout} 2>&1" )
        job_name = f"{NAME_JOB}_{original_dir[-6:]}_{i}"
        prev_job_id = submit_job(
            project=PROJECT, queue=QUEUE, walltime=WALLTIME, ncpus=NCPUS,
            mem=MEM, dir=original_dir, action=action, script_name=script_name,
            job_name=job_name, prev_job_id=prev_job_id)

    print("exiting flashtools.py...")

def clean_hdf5(original_dir, seed=140281, keep_files_at_sfe=KEEP_FILES_AT_SFE,
               delete_plotfiles=True, delete_partfiles=True, delete_chkfiles=True) :
    print(f"inspecting {original_dir}_{seed:06d}_sink for cleanning...")
    # first find all the plot and particle files
    filenames_plt = glob.glob(
        os.path.join(f"{original_dir}_{seed:06d}_sink", f"{PLOTFILE_PREFIX}_????"))
    filenames_part = glob.glob(
        os.path.join(f"{original_dir}_{seed:06d}_sink", f"{PARTFILE_PREFIX}_????"))
    filenames_plt.sort()
    filenames_part.sort()
    n_files = len(filenames_part)
    print(f"found {n_files} files!")
    if n_files == 0 :
        return 0.0

    # loop over the file
    j_SFE = 0
    SFEs_all_found = False
    for i in range(1, n_files-1) :
        # reset the filenames for deletion
        plotfile_deleted = None
        partfile_deleted = None

        # check the SFE of the file
        cst = measures.CST(filename=filenames_part[i], verbose=False)
        SFE = cst.SFE

        # if there are no sinks, delete the previous file
        if SFE == 0.0 :
            if delete_plotfiles :
                plotfile_deleted = filenames_plt[i-1]
            if delete_partfiles :
                partfile_deleted = filenames_part[i-1]
        # if the SFE did not reach the flag yet, delete the current file
        elif SFE < KEEP_FILES_AT_SFE[j_SFE] :
            if delete_plotfiles :
                plotfile_deleted = filenames_plt[i]
            if delete_partfiles :
                partfile_deleted = filenames_part[i]
        # if the SFE did reach the flag and not all flags are found,
        # keep the file and move to the next flag
        elif not SFEs_all_found :
            print(f"SFE reached at the specified value!")
            j_SFE += 1

            if j_SFE == len(KEEP_FILES_AT_SFE) :
                SFEs_all_found = True
                j_SFE = 0
        # if all flags are found, then delete the rest of the files
        else :
            if delete_plotfiles :
                plotfile_deleted = filenames_plt[i]
            if delete_partfiles :
                partfile_deleted = filenames_part[i]

        if plotfile_deleted is not None :
            os.remove(plotfile_deleted)
        if partfile_deleted is not None :
            os.remove(partfile_deleted)
        print(f"SFE:{100.0*SFE:5.2f}%, deleted: {plotfile_deleted} {partfile_deleted}")
    #endfor
    return SFE

"""
================================================================================
MAIN
================================================================================
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('original_dir', type=str,
                        help='the directory to flash exec and flash.par')
    parser.add_argument('-seeds', type=int, default=140281, nargs='*',
                        help='the random seeds for the forcing file')

    parser.add_argument('-restart', type=int, default=0,
                        help='submit the restarts')

    parser.add_argument('--first', action='store_true', default=False,
                        help=('generate the forcing file and'
                              'submit the driving phase simulation'))
    parser.add_argument('--res_grav1', action='store_true', default=False,
                        help=("create a sink simulation and"
                              "submit a short job for the re-initialisation"
                              "required for turning on the gravity"))
    parser.add_argument('--res_grav2', action='store_true', default=False,
                        help='submit a second restart for gravity simulation')
    parser.add_argument('--res_grav', action='store_true', default=False,
                        help='runs res_grav1 and res_grav2 at the same time')
    parser.add_argument('--res_gravamr', action='store_true', default=False,
                        help='submit a sink simulation for the AMR case')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='clean most of the plotfile and particle files')

    parser.add_argument('--wo_plt', action='store_true', default=False,
                        help='do not produce the plt_cnt files')
    parser.add_argument('--nofork', action='store_true', default=False,
                        help='do not fork the gravity simulations')
    parser.add_argument('-depend', type=str, default=None,
                        help='the id of the depending job')

    args = parser.parse_args()

    # get rid of the last /
    original_dir = args.original_dir
    if original_dir[-1] is "/":
        original_dir = original_dir[:-1]

    # initialisation and turbulence driving
    if args.first:
        for seed in args.seeds :
            first_sim(original_dir, seed=seed, depend=args.depend)

    # restart using prep_restart.py
    if args.restart != 0:
        restart(original_dir, args.restart, depend=args.depend, flash_name=FLASH_EXE)

    # first restart for the sink simulation
    if args.res_grav1:
        for seed in args.seeds :
            restart_grav1(original_dir, seed=seed, depend=args.depend, wo_plt=args.wo_plt)

    # second restart for the sink simulation
    if args.res_grav2:
        for seed in args.seeds :
            restart_grav2(original_dir, seed=seed, depend=args.depend)

    # submit both restarts for the sink simulation
    if args.res_grav:
        for seed in args.seeds :
            # first restart
            job_id = restart_grav1(original_dir, seed=seed, depend=args.depend,
                    wo_plt=args.wo_plt)

            # qsub the second restart (run after the first restart is finished)
            cwd = os.getcwd()
            stdout = "shell.out.rs2_{seed}"
            action = (f"cd {cwd} \nrunflash.py {original_dir}"
                      f" -seeds {seed} --res_grav2 1>{stdout} 2>&1")
            submit_job(project=PROJECT, queue=QUEUE, walltime="00:15:00",
                ncpus=2, mem="4GB", dir=cwd, action=action,
                script_name=f"job.sh.rs2_{seed}", job_name=f"submit2_{seed}",
                prev_job_id=job_id)

    # submit the restart for the AMR gravity simulation
    if args.res_gravamr:
        for seed in args.seeds :
            restart_gravamr(original_dir,
                seed=seed, depend=args.depend, fork=(not args.nofork)
                )

    # clean the files
    if args.clean :
        n_seeds = len(args.seeds)
        SFEs_max = np.zeros(n_seeds)
        for i, seed in zip(range(n_seeds), args.seeds) :
            SFEs_max[i] = clean_hdf5(original_dir, seed=seed)

        print("================================================================")
        print("cleaning finished!")
        print("{:>10s} {:>10s}".format("seed", "max SFE"))
        for SFE_max, seed in zip(SFEs_max, args.seeds) :
            print(f"{seed:10d} {100*SFE_max:9.2f}%")
