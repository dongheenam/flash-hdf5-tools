#! /usr/bin/env python3

# dependencies
import argparse
import os, sys
import shutil
import subprocess
import tempfile

""" PARAMETERS - forcing generator """
forcing_generator_path = "/short/ek9/dn8909/forcing_generator/"

st_power_law_exp = "-2.0"
velocity         = "1.0e5"
st_energy        = "0.9e-2 * velocity**3/L"
st_stirmax       = "(256+eps) * twopi/L"

""" PARAMETERS - FLASH """
flash_path             = "/short/ek9/dn8909/flash4.0.1-rsaa_fork/"
output_path            = "/short/ek9/dn8909/data/"

""" PARAMETERS - QSUB """
project = "jh2"
queue = "express"
walltime = "04:00:00"
ncpus = 512
mem = "{}GB".format(ncpus*2)


def replace_file(filename, begin_at=0, comment_symbol='#', **kwargs) :
    """
    DESCRIPTION
        finds the parameters in kwargs and replace its value as specified in kwargs
    """

    # open files
    print("opening {} for replacing...".format(filename))
    temp_file = tempfile.NamedTemporaryFile(mode='w+t')
    existing_file = open(filename, 'r')

    line_no = 1
    for line in existing_file :

        # replacement begins only when we arrive at the specified line number
        if line_no >= begin_at :
            # check whether the parameters exist
            for key, value in kwargs.items() :
                # if the key is found in the line
                if line.strip().startswith(key) :
                    print("(line {}) found: {}".format(line_no, line[:-2]))
                    i_start = line.find("=")            # replace from one space after the equal sign
                    i_end = line.find(comment_symbol)   # until the comment starts (if found)
                    if i_end is -1 :
                        line = line[:i_start+1] + " {}".format(value) + "\n"
                    else :
                        line = line[:i_start+1] + " {0:<{1}}".format(value, i_end-i_start-2) + line[i_end:]
                    print("(line {}) =>     {}".format(line_no, line[:-2]))
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
    print("replaced {}!".format(filename))

def generate_forcingfile(seed=140281) :
    """
    DESCRIPTION
        runs the forcing generator with the given random seed number
        the other default parameters are defined as global variables (see top)
    """
    print("making the forcingfile with seed : {}...".format(seed))
    # update the Fortran code
    replace_file(forcing_generator_path+"forcing_generator.F90", begin_at=623, comment_symbol='!',
                 st_power_law_exp = st_power_law_exp,
                 velocity         = velocity,
                 st_energy        = st_energy,
                 st_stirmax       = st_stirmax,
                 st_seed          = seed )

    # make the forcing generator and run
    print("making and running the forcing generator...")
    cp = subprocess.run(["make", "-C", forcing_generator_path], capture_output=True, check=True)
    cp = subprocess.run([forcing_generator_path+"forcing_generator"], capture_output=True, check=True, text=True)

    # the last line of the stdout contains the name of the forcing file
    last_stdout = cp.stdout.split("\n")[-2]

    forcingfile_name = last_stdout[last_stdout.find('"')+1:last_stdout.rfind('"')]
    print("forcing file created: {}!".format(forcingfile_name))
    return forcingfile_name

def submit_job(project=project, queue=queue, walltime=walltime, ncpus=ncpus, mem=mem, dir="",
               action="", script_name="job.sh", job_name="test", previous_job=None) :
    """
    DESCRIPTION
        writes a pbs script and submit the job and returns the job name
    """

    print("writing jobscript to {}...".format(dir+script_name))
    job = open(dir+script_name, 'w')

    job.write("#!/bin/bash \n")
    job.write("#PBS -P {} \n".format(project))
    job.write("#PBS -q {} \n".format(queue))
    job.write("#PBS -l walltime={} \n".format(walltime))
    job.write("#PBS -l ncpus={} \n".format(ncpus))
    job.write("#PBS -l mem={} \n".format(mem))
    job.write("#PBS -l wd \n")
    job.write("#PBS -N {} \n".format(job_name))
    job.write("#PBS -j oe \n")
    if previous_job is not None:
        job.write("#PBS -W depend=afterany:{} \n".format(previous_job))

    for line in action.split("\n") :
        job.write(line+"\n")
    job.close()

    print("submitting...")
    cp = subprocess.run(["qsub", dir+script_name], capture_output=True, check=True, text=True)
    job_id = cp.stdout.split()[-1]

    print("completed! job id: {}".format(job_id))
    return job_id

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('original_flash_dir', type=str,
                        help='the directory containing the flash file and flash.par')
    parser.add_argument('-seed', type=int, default=140281,
                        help='the random seed for the forcing file')

    parser.add_argument('--begin', action='store_true', default=False,
                        help='generate the forcing file and submit the driving phase simulation')
    parser.add_argument('--restart', action='store_true', default=False,
                        help='prepare for and submit gravity simulations')
    args = parser.parse_args()

    # check whether the flash dir exists
    original_flash_dir = args.original_flash_dir
    flash4_is_here = os.path.exists(original_flash_dir+"/flash4")
    flash_par_is_here = os.path.exists(original_flash_dir+"/flash.par")
    if (not flash4_is_here) or (not flash_par_is_here) :
        sys.exit("flash files not found! check the directory again.")

    # initialisation and turbulence driving
    if args.begin :
        # generate the forcing file
        forcingfile_name = generate_forcingfile(args.seed)

        # create the directory and copy the flash files
        new_flash_dir = original_flash_dir+"_{:06d}/".format(args.seed)
        os.mkdir(new_flash_dir)
        shutil.copy2(original_flash_dir+"/flash4", new_flash_dir)
        shutil.copy2(original_flash_dir+"/flash.par", new_flash_dir)

        # modify the flash.par file
        replace_file(new_flash_dir+"/flash.par",
                     st_infilename = '"../'+forcingfile_name+'"',
                     useGravity    = '.false.',
                     useParticles  = '.false.',
                     restart       = '.false.',
                     checkpointFileNumber    = "0",
                     plotFileNumber          = "0",
                     particleFileNumber      = "0",
                     plotFileIntervalTime    = "3.086e12",
                     particleFileIntervalTime= "3.086e12",
                     tmax                    = "6.171e13",
                     usePolytrope            = ".false."
                     )

        # submit the simulation
        action = "cd {dir} \nmpirun flash4 1>{stdout} 2>&1".format(dir=new_flash_dir, stdout="shell.out")

        submit_job(project=project, queue=queue, walltime=walltime, ncpus=ncpus, mem=mem, dir=new_flash_dir,
                   action=action, script_name="job.sh", job_name="test", previous_job=None)
    #endif args.begin
