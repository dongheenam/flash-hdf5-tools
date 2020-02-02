#! /usr/bin/env python3

# DEPENDENCIES
import argparse
import glob
import os
import sys

import h5py
import numpy as np

import measures
import h5tools

# CONSTANTS

PARTFILE_NAME = "Turb_hdf5_part_????"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filename', type=str,
                        help='the name to the file to be loaded')

    parser.add_argument('-ps1d', type=str,
                        help='Calculate the 1D power spectrum (of the specified field)')
    parser.add_argument('-part_at_sfe', type=float,
                        help='Collect all particle data at speficied SFE')

    parser.add_argument('-o', type=str, default=None,
                        help='the outfile name')
    parser.add_argument('--reg', action='store_true', default=False,
                        help='Use regexp instead of a single filename')


    args = parser.parse_args()

    # check if the user put regular expression
    # and find all files/folders that match
    if args.reg is True :
        print(f"received the regexp {args.filename}...")
        filenames = glob.glob(args.filename)
        filenames.sort()

        if len(filenames) == 0 :
            sys.exit("no files that match the regexp!")
        print(f"files found: {filenames}")
    else :
        filenames = [args.filename]

    # calculate the 1D power spectrum
    if args.ps1d is not None :
        # load the file and create a dataset
        for filename in filenames :
            a = h5tools.H5File(filename)
            ds = a.new_dataset(args.ps1d, small_mem=True)

            # calculate the power spectrum
            ds.calc_ps(save=True, filename_out=args.o)
        sys.exit("completed! exiting runtools.py...")

    # collect all particle data at a given SFE
    if args.part_at_sfe is not None :
        target_sfe = args.part_at_sfe

        # exit the program if the target SFE is not in the range 0 - 100%
        if (target_sfe<=0.0 or target_sfe>=1.0) :
            sys.exit("SFE invalid!")

        # create an empty array to collect all particles
        part_masses = np.array([], dtype='f8')
        print(f"finding the particle files with SFE={target_sfe}...")

        # loop over the folders
        folders = filenames
        for folder in folders :
            print(f"searching {folder}...")

            # find all particle files
            filenames_part = glob.glob(
                os.path.join(folder, "Turb_hdf5_part_????"))
            filenames_part.sort()

            # if there are no particle files, then skip
            n_files = len(filenames_part)
            if n_files == 0 :
                print("no particle files! moving on...")
                continue
            else :
                print(f"{n_files} particle files found!")

            # see if the cache exists
            try :
                filename_cache = os.path.join(folder, f".part_at_{target_sfe}")
                cachefile = open(filename_cache,'r')
                print(f"cache file {filename_cache} exists!")

                # use the content of the cache to add the particle file
                filename_part = os.path.join(folder,cachefile.read())
                pf = h5tools.PartFile(filename_part, verbose=False)
                cst = measures.CST.fromH5File(pf, verbose=True)

                print(f"\nappending {len(pf.particles)} sinks to the IMF...")
                part_masses = np.append(part_masses, pf.particles['mass'])
                print(f"number of sinks so far : {len(part_masses)}")
                continue

            # proceed if the cache does not exist
            except FileNotFoundError :
                print("cache file not found...")

            # search all particle files to find the partfile with SFE given
            i = 0
            while True :
                filename_part = filenames_part[i]
                cst = measures.CST.fromfile(filename_part, verbose=False)
                SFE = cst.SFE

                print(f"{filename_part} read! SFE = {100*SFE:.2f}%...", end='\r')
                if SFE < target_sfe :
                    i += 1
                    if i == n_files :
                        print("\nthis simulation did not reach the specified SFE!")
                        break
                    continue
                # if the particle file has reached the desired SFE
                else :
                    # append the particle info
                    pf = h5tools.PartFile(filename_part, verbose=False)
                    print(f"\nappending {len(pf.particles)} sinks to the IMF...")
                    part_masses = np.append(part_masses, pf.particles['mass'])
                    print(f"number of sinks so far : {len(part_masses)}")

                    # cache the partfile name for later access
                    filename_cache = os.path.join(folder, f".part_at_{target_sfe}")
                    cachefile = open(filename_cache,'w+')
                    cachefile.write(os.path.split(filename_part)[-1])
                    cachefile.close()
                    break
            # endfor filenames_part
        #endfor folders

        # create a HDF5 flie for storage
        if args.o is not None :
            filename_out = args.o
        else :
            filename_out = f"{folders[0]}_imf{target_sfe}.h5"

        # store the mass
        h5f = h5py.File(filename_out,'w')
        h5f.create_dataset("mass", data=part_masses)
        print(f"saved the particle mass data to: {filename_out}")
