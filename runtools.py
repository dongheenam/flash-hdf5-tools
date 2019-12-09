#! /usr/bin/env python3

# DEPENDENCIES
import argparse
import numpy as np

import h5tools

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filename', type=str,
                        help='the name to the file to be loaded')
    parser.add_argument('fieldname', type=str, nargs='?', default=None,
                        help='name the field that will be analysed')

    parser.add_argument('--ps1d', action='store_true', default=False,
                        help='Set this flag to calculate the 1D power spectrum')

    parser.add_argument('-o', type=str, default=None,
                        help='the output path')

    args = parser.parse_args()


    # calculate the 1D power spectrum
    if args.ps1d is True:
        # load the file and create a dataset
        a = h5tools.H5File(args.filename)
        ds = a.new_dataset(args.fieldname, small_mem=True)

        # calculate the power spectrum
        ds.calc_ps(save=True, filename_out=args.o)
