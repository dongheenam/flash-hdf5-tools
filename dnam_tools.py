#!/usr/bin/env python3

# DEPENDENCIES
import numpy as np

"""
FUNCTION merged_dicts = merge_dicts(*dict_args)

DESCRIPTION
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
"""
def merge_dicts(*dict_args) :
    result = {}
    for dictionary in dict_args :
        result.update(dictionary)
    return result

def read_dat(path) :
    result = np.genfromtxt(path, names=True)
    return result

"""
================================================================================
Predefined macros
================================================================================
"""
if __name__== "__main__":
    # DEPENDENCIES
    import argparse, sys
    import numpy as np

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('path', type=str, nargs='+',
                        help='the path to the file to be loaded')

    parser.add_argument('--mean', action='store_true', default=False,
                        help='Calculate the mean of the files')

    parser.add_argument('-o', type=str,
                        help='the output file name')
    args = parser.parse_args()

    if args.mean == True:
        list_data = []
        dtype = np.genfromtxt(args.path[0],names=True).dtype
        for path in args.path :
            list_data.append(np.genfromtxt(path, names=True))

        average_data = np.zeros(list_data[0].shape, dtype=dtype)
        for i in range(len(average_data)) :
            average_data[i] = np.sum([list(data[i]) for data in list_data], axis=0) / len(list_data)

        list_name = average_data.dtype.names

        if args.o == None :
            path_to_save = args.path[0]+'_average'
        else :
            path_to_save = args.o

        dat_out = open(path_to_save, mode='w')

        col_length = len(list_name)
        row_length = len(average_data)

        dat_out.write(("{:>20s}"*col_length+'\n').format(*tuple(list_name)))
        for i in range(row_length) :
            tuple_row = tuple( average_data[i] )
            dat_out.write(("{:20.11E}"*col_length+'\n').format(*tuple_row))

        dat_out.close()
