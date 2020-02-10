#!/usr/bin/env python3

# DEPENDENCIES
import numpy as np
import glob


def merge_dicts(*dict_args) :
    """
    DESCRIPTION
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args :
        result.update(dictionary)
    return result

def get_file(path, loc='last') :
    """
    DESCRIPTION
        From the specified path (with wildcards),
        return the filename at the specific lococation (first, last, or given index)
    """
    files = glob.glob(path)
    files.sort()
    if files == [] :
        return None
    elif loc == 'last' :
        return files[-1]
    elif loc == 'first' :
        return files[0]
    elif isinstance(loc, int) :
        try :
            return files[loc]
        except IndexError :
            print("loc out of range!")
            return None
    else :
        print("loc not valid!")
        return None
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
