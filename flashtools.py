#! /usr/bin/env python3

# dependencies
import subprocess
import argparse
import os
import tempfile

def replace_value(path_to_file, begin_at=0, **kwargs) :
    """
    DESCRIPTION
        finds the parameters in kwargs and replace its value as specified in kwargs
    """

    # open files
    temp_file = tempfile.TemporaryFile(mode='w+t')
    existing_file = open(path_to_file, 'r')

    line_no = 0
    for line in existing_file :

        # if we haven't reached the specified line number yet
        if line_no < begin_at :
            temp_file.writelines(line)

        else :
            # check whether the parameters exist
            for keys, values in kwargs.items() :

        # endif (replacement block)
        line_no += 1

    # endfor line in existing_file
