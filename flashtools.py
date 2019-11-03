#! /usr/bin/env python3

# dependencies
import argparse
import os
import shutil
import subprocess
import tempfile

def replace_file(path_to_file, begin_at=0, comment_symbol='#', **kwargs) :
    """
    DESCRIPTION
        finds the parameters in kwargs and replace its value as specified in kwargs
    """

    # open files
    print("opening {} for replacing...".format(path_to_file))
    temp_file = tempfile.NamedTemporaryFile(mode='w+t')
    existing_file = open(path_to_file, 'r')

    line_no = 1
    for line in existing_file :

        # replacement begins only when we arrive at the specified line number
        if line_no >= begin_at :
            # check whether the parameters exist
            for key, value in kwargs.items() :
                # if the key is found in the line
                if line.strip().startswith(key) :
                    print("(line {}) found        : {}".format(line_no, line))
                    i_start = line.find("=")            # replace from one space after the equal sign
                    i_end = line.find(comment_symbol)   # until the comment starts
                    line = line[:i_start+2] + "{0:<{1}}".format(value, i_end-i_start-2) + line[i_end:]
                    print("(line {}) replaced with: {}".format(line_no, line))
                    break
            # endfor kwargs.items()
        # endif line_no >= begin_at

        temp_file.writelines(line)
        line_no += 1
    # endfor existing_file

    # replace the file
    temp_file.seek(0)
    shutil.copy2(temp_file.name, path_to_file)
    temp_file.close()
    os.chmod(path_to_file, 0o644)
    print("finished writing!")
    print("replaced {}!".format(path_to_file))
