#! /usr/bin/env python3

# dependencies
import argparse

import numpy as np

import h5tools
from constants import M_SOL, C_S, G, PC_flash

class CST(object) :
    """
    container for the constants
    call with keyword arguments (e.g. M_TOT=...) for the input

    if you want to use h5tools.H5File object or a hdf5 file directly
    use CST.fromfile(filename) or CST.fromH5File(h5file)
    """
    def __init__(self, H5File=None, filename=None, verbose=True, **params_custom) :
        # initial variables
        self.M_TOT = 387.5 * M_SOL  # total mass of the system
        self.L = 2 * PC_flash       # box length
        self.MACH = 5               # rms mach number
        self.RHO = self.M_TOT / self.L**3     # mean density
        self.SFE = 0.0              # star formation efficiency

        # read the custom parameters
        if 'L' in params_custom :
            self.L = params_custom['L']
            if verbose :
                print(f"L updated to be: {self.L}")
        if 'MACH' in params_custom :
            self.MACH = params_custom['MACH']
            if verbose :
                print(f"MACH updated to be: {self.MACH}")
        if 'SFE' in params_custom :
            self.SFE = params_custom['SFE']
            if verbose :
                print(f"SFE updated to be: {self.SFE}")

        if 'RHO' in params_custom :
            self.RHO = params_custom['RHO']
            if verbose :
                print(f"RHO updated to be: {self.RHO}")
            self.M_TOT = self.RHO * self.L**3
        elif 'M_TOT' in params_custom :
            self.M_TOT = params_custom['M_TOT']
            if verbose :
                print(f"M_TOT updated to be: {self.M_TOT}")
            self.RHO = self.M_TOT / self.L**3

        # store the variables
        self.SIGMA_V = C_S * self.MACH # (3D) velocity dispersion
        self.T_TURB = self.L / (2*self.SIGMA_V) # turbulent crossing time
        self.T_FF = np.sqrt(3*np.pi/(32*G*self.RHO)) # free-fall time
        self.VIR = (5*self.SIGMA_V**2*self.L/2)/(3*G*self.M_TOT) # virial parameter

    @classmethod
    def fromH5File(cls, H5File, verbose=True) :
        if verbose :
            print("using the H5File instance for the constants...")
        # read the parameters from the h5file
        L = H5File.params['L'][0]
        RHO = H5File.params['rho_ambient']

        # if particle file is read, then calculate the SFE
        if isinstance(H5File, h5tools.PartFile) :
            if H5File.particles is None :
                SFE = 0.0
            else :
                SFE = np.sum(H5File.particles['mass']) / M_TOT
        else :
            SFE = 0.0

        return cls(L=L, RHO=RHO, verbose=verbose, SFE=SFE)

    @classmethod
    def fromfile(cls, filename, verbose=True) :
        if verbose :
            print(f"reading {filename}...")

        # create a H5File instance if a filename is given
        try :
            H5File = h5tools.H5File(filename, verbose=verbose)
            if verbose :
                print("created a H5File instance! passing it...")
        except KeyError :
            H5File = h5tools.PartFile(filename, verbose=verbose)
            if verbose :
                print("created a PartFile instance! passing it...")

        # call the fromH5File init function
        return cls.fromH5File(H5File, verbose=verbose)

    def show(self) :
        """
        function to print the calculated constants to stdout
        """
        print(f"Turbulent crossing time: {self.T_TURB:.4E}")
        print(f"free-fall time         : {self.T_FF:.4E}")
        print(f"mean density           : {self.RHO:.4E}")
        print(f"virial parameter       : {self.VIR:.4f}")
        print(f"SFE                    : {100.0*self.SFE:.2f}%")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', type=str, nargs='?', default=None,
                        help='the name to the file to be inspected')
    args = parser.parse_args()

    # create the instance
    print("initialising constants.py...")
    cst = CST(filename=args.filename)
    print("...completed!")

    # print the constants and exit
    cst.show()

    print("exiting constants.py...")
