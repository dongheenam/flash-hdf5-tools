#! /usr/bin/env python3

# dependencies
import argparse

import numpy as np

import h5tools

class CST(object) :
    """
    container for the constants
    one may calculate from the parameters passed on as a dictionary
    or a h5tools.H5File object
    """
    def __init__(self, H5File=None, filename=None, **params_custom) :
        # initial variables
        M_SOL = 1.98847542e33 # solar mass
        M_TOT = 387.5 * M_SOL # total mass of the system
        L = 2 * 3.086e18      # box length
        C_S = 0.2e5           # isothermal sound speed
        MACH = 5              # rms mach number
        RHO = M_TOT / L**3    # mean density

        G = 6.67408e-8        # gravitational constant

        # read the custom parameters
        if 'L' in params_custom :
            L = params_custom['L']
            print(f"L updated from kwargs to be: {L}")
        if 'C_S' in params_custom :
            C_S = params_custom['C_S']
            print(f"C_S updated from kwargs to be: {C_S}")
        if 'MACH' in params_custom :
            MACH = params_custom['MACH']
            print(f"MACH updated from kwargs to be: {MACH}")

        if 'RHO' in params_custom :
            RHO = params_custom['RHO']
            print(f"RHO updated from kwargs to be: {RHO}")
            M_TOT = RHO * L**3
        elif 'M_TOT' in params_custom :
            M_TOT = params_custom['M_TOT']
            print(f"M_TOT updated from kwargs to be: {M_TOT}")
            RHO = M_TOT / L**3

        # create a H5File instance if a filename is given
        if filename is not None :
            try :
                H5File = h5tools.H5File(filename)
            except KeyError :
                H5File = h5tools.PartFile(filename)

        # read from the H5File instance if it exists
        if H5File is not None :
            print("using the H5File instance for the constants...")
            L = H5File.params['L'][0]
            RHO = H5File.params['rho_ambient']
            M_TOT = L**3 * RHO
            print(f"M_TOT updated from H5File to be: {M_TOT}")
            print(f"L updated from H5File to be: {L}")

        # store the variables
        self.M_SOL = M_SOL
        self.M_TOT = M_TOT
        self.L = L
        self.C_S = C_S
        self.MACH = MACH
        self.RHO = RHO

        self.SIGMA_V = C_S * MACH
        self.T_TURB = L / (2*self.SIGMA_V) # turbulent crossing time
        self.T_FF = np.sqrt(3*np.pi/(32*G*self.RHO)) # free-fall time
        self.VIR = (5*self.SIGMA_V**2*L/2)/(3*G*M_TOT)

    def show(self) :
        print(f"Turbulent crossing time: {self.T_TURB:.4E}")
        print(f"free-fall time         : {self.T_FF:.4E}")
        print(f"mean density           : {self.RHO:.4E}")
        print(f"virial parameter       : {self.VIR:.4f}")

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
