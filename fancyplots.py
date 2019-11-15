#!/usr/bin/env python3

# dependencies
import argparse
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import h5tools
import mpltools

# constants
class CST(object) :
    def __init__(self) :
        self.M_SOL = 1.989e33              # solar mass
        self.M_TOT = 2 * 775 * self.M_SOL   # total mass

        self.L = 2 * 3.086e18     # box length
        self.MACH = 5.0           # rms mach number
        self.C_S = 0.2e5          # isothermal sound speed
        self.T_TURB = self.L / (2*self.MACH*self.C_S) # turbulent crossing time

        self.RHO = self.M_TOT / (self.L)**3       # mean density
        self.T_FF = np.sqrt(3*np.pi/(32*6.674e-8*self.RHO)) # free-fall time

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filename', type=str,
                        help='the name to the file to be loaded')

    parser.add_argument('--imf', action='store_true', default=False,
                        help='Set this flag to plot the IMF from the particle file')

    parser.add_argument('-o', type=str,
                        help='the output path')
    parser.add_argument('-ext', type=str,
                        help='the extension for the output (ignored if -o is set up)')

    args = parser.parse_args()

    # container for the constants
    cst = CST()

    if args.imf is True :
        print(f"plotting the imf for {args.filename}...")
        # load the particle masses
        pf = h5tools.PartFile(args.filename)
        if pf.particles is not None :
            masses_in_msol = pf.particles['mass'] / cst.M_SOL
            part_exists = True
        else :
            masses_in_msol = []
            part_exists = False

        # set up the bins
        m_min = 0.5         # in solar masses
        m_max = 100
        n_bins = 20
        logbins = np.logspace(np.log10(m_min),np.log10(m_max),n_bins)

        # annotation of the plot
        time_in_T = pf.params['time'] / cst.T_TURB
        time_in_Tff = pf.params['time'] / cst.T_FF
        SFE = 100.0* np.sum(pf.particles['mass']) / cst.M_TOT if part_exists else 0.0
        N_sink = len(pf.particles) if part_exists else 0
        annotation = ( rf'\begin{{align*}}'
                       rf'&t={time_in_T:.1f}T={time_in_Tff:.1f}T_\mathrm{{ff}}\\ '
                       rf'&\mathrm{{SFE}}={SFE:.1f}\%\\ '
                       rf'&N_\mathrm{{sink}}={N_sink} \end{{align*}}' )
        # plot the IMF
        mpltools.mpl_init()
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)
        mpltools.plot_hist(masses_in_msol, ax=ax, log=True, yrange=(0.5, 10),
            annotation=annotation, title=r'$\beta=1, \rho=2\rho_0, N=1024^3$',
            xlabel=r'$M\,[M_\odot]$', ylabel=r'$\dfrac{dN}{d\log{M}}$',
            bins=logbins, histtype='step', color='black', linewidth=2)

        # export the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = args.filename + '_imf.' + args.ext
        else :
            filename_out = args.filename + '_imf.png'
        plt.savefig(filename_out)
        print("IMF printed to: "+filename_out)
