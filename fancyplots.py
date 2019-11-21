#!/usr/bin/env python3

# dependencies
import argparse
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

import h5tools
import mpltools
from constants import *

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='the name to the file(s) to be loaded')

    parser.add_argument('--proj_mpi', action='store_true', default=False,
                        help='read the hdf5 file from projeciton_mpi and create a plot')
    parser.add_argument('--imf', action='store_true', default=False,
                        help='Set this flag to plot the IMF from the particle file')

    parser.add_argument('-o', type=str,
                        help='the output path')
    parser.add_argument('-ext', type=str,
                        help='the extension for the output (ignored if -o is set up)')

    args = parser.parse_args()
    if len(args.filenames) == 1 :
        args.filename = args.filenames[0]

    if args.proj_mpi :
        print(f"plotting the projected field from {args.filename}...")
        h5f = h5py.File(args.filename, 'r')

        filename_wo_ext = args.filename.split(".")[0]
        proj_axis = filename_wo_ext[-1]
        proj_title = "_".join(filename_wo_ext.split("_")[-3:-1])
        filename_plt = "_".join(filename_wo_ext.split("_")[:-3])

        # find for a particle file assiciated with the proj_mpi result
        i = filename_wo_ext.find("plt_cnt")
        filename_part = filename_wo_ext[:i]+'part'+filename_wo_ext[i+7:i+12]
        try :
            pf = h5tools.PartFile(filename_part)
            part_exists = False if pf.particles is None else True

            # container for the constants
            cst = CST(H5File=pf)
        except OSError :
            print(f"particle file {part_filename} not found...")
            part_exists = False

            # container for the constants
            cst = CST(filename=filename_plt)

        dens_proj = h5f[proj_title]
        xyz_lim = np.array(h5f['minmax_xyz']) / 3.086e18

        if proj_axis == 'x' :
            xlabel, ylabel = (r'$y \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
            xrange, yrange = (xyz_lim[1], xyz_lim[2])
            if part_exists :
                part_xlocs = pf.particles['posy'] / 3.086e18
                part_ylocs = pf.particles['posz'] / 3.086e18
        elif proj_axis == 'y' :
            xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
            xrange, yrange = (xyz_lim[0], xyz_lim[2])
            if part_exists :
                part_xlocs = pf.particles['posx'] / 3.086e18
                part_ylocs = pf.particles['posz'] / 3.086e18
        elif proj_axis == 'z' :
            xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$y \,[\mathrm{pc}]$')
            xrange, yrange = (xyz_lim[0], xyz_lim[1])
            if part_exists :
                part_xlocs = pf.particles['posx'] / 3.086e18
                part_ylocs = pf.particles['posy'] / 3.086e18

        if not part_exists :
            part_xlocs = []
            part_ylocs = []

        # plotting
        print("plotting...")
        mpltools.mpl_init()
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)

        time_in_T = pf.params['time'] / cst.T_TURB
        time_in_Tff = pf.params['time'] / cst.T_FF
        SFE = 100.0* np.sum(pf.particles['mass']) / cst.M_TOT if part_exists else 0.0
        N_sink = len(pf.particles) if part_exists else 0

        annotation = ( rf'\begin{{align*}}'
                       rf'&t={time_in_T:.1f}T={time_in_Tff:.1f}T_\mathrm{{ff}}\\ '
                       rf'&\mathrm{{SFE}}={SFE:.1f}\%\\ '
                       rf'&N_\mathrm{{sink}}={N_sink} \end{{align*}}' )
        mpltools.plot_proj(dens_proj, ax=ax,
            xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel,
            title=r'$\beta=1, \rho=\rho_0, N=1024^3$', annotation=annotation,
            colorbar_title=r"Column Density $[\mathrm{g}\,\mathrm{cm}^{-2}]$",
            colorbar=True, log=True, color_range=(0.01,0.5))

        mpltools.plot_scatter(part_ylocs, part_xlocs, ax=ax,
            xrange=xrange, yrange=yrange,
            overplot=True, marker=r'$\odot$', s=80, color='limegreen')

        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename_wo_ext + '.' + args.ext
        else :
            filename_out = filename_wo_ext + '.png'

        fig.savefig(filename_out)
        print(f"plot saved to: {filename_out}")
    #endif args.proj_mpi

    if args.imf :
        print(f"plotting the imf for {args.filename}...")
        # container for the constants
        cst = CST(filename=args.filename)

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
