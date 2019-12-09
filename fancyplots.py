#!/usr/bin/env python3

# dependencies
import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
import h5py

from constants import M_SOL, PC_flash
import measures
import h5tools
import mpltools

""" CONSTANTS - FLASH """


""" CONSTNATS - IMF """
BUILD_IMF_AT_SFE = 0.10
M_MIN = 0.01
M_MAX = 100.0
N_BINS = 30
M_SHIFT_FACTOR = 1

PLOT_TITLE = r'$\beta=1, \alpha_\mathrm{vir}=0.5, N=512^3$'
IMF_TEXT = rf'$\mathrm{{SFE}} = {100*BUILD_IMF_AT_SFE:.0f}\%$'

"""
================================================================================
Macro Funcitons
================================================================================
"""
def proj_mpi(filename, filename_out, save=True, ax=plt.gca(), **kwargs) :
    print(f"plotting the projected field from {filename}...")
    h5f = h5py.File(filename, 'r')

    filename_wo_ext = filename.split(".")[0]
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
    xyz_lim = np.array(h5f['minmax_xyz']) / PC

    if proj_axis == 'x' :
        xlabel, ylabel = (r'$y \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
        xrange, yrange = (xyz_lim[1], xyz_lim[2])
        if part_exists :
            part_xlocs = pf.particles['posy'] / PC_flash
            part_ylocs = pf.particles['posz'] / PC_flash
    elif proj_axis == 'y' :
        xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
        xrange, yrange = (xyz_lim[0], xyz_lim[2])
        if part_exists :
            part_xlocs = pf.particles['posx'] / PC_flash
            part_ylocs = pf.particles['posz'] / PC_flash
    elif proj_axis == 'z' :
        xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$y \,[\mathrm{pc}]$')
        xrange, yrange = (xyz_lim[0], xyz_lim[1])
        if part_exists :
            part_xlocs = pf.particles['posx'] / PC_flash
            part_ylocs = pf.particles['posy'] / PC_flash

    if not part_exists :
        part_xlocs = []
        part_ylocs = []

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
        annotation=annotation, log=True, colorbar=True, **kwargs)

    mpltools.plot_scatter(part_ylocs, part_xlocs, ax=ax,
        xrange=xrange, yrange=yrange,
        overplot=True, marker=r'$\odot$', s=80, color='limegreen')

    if save :
        fig.savefig(filename_out)
        print(f"plot saved to: {filename_out}")
#enddef proj_mpi

def imfs(folders_regexp, filename_out, save=True, ax=plt.gca(), **kwargs) :
    # find all the folders matching the regular expression
    print(f"received the regexp {folders_regexp}...")
    folders = glob.glob(folders_regexp)
    folders.sort()
    print(f"folders found: {folders}")
    if len(folders) == 0 :
        sys.exit("no folders matching the expression! exitting...")

    # find the particle files
    part_masses = np.array([], dtype='f8')
    print(f"finding the particle files with SFE={BUILD_IMF_AT_SFE}...")
    for folder in folders :
        print(f"inspecting {folder}...")
        filenames_part = glob.glob(
            os.path.join(folder, "Turb_hdf5_part_????"))
        filenames_part.sort()
        n_files = len(filenames_part)
        if n_files == 0 :
            print("no particle files! moving on...")
            continue
        else :
            print(f"{n_files} particle files found!")

        i = 0
        while True :
            filename_part = filenames_part[i]
            cst = measures.CST(filename=filename_part, verbose=False)
            SFE = cst.SFE

            print(f"{filename_part} read! SFE = {100*SFE:.2f}%...")
            if SFE < BUILD_IMF_AT_SFE :
                i += 1
                if i == len(filenames_part) :
                    print("this simulation did not reach the specified SFE!")
                    break
                continue
            else :
                pf = h5tools.PartFile(filename_part, verbose=False)
                print(f"appending {len(pf.particles)} sinks to the IMF...")
                part_masses = np.append(part_masses, pf.particles['mass'])
                print(f"number of sinks so far : {len(part_masses)}")
                break
        # endfor filenames_part
    # endfor folders

    # in terms of solar masses
    masses_in_msol = part_masses / M_SOL

    # set up the bins
    m_min = M_MIN *M_SHIFT_FACTOR
    m_max = M_MAX *M_SHIFT_FACTOR
    n_bins = N_BINS
    logbins = np.logspace(np.log10(m_min),np.log10(m_max),n_bins)

    # annotation
    #annotation = ( rf'\begin{{align*}}'
    #               rf'&\mathrm{{SFE}}={BUILD_IMF_AT_SFE:.0f}\%\\ '
    #               rf'&N_\mathrm{{sink}}={len(masses_in_msol)} \end{{align*}}' )
    annotation = f"sinks:{len(masses_in_msol)}, SFE={100*BUILD_IMF_AT_SFE:.0f}%"

    # plot the IMF
    print("plotting...")
    mpltools.plot_hist(masses_in_msol, ax=ax,
        annotation=annotation, bins=logbins, log=True,
        xrange=(m_min,m_max), yrange=(0.5, 50),
        xlabel=r'$M\,[M_\odot]$', ylabel=r'$\dfrac{dN}{d\log{M}}$',
        **kwargs)
    # salpeter slope
    ax.plot(logbins, 3*(logbins/m_max)**(-1.35), color='red', linestyle='--')
    print("plotting complete!")

    # export the plot
    if save :
        plt.savefig(filename_out)
        print("IMF printed to: "+filename_out)
#enddef imfs

def machs() :
    pass
#enddef machs

def sfrs(folders_regexp, filename_out, save=True, ax=plt.gca(), **kwargs) :
    # find all the folders matching the regular expression
    print(f"received the regexp {folders_regexp}...")
    folders = glob.glob(folders_regexp)
    folders.sort()
    print(f"folders found: {folders}")
    if len(folders) == 0 :
        sys.exit("no folders matching the expression! exitting...")

    overplot = True
    SFRs_at_the_SFE = np.array([],dtype='f8')
    for folder in folders :
        try :
            dat = h5tools.DatFile(os.path.join(folder, 'Turb.dat'))
        except OSError :
            print(f"dat file does not exist in {folder}. skipping...")
            continue

        filenames_plt = os.path.join(folder, 'Turb_hdf5_plt_cnt_????')
        try :
            cst = measures.CST(filename=glob.glob(filenames_plt)[0])
        except IndexError :
            print(f"plotfile does not exist in {folder}. skipping...")
            continue

        # calculate the star formation efficiency and rate
        SFEs = (cst.M_TOT - dat.data['mass'])/cst.M_TOT
        time_in_Tff = dat.data['time']/cst.T_FF
        SFR_in_Tff = np.diff(SFEs) / np.diff(time_in_Tff)

        # record the SFR at the specified SFE
        i_SFR = np.searchsorted(SFEs, BUILD_IMF_AT_SFE, sorter=None)
        if SFEs[-1] > BUILD_IMF_AT_SFE  :
            SFRs_at_the_SFE = np.append(SFRs_at_the_SFE, SFR_in_Tff[i_SFR])

        # smooth the function
        SFR_in_Tff = gaussian_filter1d(SFR_in_Tff, 20)
        #SFR_in_Tff = median_filter(SFR_in_Tff, size=100)
        #SFR_in_Tff = savgol_filter(SFR_in_Tff, 11, 2)

        # plot SFR vs SFE
        mpltools.plot_1D(SFEs[1:]*100.0, SFR_in_Tff,
            ax=ax, overplot=overplot, xrange=(0.0, 20.0), yrange=(0.0, 1.00),
            xlabel='SFE [%]', ylabel='SFR_ff', alpha=0.2, linewidth=2)
        # plot SFR vs t_FF
        #mpltools.plot_1D(time_in_Tff[1:], SFR_in_Tff,
        #    ax=ax, overplot=overplot, xrange=(None, None), yrange=(0.0, 1.00),
        #    xlabel='t [T_ff]', ylabel='SFR_ff', title=PLOT_TITLE, alpha=0.2, linewidth=2)

        if overplot is True :
            overplot = False

    if save :
        plt.savefig(filename_out)
        print("plot printed to: "+filename_out)

    # the distribution of SFR
    fig = plt.figure()
    ax = fig.add_subplot()
    mpltools.plot_hist(SFRs_at_the_SFE, ax=ax, xlabel='SFR_ff', ylabel='N_sim',
                       color='black', annotation=IMF_TEXT, title=PLOT_TITLE)
    if save :
        plt.savefig("SFRs.pdf")
#enddef sfrs

"""
================================================================================
MAIN
================================================================================
"""
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='the name to the file(s) to be loaded')

    parser.add_argument('--proj_mpi', action='store_true', default=False,
                        help='read the hdf5 file from projeciton_mpi and create a plot')
    parser.add_argument('--imf', action='store_true', default=False,
                        help='Set this flag to plot the IMF from the particle file')
    parser.add_argument('--imfs', action='store_true', default=False,
                        help='Build IMF from all folders that match the regexp')
    parser.add_argument('--machs', action='store_true', default=False,
                        help='Plot Mach numbers from all folders that match the regexp')
    parser.add_argument('--sfrs', action='store_true', default=False,
                        help='Plot SFR from all folders that match the regexp')

    parser.add_argument('-o', type=str,
                        help='the output path')
    parser.add_argument('-ext', type=str,
                        help='the extension for the output (ignored if -o is set up)')

    args = parser.parse_args()
    if len(args.filenames) == 1 :
        filename = args.filenames[0]
    else :
        filenames = args.filenames

    # setting up the plotting environment
    print("initiating the plotting sequence...")
    mpltools.mpl_init()
    matplotlib.rcParams['text.usetex'] = False
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)

    if args.proj_mpi :
        # the file name of the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename_wo_ext + '.' + args.ext
        else :
            filename_out = filename_wo_ext + '.png'

        # plot the projection
        proj_mpi(filename, filename_out,
            ax=ax, title=PLOT_TITLE,
            colorbar_title=r"Column Density $[\mathrm{g}\,\mathrm{cm}^{-2}]$",
            color_range=(0.01,0.5))

    if args.imfs :
        # the file name of the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename + '_imf.' + args.ext
        else :
            filename_out = filename + '_imf.png'

        # plot the imf
        imfs(filename, filename_out,
            ax=ax, title=PLOT_TITLE, histtype='step', color='black', linewidth=2)

    if args.machs :
        pass

    if args.sfrs :
        # the file name of the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename + '_sfr.' + args.ext
        else :
            filename_out = filename + '_sfr.png'

        # plot the sfrs
        sfrs(filename, filename_out,
             ax=ax, title=PLOT_TITLE)
