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

import bayestools
from constants import M_SOL, PC_flash
import measures
import h5tools
import mpltools

""" CONSTANTS - FLASH """


""" CONSTNATS - IMF """
M_MIN = 5e-3
M_MAX = 1e2
N_BINS = 20
SAL_LOC = (4e0, 3e-1)

PLOT_TITLE = r'$\alpha_\mathrm{vir}=0.25, N=512^3-2048^3$'
#PLOT_TITLE = r"$E_v\propto k^{-1}," + PLOT_TITLE[1:]
PLOT_TITLE = PLOT_TITLE[:-1] + r",\mathrm{SFE}=10\%$"
PLOT_CDF = False

SHIFT_X = False
SHIFT_Y = False

"""
================================================================================
Macro Functions
================================================================================
"""
def proj_mpi(filename, filename_out, save=True, ax=plt.gca(), shift=(False,False), **kwargs) :
    print(f"plotting the projected field from {filename}...")
    h5f = h5py.File(filename, 'r')

    filename_wo_ext = filename.split(".")[0]
    proj_axis = filename_wo_ext[-1]
    #proj_title = "_".join(filename_wo_ext.split("_")[-3:-1])
    proj_title = 'dens_proj'
    filename_plt = "_".join(filename_wo_ext.split("_")[:-3])

    # find for a particle file assiciated with the proj_mpi result
    i = filename_wo_ext.find("plt_cnt")
    filename_part = filename_wo_ext[:i]+'part'+filename_wo_ext[i+7:i+12]
    try :
        pf = h5tools.PartFile(filename_part, verbose=False)
        print(f"particle file {filename_part} found!")
        part_exists = False if pf.particles is None else True

        # container for the constants
        cst = measures.CST.fromH5File(pf)
    except OSError :
        print(f"particle file {filename_part} not found...")
        pf = h5tools.H5File(filename_plt, verbose=False)
        part_exists = False

        # container for the constants
        cst = measures.CST.fromfile(filename_plt)

    # read the density data
    dens_proj = np.transpose(h5f[proj_title])
    xyz_lim = np.array(h5f['minmax_xyz']) / PC_flash

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

    #try :
    if shift[0] is True :
        print("shifting the x axis...")
        part_xlocs += (xrange[1]-xrange[0])/2
        part_xlocs = [np.mod(x-xrange[0],xrange[1]-xrange[0])+xrange[0] for x in part_xlocs]
    if shift[1] is True :
        print("shifting the y axis...")
        part_ylocs += (yrange[1]-yrange[0])/2
        part_ylocs = [np.mod(y-yrange[0],yrange[1]-yrange[0])+yrange[0] for y in part_ylocs]

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
        annotation=annotation, log=True, colorbar=True, shift=shift, **kwargs)

    mpltools.plot_scatter(part_xlocs, part_ylocs, ax=ax,
        xrange=xrange, yrange=yrange,
        overplot=True, marker=r'$\odot$', s=80, color='limegreen')

    if save :
        fig.savefig(filename_out)
        print(f"plot saved to: {filename_out}")
#enddef proj_mpi

def imfs(filename_imf, filename_out,
         plot_cdf=False, save=True, ax=plt.gca(), label="", **kwargs) :
    # read the imf file
    h5f = h5py.File(filename_imf, 'r')
    part_masses = np.array(h5f['mass'])
    masses_in_msol = part_masses / M_SOL
    n_sink = len(masses_in_msol)

    # annotation
    label += rf" $(N_\mathrm{{sink}}={n_sink})$"

    # set up the bins
    bins = np.zeros(n_sink+2)
    bins[0] = M_MIN
    bins[1:-1] = np.sort(masses_in_msol)
    bins[-1] = M_MAX

    # make the cdf
    cdf = np.zeros(n_sink+2)
    cdf[0] = 0.0
    cdf[1:-1] = np.arange(n_sink) / n_sink
    cdf[-1] = 1.0

    # fit the cdf
    fitting_range = [(0.8<x<10.0) for x in bins]
    bins_tofit = bins[fitting_range]
    cdf_tofit = cdf[fitting_range]
    fit_params = bayestools.fit_cdf(
        bins_tofit, cdf_tofit, np.zeros(len(bins_tofit)))

    # plot the cdf
    if plot_cdf is True :
        print("plotting cdf...")
        mpltools.plot_1D(ax.step, bins, cdf,
            ax=ax, xrange=(M_MIN,M_MAX), log=(True,False),
            xlabel=r'$M\,[M_\odot]$', ylabel='CDF',
            label=label, where='pre', **kwargs)

        # fitted slope
        a, b, ln_f = fit_params[:,0]
        ax.plot(bins_tofit, 1 - a*bins_tofit**b,
            color=kwargs['color'],linestyle='--', linewidth=2)

        print("plotting completed!")

    # plot the pdf
    else :
        # set up the bins
        logbins = np.logspace(np.log10(M_MIN),np.log10(M_MAX),N_BINS)

        # create histogram and normalise it
        log_imf, _ = np.histogram(masses_in_msol, bins=logbins)
        bin_widths = np.diff(logbins)
        bin_medians = (logbins[1:]+logbins[:-1])/2
        imf = log_imf / bin_medians
        norm_factor = np.sum(imf*bin_widths)

        log_imf = log_imf / norm_factor

        # plot the IMF
        print("plotting imf...")
        mpltools.plot_1D(ax.step, bin_medians, log_imf,
            ax=ax, log=True, xrange=(M_MIN,M_MAX),
            xlabel=r'$M\,[M_\odot]$', ylabel=r'$\dfrac{dN}{d\log{M}}$',
            label=label, where='pre', **kwargs)

        # fitted slope
        a, b, ln_f = fit_params[:,0]
        ax.plot(bins_tofit, -a*b*bins_tofit**b,
            color=kwargs['color'], linestyle='--', linewidth=2)

        # salpeter slope
        if save :
            ax.plot(logbins, SAL_LOC[1]*(logbins/SAL_LOC[0])**(-1.35),
                    color='red', linestyle='--', label="Salpeter")
        print("plotting complete!")

    # export the plot
    if save :
        plt.legend(loc='upper left', prop={'size': 16})
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
            cst = measures.CST.fromfile(glob.glob(filenames_plt)[0], verbose=False)
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
        mpltools.plot_1D(ax.plot, SFEs[1:]*100.0, SFR_in_Tff,
            overplot=overplot, xrange=(0.0, 20.0), yrange=(0.0, 1.00),
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
                        help='Build IMF from the particle data')
    parser.add_argument('--cdfs', action='store_true', default=False,
                        help='Build CDF from the particle data')
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
    #matplotlib.rcParams['text.usetex'] = False
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)

    if args.proj_mpi :
        # the file name of the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename.split(".")[0] + '.' + args.ext
        else :
            filename_out = filename.split(".")[0] + '.png'

        # plot the projection
        proj_mpi(filename, filename_out,
            ax=ax, title=PLOT_TITLE,
            colorbar_title=r"Column Density $[\mathrm{g}\,\mathrm{cm}^{-2}]$",
            color_range=(0.02,1.0), shift=(SHIFT_X,SHIFT_Y))

    if args.imfs or args.cdfs :
        # the file name of the plot
        if args.o is not None :
            filename_out = args.o
        elif args.ext is not None :
            filename_out = filename + '_imf.' + args.ext
        else :
            filename_out = filename + '_imf.png'

        # plot the imf
        label_1 = r"$E_v \propto k^{-1}$"
        label_2 = r"$E_v \propto k^{-2}$"
        if args.cdfs :
            alpha = 0.5
        else :
            alpha = 1.0
        imfs("parts_beta1_sfe0.1.h5", filename_out, label=label_1,
            ax=ax, plot_cdf=args.cdfs, color='blue', linewidth=2, alpha=alpha, save=False)
        imfs("parts_beta2_sfe0.1.h5", filename_out, label=label_2,
            ax=ax, plot_cdf=args.cdfs, title=PLOT_TITLE, color='black', linewidth=2, alpha=alpha)

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
