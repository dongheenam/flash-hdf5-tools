#!/usr/bin/env python3

# dependencies
import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

import bayestools
import dnam_tools
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

""" CONSTANTS - projection """
SHIFT_X = False
SHIFT_Y = True

""" CONSTANTS - sfr """
SFE_CUTOFF = 0.1

""" CONSTANTS - general """
PLOT_TITLE = r'$\alpha_\mathrm{vir}=0.25, N=512^3-2048^3$'
#PLOT_TITLE = r"$n=1," + PLOT_TITLE[1:]
#PLOT_TITLE = PLOT_TITLE[:-1] + r",\mathrm{SFE}=10\%$"
FIT_WITH_CDF = False

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

    mpltools.plot_1D(ax.scatter, part_xlocs, part_ylocs, ax=ax,
        xrange=xrange, yrange=yrange,
        overplot=True, marker='.', s=1, color='limegreen')
    mpltools.plot_1D(ax.scatter, part_xlocs, part_ylocs, ax=ax,
        xrange=xrange, yrange=yrange,
        overplot=True, marker='o', s=50, facecolors='none', color='limegreen')

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
    print(f"particle file {filename_imf} read!")
    print(f"number of sinks: {n_sink}")
    print(f"biggest sink   : {np.max(masses_in_msol):.2E} M_sol")
    print(f"smallest sink  : {np.min(masses_in_msol):.2E} M_sol")

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
    if FIT_WITH_CDF :
        fit_params = bayestools.fit_cdf_chab(
            bins, cdf, np.zeros(len(bins)))

    # plot the cdf
    if plot_cdf is True :
        print("plotting cdf...")
        mpltools.plot_1D(ax.step, bins, cdf,
            ax=ax, xrange=(M_MIN,M_MAX), log=(True,False),
            xlabel=r'$M\,[M_\odot]$', ylabel='CDF',
            label=label, where='pre', **kwargs)

        # fitted slope
        cdf_fit = np.array([bayestools.cdf_chab(fit_params[:,0], x_elem) for x_elem in bins])
        ax.plot(bins, cdf_fit,
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

        if not FIT_WITH_CDF :
            fit_params = bayestools.fit_imf(masses_in_msol)

        # append the index to the label
        gamma = fit_params[3,0]
        gamma_err = fit_params[3,1:]
        label = label[:-1] + rf": \Gamma= {gamma:.2f}^{{+{gamma_err[0]:.2f} }}_{{-{gamma_err[1]:.2f} }} $"

        # plot the IMF
        print("plotting imf...")
        mpltools.plot_1D(ax.step, bin_medians, log_imf,
            ax=ax, log=True, xrange=(M_MIN,M_MAX), yrange=(5e-3, 5e0),
            xlabel=r'$M\,[M_\odot]$', ylabel=r'$\dfrac{dN}{d\log{M}}$',
            label=label, where='mid', **kwargs)

        # fitted slope
        fitted_IMF = bayestools.gen_imf_chab(fit_params[:,0], bin_medians) * bin_medians
        ax.plot(bin_medians, fitted_IMF,
            color=kwargs['color'], linestyle='--', linewidth=2)

        # salpeter slope
        if save :
            ax.plot(logbins, SAL_LOC[1]*(logbins/SAL_LOC[0])**(-1.35),
                    color='red', linestyle='--', label=r"Salpeter: $\Gamma= -1.35$")
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

    overplot = False
    for folder in folders :
        # read the .dat file
        try :
            dat = h5tools.DatFile(os.path.join(folder, "Turb.dat"))
            print(f"Read Turb.dat within {folder}!")
        except OSError :
            print(f"Turb.dat file does not exist in {folder}. skipping...")
            continue

        # reach out to the last plotfile in the folder for the constants
        filenames_plt = os.path.join(folder, 'Turb_hdf5_plt_cnt_????')
        filename_plt = dnam_tools.get_file(filenames_plt, loc='last')
        if filename_plt is not None :
            cst = measures.CST.fromfile(filename_plt, verbose=False)
            print(f"using {filename_plt} for the constants...")
        else :
            print(f"plotfile does not exist in {folder}. skipping...")
            continue

        # calculate the star formation efficiency and pull time data
        SFEs = (cst.M_TOT - dat.data['mass'])/cst.M_TOT
        time_sec = dat.data['time']

        # cut off SFE
        #data_below_cutoff = SFEs<SFE_CUTOFF
        #SFEs = SFEs[data_below_cutoff]
        #time_sec = time_sec[data_below_cutoff]

        # shift the SFE data if necessary
        if False :
            time_sec -= time_sec[0]
        else :
            sink_exists = SFEs>1e-10
            SFEs = SFEs[sink_exists]
            time_sec = time_sec[sink_exists]
            time_sec -= time_sec[0]

        # calculate time in terms of crossing time and free-fall time
        time_in_T = time_sec/cst.T_TURB
        time_in_Tff = time_sec/cst.T_FF

        # average the SFR
        smooth_len = 100
        x = time_in_Tff[smooth_len//2:-smooth_len//2:smooth_len]
        y = np.diff(SFEs[::smooth_len])/np.diff(time_in_Tff[::smooth_len])

        # add seed to label
        seed = str(folder)[-11:-5]
        label_old = kwargs['label']
        kwargs['label'] = kwargs['label'][:-1] + rf", \mathrm{{seed}}={seed}$"

        # plot SFE vs t_FF
        mpltools.plot_1D(ax.plot, time_in_Tff, SFEs*100.0,
            ax=ax, overplot=overplot, yrange=(0.0,15.0),
            xlabel=r'$t\,[T_\mathrm{ff}]$', ylabel=r'$\mathrm{SFE}\,[\%]$',  **kwargs)
        # plot SFR vs t_FF
        #mpltools.plot_1D(ax.plot, x, y,
        #    ax=ax, overplot=overplot,
        #    xlabel=r'$t\,[T_\mathrm{ff}]$', ylabel=r'$\mathrm{SFR}_\mathrm{ff}$', **kwargs)

        if overplot is False :
            overplot = True
            
        kwargs['label'] = label_old

    if save :
        plt.legend(loc='upper right', prop={'size': 16})
        plt.savefig(filename_out)
        print("plot printed to: "+filename_out)
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
        label_1 = r"$n=1$"
        label_2 = r"$n=2$"
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

        label_1 = r"$n=1$"
        label_2 = r"$n=2$"
        # plot the sfrs
        sfrs("AMR_beta1_*_sink", filename_out, label=label_1,
             ax=ax, linestyle='-', linewidth=2, alpha=0.5, save=False)
        sfrs("AMR_beta2_*_sink", filename_out, label=label_2,
             ax=ax, linestyle='--', linewidth=2, alpha=0.5,
             title=PLOT_TITLE)
