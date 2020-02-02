#! /usr/bin/env python3

# MATPLOTLIB SETUP
import matplotlib

import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib.ticker import FormatStrFormatter

# OTHER DEPENDENCIES
import numpy as np
import h5tools

def mpl_init() :
    """ loads predefined plotting parameters """

    # overall plot style
    matplotlib.style.use('classic')
    matplotlib.rcParams['figure.figsize'] = (8,7)
    matplotlib.rcParams['savefig.bbox'] = 'tight'

    # text
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    matplotlib.rcParams['font.family'] = "DejaVu Sans"
    matplotlib.rcParams['font.size'] = 18

    # ticks and lines
    matplotlib.rcParams['xtick.major.size'] = 12
    matplotlib.rcParams['ytick.major.size'] = 12
    matplotlib.rcParams['xtick.minor.size'] = 6
    matplotlib.rcParams['ytick.minor.size'] = 6

    matplotlib.rcParams['axes.linewidth'] = 1
    matplotlib.rcParams['xtick.major.width'] = 1
    matplotlib.rcParams['ytick.major.width'] = 1
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.minor.width'] = 1

    matplotlib.rcParams['lines.markeredgewidth'] = 0

    # legend
    matplotlib.rcParams['legend.numpoints']     = 1
    matplotlib.rcParams['legend.frameon']       = False
    matplotlib.rcParams['legend.handletextpad'] = 0.3
# enddef mpl_init


"""
================================================================================
PLOTTER FUNCTIONS
================================================================================
"""
def plot_1D(plot_func, x, y,
    ax=plt.gca(), overplot=False, xrange=(None, None), yrange=(None, None),
    xlabel=None, ylabel=None, title=None, annotation=None, log=False,
    fit=None, fit_range=None, **plot_kwargs) :
    """
    DESCRIPTION
        Plots a single line plot atop the specified axis

    INPUTS
        plot_func (function)
            instance method of Axes or pyplot, such as plt.plot or ax.hist
        x (array_like[N])
            the x-coordinates
        y (array_like[N])
            the y-coordinates

        ax (matplotlib.axes)
            the axis the data will be plotted on
        overplot (boolean)
            set this to True not to reinitialise the axes ranges and ticks
        xrange, yrange (array_like[2])
            [min, max] of the axes
        xlabel, ylabel (str)
            axes labels (to be passed on to ax.set_xlabel and ax.set_ylabel)
        title (str)
            plot title
        annotation (str)
            extra text to be displayed as anchored text
        fit (str)
            adds a linear ("linear") or log-log ("loglog") fit
        fit_range (array_like[2])
            the range of x and y to be fitted for

        plot_kwargs (keyword arguments)
            keyward arguments to be passed to matplotlib.axes.plot
    """
    from matplotlib.offsetbox import AnchoredText
    from scipy.optimize import curve_fit

    # plot the data
    plot_func(x, y, **plot_kwargs)

    if not overplot :
        # set the log scale
        try :
            if log[0] is True :
                ax.set_xscale("log")
            if log[1] is True :
                ax.set_yscale("log")

        except TypeError :
            if log is True :
                ax.set_xscale("log")
                ax.set_yscale("log")

        # set the axes ranges
        ax.set_xlim(left=xrange[0], right=xrange[1])
        ax.set_ylim(bottom=yrange[0], top=yrange[1])

        # set the axes labels
        if xlabel is not None :
            ax.set_xlabel(xlabel)
        if ylabel is not None :
            ax.set_ylabel(ylabel)

        # set title
        if title is not None :
            ax.set_title(title)

        # annotation
        if annotation is not None :
            at = AnchoredText(annotation, loc='upper left', frameon=True)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
    # endif not overplot

    # fitting
    if fit == "linear" :
        pass
    elif fit == "loglog" :
        # perform a linear fit in log-log space
        def logf(logx, a, b) :
            return a + logx*b

        # crop the x and y according to the fitting range
        if fit_range is not None :
            x_fit = x[fit_range[0]:fit_range[1]]
            y_fit = y[fit_range[0]:fit_range[1]]
        else :
            x_fit = x
            y_fit = y

        # perform the fitting
        print("begin fitting...")
        popt, pcov = curve_fit(logf, np.log10(x_fit), np.log10(y_fit), p0=(1,0))

        # overplot the fitted curve
        ax.plot(x, np.power(10, logf(np.log10(x), *popt)),
                color=plot_kwargs['color'], linewidth=plot_kwargs['linewidth']*2, alpha=0.3,
                label=rf"$k^{{ {popt[1]:5.3f}\pm {np.sqrt(np.diag(pcov))[1]:5.3f} }}$")

    # endif fit
    plt.tight_layout()
# enddef plot_1D

def plot_hist(data,
    ax=plt.gca(), overplot=False, xrange=(None, None), yrange=(None, None),
    xlabel=None, ylabel=None, title=None, annotation=None, log=False,
    **plot_kwargs) :
    """
    DESCRIPTION
        Plots a single histogram atop the specified axis

    INPUTS
        data (array_like[N])
            the data

        ax (matplotlib.axes)
            the axis the data will be plotted on
        overplot (boolean)
            set this to True not to reinitialise the axes ranges and ticks
        xrange, yrange (array_like[2])
            [min, max] of the axes
        xlabel, ylabel (str)
            axes labels (to be passed on to ax.set_xlabel and ax.set_ylabel)
        title (str)
            plot title
        annotation (str)
            extra text to be displayed as anchored text

        plot_kwargs (keyword arguments)
            keyward arguments to be passed to matplotlib.axes.plot
    """
    from matplotlib.offsetbox import AnchoredText
    from scipy.optimize import curve_fit

    # plot the data
    ax.hist(data, **plot_kwargs)

    if not overplot :
        # set the log scale :
        if log is True :
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif log is False :
            pass
        else :
            if log[0] is True :
                ax.set_xscale("log")
            if log[1] is True :
                ax.set_yscale("log")

        # set the axes ranges
        ax.set_xlim(left=xrange[0], right=xrange[1])
        ax.set_ylim(bottom=yrange[0], top=yrange[1])

        # set the axes labels
        if xlabel is not None :
            ax.set_xlabel(xlabel)
        if ylabel is not None :
            ax.set_ylabel(ylabel)

        # set title
        if title is not None :
            ax.set_title(title)

        # annotation
        if annotation is not None :
            at = AnchoredText(annotation, loc='upper left', frameon=True)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
    # endif not overplot

    plt.tight_layout()
# enddef plot_hist

def plot_proj(proj, ax=plt.gca(), overplot=False,
              xrange=None, yrange=None, xlabel=None, ylabel=None,
              title=None, colorbar=False, colorbar_title=None, annotation=None,
              log=False, shift=False, color_range=[None,None], **imshow_kwargs):
    """
    DESCRIPTION
        Plots a projection plot atop the specified axis

    INPUTS
        ax (matplotlib.axes)
            the axis the projection will be plotted on
        proj (array_like[N_x, N_y])
            the two-dimensional projection data

        overplot (boolean)
            set this to True not to reinitialise the axes ranges and ticks
        xrange, yrange (array_like[2])
            [min, max] of the axes
        xlabel, ylabel (str)
            axes labels (to be passed on to ax.set_xlabel and ax.set_ylabel)
        title (str)
            plot title
        colorbar (boolean)
            if True adds a colorbar inside the axes
            (alternatively add it to the Figure directly)
        colorbar_title (str)
            title of the colorbar
        annotation (str)
            extra text to be displayed as anchored text
        log (boolean)
            set logarithmic colormap
        shift (boolean or array_like[2])
            shifts the x or y axis (a single boolean with control both)
        color_range (array_like[2])
            minimum and maximum values to be plotted for imshow

        imshow_kwargs (keyword arguments)
            keyward arguments to be passed to matplotlib.axes.imshow

    OUTPUTS
        im (matplotlib.image.AxesImage)
            returns the mappable image (to be used for colorbar outside the axes)
    """
    from matplotlib.colors import LogNorm
    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    if log == True :
        norm = LogNorm()
    else :
        norm = None

    # transpose the image so that axes.imshow will work properly
    image = np.transpose(proj)

    # shift the image if necessary
    try :
        if shift[0] is True :
            image = np.roll(image, image.shape[0]//2, axis=1)
        if shift[1] is True :
            image = np.roll(image, image.shape[1]//2, axis=0)
    except TypeError :
        if shift is True :
            image = np.roll(image, (n//2 for n in image.shape))

    # set the physical size of the image
    if (xrange is not None) and (yrange is not None) :
        extent = [xrange[0], xrange[1], yrange[0], yrange[1]]
    else :
        extent = None

    # plot the projection data
    im = ax.imshow(image,
                   norm=norm, origin='lower', cmap='magma', aspect='equal',
                   vmin=color_range[0], vmax=color_range[1], extent=extent,
                   **imshow_kwargs)

    if not overplot :
        # set the axes ranges
        ax.set_xlim(left=xrange[0], right=xrange[1])
        ax.set_ylim(bottom=yrange[0], top=yrange[1])

        # set the axes labels
        if xlabel is not None :
            ax.set_xlabel(xlabel)
        if ylabel is not None :
            ax.set_ylabel(ylabel)

        # set title
        if title is not None :
            ax.set_title(title)

        # annotation
        if annotation is not None :
            at = AnchoredText(annotation, loc='upper left', frameon=False, prop={'color':'w'})
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

        # set colorbar
    if colorbar == True :
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='10%', pad=0)
        plt.colorbar(im, cax=cax)
        if colorbar_title is not None :
            cax.set_ylabel(colorbar_title, rotation=90)

    plt.tight_layout()
    return im
# end def plot_proj()

if __name__ == "__main__" :
    """
    ================================================================================
    SOME MACROS
    ================================================================================
    """
    import argparse
    import h5py

    # matplotlib initialisation
    mpl_init()

    # argument parsing
    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='the filenames to be loaded')

    parser.add_argument('--ps1d', action='store_true', default=False,
                        help='create a plot of the 1D power spectrum')

    parser.add_argument('-o', type=str, default=None,
                        help='the output path')
    parser.add_argument('-ext', type=str,
                        help='the extension for the output (ignored if -o is set up)')

    args = parser.parse_args()
    filenames = args.filenames

    # draws the 1d power spectrum
    if args.ps1d :
        styles = ([{'marker':'o', 'markersize':5, 'color':'red', 'alpha':1.0, 'label':r'$\mathrm{targ\_beta}=1$', 'linewidth':3, 'linestyle':'-'},
                   {'marker':'o', 'markersize':5, 'color':'blue', 'alpha':1.0, 'label':r'$\mathrm{targ\_beta}=2$', 'linewidth':3, 'linestyle':'-'}])

        fig = plt.figure(figsize=(4,3.5))
        ax = fig.add_subplot(111)

        for location, style in zip(filenames, styles) :
            data = np.genfromtxt(location)
            x, y = data.T

            overplot = False if location == filenames[0] else True
            print("plotting {}...".format(location))
            plot_1D(x, y, ax=ax, overplot=overplot,
                    xrange=(1, 128), yrange=(1e6, 5e9), xlabel='$k$', ylabel='$P_v(k)$',
                    title=None, annotation=None,
                    log=(True, True), fit='loglog', fit_range=(3, 20),
                    **style)

        if args.o is None :
            if args.ext is None :
                save_path = filenames[0][:-3] + 'png'
            else :
                save_path = filenames[0][:-3] + args.ext
        else :
            save_path = args.o
        fig.savefig(save_path)
        print("plot saved to: {}".format(save_path))
    # endif args.ps1d
