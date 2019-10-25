#! /usr/bin/env python

# MATPLOTLIB SETUP
import matplotlib

import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib.ticker import FormatStrFormatter

# OTHER DEPENDENCIES
import numpy as np

"""
================================================================================
PLOTTER FUNCTIONS
================================================================================
"""
def plot_1D(ax, x, y, overplot=False,
            xrange=(None, None), yrange=(None, None), xlabel=None, ylabel=None,
            title=None, annotation=None,
            log=(False, False), fit=None, fit_range=None,
            **plot_kwargs) :
  """
  DESCRIPTION
      Plots a single line plot atop the specified axis

  INPUTS
      ax (matplotlib.axes)
          the axis the projection will be plotted on
      x (array_like[N])
          the x-coordinates
      y (array_like[N])
          the y-coordinates

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
  ax.plot(x, y, **plot_kwargs)

  if not overplot :
    # set the log scale :
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
            label=r"$k^{%5.3f\pm%5.3f}$" % (popt[1],np.sqrt(np.diag(pcov))[1]))

    ax.legend()
    plt.tight_layout()
  # endif fit
# enddef plot_1D

def plot_proj(ax, proj, overplot=False,
              xrange=None, yrange=None, xlabel=None, ylabel=None,
              title=None, colorbar=False, colorbar_title=None, annotation=None,
              log=False, color_range=[None,None], **imshow_kwargs) :
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

  # plot the projection data
  im = ax.imshow(np.transpose(proj),
                 norm=norm, origin='lower', cmap='magma', aspect='equal',
                 vmin=color_range[0], vmax=color_range[1],
                 **imshow_kwargs)

  if not overplot :
    # set the ticks
    ny, nx = proj.shape
    ax.set_xticks(np.linspace(-0.5, nx+0.5, 5))
    ax.set_yticks(np.linspace(-0.5, ny+0.5, 5))

    if xrange is None :
      xrange = [0,nx]
    if yrange is None :
      yrange = [0,ny]

    xticks = [r"${:.1f}$".format(tick) for tick in np.linspace(xrange[0], xrange[1], 5)]
    yticks = [r"${:.1f}$".format(tick) for tick in np.linspace(yrange[0], yrange[1], 5)]
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)

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
  matplotlib.style.use('classic')
  matplotlib.rcParams['text.usetex'] = True
  matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
  matplotlib.rc('font', **{'family': 'DejaVu Sans', 'weight':'normal', 'size': 18})

  # argument parsing
  parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
  parser.add_argument('path', type=str, nargs='+',
                      help='the path to the file to be loaded')

  parser.add_argument('--proj_mpi', action='store_true', default=False,
                      help='read the hdf5 file from projeciton_mpi and create a plot')
  parser.add_argument('--ps1d', action='store_true', default=False,
                      help='create a plot of the 1D power spectrum')

  parser.add_argument('-o', type=str, default=None,
                      help='the output path')
  parser.add_argument('-ext', type=str,
                      help='the extension for the output (ignored if -o is set up)')

  args = parser.parse_args()
  path = args.path

  # draws the 1d power spectrum
  if args.ps1d :
    styles = ([{'marker':'o', 'markersize':5, 'color':'red', 'alpha':1.0, 'label':r'$\mathrm{targ\_beta}=1$', 'linewidth':3, 'linestyle':'-'},
              {'marker':'o', 'markersize':5, 'color':'blue', 'alpha':1.0, 'label':r'$\mathrm{targ\_beta}=2$', 'linewidth':3, 'linestyle':'-'}])

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)

    for location, style in zip(path, styles) :
      data = np.genfromtxt(location)
      x, y = data.T

      overplot = False if location == path[0] else True
      print("plotting {}...".format(location))
      plot_1D(ax, x, y, overplot=overplot,
              xrange=(1, 128), yrange=(1e6, 5e9), xlabel='$k$', ylabel='$P_v(k)$',
              title=None, annotation=None,
              log=(True, True), fit='loglog', fit_range=(3, 20),
              **style)

    if args.o is None :
      if args.ext is None :
        save_path = path[0][:-3] + 'png'
      else :
        save_path = path[0][:-3] + args.ext
    else :
      save_path = args.o
    fig.savefig(save_path)
    print("plot saved to: {}".format(save_path))
  # endif args.ps1d

  # draws the projeciton_mpi hdf5 file
  if args.proj_mpi :
    print("plotting the projected field from {}...".format(path[0]))
    h5f = h5py.File(args.path[0])

    proj_axis = path[0][-4]
    proj_title = path[0][-14:-3]
    proj_field = path[0][-14:-5]

    dens_proj = h5f[proj_field]
    xyz_lim = np.array(h5f['minmax_xyz']) / 3.086e18

    if proj_axis == 'x' :
      xlabel, ylabel = (r'$y \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
      xrange, yrange = (xyz_lim[1], xyz_lim[2])
    elif proj_axis == 'y' :
      xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$z \,[\mathrm{pc}]$')
      xrange, yrange = (xyz_lim[0], xyz_lim[2])
    elif proj_axis == 'z' :
      xlabel, ylabel = (r'$x \,[\mathrm{pc}]$', r'$y \,[\mathrm{pc}]$')
      xrange, yrange = (xyz_lim[0], xyz_lim[1])

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)

    time_in_T = h5f['time'][0] / 3.086e13

    plot_proj(ax, dens_proj,
                  xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel,
                  title=r'$t={:.1f}T$'.format(time_in_T), annotation=r"$\beta=2$",
                  colorbar_title=r"Density $[\mathrm{g}\,\mathrm{cm}^{-3}]$",
                  colorbar=True, log=True, color_range=(None,None))

    if args.o is None :
      if args.ext is None :
        save_path = path[0][:-2] + 'png'
      else :
        save_path = path[0][:-2] + args.ext
    else :
      save_path = args.o

    fig.savefig(save_path)
    print("plot saved to: {}".format(save_path))
  # endif args.proj_mpi
