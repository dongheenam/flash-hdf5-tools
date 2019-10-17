#! /usr/bin/env python

# MATPLOTLIB SETUP
import matplotlib

import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# OTHER DEPENDENCIES
import numpy as np

def plot_proj(ax, proj,
              xrange=None, yrange=None, xlabel=None, ylabel=None,
              title=None, annotation=None, colorbar_title=None,
              colorbar=False, log=False, plot_range=[None,None], **imshow_kwargs) :
  """
  DESCRIPTION
      Plots a projection plot atop the specified axis

  INPUTS
      ax (matplotlib.axes)
          the axis the projection will be plotted on
      proj (array_like[N_x, N_y])
          the two-dimensional projection data

      xrange, yrange (array_like[2])
          [min, max] for the ticks
      xlabel, ylabel (str)
          axes labels (to be passed on to ax.set_xlabel and ax.set_ylabel)
      title (str)
          plot title
      annotation (str)
          extra text to be displayed
      colorbar (boolean)
          if True adds a colorbar inside the axes
          (alternatively add it to the Figure directly)
      colorbar_title (str)
          title of the colorbar
      log (boolean)
          set logarithmic colormap
      plot_range (array_like[2])
          minimum and maximum values to be plotted

  OUTPUTS
      im (matplotlib.image.AxesImage)
          returns the mappable image (to be used for colorbar outside the axes)
  """
  if log == True :
    norm = LogNorm()
  else :
    norm = None

  # plot the projection data
  im = ax.imshow(np.transpose(proj),
                 norm=norm, origin='lower', cmap='magma', aspect='equal',
                 vmin=plot_range[0], vmax=plot_range[1],
                 **imshow_kwargs)

  # set the ticks
  ny, nx = proj.shape
  ax.set_xticks(np.linspace(-0.5, nx+0.5, 5))
  ax.set_yticks(np.linspace(-0.5, ny+0.5, 5))

  if xrange is None :
    xrange = [0,nx]
  if yrange is None :
    yrange = [0,ny]

  xticks = [r"${:.1f}$".format(f) for f in np.linspace(xrange[0], xrange[1], 5)]
  yticks = [r"${:.1f}$".format(f) for f in np.linspace(yrange[0], yrange[1], 5)]
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

  # set annotation


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
  """ SOME MACROS """
  import argparse
  import h5py

  matplotlib.style.use('classic')
  matplotlib.rcParams['text.usetex'] = True
  matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
  matplotlib.rc('font', **{'family': 'DejaVu Sans', 'weight':'normal', 'size': 18})
  matplotlib.rc('text', usetex=True)

  parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
  parser.add_argument('path', type=str,
                      help='the path to the file to be loaded')

  parser.add_argument('--proj_mpi', action='store_true', default=False,
                      help='read the hdf5 file from projeciton_mpi and create a plot')

  parser.add_argument('-o', type=str, default=None,
                      help='the output path')
  parser.add_argument('-ext', type=str,
                      help='the extension for the output (ignored if -o is set up)')

  args = parser.parse_args()
  path = args.path

  # draws the projeciton_mpi hdf5 file
  if args.proj_mpi :
    print("plotting the projected field from {}...".format(path))
    h5f = h5py.File(args.path)

    proj_axis = path[-4]
    proj_title = path[-14:-3]
    proj_field = path[-14:-5]

    dens_proj = h5f['dens_proj']
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

    plot_proj(ax, h5f[proj_field],
                  xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel,
                  title=r'$t={:.1f}T$'.format(time_in_T), annotation=None,
                  colorbar_title=r"Projected density $[\mathrm{g}\,\mathrm{cm}^{-2}]$",
                  colorbar=True, log=True, plot_range=[1e-2,2e-1])

    if args.o is None :
      if args.ext is None :
        save_path = path[:-2] + 'png'
      else :
        save_path = path[:-2] + args.ext
    else :
      save_path = args.o
    
    fig.savefig(save_path)
    print("plot saved to: {}".format(save_path))
  # endif args.proj_mpi
