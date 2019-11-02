#!/usr/bin/env python3

# DEPENDENCIES
import gc                     # garbage collection
from datetime import datetime # time measurement
import h5py                   # hdf5 handling
import numpy as np

import pyfftw                 # python wrapper for FFTW

import dnam_tools                # my tools for python

def read_parameters(h5f, dataname) :
  """
  DESCRIPTION
      return the specified parameter group as a dictionary from the FLASH hdf5 file

  INPUTS
      h5f (h5py.File)
          the h5py File object containing the hdf5 file
      dataname (string)
          the keyword for the parameter group
            e.g.: ['integer runtime parameters', 'real runtime parameters',
                   'integer scalars', 'real scalars']

  OUTPUTS
      dict (dictionary)
          contains the parameters, which can be accessed with their names e.g. dict['nxb']
  """

  # parameters
  # python3 : keys are now in bytes
  params = dict(h5f[dataname])

  # strip the spaces for easier access
  params = {x.decode("utf-8").replace(' ','') : v for x,v in params.items()}

  return params

def read_array(h5f, dataname):
  """
  DESCRIPTION
      return the specified array that does not need to be sorted
      as a NumPy from the FLASH hdf5 file

  INPUTS
      h5f (h5py.File)
          the h5py File object containing the hdf5 file
      dataname (string)
          the keyword for the array
              e.g. ['block size', 'coordinates']

  OUTPUTS
      array (float[: , ... , :])
          the NumPy array containing the data
  """

  # the parameters
  array = np.array(h5f[dataname])

  return array

def sort(H5file, data) :
  """
  DESCRIPTION
      by default the FLASH file prints the field in [nProcs, nzb, nyb, nxb]
      this method is returns the field in [N_x, N_y, N_z] dimensions
      using the parameters stored in the H5file instance

  INPUTS
      H5file
          the H5file instance
      data (float[nProcs, nzb, nyb, nxb])
          the raw data from the FLASH hdf5 file

  OUTPUTS
      data_sorted (float[N_x, N_y, N_z])
          the 3D NumPy array containing the sorted data
  """

  # some parameters required for the sorting
  # they are read from the parent class (H5file)
  dims = np.array( H5file.params['dims'] )
  L_bl = np.array( H5file.params['Lmin'] )
  L_tr = np.array( H5file.params['Lmax'] )
  L = L_tr - L_bl
  L_cell = L / dims

  block_sizes = H5file.block_sizes
  box_coords = H5file.block_coords

  # the raw field array has structure of [nProcs, nzb, nyb, nxb]
  # so transposing is necessary to have shape of [nProcs, nxb, nyb, nzb]
  data = np.transpose(data, (0,3,2,1))

  # place the boxes in the 3D array
  data_sorted = np.zeros(dims)

  iprocs = H5file.params['iprocs']
  jprocs = H5file.params['jprocs']
  kprocs = H5file.params['kprocs']
  for proc_id in range(iprocs*jprocs*kprocs) :

    # bottom left corner
    box_loc_bl = box_coords[proc_id] - block_sizes[proc_id]/2
    index_bl = np.rint( (box_loc_bl - L_bl)/L_cell ).astype(int)

    # top right corner
    box_loc_tr = box_coords[proc_id] + block_sizes[proc_id]/2
    index_tr = np.rint( (box_loc_tr - L_bl)/L_cell ).astype(int)

    # insert the block
    data_sorted[index_bl[0]:index_tr[0], index_bl[1]:index_tr[1], index_bl[2]:index_tr[2]] \
                = data[proc_id]

  # update the sorted data
  return data_sorted

class H5file :
  """
  ================================================================================
  DESCRIPTION
      container for a FLASH hdf5 file and its parameters

  INPUTS
      path (string)
          the full path to the hdf5 file

  VARIABLES
      H5file.dir_to_file (string)
          the directory where the file is located
      H5file.filename (string)
          the name of the hdf5 file
      H5file.h5f (h5py.File)
          the h5py File object
      H5file.params (dictionary)
          the integer/real parameters/scalars from the file

  METHODS
      H5file.new_dataset(dataname)
          returns a scalar or vector dataset instance, depending on the data name
  ================================================================================
  """

  def __init__(self, path) :
    slash_loc = path.rfind('/')
    if slash_loc == -1 :
      self.dir_to_file = ''
      self.filename = path
    else :
      self.dir_to_file = path[slash_loc+1 : ]
      self.filename =    path[ : slash_loc+1]
    self.h5f = h5py.File(path, 'r')

    int_params = read_parameters(self.h5f, 'integer runtime parameters')
    real_params = read_parameters(self.h5f, 'real runtime parameters')
    int_scalars = read_parameters(self.h5f, 'integer scalars')
    real_scalars = read_parameters(self.h5f, 'real scalars')
    self.params = dnam_tools.merge_dicts(int_params, real_params, int_scalars, real_scalars)

    nxb = self.params['nxb']
    nyb = self.params['nyb']
    nzb = self.params['nzb']
    iprocs = self.params['iprocs']
    jprocs = self.params['jprocs']
    kprocs = self.params['kprocs']
    self.params['dims'] = [nxb*iprocs, nyb*jprocs, nzb*kprocs]

    xmin = self.params['xmin']
    xmax = self.params['xmax']
    ymin = self.params['ymin']
    ymax = self.params['ymax']
    zmin = self.params['zmin']
    zmax = self.params['zmax']
    self.params['Lmin'] = [xmin, ymin, zmin]
    self.params['Lmax'] = [xmax, ymax, zmax]
    self.params['L'] = [xmax-xmin, ymax-ymin, zmax-zmin]

    self.block_sizes = read_array(self.h5f, 'block size')
    self.block_coords = read_array(self.h5f, 'coordinates')

    print("{} read!".format(path))
    print("simulation time: {}".format(self.params['time']))
    print("size           : {}".format(self.params['dims']))
    print("domain         : {}".format(self.params['L']))

  def new_dataset(self, dataname, small_mem=False) :
    if dataname == 'dens' :  # for density field              => create a density data set
      print("attempting to create a density dataset...")
      return self.DensityDataset(self, small_mem)
    else :
      try :         # see if the dataname exists              => create a scalar data set
        print("attempting to create a scalar dataset: {}...".format(dataname))
        return self.ScalarDataset(self, dataname, small_mem)
      except :
        try :       # see if the dataname+"x","y","z" exists  => create a vector data set
          print("attempting to create a vector dataset: {}...".format(dataname))
          return self.VectorDataset(self, dataname, small_mem)
        except :    # if the dataname does not exist within the hdf5 file => do nothing
          print("dataname not recognised")

  class Dataset(object) :
    """
    ================================================================================
    DESCRIPTION
        container for a field

    INPUTS
        H5file
            the outer class H5file

    VARIABLES
        Dataset.H5file
            the outer class
        Dataset.dataname (string)
            the key to access the field in the hdf5 field

        Dataset.ft (complex[k_x, k_y, k_z])     <== Dataset.calc_ft()
            the three-dimensional complex fourier field of .data
        Dataset.ps_3D (float[k_x, k_y, k_z])     <== Dataset.calc_ps()
            the three-dimensional power spectrum of .data
        Dataset.ps (np.array[k_max])            <== Dataset.calc_ps(calc_1D=True)
            the one-dimensional power spectrum of .data

    METHODS
        Dataset.save_to_hdf5(list_data_to_save, list_name, path_to_save=None)
            save the set of data to a hdf5 file (optional path and filename)
        Dataset.save_to_dat(list_data_to_save, list_name, path_to_save=None)
            save the set of data to a dat file (optional path and filename)
            data is stored in rows with list_name as the header

        Dataset.calc_ps(self, calc_1D=True,
                    save_1D=False, save_1D_path=None, save_3D=False, save_3D_path=None)
            calculate the 1D and/or 3D power spetrum
            requires the fourier spectrum (self.ft) to exist
    ================================================================================
    """

    def __init__(self, H5file, dataname, small_mem=False) :
      self.H5file = H5file
      self.dataname = dataname
      self.small_mem = small_mem

    def save_to_hdf5(self, list_data_to_save, list_name, path_to_save=None) :
        if path_to_save is None :
          path_to_save = "{}{}_{}.hdf5".format(
                      self.H5file.dir_to_file, self.H5file.filename, list_name[-1] )

        h5_out = h5py.File(path_to_save,'w')
        for data_to_save, name in zip(list_data_to_save, list_name) :
          print("{} saving to: {}...".format( name, path_to_save ))
          h5_out.create_dataset(name, data=data_to_save)
        h5_out.close()
        print("file saved!")

    def save_to_dat(self, list_data_to_save, list_name, path_to_save=None) :
      if path_to_save is None :
        path_to_save = "{}{}_{}.dat".format(
                    self.H5file.dir_to_file, self.H5file.filename, list_name[-1] )

      dat_out = open(path_to_save, mode='w')
      print("{} saving to: {}...".format( list_name[-1], path_to_save ))

      col_length = len(list_name)
      row_length = len(list_data_to_save[-1])
      dat_out.write(("{:>20s}"*col_length+'\n').format(*tuple(list_name)))

      for i in range(row_length) :
        tuple_row = tuple( [data[i] for data in list_data_to_save] )
        dat_out.write(("{:20.11E}"*col_length+'\n').format(*tuple_row))

      dat_out.close()
      print("file saved!")

    def calc_ps(self, save=False, save_path=None) :
      t_start = datetime.now()

      small_mem = self.small_mem
      try :
        self.ps_3D
        print("3D power spectrum found!")
      except AttributeError :
        print("3D power spectrum not found!")
        self.calc_ps_3D()

      print("calculating the power spectrum...")
      sum_power = np.sum(self.ps_3D)

      # some physical properties
      dims = np.array(self.H5file.params['dims'])
      nx, ny, nz = dims
      L = np.array(self.H5file.params['L'])

      # calculate the wavenumber space
      # [-N, -(N-1), ..., -1, 0, 1, ..., N-2, N-1]
      kx_list = ( np.arange(-nx//2,nx//2,1) )/L[0]
      ky_list = ( np.arange(-ny//2,ny//2,1) )/L[1]
      kz_list = ( np.arange(-nz//2,nz//2,1) )/L[2]

      # the wavenumber of the corresponding element of the 3D power spectrum
      k_x, k_y, k_z = np.meshgrid(kx_list, ky_list, kz_list, indexing="ij")
      k = np.sqrt(k_x**2 + k_y**2 + k_z**2)

      # physical limits to the wavenumbers
      kmin = np.min(1.0/L)
      kmax = np.min(0.5*dims/L)
      # bins of wavenumbers
      # first bin = (0.5 to 1.5), second bin = (1.5 to 2.5), ...
      k_bins = np.arange(kmin, kmax, kmin) - 0.5*kmin

      # calculate the power spectrum
      ps_1D = np.zeros(len(k_bins))

      # sorting the power spectrum in the increasing order of wavenumber
      sorting_ind = np.argsort(k.flat)
      k_sorted = k.flat[sorting_ind]
      if small_mem :
        k = None
      ps_sorted = self.ps_3D.flat[sorting_ind]
      if small_mem :
        self.ps_3D = None
      sorting_ind = None
      gc.collect()

      # determine the location of the bins
      loc_bins = np.searchsorted(k_sorted, k_bins, sorter=None)
      # sum the power spectrum under the bins
      ps_1D = [np.sum(ps_sorted[loc_bins[i]:loc_bins[i+1]]) for i in range(len(k_bins)-1)]

      self.ps = ps_1D
      self.k = k_bins[:-1]/kmin + 0.5

      t_end = datetime.now()
      delta = t_end - t_start
      print("completed!")
      print("time taken: {}".format(delta.total_seconds()))

      if save == True :
        dataset_name = self.dataname + "ps"
        print("saving power spectrum to {}...".format(dataset_name))
        self.save_to_dat([self.k,self.ps], ['k',dataset_name], path_to_save=save_path)
    # end def calc_ps

  class ScalarDataset(Dataset) :
    """
    ================================================================================
    DESCRIPTION
        container for a scalar field

    INPUTS
        H5file
            the outer class H5file

    VARIABLES
        Dataset.H5file
            the outer class
        Dataset.data (np.array[N_x,N_y,N_z])
            the raw field values
        Dataset.dataname (string)
            the key to access the field in the hdf5 field

        Dataset.ft (complex[k_x, k_y, k_z])     <== Dataset.calc_ft()
            the three-dimensional complex fourier field of .data
        Dataset.ps_3D (float[k_x, k_y, k_z])     <== Dataset.calc_ps()
            the three-dimensional power spectrum of .data
        Dataset.ps (np.array[k_max])            <== Dataset.calc_ps(calc_1D=True)
            the one-dimensional power spectrum of .data

    METHODS

    ================================================================================
    """

    def __init__(self, H5file, dataname, small_mem=False) :
      self.H5file = H5file
      self.dataname = dataname
      self.small_mem = small_mem
      self.data = sort(H5file, H5file.h5f[dataname])

    def calc_proj(self, axis) :
      if axis =='x' :
        axis_no = 0
      elif axis == 'y' :
        axis_no = 1
      elif axis == 'z' :
        axis_no = 2
      else :
        print("axis input not set up properly!")

      self.data = np.sum(self.data, axis=axis_no)

    def calc_ps_3D(self, save=False, save_path=None) :
      t_start = datetime.now()
      small_mem=self.small_mem

      print("performing fast fourier transform...")
      rms_field = np.sqrt(np.average(self.data**2))

      fft_object = pyfftw.builders.fftn(self.data)
      self.ft = np.fft.fftshift(fft_object()) / np.product(self.H5file.params['dims'])

      print("... fft completed!")

      if small_mem==True : self.data = None

      self.ps_3D = np.abs(self.ft)**2
      sum_power = np.sum(self.ps_3D)
      print("sum_power        : {}".format(sum_power))
      print("rms_squared_field: {}".format(rms_field**2))

      t_end = datetime.now()
      delta = t_end - t_start
      print("time taken: {}".format(delta.total_seconds()))

      if save == True :
        dataset_name = self.dataname + "ps3d"
        self.save_to_hdf5([self.ps_3D], [dataset_name], path_to_save=save_path)

  class DensityDataset(ScalarDataset) :
    """
    ================================================================================
    DESCRIPTION
        container for the density field

    INPUTS
        H5file
            the outer class H5file

    VARIABLES
        Dataset.H5file
            the outer class
        Dataset.data (np.array[N_x,N_y,N_z])
            the raw field values

    METHODS

    ================================================================================
    """

    def __init__(self, H5file, small_mem=True) :
      super(H5file.DensityDataset,self).__init__(H5file, 'dens', small_mem=small_mem)

    def set_log(self) :
      self.data = np.log(self.data/self.mean())

    def set_delta(self) :
      self.set_log()
      self.data = self.data - self.mean()

  class VectorDataset(Dataset) :
    """
    ================================================================================
    DESCRIPTION
        container for a vector field of data

    INPUTS
        H5file
            the outer class H5file
        dataname (string)
            the keyword of the dataset within the file without x,y,z suffixes
                e.g. ['vel', 'mag']

        save_mem = False
            if working on large data, it is often impossible
            to work with all three field components
            set True when you do not want to load all field components at once

    VARIABLES
        Dataset.H5file
            the outer class
        Dataset.datax (np.array[N_x,N_y,N_z])
            the raw values of the x-component of the field
        Dataset.datay (np.array[N_x,N_y,N_z])
            the raw values of the y-component of the field
        Dataset.dataz (np.array[N_x,N_y,N_z])
            the raw values of the z-component of the field
    ================================================================================
    """

    def __init__(self, H5file, dataname, small_mem=False) :
      self.H5file = H5file
      self.dataname = dataname
      self.small_mem = small_mem
      if small_mem :
        self.datax = None
        self.datay = None
        self.dataz = None
      else :
        self.datax = sort(H5file, H5file.h5f[dataname+'x'])
        self.datay = sort(H5file, H5file.h5f[dataname+'y'])
        self.dataz = sort(H5file, H5file.h5f[dataname+'z'])

    def calc_ps_3D(self, save=False, save_path=None) :
      t_start = datetime.now()

      dims = np.array(self.H5file.params['dims'])
      nx, ny, nz = dims

      ps_3D = np.zeros(dims)
      ms_field = 0.0
      gc.collect()

      list_comp = [self.datax, self.datay, self.dataz]      # the vector components
      list_name = [self.dataname+s for s in ['x','y','z']]  # the name of the components
      for comp, comp_name in zip(list_comp, list_name) :
      # begin loop over components
        if comp is None :
          print("loading the field...")
          comp = sort(self.H5file, self.H5file.h5f[comp_name])
          print("field loaded!")
        else :
          print("field found within the dataset!")

        ms_field += np.average(comp**2)

        print("performing fast fourier transform...")

        FF = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(comp)) / np.product(dims)

        ps_3D += np.abs(FF)**2
        print("completed!")

        comp = None
        gc.collect()
      # end loop over components

      self.ps_3D = ps_3D

      t_end = datetime.now()
      delta = t_end - t_start
      print("time taken: {}".format(delta.total_seconds()))

      if save == True :
        dataset_name = self.dataname + "3dps"
        self.save_to_hdf5([self.ps_3D], [dataset_name], path_to_save=save_path)
    # end def calc_ps_3D

if __name__== "__main__":
  """
  ================================================================================
  Predefined macros
  ================================================================================
  """
  # DEPENDENCIES
  import argparse
  import numpy as np

  parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
  parser.add_argument('path', type=str,
                      help='the path to the file to be loaded')
  parser.add_argument('field', type=str,
                      help='name the field that will be analysed')

  parser.add_argument('--raw', action='store_true', default=False,
                      help='Save the raw field as hdf5 file (scalar only)')
  parser.add_argument('--proj', type=str, default=None, choices=['x','y','z'],
                      help='Project the field to along an axis (scalar only)')
  parser.add_argument('--ps1d', action='store_true', default=False,
                      help='Set this flag to calculate the 1D power spectrum')
  parser.add_argument('--ps3d', action='store_true', default=False,
                      help='Set this flag to calculate the 3D power spectrum')

  parser.add_argument('-o', type=str,
                      help='the output path')

  args = parser.parse_args()

  # load the file and create a dataset
  a = H5file(args.path)
  ds = a.new_dataset(args.field, small_mem=True)

  # print the raw field
  if args.raw :
    print("saving the field: {}...".format(ds.dataname))
    try :
      ds.data
    except AttributeError :
      sys.exit("set a scalar field for this operation!")

    ds.save_to_hdf5([ds.data], [ds.dataname], path_to_save=args.o)

  # project the field
  if args.proj is not None:
    print("calculating the projection of {} along {}...".format(ds.dataname, args.proj))
    ds.calc_proj(args.proj)
    ds.save_to_hdf5([ds.data], [ds.dataname+'_proj'+args.proj], path_to_save=args.o)

  # calculate the 1D power spectrum
  if args.ps1d == True :
    ds.calc_ps(save=True, save_path=args.o)
