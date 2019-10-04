# DEPENDENCIES
import gc                     # garbage collection

import h5py                   # hdf5 handling
import numpy as np

import pytools                # my tools for python

"""
FUNCTION dict = read_parameters(h5f, dataname)

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
def read_parameters(h5f, dataname) :

  # parameters
  params = dict(h5f[dataname])

  # strip the spaces for easier access
  keys_stripped = [keys.strip() for keys in params.keys()]
  for newkey, oldkey in zip(keys_stripped, params.keys()) :
    params[newkey] = params.pop(oldkey)

  return params

"""
FUNCTION array = read_array(h5f, dataname)

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
def read_array(h5f, dataname):

  # the parameters
  array = np.array(h5f[dataname])

  return array

"""
FUNCTION data_sorted = sort(H5file, data)

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
def sort(H5file, data) :

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

"""
================================================================================
CLASS H5file(filename)

DESCRIPTION
    container for a FLASH hdf5 file and its parameters

INPUTS
    filename (string)
        the name of the hdf5 file

VARIABLES
    H5file.filename (string)
        the specified filename of the hdf5 file
    H5file.h5f (h5py.File)
        the h5py File object
    H5file.params (dictionary)
        the integer/real parameters/scalars from the file

METHODS
    H5file.new_dataset(dataname)
        returns a scalar or vector dataset instance, depending on the data name
================================================================================
"""
class H5file :

  def __init__(self, filename) :
    self.filename = filename
    self.h5f = h5py.File(filename, 'r')

    int_params = read_parameters(self.h5f, 'integer runtime parameters')
    real_params = read_parameters(self.h5f, 'real runtime parameters')
    int_scalars = read_parameters(self.h5f, 'integer scalars')
    real_scalars = read_parameters(self.h5f, 'real scalars')
    self.params = pytools.merge_dicts(int_params, real_params, int_scalars, real_scalars)

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

    print("{} read!".format(filename))
    print("simulation time: {}".format(self.params['time']))
    print("size           : {}".format(self.params['dims']))
    print("domain         : {}".format(self.params['L']))

  def new_dataset(self, dataname, save_mem=False) :
    if dataname == 'dens' :  # for density field              => create a density data set
      print("attempting to create a density dataset...")
      return self.DensityDataset(self)
    else :
      try :         # see if the dataname exists              => create a scalar data set
        print("attempting to create a scalar dataset: {}...".format(dataname))
        return self.ScalarDataset(self, dataname)
      except :
        try :       # see if the dataname+"x","y","z" exists  => create a vector data set
          print("attempting to create a vector dataset: {}...".format(dataname))
          return self.VectorDataset(self, dataname, save_mem)
        except :    # if the dataname does not exist within the hdf5 file => do nothing
          print("dataname not recognised")

  """
  ================================================================================
  CLASS H5file.ScalarDataset(object)

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

      Dataset.ft (complex[k_x, k_y, k_z])     <== Dataset.calc_ft()
          the three-dimensional complex fourier field of .data
      Dataset.ps_3D (float[k_x, k_y, k_z])     <== Dataset.calc_ps()
          the three-dimensional power spectrum of .data
      Dataset.ps (np.array[k_max])            <== Dataset.calc_ps(calc_1D=True)
          the one-dimensional power spectrum of .data

  METHODS
      mean = Dataset.mean()
          returns the mean value of the data
  ================================================================================
  """
  class ScalarDataset(object) :

    def __init__(self, H5file, dataname) :
      self.H5file = H5file
      self.dataname = dataname
      self.data = sort(H5file, H5file.h5f[dataname])

    def mean(self) :
      return np.mean(self.data)

    def calc_ft(self) :
      print("performing fast fourier transform...")
      rms_field = np.sqrt(np.average(self.data**2))

      self.ft = np.fft.fftshift(np.fft.fftn(self.data)) / np.product(self.H5file.params['dims'])
      print("... fft completed!")

      sum_power = np.sum(np.abs(self.ft)**2)
      print("sum_power        : {}".format(sum_power))
      print("rms_squared_field: {}".format(rms_field**2))

    def calc_ps(self, calc_1D=True) :
      try :
        self.ft
        print("fourier field found!")
      except AttributeError :
        print("fourier field not found!")
        self.calc_ft()

      print("calculating the power spectrum...")
      ps_3D = np.abs(self.ft)**2
      sum_power = np.sum(ft_squared)

      dims = np.array(self.H5file.params['dims'])
      nx, ny, nz = dims
      L = np.array(self.H5file.params['L'])
      # [-N, -(N-1), ..., -1, 0, 1, ..., N-2, N-1]
      kx_list = ( np.arange(-nx//2,nx//2,1) )/L[0]
      ky_list = ( np.arange(-ny//2,ny//2,1) )/L[1]
      kz_list = ( np.arange(-nz//2,nz//2,1) )/L[2]

      # tells the k of the corresponding element of 'power_field'
      k_x, k_y, k_z = np.meshgrid(kx_list, ky_list, kz_list, indexing="ij")
      k_3Darray = np.sqrt(k_x**2 + k_y**2 + k_z**2)

      # physical limits to the wavenumbers
      kmin = np.min(1.0/L)
      kmax = np.min(0.5*dims/L)

      # bins of wavenumbers
      # first bin = (0.5 to 1.5), second bin = (1.5 to 2.5), ...
      k_bins = np.arange(kmin, kmax, kmin) - 0.5*kmin

      # returns the index of k_bins which each k_3Darray fits inside
      bin_index = np.digitize(k_3Darray.flat, k_bins)

      self.ps_3D = ps_3D
      self.k = k_bins
      print("three-dimensional power spectrum saved!")

      if calc_1D :
        ps_1D = np.zeros(len(k_bins))

        print("summing the power spectrum...")
        for i in range(1,len(ps_1D)) :
            ps_1D[i-1] = np.sum(ps_3D.flat[bin_index==i])
            if i%20 == 0 :
                print("power spectrum summed until k= {} out of {}".format(i, len(ps_1D)))
        print("completed!")
        print("sum_powerspec    : {}".format(np.sum(ps_1D)))

        self.ps = ps_1D

  """
  ================================================================================
  CLASS H5file.DensityDataset(H5file.ScalarDataset)

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
  class DensityDataset(ScalarDataset) :

    def __init__(self, H5file) :
      super(H5file.DensityDataset,self).__init__(H5file, 'dens')

    def set_log(self) :
      self.data = np.log(self.data/self.mean())

    def set_delta(self) :
      self.set_log()
      self.data = self.data - self.mean()

  """
  ================================================================================
  CLASS H5file.VectorDataset(H5file, dataname)

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
  class VectorDataset(object) :

    def __init__(self, H5file, dataname, save_mem=False) :
      self.H5file = H5file
      self.dataname = dataname
      self.save_mem = save_mem
      if save_mem :
        self.datax = None
        self.datay = None
        self.dataz = None
      else :
        self.datax = sort(H5file, H5file.h5f[dataname+'x'])
        self.datay = sort(H5file, H5file.h5f[dataname+'y'])
        self.dataz = sort(H5file, H5file.h5f[dataname+'z'])

    def calc_ps(self, calc_1D=True) :

      print("calculating the power spectrum...")
      dims = np.array(self.H5file.params['dims'])
      nx, ny, nz = dims

      ps_3D = np.zeros(dims)
      ms_field = 0.0
      gc.collect()

      list_comp = [self.datax, self.datay, self.dataz]
      list_name = [self.dataname+s for s in ['x','y','z']]
      for comp, comp_name in zip(list_comp, list_name) :

        if comp is None :
          print("loading the field...")
          comp = sort(self.H5file, self.H5file.h5f[comp_name])
          print("field loaded!")
        else :
          print("field found within the dataset!")

        ms_field += np.average(comp**2)

        print("performing fast fourier transform...")
        FF = np.fft.fftshift(np.fft.fftn(comp)) / np.product(dims)

        ps_3D += np.abs(FF)**2
        print("completed!")

        comp = None
        gc.collect()

      sum_power = np.sum(ps_3D)
      print("sum_power        : {}".format(sum_power))
      print("rms_squared_field: {}".format(ms_field))

      L = np.array(self.H5file.params['L'])
      # [-N, -(N-1), ..., -1, 0, 1, ..., N-2, N-1]
      kx_list = ( np.arange(-nx//2,nx//2,1) )/L[0]
      ky_list = ( np.arange(-ny//2,ny//2,1) )/L[1]
      kz_list = ( np.arange(-nz//2,nz//2,1) )/L[2]

      # tells the k of the corresponding element of 'power_field'
      k_x, k_y, k_z = np.meshgrid(kx_list, ky_list, kz_list, indexing="ij")
      k_3Darray = np.sqrt(k_x**2 + k_y**2 + k_z**2)

      # physical limits to the wavenumbers
      kmin = np.min(1.0/L)
      kmax = np.min(0.5*dims/L)

      # bins of wavenumbers
      # first bin = (0.5 to 1.5), second bin = (1.5 to 2.5), ...
      k_bins = np.arange(kmin, kmax, kmin) - 0.5*kmin

      # returns the index of k_bins which each k_3Darray fits inside
      bin_index = np.digitize(k_3Darray.flat, k_bins)

      self.ps3D = ps_3D
      self.k = k_bins
      print("three-dimensional power spectrum saved!")

      if calc_1D :
        ps_1D = np.zeros(len(k_bins))

        print("summing the power spectrum...")
        for i in range(1,len(ps_1D)) :
            ps_1D[i-1] = np.sum(ps_3D.flat[bin_index==i])
            if i%20 == 0 :
                print("power spectrum summed until k= {} out of {}".format(i, len(ps_1D)))
        print("completed!")
        print("sum_powerspec    : {}".format(np.sum(ps_1D)))

        self.ps = ps_1D
