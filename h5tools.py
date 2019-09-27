"""
MODULE h5tools

DESCRIPTION
    contains a list of helper functions that utilises h5py
    to read FLASH hdf5 output files

AUTHOR
    Donghee NAM, 2019
===============================================================================
"""
# DEPENDENCIES
import h5py
import numpy as np


"""
FUNCTION dict = h5tools.read_integer_parameters, read_integer_scalars,
                        read_real_parameters, read_real_scalars,

DESCRIPTION
    reads the parameters/scalars from a FLASH file and returns them as a dictionary

INPUTS
    file (string)
        the name of the hdf5 file
    file (h5py.File)
        the h5py file object of the hdf5 file
(the type should be given with the isFilename parameter below;
it is assumed to be a h5py object by default)

PARAMETERS
    isFilename (boolean) = False
        if false, file is assumed to be a h5py object
        if true, file is assumed to be the filename for the FLASH file
    verbose (boolean) = False
        if True then the program will print the information

OUTPUTS
    dict (dictionary)
        the dictionary containing the parameters/scalars
        the values can be accessed via their names e.g. dict['nxb']
"""
def read_integer_parameters(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  params = dict(h5f['integer runtime parameters'])

  # strip the spaces for easier access
  keys_stripped = [keys.strip() for keys in params.keys()]
  for newkey, oldkey in zip(keys_stripped, params.keys()) :
    params[newkey] = params.pop(oldkey)

  if verbose : print("integer parameters read")
  return params

def read_integer_scalars(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  params = dict(h5f['integer scalars'])

  # strip the spaces for easier access
  keys_stripped = [keys.strip() for keys in params.keys()]
  for newkey, oldkey in zip(keys_stripped, params.keys()) :
    params[newkey] = params.pop(oldkey)

  if verbose : print("integer scalars read")
  return params

def read_real_parameters(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  params = dict(h5f['real runtime parameters'])

  # strip the spaces for easier access
  keys_stripped = [keys.strip() for keys in params.keys()]
  for newkey, oldkey in zip(keys_stripped, params.keys()) :
    params[newkey] = params.pop(oldkey)

  if verbose : print("real parameters read")
  return params

def read_real_scalars(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  params = dict(h5f['real scalars'])

  # strip the spaces for easier access
  keys_stripped = [keys.strip() for keys in params.keys()]
  for newkey, oldkey in zip(keys_stripped, params.keys()) :
    params[newkey] = params.pop(oldkey)

  if verbose : print("real scalars read")
  return params

"""
FUNCTION block_sizes = h5tools.read_block_sizes(file, isFilename=False, verbose=False)

DESCRIPTION
    reads the box sizes from a FLASH file and returns them as an array

INPUTS
    file (string or h5py.File)
        see above

PARAMETERS
    isFilename=False, verbose=False
        see above

OUTPUTS
    block_sizes (float[:,3])
        the NumPy array containing the box coordinates
"""
def read_block_sizes(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  block_sizes = np.array(h5f['block size'])

  if verbose : print("block sizes read")
  return block_sizes

"""
FUNCTION box_coords = h5tools.read_box_coordinates(file, isFilename=False, verbose=False)

DESCRIPTION
    reads the box coordinates from a FLASH file and returns them as an array

INPUTS
    file (string or h5py.File)
        see above

PARAMETERS
    isFilename=False, verbose=False
        see above

OUTPUTS
    box_coords (float[:,3])
        the NumPy array containing the box coordinates
"""
def read_box_coordinates(file, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  # the parameters
  box_coords = np.array(h5f['coordinates'])

  if verbose : print("box coordinates read")
  return box_coords

"""
FUNCTION field = h5tools.read_flash(file, field_name, isFilename=False, verbose=False)

DESCRIPTION
    reads a HDF5 file and return the specified 3D field

INPUTS
    file (string)
        the name of the hdf5 file
    file (h5py.File)
        the h5py file object of the hdf5 file
(the type should be given with the isFilename parameter below;
it is assumed to be a h5py object by default)
    field_name (string)
        the name of the field as specified in the hdf5 file

PARAMETERS
    isFilename (boolean) = False
        if false, file is assumed to be a h5py object
        if true, file is assumed to be the filename for the FLASH file
    verbose (boolean) = False
        if True then the program will print the information

OUTPUTS
    field (float[:, :, :])
        the 3D numpy array containing the field information
        the dimension should equal nxb*iprocs, nyb*jprocs, nzb*kprocs
"""
def read_flash(file, field_name, isFilename=False, verbose=False) :

  if isFilename :
    if verbose : print("opening the hdf5 file...")
    h5f = h5py.File(file,'r')
  else :
    h5f = file

  if verbose : print("reading {} from the file...".format(field_name))

  # read the key parameters
  int_param = read_integer_scalars(h5f)
  nxb = int_param['nxb']
  nyb = int_param['nyb']
  nzb = int_param['nzb']
  iprocs = int_param['iprocs']
  jprocs = int_param['jprocs']
  kprocs = int_param['kprocs']
  dims = np.array([nxb*iprocs, nyb*jprocs, nzb*kprocs])

  real_param = read_real_parameters(h5f)
  xmin = real_param['xmin']
  xmax = real_param['xmax']
  ymin = real_param['ymin']
  ymax = real_param['ymax']
  zmin = real_param['zmin']
  zmax = real_param['zmax']
  L_bl = np.array([xmin, ymin, zmin])
  L_tr = np.array([xmax, ymax, zmax])
  L = L_tr - L_bl
  L_cell = L / dims

  block_sizes = read_block_sizes(h5f)
  box_coords = read_box_coordinates(h5f)

  # the raw field array
  # it has structure of [nProcs, nzb, nyb, nxb]
  # so transposing is necessary to have shape of [nProcs, nxb, nyb, nzb]
  if verbose : print("loading {} ...".format(field_name))
  field_raw = np.transpose(np.array(h5f[field_name]), (0,3,2,1))
  if verbose : print("{} loaded!".format(field_name))

  # place the boxes in the 3D array
  field_sorted = np.zeros(dims)

  if verbose : print("sorting commenced ...")
  for proc_id in range(iprocs*jprocs*kprocs) :

    # bottom left corner
    box_loc_bl = box_coords[proc_id] - block_sizes[proc_id]/2
    index_bl = np.rint( (box_loc_bl - L_bl)/L_cell ).astype(int)

    # top right corner
    box_loc_tr = box_coords[proc_id] + block_sizes[proc_id]/2
    index_tr = np.rint( (box_loc_tr - L_bl)/L_cell ).astype(int)

    if verbose : print("proc_id: {}  indices from {} to {}".format(proc_id, index_bl, index_tr))

    # insert the block
    field_sorted[index_bl[0]:index_tr[0], index_bl[1]:index_tr[1], index_bl[2]:index_tr[2]] \
                = field_raw[proc_id]

  if verbose : print("sorting completed!")

  return field_sorted
