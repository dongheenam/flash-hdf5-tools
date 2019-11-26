#!/usr/bin/env python3

# DEPENDENCIES
import os
import gc  # garbage collection
from datetime import datetime  # time measurement

import h5py  # hdf5 handling
import numpy as np
import pyfftw  # python wrapper for FFTW

import dnam_tools  # my tools for python


def read_parameters(h5f, keyword):
    """
    DESCRIPTION
      return the specified parameter group
      as a dictionary from the FLASH hdf5 file

    INPUTS
        h5f (h5py.File)
            the h5py File object containing the hdf5 file
        keyword (string)
            the keyword for the parameter group
                e.g. 'integer runtime parameters', 'real runtime parameters',
                     'integer scalars', 'real scalars'

    OUTPUTS
        dict (dictionary)
            contains the parameters, which can be accessed with their names
                e.g. dict['nxb']
    """
    # parameters
    params = dict(h5f[keyword])

    # strip the spaces for easier access
    # python3 : keys are now in bytes
    params = {x.decode("utf-8").rstrip(): v for x, v in params.items()}
    return params


def read_data(h5f, dataname):
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
    data = np.array(h5f[dataname])
    return data


def sort_field(H5file, data_unsorted):
    """
    DESCRIPTION
        by default the FLASH file prints the field in [nProcs, nzb, nyb, nxb]
        this method is returns the field in [N_x, N_y, N_z] dimensions
        using the parameters stored in the H5file instance

    INPUTS
    H5file
        the H5file instance
    data_unsorted (float[nProcs, nzb, nyb, nxb])
        the raw data from the FLASH hdf5 file

    OUTPUTS
        data_sorted (float[N_x, N_y, N_z])
            the 3D NumPy array containing the sorted data
    """
    # some parameters required for the sorting
    # they are read from the parent class (H5file)
    dims = np.array(H5file.params['dims'])  # simulation dimensions
    L_bl = np.array(H5file.params['Lmin'])  # bottom left corner coordinates
    L_tr = np.array(H5file.params['Lmax'])  # top right corner coordinates
    L = L_tr - L_bl  # simulation box size
    L_cell = L / dims  # size of a single cell

    block_sizes = H5file.block_sizes
    block_coords = H5file.block_coords

    # the raw field array has structure of [nProcs, nzb, nyb, nxb]
    # so transposing is necessary to have shape of [nProcs, nxb, nyb, nzb]
    data_unsorted = np.transpose(data_unsorted, (0, 3, 2, 1))

    # first create the space for the entire data
    data_sorted = np.zeros(dims)

    # loop over each block
    iprocs = H5file.params['iprocs']
    jprocs = H5file.params['jprocs']
    kprocs = H5file.params['kprocs']
    for block_id in range(iprocs * jprocs * kprocs):
        # bottom left corner
        block_bl_loc = block_coords[block_id] - block_sizes[block_id] / 2
        block_bl_ind = np.rint((block_bl_loc - L_bl) / L_cell).astype(int)
        # top right corner
        block_tr_loc = block_coords[block_id] + block_sizes[block_id] / 2
        block_tr_ind = np.rint((block_tr_loc - L_bl) / L_cell).astype(int)

        # insert the block
        data_sorted[block_bl_ind[0]:block_tr_ind[0], \
        block_bl_ind[1]:block_tr_ind[1], \
        block_bl_ind[2]:block_tr_ind[2]] \
            = data_unsorted[block_id]

    # update the sorted data
    return data_sorted


def save_to_hdf5(filename_out, *datas_and_names):
    """
    DESCRIPTION
        save the sets of data to a hdf5 file

    INPUTS
    filename_out
        the name of the hdf5 file that will be created
    datas_and_names (array_like[:])
        tuples of the data and its name
            e.g. [(data1, name1), (data2, name2), ... ]
    """
    h5f_out = h5py.File(filename_out, 'w')
    for data, dataname in datas_and_names:
        print(f"saving {dataname} to: {filename_out}...")
        h5f_out.create_dataset(name, data=data_to_save)
    h5f_out.close()
    print("file saved!")


def save_to_dat(filename_out, *datas_and_names):
    """
    DESCRIPTION
        save the sets of data to a .dat file

    INPUTS
    filename_out
        the name of the hdf5 file that will be created
    datas_and_names (array_like[:])
        tuples of the data and its name
            e.g. [(data1, name1), (data2, name2), ... ]
        the data must be one-dimensional
    """
    dat_out = open(filename_out, mode='w')

    # split the data and datanames
    datas = [dn[0] for dn in datas_and_names]
    datanames = [dn[1] for dn in datas_and_names]
    print(f"saving {datanames} to: {filename_out}...")
    # row and column lengths of the dat file
    n_cols = len(datanames)
    n_rows = np.min([len(data) for data in datas])

    # print the data
    dat_out.write(("{:>20s}" * n_cols + '\n').format(*tuple(datanames)))
    for i in range(n_rows):
        single_row = tuple([data[i] for data in datas])
        dat_out.write(("{:20.11E}" * n_cols + '\n').format(*single_row))

    dat_out.close()
    print("file saved!")


class H5File(object):
    """
    ================================================================================
    DESCRIPTION
        container for a FLASH hdf5 file and its parameters

    INPUTS
        filename (string)
            the name of the hdf5 file

    VARIABLES
        H5File.filename (string)
            the name of the hdf5 file
        H5File.h5f (h5py.File)
            the h5py File object
        H5File.params (dictionary)
            the integer/real parameters/scalars from the file
        H5File.block_sizes (array_like[nprocs, 3])
            the size of each block
        H5File.block_coords (array_like[nprocs, 3])
            the coordinates of the centre of each block

    METHODS
        H5file.new_dataset(dataname)
            returns a scalar or vector dataset instance, depending on the data name
    ================================================================================
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.h5f = h5py.File(filename, 'r')

        # read the parameters
        int_params = read_parameters(self.h5f, 'integer runtime parameters')
        real_params = read_parameters(self.h5f, 'real runtime parameters')
        int_scalars = read_parameters(self.h5f, 'integer scalars')
        real_scalars = read_parameters(self.h5f, 'real scalars')
        self.params = dnam_tools.merge_dicts(
            int_params, real_params, int_scalars, real_scalars)

        # calculate basic properties
        # number of cells in each direction
        self.params['dims'] = [self.params['nxb'] * self.params['iprocs'],
                               self.params['nyb'] * self.params['jprocs'],
                               self.params['nzb'] * self.params['kprocs']]
        # leftmost, rightmost coordinates of the simulation
        Lmin = [self.params['xmin'], self.params['ymin'], self.params['zmin']]
        Lmax = [self.params['xmax'], self.params['ymax'], self.params['zmax']]
        self.params['Lmin'] = Lmin
        self.params['Lmax'] = Lmax
        # box size of the simulation
        self.params['L'] = [max - min for max, min in zip(Lmax, Lmin)]

        self.block_sizes = read_data(self.h5f, 'block size')
        self.block_coords = read_data(self.h5f, 'coordinates')

        if verbose :
            print(f"{filename} read!")
            print(f"simulation time: {self.params['time']:.4E}")
            print(f"resolution     : {self.params['dims']}")
            print(f"domain         : {self.params['L']}")

    def new_dataset(self, dataname, small_mem=False):
        if dataname == 'dens':
            d = self.DensityDataset(self, small_mem)
            print("created a density dataset!")
            return d
        else:
            try:
                d = self.ScalarDataset(self, dataname, small_mem)
                print(f"created a scalar dataset for {dataname}!")
                return d
            except:
                pass
            try:
                d = self.VectorDataset(self, dataname, small_mem)
                print(f"created a vector dataset for {dataname}!")
                return d
            except:
                print("dataname not recognised!")
                return

    class Dataset(object):
        """
        ================================================================================
        DESCRIPTION
            container for a dataset

        INPUTS
            H5File
                the outer class H5File
            dataname(string)
                the name of the dataset

            small_mem(boolean) (default=False)
                set it to True if dataset is too big

        VARIABLES
            Dataset.H5File
                the outer class
            Dataset.dataname (string)
                the name of the data

            Dataset.ps (array_like[k_max])            <== Dataset.calc_ps(calc_1D=True)
                the one-dimensional power spectrum of .data

        METHODS
            Dataset.calc_ps(self, save=False, save_path=None)
                calculate the 1D spectrum
                requires the fourier spectrum (self.ft) to exist
        ================================================================================
        """

        def __init__(self, H5File, dataname, small_mem=False):
            self.H5File = H5File
            self.dataname = dataname
            self.small_mem = small_mem

        def calc_ps(self, save=False, filename_out=None):
            t_start = datetime.now()  # count the time
            small_mem = self.small_mem

            try:
                self.ps_3D
                print("3D power spectrum found!")
            except AttributeError:
                print("3D power spectrum not found!")
                self.calc_ps_3D()

            print("calculating the power spectrum...")
            sum_power = np.sum(self.ps_3D)

            # some physical properties
            dims = np.array(self.H5File.params['dims'])
            nx, ny, nz = dims
            L = np.array(self.H5File.params['L'])

            # calculate the wavenumbers
            # [-N, -(N-1), ..., -1, 0, 1, ..., N-2, N-1]
            kx_list = (np.arange(-nx // 2, nx // 2, 1)) / L[0]
            ky_list = (np.arange(-ny // 2, ny // 2, 1)) / L[1]
            kz_list = (np.arange(-nz // 2, nz // 2, 1)) / L[2]

            # the wavenumber array
            k_x, k_y, k_z = np.meshgrid(kx_list, ky_list, kz_list, indexing="ij")
            k = np.sqrt(k_x ** 2 + k_y ** 2 + k_z ** 2)

            # physical limits to the wavenumbers
            kmin = np.min(1.0 / L)
            kmax = np.min(0.5 * dims / L)

            # bins of wavenumbers
            # first bin = (0.5 to 1.5), second bin = (1.5 to 2.5), ...
            k_bins = np.arange(kmin, kmax, kmin) - 0.5 * kmin

            # holder for the power spectrum
            ps_1D = np.zeros(len(k_bins))

            # sorting the power spectrum in the increasing order of wavenumber
            sorting_ind = np.argsort(k.flat)
            k_sorted = k.flat[sorting_ind]
            if small_mem:
                k = None
            ps_sorted = self.ps_3D.flat[sorting_ind]
            if small_mem:
                self.ps_3D = None
            sorting_ind = None
            gc.collect()

            # determine the location of the bins
            loc_bins = np.searchsorted(k_sorted, k_bins, sorter=None)
            # sum the power spectrum under the bins
            ps_1D = [np.sum(ps_sorted[loc_bins[i]:loc_bins[i + 1]]) for i in range(len(k_bins) - 1)]

            self.ps = ps_1D
            self.k = k_bins[:-1] / kmin + 0.5

            t_end = datetime.now()
            delta = t_end - t_start
            print("completed!")
            print("time taken: {}".format(delta.total_seconds()))

            if save is True:
                dataname_ps = self.dataname + "ps"
                if filename_out is None:
                    filename_out = self.H5File.filename + f"_{dataname_ps}.dat"
                save_to_dat(filename_out,
                            (self.k, 'k'), (self.ps, dataname_ps))
        # end def calc_ps

    class ScalarDataset(Dataset):
        """
        ================================================================================
        DESCRIPTION
            container for a scalar field

        INPUTS
            H5File
                the outer class H5File

        VARIABLES
            Dataset.H5File
                the outer class
            Dataset.data (np.array[N_x,N_y,N_z])
                the raw field values
            Dataset.dataname (string)
                the key to access the field in the hdf5 field

            Dataset.ps_3D (float[k_x, k_y, k_z])     <== Dataset.calc_ps()
                the three-dimensional power spectrum of .data
            Dataset.ps (np.array[k_max])            <== Dataset.calc_ps(calc_1D=True)
                the one-dimensional power spectrum of .data

        METHODS
            Dataset.calc_proj(axis={"x", "y", "z"})
                replace the data with its projection
        ================================================================================
        """

        def __init__(self, H5File, dataname, small_mem=False):
            super().__init__(H5File, dataname, small_mem=small_mem)
            self.data = sort_field(H5File, H5File.h5f[dataname])

        def calc_proj(self, axis):
            if axis == 'x':
                axis_no = 0
            elif axis == 'y':
                axis_no = 1
            elif axis == 'z':
                axis_no = 2
            else:
                print("axis input not set up properly!")
                return
            self.data = np.sum(self.data, axis=axis_no)

        def calc_ps_3D(self, save=False, filename_out=None):
            t_start = datetime.now()
            small_mem = self.small_mem

            print("performing fast fourier transform...")
            rms_field = np.sqrt(np.average(self.data ** 2))

            fft_object = pyfftw.builders.fftn(self.data, threads=16)
            self.ft = np.fft.fftshift(fft_object()) / np.product(self.H5File.params['dims'])

            print("... fft completed!")
            if small_mem is True:
                self.data = None

            self.ps_3D = np.abs(self.ft) ** 2
            if small_mem is True:
                self.ft = None

            sum_power = np.sum(self.ps_3D)
            print("sum_power        : {}".format(sum_power))
            print("rms_squared_field: {}".format(rms_field ** 2))

            t_end = datetime.now()
            delta = t_end - t_start
            print("time taken: {}".format(delta.total_seconds()))

            if save is True:
                dataname_ps = self.dataname + "ps3D"
                if filename_out is None:
                    filename_out = self.H5File.filename + f"_{dataname_ps}.hdf5"
                save_to_hdf5(filename_out, (self.ps3D, dataname_ps))

    class DensityDataset(ScalarDataset):
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

        def __init__(self, H5file, small_mem=True):
            super().__init__(H5file, 'dens', small_mem=small_mem)

        def set_log(self):
            self.data = np.log(self.data / np.mean(self.data))

        def set_delta(self):
            self.set_log()
            self.data = self.data - np.mean(self.data)

    class VectorDataset(Dataset):
        """
        ================================================================================
        DESCRIPTION
            container for a vector field of data

        INPUTS
            H5File
                the outer class H5File
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

        def __init__(self, H5File, dataname, small_mem=False):
            super().__init__(H5File, dataname, small_mem=small_mem)
            if small_mem:
                self.datax = None
                self.datay = None
                self.dataz = None
            else:
                self.datax = sort_field(H5File, H5File.h5f[dataname + 'x'])
                self.datay = sort_field(H5File, H5File.h5f[dataname + 'y'])
                self.dataz = sort_field(H5File, H5File.h5f[dataname + 'z'])

        def calc_ps_3D(self, save=False, filename_out=None):
            t_start = datetime.now()

            dims = np.array(self.H5File.params['dims'])
            ps_3D = np.zeros(dims)
            ms_field = 0.0
            gc.collect()

            # the vector components and their names
            datas = [self.datax, self.datay, self.dataz]
            datanames = [self.dataname + s for s in ['x', 'y', 'z']]
            for data, dataname in zip(datas, datanames):
                # begin loop over components
                if data is None:
                    print("loading the field...")
                    data = sort_field(self.H5File, self.H5File.h5f[dataname])
                    print("field loaded!")
                else:
                    print("field found within the dataset!")

                ms_field += np.average(data ** 2)

                print("performing fast fourier transform...")
                fft_object = pyfftw.builders.fftn(data, threads=16)
                FF = np.fft.fftshift(fft_object()) / np.product(dims)

                ps_3D += np.abs(FF) ** 2
                print("completed!")

                data = None
                gc.collect()
            # end loop over components

            self.ps_3D = ps_3D

            t_end = datetime.now()
            delta = t_end - t_start
            print("time taken: {}".format(delta.total_seconds()))

            if save is True:
                dataname_ps = self.dataname + "ps3D"
                if filename_out is None:
                    filename_out = self.H5File.filename + f"_{dataname_ps}.hdf5"
                save_to_hdf5(filename_out, (self.ps3D, dataname_ps))
        # end def calc_ps_3D


class PartFile(object):
    """
    ================================================================================
    DESCRIPTION
        container for a FLASH particle file

    INPUTS
        filename (string)
            the name of the hdf5 particle file

    VARIABLES
        PartFile.filename (string)
            the name of the hdf5 particle file
        PartFile.h5f (h5py.File)
            the h5py File object
        PartFile.params (dictionary)
            the integer/real parameters/scalars from the file

    METHODS

    ================================================================================
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.h5f = h5py.File(filename, 'r')

        int_params = read_parameters(self.h5f, 'integer runtime parameters')
        real_params = read_parameters(self.h5f, 'real runtime parameters')
        int_scalars = read_parameters(self.h5f, 'integer scalars')
        real_scalars = read_parameters(self.h5f, 'real scalars')
        self.params = dnam_tools.merge_dicts(int_params, real_params, int_scalars, real_scalars)

        # read the particles (if exists)
        try:
            self.particles = self.read_parts()
        except KeyError:
            self.particles = None

        # read the parameters
        int_params = read_parameters(self.h5f, 'integer runtime parameters')
        real_params = read_parameters(self.h5f, 'real runtime parameters')
        int_scalars = read_parameters(self.h5f, 'integer scalars')
        real_scalars = read_parameters(self.h5f, 'real scalars')
        self.params = dnam_tools.merge_dicts(
            int_params, real_params, int_scalars, real_scalars)

        # calculate basic properties
        # number of cells in each direction
        self.params['dims'] = [self.params['nxb'] * self.params['iprocs'],
                               self.params['nyb'] * self.params['jprocs'],
                               self.params['nzb'] * self.params['kprocs']]
        # leftmost, rightmost coordinates of the simulation
        Lmin = [self.params['xmin'], self.params['ymin'], self.params['zmin']]
        Lmax = [self.params['xmax'], self.params['ymax'], self.params['zmax']]
        self.params['Lmin'] = Lmin
        self.params['Lmax'] = Lmax
        # box size of the simulation
        self.params['L'] = [max - min for max, min in zip(Lmax, Lmin)]

        if verbose :
            print(f"{filename} read!")
            print(f"simulation time: {self.params['time']:.4E}")
            print(f"resolution     : {self.params['dims']}")
            print(f"domain         : {self.params['L']}")
            n_parts = 0 if self.particles is None else len(self.particles)
            print(f"No. of sinks   : {n_parts}")
            total_mass = 0 if self.particles is None else np.sum(self.particles['mass']) / 1.989e33
            print(f"Mass of sinks  : {total_mass} solar masses")

    def read_parts(self):
        # names of the particle parameters
        pnames = self.h5f['particle names']
        dtype = [(pname[0].decode("utf-8").rstrip(), 'f8') for pname in pnames]

        particles_raw = [tuple(p) for p in self.h5f['tracer particles']]
        particles = np.array(particles_raw, dtype=dtype)
        return particles

class DatFile(object):
    """
    ================================================================================
    DESCRIPTION
        container for a Turb.dat file

    INPUTS
        filename (string)
            the name of .dat file

    VARIABLES
        PartFile.filename (string)
            the name of the .dat file

    METHODS

    ================================================================================
    """
    def __init__(self, filename, verbose=True) :
        self.filename = filename
        self.data = np.genfromtxt(filename, names=True)

        if verbose :
            print(f"data file {filename} read!")

        # strip the first three letters of the keys
        keys = self.data.dtype.names
        keys = [s[3:] for s in keys]
        self.keys = keys
        self.data.dtype.names = keys


if __name__ == "__main__":
    """
    ================================================================================
    Predefined macros
    ================================================================================
    """
    # DEPENDENCIES
    import argparse
    import numpy as np

    import dnam_tools

    parser = argparse.ArgumentParser(description='predefined macros for common tasks.')
    parser.add_argument('filename', type=str, nargs='?', default=None,
                        help='the name to the file to be inspected')

    args = parser.parse_args()

    # if the tool is used with a specific filename
    if args.filename is not None :
        try :
            H5File(args.filename)
        except KeyError:
            PartFile(args.filename)
        finally :
            print("exiting h5tools.py...")

    # otherwise search for the last plt and part files
    else :
        last_chkfile = dnam_tools.get_file("Turb_hdf5_chk_????", loc='last')
        if last_chkfile is not None :
            H5File(last_chkfile)
        else :
            print("checkfile not found!")

        last_pltfile = dnam_tools.get_file("Turb_hdf5_plt_cnt_????", loc='last')
        if last_pltfile is not None :
            H5File(last_pltfile)
        else :
            print("plotfile not found!")

        last_partfile = dnam_tools.get_file("Turb_hdf5_part_????", loc='last')
        if last_partfile is not None :
            PartFile(last_partfile)
        else :
            print("particlefile not found!")
        print("exiting h5tools.py")
