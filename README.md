# flash-hdf5-tools
This repo contains python modules I use to analyse FLASH hydrodynamical simulation data.

## module description

### constants.py
This is a small container for physical constants (just like `astropy.constants` but without units). All values are in cgs units.

### dnam_tools.py
This contains a few handy functions (such as finding the last file that matches a regexp).

### h5tools.py
This contains wrapper classes for FLASH output files: `H5File` for plotfiles, `PartFile` for particle files, and `DatFile` for .dat output. `H5File` class provies inner class `Dataset` for accessing the fieids stored in the HDF5 file.

### runtools.py
Macro for `h5tools.py`.

### runflash.py
Performs essential operations for running FLASH simulations.

### measures.py
This module calculates essential turbulence and gravity constants (free-fall time, turbulent crossing time, ...) from a direct input or from a FLASH HDF5 file.

### mpltools.py
This module initialises the `matplotlib` library and also provides a wrapper for plot function calls on `matplotlib`.

### fancyplots.py
Macro for `mpltools.py`.

