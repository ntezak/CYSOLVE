CYSOLVE
=======

Python/Cython code to use the GSL ode integration routines efficiently.
So far, all code was written inside the main CYSOLVE.ipynb ipython notebook file.
You can see this notebook (which also includes some usage examples) [here](http://nbviewer.ipython.org/urls/raw.github.com/ntezak/CYSOLVE/master/CYSOLVE.ipynb).

To use this package, you will need Cython (tested with version >= 18) as well as CythonGSL and the actual GSL.


To compile on OSX, you may need to set your compiler to be gcc in order to be able to use openmp:

	export CC=gcc
	python setup.py build_ext --inplace
