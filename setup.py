
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import cython_gsl

ext_modules = [
    Extension("cysolve.ode",
              ["cysolve/ode.pyx"],
              include_dirs=[np.get_include()],
              libraries=cython_gsl.get_libraries(),
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp']
              ),
    Extension("cysolve.callback",
              ["cysolve/callback.pyx"],
              include_dirs=[np.get_include()],
              libraries=cython_gsl.get_libraries(),
#              extra_compile_args=['-fopenmp'],
#              extra_link_args=['-fopenmp']
              ),

]

setup(
    name='CYSOLVE',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)