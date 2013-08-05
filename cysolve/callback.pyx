
cimport cython
from cython cimport view
from cython_gsl cimport GSL_SUCCESS
from cysolve.ode cimport ODE, copy_array


cimport numpy as np
import numpy as np

from cpython cimport PyObject

cdef struct CINFO:
    int dim
    PyObject * callback
    
    

cdef view.array yview

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _callback_c_ode(double t, double y[], double f[], void * cinfo) nogil:
    cdef CINFO p  = (<CINFO *> cinfo)[0]
    cdef double[::1] retV
    with gil:
        yview = <double[:2]> y
        retV = (<object> p.callback)(t, yview)
    copy_array(<double *> &retV[0], f, p.dim)
    return GSL_SUCCESS


cdef class ODECallback(ODE):
    """
    Use the cysolve.ode methods with an ode defined by a simple python function f(t, y).
    
    Instantiate as::
        
        ode = ODECallback(dim, y, callback, t=0.)
    
    Integrate::
        
        Yvals = ode.integrate(tf, Nsteps)

    """
    
    cdef public object callback
    cdef CINFO cinfo
    
    def __cinit__(self, int dim, np.ndarray y, object callback, double t = 0.):
       
        self.dim = dim
        self.y = y
        self.t = t
        
        self.c_ode = _callback_c_ode
        
        # this ensures that the callback object
        # as well as the CINFO object are not garbage-collected
        self.cinfo.dim = dim
        self.cinfo.callback = <PyObject*> callback
        self.callback = callback

        self.c_params = <void*> &self.cinfo
        
        
