#cython: embedsignature=True

cimport numpy as np
import numpy as np

cimport cython
import cython

cdef inline int imin(int a, int b) nogil: 
    return a if a <= b else b

cdef inline int imax(int a, int b) nogil: 
    return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copy_array(double * from_p, double * to_p, int Np) nogil:
    cdef int kk
    for kk in range(Np):
        to_p[kk] = from_p[kk]
        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double normsq(double * x, int N, int dx) nogil:
    cdef double ret = 0.
    cdef int kk
    for kk in range(N):
        ret += x[kk * dx] * x[kk * dx]
    return ret


cdef int c_integrate_ode(double tf, int N, int dim, double y0[], double * Y, int (*ode)(double, double [], double [], void *) nogil, void * params, double t0=*) nogil

cdef int c_ode_steady_state(double T_max, double tol, int dim, double y0[], double * yss, int (*ode)(double, double [], double [], void *) nogil, void * params) nogil

cdef np.ndarray c_eval_ode(double t, np.ndarray[np.float64_t, ndim=1] y, void * params, int dim, int (*ode)(double, double [], double [], void *))

cdef int _c_ode_ODE(double t, double * y, double * f, void * params) nogil


cdef class ODE:
    cdef public int dim
    cdef public double t
    cdef public np.ndarray y
    cdef int c_ode(ODE self, double t, double * y, double * f) nogil
    cpdef np.ndarray integrate(self, double delta_t, int N, int include_initial=*, int update_state=*)
    cpdef np.ndarray steady_state(self, double T_max, double tol)
    cpdef np.ndarray evaluate(self)


cdef class ODEs:
    cdef np.ndarray odes
    cpdef object integrate(self, double delta_t, int N, int include_initial=*, int update_state=*)
    cpdef object steady_state(self, double T_max, double tol)
    cpdef np.ndarray evaluate(self)