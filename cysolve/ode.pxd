
cimport numpy as np
import numpy as np

cdef inline int imin(int a, int b) nogil: 
    return a if a <= b else b
cdef inline int imax(int a, int b) nogil: 
    return a if a >= b else b

cdef inline void copy_array(double * from_p, double * to_p, int Np) nogil:
    cdef int kk
    for kk in range(Np):
        to_p[kk] = from_p[kk]

cdef int c_integrate_ode(double tf, int N, int dim, double y0[], double * Y, int (*ode)(double, double [], double [], void *) nogil, void * params, double t0=*) nogil

cdef np.ndarray c_eval_ode(double t, np.ndarray[np.float64_t, ndim=1] y, void * params, int dim, int (*ode)(double, double [], double [], void *))

cdef class ODE:
    
    cdef int (*c_ode)(double, double [], double [], void *) nogil
    cdef void * c_params
    cdef public int dim
    cdef public double t
    cdef public np.ndarray y
    cdef int free_params_on_dealloc
    cpdef np.ndarray integrate(self, double delta_t, int N, int include_initial=*, int update_state=*)
    cpdef np.ndarray evaluate(self)

cdef class ODEs:
    
    cdef int (*c_ode)(double, double [], double [], void *) nogil
    cdef void * c_params
    cdef int param_size
    cdef int N_params
    cdef public int dim
    cdef public double t
    cdef public np.ndarray y
    cdef int free_params_on_dealloc
    cpdef object integrate(self, double delta_t, int N, int include_initial=*, int update_state=*)
    cpdef np.ndarray evaluate(self)
