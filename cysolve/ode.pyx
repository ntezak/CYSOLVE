
cimport cython
import numpy as np
cimport numpy as np

from cython_gsl cimport (gsl_matrix_view, gsl_matrix_view_array, gsl_matrix, gsl_matrix_set, gsl_odeiv2_system,
                            GSL_SUCCESS, gsl_odeiv2_driver, gsl_odeiv2_driver_alloc_y_new, gsl_odeiv2_driver_apply,
                            gsl_odeiv2_driver_free, gsl_odeiv2_step_rk8pd, gsl_odeiv2_driver_apply_fixed_step,
                            gsl_matrix_alloc, gsl_matrix_free
                            )

from libc.math cimport floor as mfloor, lround as mlround, sqrt, sin, cos
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange


@cython.cdivision(True)
@cython.boundscheck(False)
cdef int c_integrate_ode(double tf, int N, int dim, double y0[], double * Y, int (*ode)(double, double [], double [], void *) nogil, void * params, double t0 = 0.) nogil:
    """
    Solve the ODE given by ode, with dimension dim, parametrized by params, in a given time interval [t0, tf] for an initial condition y0 and compute the state at N steps.
    The result is stored in the array Y as [y_1[t0] ... y_dim[t0], y_2[t0 + dt], ..., y_dim[t0+dt],...].
    """
    
    # initialize GSL ode specification
    cdef gsl_odeiv2_system sys
    sys.function = ode
    sys.dimension = dim
    sys.params = params
        
    # initialize GSL ode driver
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0
        )
    
    
    cdef int kk, jj, status
    cdef double t = t0
    cdef double tkk, stepsize 
    cdef double * y = <double *> malloc(sys.dimension * sizeof(np.float64_t))
    
    # initialize current state
    copy_array(y0, y, sys.dimension)

    stepsize = (tf - t0) / N
    
    for kk in range(N):
        
        tkk = t0 + (kk + 1) * stepsize
        # integrate over [t, tkk = t + stepsize]
        status = gsl_odeiv2_driver_apply (d, &t, tkk, y)
        
        if (status != GSL_SUCCESS):
            return status
        
        # store current state in output array
        copy_array(y, &Y[sys.dimension * kk], sys.dimension)
    
    # free up resources
    gsl_odeiv2_driver_free(d)
    free(y)
    return GSL_SUCCESS



cdef np.ndarray c_eval_ode(double t, np.ndarray[np.float64_t, ndim=1] y, void * params, int dim, int (*ode)(double, double [], double [], void *)):
    """
    Evaluate and return the ODE given by ode, with dimension dim, parametrized by params, at a given time t and state y.
    """
    cdef np.ndarray ret = np.zeros(dim)
    cdef double[::1] retView = ret
    cdef double[::1] yView = y
    ode(t, <double *> &yView[0], <double *> &retView[0], params)
    return ret
    
cdef class ODE:

    def __cinit__(self):
        self.free_params_on_dealloc = 0

    cpdef np.ndarray integrate(self, double delta_t, int N, int include_initial=0, int update_state=1):

        cdef double[::1] YView
        cdef double[::1] yview = self.y
        cdef np.ndarray Y
        
        if include_initial:
            Y = np.zeros((N + 1, self.dim))
            Y[0] = self.y
        else:
            Y = np.zeros((N, self.dim))
            
        YView = Y[-N:].ravel()
        success = c_integrate_ode(self.t + delta_t, N, self.dim, <double*>&yview[0], <double*>&YView[0], self.c_ode, self.c_params, self.t)
        
        if success == GSL_SUCCESS:
            if update_state:
                self.y = Y[-1]
                self.t += delta_t
            return Y
        else:
            raise Exception('Execution error')
            
    cpdef np.ndarray evaluate(self):
        return c_eval_ode(self.t, self.y, self.c_params, self.dim, self.c_ode)
        
    def __dealloc__(self):
        if self.free_params_on_dealloc:
            free(self.c_params)
        