#cython: embedsignature=True

cimport cython
cimport numpy as np
import numpy as np


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
@cython.wraparound(False)
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



    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int c_ode_steady_state(double T_max, double tol, int dim, double y0[], double yss[], int (*ode)(double, double [], double [], void *) nogil,  void * params) nogil:
    """
    Integrate an ode until either T_max or the ODE converges ||\dot{y}||^2/||y||^2 < tol. The result is stored in yss.
    """
    cdef gsl_odeiv2_system sys
    sys.function = ode
    sys.dimension = dim
    sys.params = params
    

    cdef gsl_odeiv2_driver * d
    d = gsl_odeiv2_driver_alloc_y_new(
            &sys, gsl_odeiv2_step_rk8pd,
            1e-6, 1e-6, 0.0)
    

    
    cdef int kk, ll
    cdef double t
    cdef double * y = <double *> malloc(dim * sizeof(double))
    cdef double * f = <double *> malloc(dim * sizeof(double))
    cdef double n
    cdef int converged = 0

    t = 0.
    copy_array(y0, y, dim)
    
    cdef int status
    cdef int steps = imax(100, <int>(T_max/.1))
    cdef double tkk, fn, yn

    
    for kk in range(steps):
        tkk = (kk + 1) * T_max / steps
        status = gsl_odeiv2_driver_apply (d, &t, tkk, y)
        

        if (status != GSL_SUCCESS):
            status = 21
            break

        status = ode(t, y, f, params)

        if (status != GSL_SUCCESS):
            status = 23
            break

        fn = normsq(f, dim, 1)
        yn = normsq(y, dim, 1)
        
        if sqrt(fn/yn) < tol:
            converged = 1
            status = GSL_SUCCESS
            break
    
    copy_array(y, yss, dim)
    gsl_odeiv2_driver_free(d)
    free(y)
    free(f)
    
    if converged == 0 and status == GSL_SUCCESS:
        status =  11

    return status



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
    """
    Convenience class to set up a single ODE in an object-oriented fashion.

    The current state is stored in self.y
    The current time is stored in self.t
    """
    
#    def __cinit__(self):
#        self.free_params_on_dealloc = 0
#        self.context = dict()

    cpdef np.ndarray integrate(self, double delta_t, int N, int include_initial=0, int update_state=1):

        cdef double[::1] YView
        cdef double[::1] yview = self.y
        cdef np.ndarray Y
        cdef int success
        
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
    
    cpdef np.ndarray steady_state(self, double T_max, double tol):
        cdef double[::1] yview = self.y
        cdef np.ndarray yss = np.zeros_like(self.y)
        cdef double[::1] yssview = yss
        cdef int success = c_ode_steady_state(T_max, tol, self.dim, <double*>&yview[0], <double*>&yssview[0], self.c_ode, self.c_params)
        if success == GSL_SUCCESS:
            return yss
        else:
            raise Exception("Execution error: {:d}".format(success))
    
    cpdef np.ndarray evaluate(self):
        return c_eval_ode(self.t, self.y, self.c_params, self.dim, self.c_ode)
        
#    def __dealloc__(self):
#        if self.free_params_on_dealloc and self.c_params != NULL:
#            free(self.c_params)

            
cdef class ODEs:
    """
    Convenience class to set up a whole set of ODEs that share the same ode function 
    (but can differ in current state and ode parameters) in an object-oriented fashion.
    """
    

#    def __cinit__(self):
#        self.free_params_on_dealloc = 0
#        self.context = dict()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object integrate(self, double delta_t, int N, int include_initial=0, int update_state=1):
        """
        Starting from the current states self.y and at the current time self.t integrate for a duration delta_t and
        evaluate the states at N time intervals. 
        
        If include_initial is non-zero, the returned array of states contains both the N integrated states and the initial state, 
        otherwise just the integrated states. If update_state is non-zero, after integration, self.y and self.t are updated.
        """
        cdef double[::1] YView
        cdef double[:,::1] yview = self.y
        cdef np.ndarray Y
        cdef int traj_index
        cdef int strd
        
#        if include_initial:
#            Y = np.zeros((self.N_params, N + 1, self.dim))
#            Y[:,0,:] = self.y
#        else:
        Y = np.zeros((self.N_params, N, self.dim))
        strd = N * self.dim
        
        YView = Y.ravel()
        cdef:
            double t = self.t
            int (*c_ode)(double, double [], double [], void *) nogil
            void * c_params = self.c_params
            int param_size = self.param_size
            int dim = self.dim
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
            
        c_ode = self.c_ode
            
        for traj_index in prange(self.N_params, nogil=True):
            successview[traj_index] = c_integrate_ode(t + delta_t, N, dim, <double*>&yview[traj_index,0], <double*>&YView[traj_index * strd], c_ode, c_params + param_size * traj_index, t)
        
        if update_state:
            self.y[success == GSL_SUCCESS] = Y[success == GSL_SUCCESS,-1, :]
            self.t += delta_t
            
        if include_initial:
            Y = np.concatenate((self.y.reshape(self.N_params, 1, -1), Y), axis=1)

        return Y, success

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object steady_state(self, double T_max, double tol):
        """
        Starting from the current states self.y at the current time self.t, 
        integrate all ODEs until convergence to a steady state or until T_max is reached.
        
        Returns both the final states and an array of integers indicating the successful convergence (==GSL_SUCCESS==0).
        """
        
        cdef:
            double[:, ::1] yview = self.y
            np.ndarray yss = np.zeros_like(self.y)
            double[:, ::1] yssview = yss
            int (*c_ode)(double, double [], double [], void *) nogil
            void * c_params = self.c_params
            int param_size = self.param_size
            int dim = self.dim
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
            int traj_index
            
        c_ode = self.c_ode
        for traj_index in prange(self.N_params, nogil=True):
            successview[traj_index] = c_ode_steady_state(T_max, tol, dim, <double*>&yview[traj_index, 0], <double*>&yssview[traj_index, 0], c_ode, c_params + param_size * traj_index)
        
        return yss, success
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray evaluate(self):
        """
        Evaluate all ODEs given the current states at the current time and return \dot{y}=f(y,t).
        """

        cdef double[::1] YView
        cdef double[:,::1] yview = self.y
        cdef np.ndarray Y
        cdef int traj_index
        
        Y = np.zeros((self.N_params, self.dim))
        
        YView = Y.ravel()
        cdef:
            double t = self.t
            int (*c_ode)(double, double [], double [], void *) nogil
            void * c_params = self.c_params
            int param_size = self.param_size
            int dim = self.dim
        
        c_ode = self.c_ode

            
        for traj_index in prange(self.N_params, nogil=True):
            c_ode(t, <double*>&yview[traj_index,0], <double*>&YView[traj_index * dim], c_params + param_size * traj_index)
        
        return Y
        
#    def __dealloc__(self):
#        if self.free_params_on_dealloc and self.c_params != NULL:
#            free(self.c_params)
        