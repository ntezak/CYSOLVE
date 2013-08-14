#cython: embedsignature=True

cimport cython
import numpy

cimport numpy as np
import numpy as np

from cython_gsl cimport (
    gsl_odeiv2_system, GSL_SUCCESS, gsl_odeiv2_driver, gsl_odeiv2_driver_alloc_y_new, gsl_odeiv2_driver_apply,
    gsl_odeiv2_driver_free, gsl_odeiv2_step_rk8pd, gsl_odeiv2_driver_apply_fixed_step
)
from libc.math cimport floor as mfloor, lround as mlround, sqrt, sin, cos
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
from cpython cimport PyObject


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
    
    # set up GSL ode system
    cdef gsl_odeiv2_system sys
    sys.function = ode
    sys.dimension = dim
    sys.params = params
    
    # set up driver
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

    # integrate till T_max or convergence
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
        
        # compute squared norms of f=\dot{y} and y
        fn = normsq(f, dim, 1)
        yn = normsq(y, dim, 1)
        
        # if ratio of norms < tol, converged!
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
    
    
cdef int _c_ode_ODE(double t, double * y, double * f, void * params) nogil:
    """
    Wrap the ODE.ode function in a suitable way to be passed to c_integrate_ode.
    """
    return (<ODE>(<PyObject *> params)[0]).c_ode(t, y, f)


cdef class ODE:
    
    """
    Convenience class to set up a single ODE in an object-oriented fashion.

    The current state is stored in self.y
    The current time is stored in self.t
    """
    
    cpdef np.ndarray integrate(self, double delta_t, int N, int include_initial=0, int update_state=1):

        cdef double[::1] YView
        cdef double[::1] yview = self.y
        cdef np.ndarray Y
        cdef int success
                
        Y = np.zeros((N, self.dim))
            
        YView = Y.ravel()
        success = c_integrate_ode(self.t + delta_t, N, self.dim, <double*>&yview[0], <double*>&YView[0], _c_ode_ODE, <void *>self, self.t)
        
        if success == GSL_SUCCESS:
            
            if include_initial:
                Y = np.vstack((np.zeros((1, self.dim)), Y))
                Y[0] = self.y
                
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
        cdef int success = c_ode_steady_state(T_max, tol, self.dim, <double*>&yview[0], <double*>&yssview[0], _c_ode_ODE, <void *>self.c_params)
        
        if success == GSL_SUCCESS:
            return yss
        else:
            raise Exception("Execution error: {:d}".format(success))
    
    cpdef np.ndarray evaluate(self):
        return c_eval_ode(self.t, self.y, <void*>self, self.dim, _c_ode_ODE)
    
    
    cdef int c_ode(ODE self, double t, double * y, double * f) nogil:
        """
        Overwrite this in a concrete subclass!!
        """
        return GSL_SUCCESS
        

            
cdef class ODEs:
    """
    Convenience class to set up a whole set of ODEs that share the same ode function 
    (but can differ in current state and ode parameters) in an object-oriented fashion.
    """
    
    def __init__(self, odes):
        self.odes = np.array(odes)
        self.dim = odes[0].dim
        if not (np.array([o.dim for o in odes]) == self.dim).all():
            raise ValueError("All odes need to have the same state space dimension.")
    
        
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
        cdef long[::1] odes = self.odes
        cdef int N_odes = self.odes.shape[0]
        
        Y = np.zeros((N_odes, N, self.dim))

        
        YView = Y.ravel()
        
        cdef:
            np.ndarray success = np.zeros(self.N_odes, dtype=int)
            long[:] successview = success
            int current_index
            int strd = N * self.dim
            double t
            
        for current_index in prange(N_odes, nogil=True):
            t = (<ODE>(<PyObject *>odes[current_index])).t
            successview[traj_index] = c_integrate_ode(t + delta_t, N, self.dim, <double*>&yview[current_index,0], <double*>&YView[current_index * strd], _c_ode_ODE, <void*>odes[current_index], t)
        
        if include_initial:
            Y = np.concatenate((self.y.reshape(N_odes, 1, -1), Y), axis=1)

        if update_state:
            pass

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
            
            long[::1] odes = self.odes
            int N_odes = self.odes.shape[0]
            
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
            
            int current_index
            
        
        for current_index in prange(N_odes, nogil=True):
            successview[current_index] = c_ode_steady_state(T_max, tol, self.dim, <double*>&yview[current_index, 0], <double*>&yssview[current_index, 0], _c_ode_ODE, <void*>odes[current_index])
        
        return yss, success
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray evaluate(self):
        """
        Evaluate all ODEs given the current states at the current time and return \dot{y}=f(y,t).
        """

        cdef int N_odes = self.odes.shape[0]
        cdef np.ndarray Y = np.zeros((N_odes, self.dim))
        cdef double[::1] YView = Y.ravel()
        cdef double[:,::1] yview = self.y
        cdef long[::1] odes = self.odes
        cdef int current_index
            
        for current_index in prange(N_odes, nogil=True):
            c_ode(current_ode.t, <double*>&yview[current_index,0], <double*>&YView[current_index * self.dim], <void*>odes[current_index])
        return Y
        