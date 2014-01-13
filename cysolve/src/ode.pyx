#cython: embedsignature=True

cimport cython
import cython

cimport numpy as np
import numpy as np

from cython_gsl cimport (
    GSL_SUCCESS, gsl_odeiv2_system, gsl_odeiv2_driver, gsl_odeiv2_driver_alloc_y_new, 
    gsl_odeiv2_driver_apply, gsl_odeiv2_driver_free, gsl_odeiv2_step_rk8pd, gsl_odeiv2_driver_apply_fixed_step,
    gsl_rng_env_setup, gsl_rng, gsl_rng_type, gsl_rng_free, gsl_rng_alloc, gsl_rng_default, gsl_ran_gaussian, gsl_rng_set,
    gsl_matrix_alloc, gsl_matrix_free, gsl_matrix, gsl_blas_dgemv, CblasRowMajor, CblasNoTrans, 
    gsl_vector, gsl_vector_alloc, gsl_vector_free, gsl_blas_dtrmv, gsl_linalg_cholesky_decomp, CblasLower, CblasNonUnit
)

from libc.math cimport floor as mfloor, lround as mlround, ceil as mceil, sqrt, sin, cos
from libc.stdlib cimport malloc, free

from cython.parallel cimport prange


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_integrate_ode(double tf, Py_ssize_t N, Py_ssize_t dim, double y0[], double * Y, ode_t ode, void * params, double t0 = 0.) nogil:
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
    
    
    cdef Py_ssize_t kk, jj, status
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

cdef struct SDEPARAMS:
    void * odeparams
    ode_t c_ode
    Py_ssize_t dim
    double * noises

cdef int c_wrapped_sde_ode(double t, double *y, double *f, void * params) nogil:
    cdef SDEPARAMS sdeparams = (<SDEPARAMS*>params)[0]
    cdef int success = sdeparams.c_ode(t, y, f, sdeparams.odeparams)
    cdef Py_ssize_t kk
    for kk in range(sdeparams.dim):
        f[kk] += sdeparams.noises[kk]
    return success

@cython.boundscheck(False)
cdef void sample_gaussian(gsl_rng * r, double sigma, double * w, Py_ssize_t m) nogil:
    cdef Py_ssize_t kk
    for kk in range(m):
        w[kk] = gsl_ran_gaussian(r, sigma)
        
# @cython.boundscheck(False)
# cdef int noise_from_covariance(Py_ssize_t n, double * BBT, double * b, double * w) nogil:
#     #TODO implement
#     return 0

# cdef int noise_from_matvec(double t, double * y, matvec_t matvec, double * b, double * w, void * params) nogil:
    



cdef int c_integrate_sde(double tf, double h, Py_ssize_t N, Py_ssize_t dim, Py_ssize_t m, double * y0,  double * Y, ode_t ode, noise_t noise_coeff, void *params, double t0=0, long rngseed=0) nogil:
    """
    Solve the Ito-SDE given by ode and noise_coeff, with dimension dim, m Wiener-noises, parametrized by params, in a given time interval [t0, tf] for an initial condition y0 and compute the state at N steps.
    The result is stored in the array Y as [y_1[t0] ... y_dim[t0], y_2[t0 + dt], ..., y_dim[t0+dt],...].
    """

    cdef const gsl_rng_type * T
    cdef gsl_rng * r

    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc(T)

    if rngseed != 0:
        gsl_rng_set(r, rngseed)

    
    cdef SDEPARAMS sdeparams
    sdeparams.odeparams = params
    sdeparams.dim = dim
    sdeparams.c_ode = ode
    
    
    # cdef gsl_vector * noises = gsl_vector_alloc(m)
    cdef double * noises = <double*> malloc(dim * sizeof(double))
    sdeparams.noises = noises
    
    # initialize GSL ode specification
    cdef gsl_odeiv2_system sys
    sys.function = c_wrapped_sde_ode
    sys.dimension = dim
    sys.params = <void*> &sdeparams
        
    # initialize GSL ode driver
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0
        )
    
    cdef int status
    cdef Py_ssize_t kk, jj, ll, stepspersample
    cdef double t = t0
    cdef double tkk, tjj, samplestep, sigma
    cdef double * y = <double *> malloc(sys.dimension * sizeof(np.float64_t))
    
    # if noise_coeff_is_covariance:
    #     if dim != m:
    #         return 300

    # cdef gsl_matrix * B = gsl_matrix_alloc(dim, m)
    # cdef gsl_vector * w = gsl_vector_alloc(m)

    cdef double * w = <double *> malloc(m * sizeof(double))
    # cdef double * B = <double *> malloc(m * dim * sizeof(double))
    

    # initialize current state
    copy_array(y0, y, sys.dimension)

    samplestep = (tf - t0) / N
    stepspersample = <Py_ssize_t> mceil(samplestep/h)
    h = samplestep * 1.0 / stepspersample

    sigma = 1./sqrt(h)

    for kk in range(N):
        tkk = t0 + (kk + 1) * samplestep
        for jj in range(stepspersample):

            tjj = tkk + (jj + 1) * h

            sample_gaussian(r, sigma, w, m)

            status = noise_coeff(t, y, w, noises, params)

            if status != GSL_SUCCESS:
                return 201
            
            
            # integrate over [t, tkk = t + stepsize]
            status = gsl_odeiv2_driver_apply (d, &t, tjj, y)
        
            if (status != GSL_SUCCESS):
                return status
        
        # store current state in output array
        copy_array(y, &Y[sys.dimension * kk], sys.dimension)
    
    # free up resources
    gsl_odeiv2_driver_free(d)
    free(y)
    free(noises)
    free(w)
    # gsl_vector_free(w)
    # gsl_vector_free(noises)
    # gsl_matrix_free(B)
    gsl_rng_free (r)
    return GSL_SUCCESS


cdef int c_integrate_sde_store_noises(double tf, double h, Py_ssize_t N, Py_ssize_t dim, Py_ssize_t m, double * y0,  double * Y, double * W, double * B, ode_t ode, noise_t noise_coeff, void *params, double t0=0, long rngseed=0) nogil:
    """
    Solve the Ito-SDE given by ode and noise_coeff, with dimension dim, m Wiener-noises, parametrized by params, in a given time interval [t0, tf] for an initial condition y0 and compute the state at N steps.
    The result is stored in the array Y as [y_1[t0] ... y_dim[t0], y_1[t0 + dt], ..., y_dim[t0+dt],...].
    The noise increments for every sampled time interval dB_j = B_j[t+dt] - B_j[t] are stored in the array B (which should have be initialized to 0.).
    """

    cdef const gsl_rng_type * T
    cdef gsl_rng * r

    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc(T)
    if rngseed != 0:
        gsl_rng_set(r, rngseed)

    
    cdef SDEPARAMS sdeparams
    sdeparams.odeparams = params
    sdeparams.dim = dim
    sdeparams.c_ode = ode
    
    
    # cdef gsl_vector * noises = gsl_vector_alloc(m)
    cdef double * noises = <double*> malloc(dim * sizeof(double))
    sdeparams.noises = noises
    
    # initialize GSL ode specification
    cdef gsl_odeiv2_system sys
    sys.function = c_wrapped_sde_ode
    sys.dimension = dim
    sys.params = <void*> &sdeparams
        
    # initialize GSL ode driver
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0
        )
    
    cdef int status = GSL_SUCCESS
    cdef Py_ssize_t kk, jj, ll, stepspersample
    cdef double t = t0
    cdef double tkk, tjj, samplestep, sigma
    cdef double * y = <double *> malloc(sys.dimension * sizeof(np.float64_t))
    
    # if noise_coeff_is_covariance:
    #     if dim != m:
    #         return 300

    # cdef gsl_matrix * B = gsl_matrix_alloc(dim, m)
    # cdef gsl_vector * w = gsl_vector_alloc(m)

    cdef double * w = <double *> malloc(m * sizeof(double))
    # cdef double * B = <double *> malloc(m * dim * sizeof(double))
    

    # initialize current state
    copy_array(y0, y, sys.dimension)

    samplestep = (tf - t0) / N
    stepspersample = <Py_ssize_t> mceil(samplestep/h)
    h = samplestep * 1.0 / stepspersample

    sigma = 1./sqrt(h)
    # with gil:
    #     print "starting integration"
    
    # return 0

    for kk in range(N):
        tkk = t0 + (kk + 1) * samplestep
        
        for jj in range(stepspersample):

            tjj = tkk + (jj + 1) * h

            sample_gaussian(r, sigma, w, m)

            status = noise_coeff(t, y, w, noises, params)

            if status != GSL_SUCCESS:
                return 201
            
            # with gil:
            #     print ".",
            # integrate over [t, tkk = t + stepsize]
            status = gsl_odeiv2_driver_apply (d, &t, tjj, y)
        
            if (status != GSL_SUCCESS):
                return status
            add_to(w, &W[kk * m], m)
            add_to(noises, &B[kk * dim], dim)
        
        # store current state in output array
        copy_array(y, &Y[sys.dimension * kk], sys.dimension)
    
    # free up resources
    gsl_odeiv2_driver_free(d)
    free(y)
    free(noises)
    free(w)
    # gsl_vector_free(w)
    # gsl_vector_free(noises)
    # gsl_matrix_free(B)
    gsl_rng_free (r)
    return GSL_SUCCESS




    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int c_ode_steady_state(double T_max, double tol, Py_ssize_t dim, double y0[], double yss[], ode_t ode,  void * params) nogil:
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
    
    cdef Py_ssize_t kk, ll
    cdef double t
    cdef double * y = <double *> malloc(dim * sizeof(double))
    cdef double * f = <double *> malloc(dim * sizeof(double))
    cdef double n
    cdef int converged = 0

    t = 0.
    copy_array(y0, y, dim)
    
    cdef int status
    cdef Py_ssize_t steps = imax(100, <int>(T_max/.1))
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



cdef np.ndarray c_eval_ode(double t, np.ndarray[np.float64_t, ndim=1] y, void * params, Py_ssize_t dim, ode_t ode):
    """
    Evaluate and return the ODE given by ode, with dimension dim, parametrized by params, at a given time t and state y.
    """
    cdef np.ndarray ret = np.zeros(dim)
    ode(t, <double *> np.PyArray_DATA(y), <double *> np.PyArray_DATA(ret), params)
    return ret
    
    
cdef class ODE:
    """
    Convenience class to set up a single ODE in an object-oriented fashion.

    The current state is stored in self.y
    The current time is stored in self.t
    """
 
    cpdef np.ndarray integrate(self, double delta_t, Py_ssize_t N, int include_initial=0, int update_state=1):

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
        
            
cdef class ODEs:
    """
    Convenience class to set up a whole set of ODEs that share the same ode function 
    (but can differ in current state and ode parameters) in an object-oriented fashion.
    """
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object integrate(self, double delta_t, Py_ssize_t N, int include_initial=0, int update_state=1):
        """
        Starting from the current states self.y and at the current time self.t integrate for a duration delta_t and
        evaluate the states at N time intervals. 
        
        If include_initial is non-zero, the returned array of states contains both the N integrated states and the initial state, 
        otherwise just the integrated states. If update_state is non-zero, after integration, self.y and self.t are updated.
        """
        cdef double[::1] YView
        cdef double[:,::1] yview = self.y
        cdef np.ndarray Y
        cdef Py_ssize_t traj_index
        cdef Py_ssize_t strd

        Y = np.zeros((self.N_params, N, self.dim))
        strd = N * self.dim
        
        YView = Y.ravel()
        cdef:
            double t = self.t
            ode_t c_ode
            void * c_params = self.c_params
            int param_size = self.param_size
            Py_ssize_t dim = self.dim
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
            
        c_ode = self.c_ode
            
        for traj_index in prange(self.N_params, nogil=True):
            successview[traj_index] = c_integrate_ode(t + delta_t, N, dim, <double*>&yview[traj_index,0], <double*>&YView[traj_index * strd], c_ode, c_params + param_size * traj_index, t)
        
        if include_initial:
            Y = np.concatenate((self.y.reshape(self.N_params, 1, -1), Y), axis=1)

        if update_state:
            self.y[success == GSL_SUCCESS] = Y[success == GSL_SUCCESS, Y.shape[1] - 1, :]
            self.t += delta_t
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
            ode_t c_ode
            void * c_params = self.c_params
            int param_size = self.param_size
            Py_ssize_t dim = self.dim
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
            Py_ssize_t traj_index
            
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
        cdef:
            np.ndarray Y = np.zeros((self.N_params, self.dim))
            double *  YView = <double*> np.PyArray_DATA(Y)
            double *  yview = <double*> np.PyArray_DATA(self.y)
            Py_ssize_t traj_index
            
            double t = self.t
            ode_t c_ode
            void * c_params = self.c_params
            int param_size = self.param_size
            Py_ssize_t dim = self.dim
        
        c_ode = self.c_ode
        
        for traj_index in prange(self.N_params, nogil=True):
#        for traj_index in range(self.N_params):
            # pass
            c_ode(t, &yview[traj_index * dim], &YView[traj_index * dim], c_params + param_size * traj_index)
        return Y
        

cdef class SDEs(ODEs):


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object integrate_sde(self, double delta_t, double h, Py_ssize_t N, 
                               int include_initial=0, int update_state=1, int return_noises=0, np.ndarray rngseeds=None):
    # cpdef object integrate(self, double delta_t, Py_ssize_t N, int include_initial=0, int update_state=1):
        """
        Starting from the current states self.y and at the current time self.t integrate for a duration delta_t and
        evaluate the states at N time intervals. 
        
        If include_initial is non-zero, the returned array of states contains both the N integrated states and the initial state, 
        otherwise just the integrated states. If update_state is non-zero, after integration, self.y and self.t are updated.
        If return_noises is true, this returns a tuple (Y, B, success), 
        where B are the noise increments during the sampled intervals.
        
        """
        
        if rngseeds is None:
            rngseeds = np.zeros(self.N_params, dtype=np.int64)

        cdef long[::1] rngseedsView = rngseeds
        

        cdef double[::1] YView, BView, WView
        cdef double[:,::1] yview = self.y
        cdef np.ndarray Y, B, W
        cdef Py_ssize_t traj_index
        cdef Py_ssize_t strd, strd2

        Y = np.zeros((self.N_params, N, self.dim))
        strd = N * self.dim
        strd2 = N * self.N_noises
        
        YView = Y.ravel()
        cdef:
            double t = self.t
            ode_t c_ode
            noise_t c_noise_coeff
            void * c_params = self.c_params
            int param_size = self.param_size
            Py_ssize_t dim = self.dim
            Py_ssize_t N_noises = self.N_noises
            np.ndarray success = np.zeros(self.N_params, dtype=int)
            long[:] successview = success
        
            
        c_ode = self.c_ode
        c_noise_coeff = self.c_noise_coeff
        
        
        if return_noises:
            B = np.zeros((self.N_params, N, self.dim))
            W = np.zeros((self.N_params, N, self.N_noises))
            BView = B.ravel()
            WView = W.ravel()
            for traj_index in prange(self.N_params, nogil=True):
                successview[traj_index] = c_integrate_sde_store_noises(t + delta_t, h, N, dim, N_noises, 
                                                                       <double*>&yview[traj_index,0], 
                                                                       <double*>&YView[traj_index * strd], 
                                                                       <double*>&WView[traj_index * strd2],  
                                                                       <double*>&BView[traj_index * strd], 
                                                                       c_ode, c_noise_coeff, 
                                                                       c_params + param_size * traj_index, t, 
                                                                       rngseedsView[traj_index])
        else:
            for traj_index in prange(self.N_params, nogil=True):
                successview[traj_index] = c_integrate_sde(t + delta_t, h, N, dim, N_noises, 
                                                          <double*>&yview[traj_index,0], 
                                                          <double*>&YView[traj_index * strd],  
                                                          c_ode, c_noise_coeff, c_params + param_size * traj_index, t,
                                                          rngseedsView[traj_index])
        
            
        if include_initial:
            Y = np.concatenate((self.y.reshape(self.N_params, 1, -1), Y), axis=1)
            if return_noises:
                B = np.concatenate((np.zeros((self.N_params, 1, self.dim), dtype=float), B), axis=1)
                W = np.concatenate((np.zeros((self.N_params, 1, self.N_noises), dtype=float), W), axis=1)

        if update_state:
            self.y[success == GSL_SUCCESS] = Y[success == GSL_SUCCESS, Y.shape[1] - 1, :]
            self.t += delta_t

        if return_noises:
            return Y, W, B, success
        else:
            return Y, success

    cpdef np.ndarray sample_noise(self):
        pass

    
    
