
#cython: embedsignature=True

cimport numpy as np
import numpy as np

cimport cython
import cython

from cython_gsl cimport GSL_SUCCESS, gsl_rng

cdef inline Py_ssize_t imin(Py_ssize_t a, Py_ssize_t b) nogil: 
    return a if a <= b else b
cdef inline Py_ssize_t imax(Py_ssize_t a, Py_ssize_t b) nogil: 
    return a if a >= b else b


# cdef extern from "cblas.h":

#     enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
#     enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    
#     void lib_dgemv "cblas_dgemv"(
#         CBLAS_ORDER Order, 
#         CBLAS_TRANSPOSE TransA,
#         int M, 
#         int N, 
#         double alpha, 
#         double *A, 
#         int lda,
#         double *x, 
#         int dx,
#         double beta,
#         double *y, 
#         int dy) nogil
    
# cdef inline void dgemv(int m, int n, double * A, double * x, double * y) nogil:
#     lib_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, x, 1, 0., y, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copy_array(double * from_p, double * to_p, Py_ssize_t Np) nogil:
    cdef Py_ssize_t kk
    for kk in range(Np):
        to_p[kk] = from_p[kk]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void init_zero(double * ptr,  Py_ssize_t Np) nogil:
    cdef Py_ssize_t kk
    for kk in range(Np):
        ptr[kk] = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void add_to(double * from_p, double * to_p, Py_ssize_t Np) nogil:
    cdef Py_ssize_t kk
    for kk in range(Np):
        to_p[kk] += from_p[kk]
        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double normsq(double * x, Py_ssize_t N, Py_ssize_t dx) nogil:
    cdef double ret = 0.
    cdef Py_ssize_t kk
    for kk in range(N):
        ret += x[kk * dx] * x[kk * dx]
    return ret

ctypedef int (* ode_t)(double t, double y[], double f[], void * params) nogil
ctypedef int (* noise_t)(double, double y[], double w[], double b[] , void * params) nogil
# ctypedef int (* matvec_t)(double t, double * y, double * w, void * params) nogil

cdef void sample_gaussian(gsl_rng * r, double sigma, double * w, Py_ssize_t m) nogil
# cdef int noise_from_covariance(Py_ssize_t n, double * BBT, double * b, double * w) nogil
# cdef int noise_from_matvec(double t, double * y, matvec_t matvec, double * b, double * w, void * params) nogil


cdef int c_integrate_ode(double tf, Py_ssize_t N, Py_ssize_t dim, double y0[], double * Y, ode_t ode, void * params, double t0=*) nogil

cdef int c_integrate_sde(double tf, double h, Py_ssize_t N, Py_ssize_t dim, Py_ssize_t m, double y0[], double * Y, ode_t ode, noise_t noise_coeff, void *params, double t0=*, long rngseed=*) nogil

cdef int c_integrate_sde_store_noises(double tf, double h, Py_ssize_t N, Py_ssize_t dim, Py_ssize_t m, double * y0,  double * Y, double * W, double * B, ode_t ode, noise_t noise_coeff, void *params, double t0=*, long rngseed=*) nogil

cdef int c_ode_steady_state(double T_max, double tol, Py_ssize_t dim, double y0[], double * yss, ode_t ode, void * params) nogil

cdef np.ndarray c_eval_ode(double t, np.ndarray[np.float64_t, ndim=1] y, void * params, Py_ssize_t dim, ode_t ode)


cdef class ODE:
    cdef ode_t c_ode
    cdef void * c_params
    cdef public Py_ssize_t dim
    cdef public double t
    cdef public np.ndarray y
    cpdef np.ndarray integrate(self, double delta_t, Py_ssize_t N, int include_initial=*, int update_state=*)
    cpdef np.ndarray steady_state(self, double T_max, double tol)
    cpdef np.ndarray evaluate(self)


cdef class ODEs:
    cdef ode_t c_ode
    cdef void * c_params
    cdef Py_ssize_t param_size
    cdef Py_ssize_t N_params
    cdef public Py_ssize_t dim
    cdef public double t
    cdef public np.ndarray y

    cpdef object integrate(self, double delta_t, Py_ssize_t N, int include_initial=*, int update_state=*)
    cpdef object steady_state(self, double T_max, double tol)
    cpdef np.ndarray evaluate(self)

cdef class SDEs(ODEs):
    cdef noise_t c_noise_coeff
    cdef public Py_ssize_t N_noises
    cpdef object integrate_sde(self, double delta_t, double h, Py_ssize_t N, int include_initial=*, int update_state=*, int return_noises=*, np.ndarray rngseeds=*)
    cpdef np.ndarray sample_noise(self)

