"""
Algorithms
----------

Implementation of the algorithms for spectral density estimation.
"""

import sys

import numpy as np
import scipy as sp

from src.kernel import gaussian_kernel
from src.interpolation import chebyshev_coefficients, squared_chebyshev_coefficients, exponentiate_chebyshev_coefficients_cosine_transform, _squared_chebyshev_coefficients_summation, chebyshev_recurrence, multiply_chebyshev_coefficients_cosine_transform
from src.eigenproblem import generalized_eigenproblem_standard, generalized_eigenproblem_pinv, generalized_eigenproblem_dggev, generalized_eigenproblem_lstsq, generalized_eigenproblem_kernelunion, generalied_eigenproblem_direct
from src.approximation import generalized_nystrom_pinv, generalized_nystrom_qr, generalized_nystrom_stable_qr, nystrom_svd
from src.utils import continued_fraction


# Increase recursion limit for Haydock's method
sys.setrecursionlimit(5000)


def DGC(A, t, M, sigma, N_v, kernel=gaussian_kernel, seed=0):
    """
    Delta-Gauss-Chebyshev method for computing the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors.
    kernel : function
        The smoothing kernel applied to the spectral density.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 2.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    # Polynomial expansion
    g = lambda x: kernel(x, N=N, sigma=sigma)
    mu = chebyshev_coefficients(t, M, function=g)

    # Do recurrence
    W = np.random.randn(N, N_v)
    phi_tilde = chebyshev_recurrence(mu, A, T_0=W, L=lambda x: np.trace(W.T @ x) / N_v)

    return phi_tilde


def KPM(A, t, M, N_v, seed=0, sigma=None):
    """
    Kernel polynomial method for computing the spectral density.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray of shape (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    N_v : int > 0
        Number of random vectors.
    seed : int >= 0
        The seed for generating the random matrix W.
    sigma : None
        Unused dummy-argument to match function signature of other algorithms.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [4] Lin, L. Approximating spectral densities of large matrices: old and new.
        Math/CS Seminar, Emory University (2015).
        Link: https://math.berkeley.edu/~linlin/presentations/201753_DOS.pdf
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    # Initializations
    mu = np.zeros(M + 1)
    W = np.random.randn(N, N_v)
    V_c = W.copy()
    V_m = np.zeros((N, N_v))
    V_p = np.zeros((N, N_v))

    # Chebyshev recursion
    for l in range(M + 1):
        mu[l] = np.trace(W.T @ V_c) / (N_v * N * np.pi)
        V_p = (1 if l == 0 else 2) * A @ V_c - V_m
        V_m = V_c.copy()
        V_c = V_p.copy()

    # Computation of the approximate spectral density
    phi_tilde = 1 / np.sqrt(1 + t**2) * np.polynomial.chebyshev.Chebyshev(mu)(t)
    return phi_tilde


def NyCheb(A, t, M, sigma, N_v, tau=1e-7, epsilon=1e-1, kernel=gaussian_kernel, eigenproblem="standard", seed=0):
    """
    Spectrum sweeping method using the Delta-Gauss-Chebyshev expansion for
    estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [2] (project out kern(K_W))
         -> kernelunion = Project out union of kern(K_W) and kern(K_Z)
         -> pinv = Directly compute pseudoinverse
         -> dggev = Use KZ algorithm
         -> lstsq = Solve leastsquares problem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Polynomial expansion
    g = lambda x: kernel(x, N=N, sigma=sigma)
    mu = chebyshev_coefficients(t, M, function=g)

    # Do recurrence
    W = np.random.randn(N, N_v)
    Z = chebyshev_recurrence(mu, A, T_0=W, final_shape=(N, N_v))

    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        if eigenproblem == "kernelunion":
            Xi = generalized_eigenproblem_kernelunion(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        elif eigenproblem == "pinv":
            Xi = generalized_eigenproblem_pinv(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        elif eigenproblem == "dggev":
            Xi = generalized_eigenproblem_dggev(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        elif eigenproblem == "lstsq":
            Xi = generalized_eigenproblem_lstsq(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        else:
            Xi = generalized_eigenproblem_standard(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        phi_tilde[i] = np.sum(Xi)

    return phi_tilde


def NyChebSI(A, t, M, sigma, N_v, tau=1e-7, epsilon=1e-1, kernel=gaussian_kernel, eigenproblem="standard", seed=0):
    """
    Spectrum sweeping method using the Delta-Gauss-Chebyshev expansion for
    estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [2] (project out kern(K_W))
         -> kernelunion = Project out union of kern(K_W) and kern(K_Z)
         -> pinv = Directly compute pseudoinverse
         -> dggev = Use KZ algorithm
         -> lstsq = Solve leastsquares problem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Polynomial expansion
    g = lambda x: kernel(x, N=N, sigma=sigma)
    mu = chebyshev_coefficients(t, M, function=g)
    #nu = squared_chebyshev_coefficients_cosine_transform(mu)

    # Do recurrence
    W = np.random.randn(N, N_v)
    Z = chebyshev_recurrence(mu, A, T_0=W, final_shape=(N, N_v))
    Y = chebyshev_recurrence(mu, A, final_shape=(N, N))

    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        phi_tilde[i] = np.trace(Z[i] @ np.linalg.pinv(Z[i].T @ Z[i]) @ Z[i].T @ Y[i])
        #if eigenproblem == "kernelunion":
        #    Xi = generalized_eigenproblem_kernelunion(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        #elif eigenproblem == "pinv":
        #    Xi = generalized_eigenproblem_pinv(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        #elif eigenproblem == "dggev":
        #    Xi = generalized_eigenproblem_dggev(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        #elif eigenproblem == "lstsq":
        #    Xi = generalized_eigenproblem_lstsq(Z[i].T @ Z[i], W.T @ Z[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        #else:
        #    Xi = generalized_eigenproblem_standard(Y[i].T @ Y[i], Z[i].T @ Y[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
        #phi_tilde[i] = np.sum(Xi)

    return phi_tilde


def GenNyCheb(A, t, M, sigma, N_v, c1=1/4, c2=1/2, nystrom_version="pinv", seed=0):
    """
    TODO
    
    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors.
    c_1 : TODO

    c_2 : TODO

    nystrom_version : TODO    
    
    seed : int >= 0
        The seed for generating the random matrix W.
    
    References
    ----------
    [X] TODO
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    N_v_1 = round(N_v * c1)
    N_v_2 = round(N_v * c2)
    N_v_3 = N_v - N_v_1 - N_v_2

    # Polynomial expansion
    g = lambda x: gaussian_kernel(x, N=N, sigma=sigma)
    mu = chebyshev_coefficients(t, M, function=g)

    # Initializations
    W_1 = np.random.rand(N, N_v_1)
    W_2 = np.random.rand(N, N_v_2)
    W_3 = np.random.rand(N, N_v_3)

    Z_1 = chebyshev_recurrence(mu, A, T_0=W_1, final_shape=(N, N_v_1))
    Z_2 = chebyshev_recurrence(mu, A, T_0=W_2, final_shape=(N, N_v_2))
    Z_3 = chebyshev_recurrence(mu, A, T_0=W_3, final_shape=(N, N_v_3))

    phi_tilde = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        if nystrom_version == "pinv":
            P = generalized_nystrom_pinv(Z_1[i], Z_2[i], W_2)
        if nystrom_version == "qr":
            P = generalized_nystrom_qr(Z_1[i], Z_2[i], W_2)
        if nystrom_version == "stable_qr":
            P = generalized_nystrom_stable_qr(Z_1[i], Z_2[i], W_2)
        phi_tilde[i] = np.trace(P) + (np.trace(W_3.T @ Z_3[i]) + np.trace(W_3.T @ P @ W_3)) / N_v_3

    return phi_tilde


def FastNyCheb(A, t, M, sigma, N_v, k=1, tau=1e-7, delta=1e-5, epsilon=1e-1, kernel=gaussian_kernel, square_coefficients="transformation", eigenproblem="standard", seed=0):
    """
    Spectrum sweeping method using the Delta-Gauss-Chebyshev expansion for
    estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the Chebyshev polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors.
    k : int > 0
        The approximation method used (1 = Nyström, 2 = RSVD, 3 = SI-Nyström)
    tau : int or float in (0, 1]
        Truncation parameter.
    delta : float > 0
        The threshold on the Hutchinson estimate of g_sigma. If it is below this
        value, instead of solving the possibly ill-conditioned generalized
        eigenvalue problem, we set the spectral density at that point to zero.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    square_coefficients : str or None
        Method by which the coefficients of the squared Gaussian are computed.
         -> transformation = Compute coefficients with discrete cosine transform
         -> summation = Explicitly square the interpolant
         -> interpolation = Interpolate the squared function
         -> None = Use SS_DGC
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [2] (project out kern(K_W))
         -> kernelunion = Project out union of kern(K_W) and kern(K_Z)
         -> direct = Directly solve the generalized eigenproblem
         -> pinv = Directly compute pseudoinverse
         -> dggev = Use QZ algorithm
         -> lstsq = Solve leastsquares problem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 5.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix
    N = A.shape[0]

    # Convert evaluation point(s) to numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Chebyshev expansion
    g = lambda x: kernel(x, N=N, sigma=sigma) + (1e-3 if eigenproblem == "cholesky" else 0)

    if square_coefficients == "interpolation":
        mu_W = chebyshev_coefficients(t, k * M, function=lambda x: g(x) ** k)
        mu_Z = chebyshev_coefficients(t, (k + 1) * M, function=lambda x: g(x) ** (k + 1))
    elif square_coefficients == "transformation":
        mu = chebyshev_coefficients(t, M, function=g)
        mu_W = exponentiate_chebyshev_coefficients_cosine_transform(mu, k=k)
        mu_Z = exponentiate_chebyshev_coefficients_cosine_transform(mu, k=k + 1)
    else:  # square_coefficients == None:
        return NyCheb(A, t, M, sigma, N_v, tau, epsilon, kernel, seed)

    # Chebyshev recurrence
    W = np.random.randn(N, N_v)

    K_W = chebyshev_recurrence(mu_W, A, T_0=W, L=lambda x: W.T @ x, final_shape=(N_v, N_v))
    K_Z = chebyshev_recurrence(mu_Z, A, T_0=W, L=lambda x: W.T @ x, final_shape=(N_v, N_v))

    # Trace computation
    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        # Check if rank of if Hutchinson (k=1) for Tr(g^M(tI-A)) is almost zero
        if np.trace(K_W[i]) / N_v < delta:
            phi_tilde[i] = 0
            continue
        else:
            if eigenproblem == "kernelunion":
                Xi = generalized_eigenproblem_kernelunion(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            elif eigenproblem == "pinv":
                Xi = generalized_eigenproblem_pinv(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            elif eigenproblem == "dggev":
                Xi = generalized_eigenproblem_dggev(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            elif eigenproblem == "lstsq":
                Xi = generalized_eigenproblem_lstsq(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            elif eigenproblem == "direct":
                Xi = generalied_eigenproblem_direct(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            elif eigenproblem == "cholesky":
                Xi = generalized_eigenproblem_cholesky(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            else:  # square_coefficients == "standard":
                Xi = generalized_eigenproblem_standard(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)[0]
            phi_tilde[i] = np.sum(Xi) - len(Xi) * (1e-3 if eigenproblem == "cholesky" else 0)

    return phi_tilde


def FastNyChebPP(A, t, M, sigma, N_v, N_v_tilde=None, k=1, tau=1e-7, delta=1e-5, epsilon=1e-1, kernel=gaussian_kernel, square_coefficients="transformation", eigenproblem="standard", seed=0):
    """
    Robust and efficient spectrum sweeping with Delta-Gauss-Chebyshev method
    for estimating the spectral density.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of Chebyshev the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors in W.
    N_v_tilde : int > 0
        Number of random vectors in W_tilde.
    k : int > 0
        The approximation method used (1 = Nyström, 2 = RSVD, 3 = SI-Nyström)
    tau : int or float in (0, 1]
        Truncation parameter.
    delta : float > 0
        The threshold on the Hutchinson estimate of g_sigma. If it is below this
        value, instead of solving the possibly ill-conditioned generalized
        eigenvalue problem, we set the spectral density at that point to zero.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma.
    kernel : function
        The smoothing kernel applied to the spectral density.
    square_coefficients : str or None
        Method by which the coefficients of the squared Gaussian are computed.
         -> transformation = Compute coefficients with discrete cosine transform
         -> summation = Explicitly square the interpolant
         -> interpolation = Interpolate the squared function
    eigenproblem : str
        Resolution method of the generalized eigenvalue problem in SS methods.
         -> standard = As proposed in [2] (project out kern(K_W))
         -> kernelunion = Project out union of kern(K_W) and kern(K_Z)
         -> direct = Directly solve the generalized eigenproblem
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 7.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    # Convert evaluation point(s) to numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    # Preprocess the number of random vectors
    if N_v == 0:
        return DGC(A, t, M, sigma, N_v_tilde, kernel, seed)
    if N_v_tilde is None:  # Evenly distribute mat-vecs
        N_v_tilde = N_v // 2
        N_v = N_v // 2
    elif N_v_tilde == 0:
        return FastNyCheb(A, t, M, sigma, N_v, k, tau, delta, epsilon, kernel, square_coefficients, eigenproblem, seed)

    # Chebyshev expansion
    g = lambda x: kernel(x, N=N, sigma=sigma)
    
    if square_coefficients == "transformation":
        mu = chebyshev_coefficients(t, M, function=g)
        mu_W = exponentiate_chebyshev_coefficients_cosine_transform(mu, k=k)
        mu_Z = exponentiate_chebyshev_coefficients_cosine_transform(mu, k=k + 1)
        mu_C = mu if k < 3 else exponentiate_chebyshev_coefficients_cosine_transform(mu, k=(k + 1) // 2)
        mu_D = mu_C if k % 2 == 1 else exponentiate_chebyshev_coefficients_cosine_transform(mu, k=(k + 2) // 2)
    else:  # square_coefficients == "interpolation":
        mu = chebyshev_coefficients(t, M, function=g)
        mu_W = chebyshev_coefficients(t, k * M, function=lambda x: g(x) ** k)
        mu_Z = chebyshev_coefficients(t, k * M, function=lambda x: g(x) ** (k + 1))
        mu_C = mu if k < 3 else chebyshev_coefficients(t, k * M, function=lambda x: g(x) ** ((k + 1) // 2))
        mu_D = mu_C if k % 2 == 1 else chebyshev_coefficients(t, k * M, function=lambda x: g(x) ** ((k + 2) // 2))

    # Initializations
    W = np.random.randn(N, N_v)
    W_tilde = np.random.randn(N, N_v_tilde)

    K_W = chebyshev_recurrence(mu_W, A, T_0=W, L=lambda x: W.T @ x, final_shape=(N_v, N_v))
    K_Z = chebyshev_recurrence(mu_Z, A, T_0=W, L=lambda x: W.T @ x, final_shape=(N_v, N_v))
    K_C = chebyshev_recurrence(mu_C, A, T_0=W, L=lambda x: W_tilde.T @ x, final_shape=(N_v_tilde, N_v))
    K_D = K_C if k % 2 == 1 else chebyshev_recurrence(mu_D, A, T_0=W, L=lambda x: W_tilde.T @ x, final_shape=(N_v_tilde, N_v))
    K_W_tilde = chebyshev_recurrence(mu, A, T_0=W_tilde, L=lambda x: np.trace(W_tilde.T @ x), final_shape=())

    phi_tilde = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        if np.trace(K_W[i]) / N_v < delta:  # Hutchinson for Tr(g^M(tI-A))
            continue

        if eigenproblem == "kernelunion":
            xi_tilde, C_tilde = generalized_eigenproblem_kernelunion(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)
        elif eigenproblem == "direct":
            xi_tilde, C_tilde = generalied_eigenproblem_direct(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)
        else:  # square_coefficients == "standard":
            xi_tilde, C_tilde = generalized_eigenproblem_standard(K_Z[i], K_W[i], N=N, sigma=sigma, tau=tau, epsilon=epsilon)
        T = np.trace(K_C[i] @ C_tilde @ C_tilde.conjugate().T @ K_D[i].T)
        phi_tilde[i] = np.sum(xi_tilde) + (K_W_tilde[i] - T) / N_v_tilde

    return phi_tilde


def hutchinson(A, N_v, seed=0):
    """
    Hutchinson trace estimator.

    Parameters
    ----------
    A : np.ndarray (N, N)
        The matrix for which the trace will be computed.
    N_v : int
        The number of random vectors to be used in total.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the Hutchinson method.
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    W = 2.0 * (np.random.rand(N, N_v) > 0.5) - 1.0

    t_A = np.trace(W.T @ A @ W) / N_v
    return t_A


def hutchpp(A, N_v, sketch_fraction=2/3, seed=0):
    """
    Hutch++ trace estimator.

    Parameters
    ----------
    A : np.ndarray (N, N)
        The matrix for which the trace will be computed.
    N_v : int
        The number of random vectors to be used in total.
    sketch_fraction : float
        The fraction of random vectors which are used for sketching A.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the Hutch++ algorithm.

    References
    ----------
    [3] Meyer et. al. Hutch++: Optimal Stochastic Trace Estimation.
        Proc SIAM Symp Simplicity Algorithms. (2021) 142-155. 
        DOI: 10.1137/1.9781611976496.16
    """   
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix and amount of random mat-vec sketches
    N = A.shape[0]
    n_sketch = round(N_v * sketch_fraction / 2)
    n_hutch  = N_v - 2 * n_sketch

    # Generate sketching matrices
    S = 2.0 * (np.random.rand(N, n_sketch) > 0.5) - 1.0
    G = 2.0 * (np.random.rand(N, n_hutch) > 0.5) - 1.0

    # Generate orthonormal basis of sketch AS
    Q, _ = np.linalg.qr(A @ S)

    # Compute Hutch++ estimate
    G -= Q @ Q.T @ G
    t_A = np.trace(Q.T @ A @ Q) + np.trace(G.T @ A @ G) / n_hutch
    return t_A


def nahutchpp(A, N_v, c1=1/4, c2=1/2, seed=0):
    """
    Non-adaptive Hutch++ trace estimator.

    Parameters
    ----------
    A : np.ndarray (N, N)
        The matrix for which the trace will be computed.
    N_v : int
        The number of random vectors to be used in total.
    c1 : float
        Fraction of random vectors used for sketching A's QR decomposition.
    c2 : float
        Fraction of random vectors used for sketching A.
    seed : int or None
        The seed for the random number generator.

    Returns
    -------
    t_A : float
        Estimate of trace by the non-adaptive Hutch++ algorithm.

    References
    ----------
    [3] Meyer et. al. Hutch++: Optimal Stochastic Trace Estimation.
        Proc SIAM Symp Simplicity Algorithms. (2021) 142-155. 
        DOI: 10.1137/1.9781611976496.16
    """ 
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix and amount of random mat-vec sketches
    N = A.shape[0]
    n_R_sketch = round(N_v * c1)
    n_S_sketch = round(N_v * c2)
    n_hutch = N_v - n_R_sketch - n_S_sketch

    R = 2.0 * (np.random.rand(N, n_R_sketch) > 0.5) - 1.0
    S = 2.0 * (np.random.rand(N, n_S_sketch) > 0.5) - 1.0
    G = 2.0 * (np.random.rand(N, n_hutch) > 0.5) - 1.0

    Z = A @ R
    W = A @ S

    P = np.linalg.pinv(S.T @ Z)
    trace = np.trace(P @ (W.T @ Z)) + (np.trace(G.T @ A @ G) - np.trace((G.T @ Z) @ P @ (W.T @ G))) / n_hutch
    return trace


def HDGC(A, t, M, sigma, N_v, estimator=hutchpp, seed=0):
    """
    Hutchinson-type estimators for Delta-Gauss-Chebyshev method.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    M : int > 0
        Degree of the polynomial.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors in W.
    estimator : function
        The trace estimator.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.
    """
    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    # Polynomial expansion
    g = lambda x: gaussian_kernel(x, N=N, sigma=sigma)
    mu = chebyshev_coefficients(t, M, function=g)

    # Initializations
    Z = np.zeros((M+1, N, N))
    V_c = np.eye(N)
    V_m = np.zeros((N, N))
    V_p = np.zeros((N, N))

    # Chebyshev recursion
    for l in range(M + 1):
        Z[l] = V_c
        V_p = (1 if l == 0 else 2) * A @ V_c - V_m
        V_m = V_c.copy()
        V_c = V_p.copy()

    # Computation of the approximate spectral density
    g_M = np.tensordot(mu, Z, axes=([1], [0]))
    phi_tilde = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        phi_tilde[i] = estimator(g_M[i], N_v, seed=seed)
    return phi_tilde


def Lanczos(A, x, k, reorth_tol=0.7):
    n = A.shape[0]

    a = np.empty(k)
    b = np.empty(k)

    U = np.empty((n, k + 1))
    U[:, 0] = x / np.linalg.norm(x)

    for j in range(k):
        w = A @ U[:, j]
        a[j] = U[:, j].T @ w
        u_tilde = w -  U[:, j] * a[j] - (U[:, j - 1] * b[j - 1] if j > 0 else 0) 

        if np.linalg.norm(u_tilde) <= reorth_tol * np.linalg.norm(w):
            # Twice is enough
            h_hat = U[:, : j + 1].T @ u_tilde
            a[j] += h_hat[-1]
            if j > 0:
                b[j - 1] += h_hat[-2]
            u_tilde -= U[:, : j + 1] @ h_hat

        b[j] = np.linalg.norm(u_tilde)
        U[:, j + 1] = u_tilde / b[j]

    return a, b


def SLQ(A, t, sigma, N_v, M=200, seed=0):
    """
    Stochastic Lanczos Quadrature.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors used in Monte-Carlo estimate.
    M : int > 0
        The number of Lanczos iterations.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [6] T. Chen, T. Trogdon, S. Ubaru. Analysis of stochastic Lanczos quadrature
        for spectrum approximation. PMLR 139:1728-1739 (2021).
        Link: http://proceedings.mlr.press/v139/chen21s/chen21s.pdf
    """

    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    phi_tilde = np.zeros_like(t)
    for _ in range(N_v):
        x = np.random.randn(N)
        a, b = Lanczos(A, x / np.linalg.norm(x), M)
        theta, S = sp.linalg.eigh_tridiagonal(a[: M], b[: M - 1])
        t_minus_theta = np.subtract.outer(t, theta)
        phi_tilde += gaussian_kernel(t_minus_theta, sigma=sigma) @ S[0]**2

    return phi_tilde / N_v


def Haydock(A, t, sigma, N_v, M, seed=0, kernel=None):
    """
    Haydock's method.

    Parameters
    ----------
    A : np.ndarray (n, n)
        Hermitian matrix with eigenvalues between (-1, 1).
    t : np.ndarray (N_t,)
        A set of points at which the DOS is to be evaluated.
    sigma : int or float > 0
        Smearing parameter.
    N_v : int > 0
        Number of random vectors used in Monte-Carlo estimate.
    M : int > 0
        The number of Lanczos iterations.
    seed : int >= 0
        The seed for generating the random matrix W.
    kernel : None
        Unused dummy argument for compatibility reasons.

    Returns
    -------
    phi_tilde : np.ndarray
        Approximations of the spectral density at the points t.

    References
    ----------
    [7] L. Lin, Y. Saad, C. Yang. Approximating Spectral Densities of Large Matrices.
        SIAM Reviev 58(1) (2016). Section 3.2.2. 
        Link: https://doi.org/10.1137/130934283
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    phi_tilde = np.zeros(len(t))
    for _ in range(N_v):
        # Compute tridiagonal matrix from Lanczos for random vector
        v = np.random.randn(N)
        a, b = Lanczos(A, v, M)
        phi_tilde += np.imag(continued_fraction((t + 1j*sigma), a, b))

    phi_tilde *= - 1 / (N_v * np.pi)
    return phi_tilde


def _randomized_lowrank_decomposition(A, r, c=10, seed=0):
    """
    Randomized low-rank decomposition of a Hermitian matrix. Format: A = ZBZ*

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Hermitian matrix which will be approximated.
    r : int > 0
        Approximate rank of the matrix A.
    c : int > 0
        Small constant by which the random matrix will be larger than r.
    seed : int >= 0
        The seed for generating the random matrix W.

    Returns
    -------
    Z : np.ndarray of shape (N, r + c)
        Basis spanning the space of the low-rank approximation.
    B : np.ndarray of shape (r + c, r + c)
        Approximate decomposition of the matrix P in the basis Z.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 3.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    # Compute randomized low-rank approximation
    W = np.random.randn(N, r + c)
    Z = A @ W
    B = np.linalg.pinv(W.T @ Z, hermitian=True)

    return Z, B


def _randomized_trace_estimation(A, N_v, N_v_tilde):
    """
    Robust and efficient method for estimating the trace of a low-rank matrix.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Hermitian matrix which will be approximated.
    N_v : int > 0
        Number of randomized vectors in random matrix.
    N_v_tilde : int > 0
        Number of randomized vectors in other random matrix.

    Returns
    -------
    trace : float
        Approximation of the trace of the matrix P.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 6.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # Determine size of matrix i.e. number of eigenvalues 
    N = A.shape[0]

    W = np.random.randn(N, N_v)
    W_tilde = np.random.randn(N, N_v_tilde)
    K_W = W.T @ (A @ W)
    K_Z = W.T @ (np.linalg.matrix_power(A, 2) @ W)
    K_C = W_tilde.T @ (A @ W)
    K_W_tilde = W_tilde.T @ (A @ W_tilde)

    xi_tilde, C_tilde = generalized_eigenproblem_standard(K_Z, K_W, N=N)

    T = K_C @ C_tilde @ C_tilde.T @ K_C.T
    trace = np.sum(xi_tilde) + (np.trace(K_W_tilde - T)) / N_v_tilde

    return trace
