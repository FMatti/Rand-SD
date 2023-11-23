"""
Interpolation
-------------

Functions for Chebyshev interpolation.
"""

import numpy as np
import scipy as sp


def chebyshev_coefficients(t, M, function, adjust_first=True):
    """
    Chebyshev expansion of a function.

    Parameters
    ----------
    t : int, float, list, or np.ndarray of shape (n,)
        Point(s) where the expansion should be evaluated.
    M : int > 0
        Degree of the Chebyshev polynomial.
    N_theta : int > M
        The (half) number of integration points.
    function : function
        The kernel used to regularize the spectral density.

    Returns
    -------
    mu : np.ndarray of shape (N_t, M + 1)
        The coefficients of the Chebyshev polynomials. Format: mu[t, l].

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 1.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    # If t is a scalar, we convert it to a 1d array to make computation work
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)

    theta = np.arange(M + 1) * np.pi / M

    # Compute the coefficients mu for all t and l simultaneously with FFT
    t_minus_theta = np.subtract.outer(t, np.cos(theta))
    mu = 1 / M * sp.fft.dct(function(t_minus_theta), type=1)

    if adjust_first:
        mu[:, 0] /= 2

    return mu


def squared_chebyshev_coefficients(t, M, function, adjust_first=True):
    """
    Determine the Che.

    Parameters
    ----------
    mu : np.ndarray (N_t, M_mu + 1)
        Chebyshev coefficients corresponding to an expansion of a function.
    M : int > 0 or None
        Degree of the squared Chebyshev polynomial.

    Returns
    -------
    nu : np.ndarray of shape (N_t, M + 1)
        Chebyshev coefficients for square of expansion defined by mu.
    """
    function_squared = lambda x: function(x) ** 2
    nu = chebyshev_coefficients(t, M, function=function_squared, adjust_first=adjust_first)

    return nu


def exponentiate_chebyshev_coefficients_cosine_transform(mu, k=2, M=None, adjust_first=True):
    """
    Squared expansion of the polynomial defined by Chebyshev coefficients mu.
    The discrete cosine transform is used to efficiently compute the
    coefficients of the squared Chebyshev polynomial.

    Parameters
    ----------
    mu : np.ndarray (N_t, M_mu + 1)
        Chebyshev coefficients corresponding to an expansion of a function.
    k : int > 0
        The power to which the Chebyshev polynomial should be raised.
    M : int > 0 or None
        Degree of the squared Chebyshev polynomial.
    adjust_first : bool
        Whether to use the convention to divide the first coefficient mu_0 by 2.

    Returns
    -------
    nu : np.ndarray of shape (N_t, M + 1)
        Chebyshev coefficients for square of expansion defined by mu.

    References
    ----------
    [4] G. Baszenski, M. Tasche. Fast Polynomial Multiplication and Convolutions
        Related to the Discrete Cosine Transform.
        Linear Algebra and its Applications 252:1-25. (1997)
        DOI: https://doi.org/10.1016/0024-3795(95)00696-6
    """
    if k == 1:
        return mu
    M_mu = mu.shape[1] - 1
    if M is None:
        M = k * M_mu
    mu_tilde = np.hstack((mu, np.zeros((mu.shape[0], M + 1 - M_mu))))
    if adjust_first:
        mu_tilde[:, 0] *= 2
    nu = (M + 1) ** (k - 1) * sp.fft.dct(sp.fft.idct(mu_tilde, type=1) ** k, type=1)[:, : M + 1]

    if adjust_first:
        nu[:, 0] /= 2

    return nu


def multiply_chebyshev_coefficients_cosine_transform(mu_1, mu_2, M=None, adjust_first=True):
    """
    Squared expansion of the polynomial defined by Chebyshev coefficients mu.
    The discrete cosine transform is used to efficiently compute the
    coefficients of the squared Chebyshev polynomial.

    Parameters
    ----------
    mu : np.ndarray (N_t, M_mu + 1)
        Chebyshev coefficients corresponding to an expansion of a function.
    M : int > 0 or None
        Degree of the squared Chebyshev polynomial.

    Returns
    -------
    nu : np.ndarray of shape (N_t, M + 1)
        Chebyshev coefficients for square of expansion defined by mu.

    References
    ----------
    [4] G. Baszenski, M. Tasche. Fast Polynomial Multiplication and Convolutions
        Related to the Discrete Cosine Transform.
        Linear Algebra and its Applications 252:1-25. (1997)
        DOI: https://doi.org/10.1016/0024-3795(95)00696-6
    """
    M_mu_1 = mu_1.shape[1] - 1
    M_mu_2 = mu_2.shape[1] - 1
    if M is None:
        M = M_mu_1 + M_mu_2
    mu_1_tilde = np.hstack((mu_1, np.zeros((mu_1.shape[0], M + 1 - M_mu_1))))
    mu_2_tilde = np.hstack((mu_2, np.zeros((mu_2.shape[0], M + 1 - M_mu_2))))

    if adjust_first:
        mu_1_tilde[:, 0] *= 2
        mu_2_tilde[:, 0] *= 2

    nu = (M + 1) * sp.fft.dct(sp.fft.idct(mu_1_tilde, type=1) * sp.fft.idct(mu_2_tilde, type=1), type=1)[:, : M + 1]

    if adjust_first:
        nu[:, 0] /= 2

    return nu


def chebyshev_recurrence(mu, A, T_0=None, L=None, final_shape=()):
    """
    Implements Chebyshev-recurrence to compute

        Z = L( \\sum_{l=0}^{M} mu_l T_l(A) T_0 )

    Parameters
    ----------
    mu : np.ndarray (N_t, M)
        The Chebyshev-coefficients at each time step.
    A : sp.sparse.matrix or np.ndarray (N, N)
        The matrix used as argument in the Chebyshev-polynomial.
    T_0 : np.ndarray
        The initial value of the recurrence (for speed).
    L : function
        Function handle to a linear function applied in recurrence (for speed).
    final_shape : tuple
        The shape the output array Z has at each time step (depends on L and T_0).

    Returns 
    -------
    Z : np.ndarray (N_t, final_shape)
        The evaluated Chebyshev matrix polynomial for A with coefficients mu.
    """
    if L is None:
        L = lambda x: x
    if T_0 is None:
        T_0 = np.eye(A.shape[0])
    if len(mu.shape) < 2:
        mu = mu.reshape(1, -1)

    Z = np.zeros((mu.shape[0], *final_shape))

    T_prev = np.zeros_like(T_0)
    T_curr = T_0.copy()
    T_next = np.zeros_like(T_0)

    for l in range(mu.shape[1]):
        X = L(T_curr)
        for i in range(mu.shape[0]):
            Z[i] += mu[i, l] * X
        T_next = (1 if l == 0 else 2) * A @ T_curr - T_prev
        T_prev = T_curr.copy()
        T_curr = T_next.copy()

    return Z


def _squared_chebyshev_coefficients_ifft(t, M, function, N_theta=None):
    if not isinstance(t, np.ndarray):
        t = np.array(t).reshape(-1)
    if N_theta is None:
        N_theta = 2 * M + 1
    theta = np.arange(2 * N_theta) * np.pi / N_theta

    # Compute the coefficients mu for all t and l simultaneously with FFT
    t_minus_theta = np.subtract.outer(t, np.cos(theta))
    function_sampled = function(t_minus_theta) ** 2
    mu_tilde = np.fft.fft(function_sampled, axis=1)

    phase = np.exp(1j * np.pi * np.arange(2*N_theta) / (2 * N_theta))
    function_half_samples = np.fft.ifft(mu_tilde * phase, axis=1)

    function_double_sampled = np.empty((len(t), 4*N_theta), dtype=np.complex64)
    function_double_sampled[:, 0::2] = function_sampled
    function_double_sampled[:, 1::2] = function_half_samples

    nu_tilde = np.fft.fft(function_double_sampled, axis=1)

    nu = np.real(nu_tilde)[:, :2 * M+1]

    # Rescale the coefficients (as required by the definition)
    nu[:, 0] /= 4 * N_theta
    nu[:, 1:] /= 2 * N_theta

    return nu


def _squared_chebyshev_coefficients_quadrature(mu, M=None, N_theta=None):
    """
    Squared expansion of the polynomial defined by Chebyshev coefficients mu 
    computed using the formula found in [2].

    Parameters
    ----------
    mu : np.ndarray (N_t, M_mu + 1)
        Chebyshev coefficients corresponding to an expansion of a function.
    M : int > 0 or None
        Degree of the squared Chebyshev polynomial.
    N_theta : int > M
        The number of integration points to be used in the quadrature.

    Returns
    -------
    nu : np.ndarray of shape (N_t, M + 1)
        Chebyshev coefficients for square of expansion defined by mu.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 1.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    M_mu = mu.shape[1] - 1
    if M is None:
        M = 2 * M_mu
    if N_theta is None:
        N_theta = M + 1
    theta = np.arange(2 * N_theta) * np.pi / N_theta

    # Compute the coefficients mu for all t and l simultaneously with FFT
    k = np.arange(M_mu + 1)
    k_times_theta = np.multiply.outer(k, theta)
    nu = np.real(np.fft.fft((mu @ np.cos(k_times_theta))**2, axis=1)[:, :M+1])

    # Rescale the coefficients (as required by the definition)
    nu[:, 0] /= 2 * N_theta
    nu[:, 1:] /= N_theta
    return nu


def _squared_chebyshev_coefficients_summation(mu, M=None):
    """
    Squared expansion of the polynomial defined by Chebyshev coefficients mu.
    Trigonometric identities are used to represent the product of two Chebyshev
    polynomials in terms of a sum of Chebyshev polynomials to determine the new
    coefficients.

    Parameters
    ----------
    mu : np.ndarray (N_t, M_mu + 1)
        Chebyshev coefficients corresponding to an expansion of a function.
    M : int > 0 or None
        Degree of the squared Chebyshev polynomial.

    Returns
    -------
    nu : np.ndarray of shape (N_t, M + 1)
        Chebyshev coefficients for square of expansion defined by mu.
    """
    M_mu = mu.shape[1] - 1
    if M is None:
        M = 2 * M_mu
    nu = np.zeros((mu.shape[0], M + 1))

    for n in range(M + 1):
        nu[:, n] = np.sum(mu[:, 0 : max(0, M_mu + 1 - n)] * mu[:, min(n, M_mu + 1) : M_mu + 1], axis=1)
        if n == 0:
            nu[:, n] /= 2
        nu[:, n] += 0.5 * np.sum(mu[:, max(0, n - M_mu) : min(M_mu, n) + 1] * mu[:, n - min(M_mu, n):n - max(0, n - M_mu) + 1][:, ::-1], axis=1)

    return nu
