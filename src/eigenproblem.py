"""
Eigenproblem
------------

Different solution strategies for solving the generalized eigenvalue problem

    K_Z * C = K_W * C * Xi
"""

import numpy as np
import scipy as sp


def generalized_eigenproblem_standard(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    s, U = np.linalg.eigh(K_W)

    idx = np.where(s >= tau * np.max(s))[0].flatten()
    s_tilde_invsqrt = s[idx] ** (-0.5)
    U_tilde = U[:, idx]
    #print("Keeping {} of s".format(np.sum(s >= tau * np.max(s)) / len(s)))
    A = np.outer(s_tilde_invsqrt, s_tilde_invsqrt) * (U_tilde.T @ K_Z @ U_tilde)

    xi, X = np.linalg.eigh(A)

    # Increase maximum allowed value slightly to avoid unwanted filtering
    max_val = (1 + epsilon) / (n * sigma * np.sqrt(2 * np.pi))

    idx_tilde = np.where(np.logical_and(0 <= xi, xi <= max_val))[0].flatten()
    xi_tilde = xi[idx_tilde]
    X_tilde = X[:, idx_tilde]
    #print("Keeping {} of xi".format(np.sum(np.logical_and(0 <= xi, xi <= max_val)) / len(xi)))

    C_tilde = U_tilde @ np.diag(s_tilde_invsqrt) @ X_tilde

    return xi_tilde, C_tilde


def generalied_eigenproblem_direct(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    xi, C_l, C_r = sp.linalg.eig(K_Z, K_W, left=True, right=True)
    conditioning = 1 / np.abs(np.diag(C_l.conjugate().T @ C_r))
    idx = conditioning > tau
    #print(np.sum(idx))
    return xi, C_r #xi[idx], C_r[:, idx]


def generalized_eigenproblem_kernelunion(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    s_Z, U_Z = np.linalg.eigh(K_Z)

    idx = np.where(s_Z > tau * np.max(s_Z))[0].flatten()
    U_Z_tilde = U_Z[:, idx]

    s_W, U_W = np.linalg.eigh(U_Z_tilde.T @ K_W @ U_Z_tilde)

    idx = np.where(s_W > tau * np.max(s_W))[0].flatten()
    s_W_tilde = s_W[idx] ** (-0.5)
    U_W_tilde = U_W[:, idx]

    A = np.outer(s_W_tilde, s_W_tilde) * (U_W_tilde.T @ U_Z_tilde.T @ K_Z @ U_Z_tilde @ U_W_tilde)

    xi, X = np.linalg.eigh(A)

    # Increase maximum allowed value slightly to avoid unwanted filtering
    max_val = (1 + epsilon) / (n * np.sqrt(2 * np.pi * sigma**2))

    idx_tilde = np.where(np.logical_and(0 <= xi, xi <= max_val))[0].flatten()
    xi_tilde = xi[idx_tilde]
    C_tilde = U_Z_tilde @ U_W_tilde @ np.diag(s_W_tilde) @ X[:, idx_tilde]

    return xi_tilde, C_tilde


def generalized_eigenproblem_pinv(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    s, U = np.linalg.eigh(K_Z)

    idx = np.where(s >= tau**2 * np.max(s))[0].flatten()
    s_tilde = s[idx]
    U_tilde = U[:, idx]
    K_Z_truncated = U_tilde @ np.diag(s_tilde) @ U_tilde.T

    Xi = np.linalg.pinv(K_W, rcond=tau) @ K_Z_truncated

    return np.diag(Xi), None


def generalized_eigenproblem_dggev(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    alphar, alphai, beta, _, _, _, _ = sp.linalg.lapack.dggev(K_Z, K_W)
    idx = np.abs(beta) > 1e-7
    xi_tilde = alphar[idx] / beta[idx]

    return xi_tilde, None


def generalized_eigenproblem_lstsq(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    Xi = np.linalg.lstsq((K_W + K_W.T) / 2, (K_Z + K_Z.T) / 2, rcond=tau)[0]

    return np.diag(Xi), None


def generalized_eigenproblem_cholesky(K_Z, K_W, n, sigma=1.0, tau=1e-7, epsilon=1e-1):
    """
    Solve the generalized eigenvalue problem for the spectrum sweeping method.

    Parameters
    ----------
    K_Z : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_Z = Z* Z = W* P^2 W.
    K_W : np.ndarray of shape (n_v, n_v)
        Reduced matrix defined as K_W = W* Z = W* P W.
    sigma : int or float > 0
        Smearing parameter.
    tau : int or float in (0, 1]
        Truncation parameter.
    epsilon : float > 0
        The tolerance for removing eigenvalues which are outside the range of
        g_sigma. 

    Returns
    -------
    xi_tilde : np.ndarray
        The generalized eigenvalues.
    C_tilde : np.ndarray
        The generalized eigenvectors.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017). Algorithm 4.
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """

    C = np.linalg.cholesky(K_W)

    C_tilde = np.linalg.pinv(C, rcond=tau)

    xi_tilde = np.diag(C_tilde @ C_tilde.T @ K_W)

    return xi_tilde, C_tilde
