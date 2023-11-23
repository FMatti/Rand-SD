"""
Utils
-----

Utility functions for the implementations.
"""

import itertools
import urllib.request
import tarfile
import os
import tempfile
import shutil

import numpy as np
import scipy as sp

from src.kernel import gaussian_kernel


def download_matrix(url, save_path="matrices", save_name=None):
    """
    Download a matrix from an online matrix market archive.

    Parameters
    ----------
    url : str
        The URL, e.g. https://www.[...].com/matrix.tar.gz.
    save_path : str
        The path under which the matrix should be saved.
    save_name : str or None
        The filename of the matrix. When None, name is inferred from url.

    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Download the archive containing the matrix
        file_name = os.path.join(temp_dir, "archive.tar.gz")
        archive_file_path, _ = urllib.request.urlretrieve(url, file_name)

        # Open the archive
        with tarfile.open(archive_file_path, "r:gz") as tar:
            # Extract only the ".mtx" files
            mtx_members = [m for m in tar.getmembers() if m.name.endswith(".mtx")]
            tar.extractall(path=temp_dir, members=mtx_members)

            # Convert and save matrices as scipy.sparse.matrix
            for m in mtx_members:
                matrix = sp.io.mmread(os.path.join(temp_dir, m.name))
                if save_name is None:
                    save_name = os.path.splitext(os.path.basename(m.name))[0]
                try:
                    sp.sparse.save_npz(os.path.join(save_path, save_name), matrix)
                except:
                    continue

    finally:
        # Clean up: Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)


def form_spectral_density(eigenvalues, kernel=gaussian_kernel, N=None, a=-1, b=1, N_t=100, sigma=0.1):
    """
    Compute the (regularized) spectral density of a (small) matrix A at N_t
    evenly spaced grid-points within the interval [a, b].

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        The eigenvalues for which the spectral density should be computed.
    N : int > 0
        Size of the matrix A. If None, then the size is assumed to be equal to
        the number of computed eigenvalues.
    a : int or float
        The starting point of the interval within which the density is computed.
    b : int or float > a
        The ending point of the interval within which the density is computed.
    N_t : int > 0
        Number of evenly spaced grid points at which the density is evaluated.
    sigma : int or float > 0
        Smearing parameter of the spectral density.

    Returns
    -------
    spectral_density : np.ndarray of shape (N_t,)
        The value of the spectral density evaluated at the grid points.
    """
    spectral_density = np.zeros(N_t)
    grid_points = np.linspace(a, b, N_t)

    for eigenvalue in eigenvalues:
        spectral_density += kernel(
            grid_points - eigenvalue, N=N if N else len(eigenvalues), sigma=sigma
        )

    return spectral_density


def density_to_distribution(phi, t=None, normalize=False):
    if t is None:
        t = np.linspace(-1, 1, len(phi))
    spectral_distribution = np.cumsum(phi) * np.append(0, np.diff(t))

    if normalize:
        spectral_distribution /= spectral_distribution[-1]
    return spectral_distribution


def form_spectral_distribution(eigenvalues, kernel=gaussian_kernel, N=None, a=-1, b=1, N_t=100, sigma=0.1):
    """
    Compute the (regularized) spectral distribution of a (small) matrix A at N_t
    evenly spaced grid-points within the interval [a, b].

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        The eigenvalues for which the spectral distribution should be computed.
    N : int > 0
        Size of the matrix A. If None, then the size is assumed to be equal to
        the number of computed eigenvalues.
    a : int or float
        The starting point of the interval within which the distribution is computed.
    b : int or float > a
        The ending point of the interval within which the distribution is computed.
    N_t : int > 0
        Number of evenly spaced grid points at which the distribution is evaluated.
    sigma : int or float > 0
        Smearing parameter of the spectral distribution.

    Returns
    -------
    spectral_distribution : np.ndarray of shape (N_t,)
        The value of the spectral distribution evaluated at the grid points.
    """
    spectral_density = np.zeros(N_t)
    grid_points = np.linspace(a, b, N_t)

    for eigenvalue in eigenvalues:
        spectral_density += kernel(
            grid_points - eigenvalue, N=N if N else len(eigenvalues), sigma=sigma
        )
    spectral_distribution = np.cumsum(spectral_density)
    return spectral_distribution


def regular_grid(a=0, b=1, N=10, dim=3):
    """
    Generate a regularly spaced grid in all dimensions.

    Parameters
    ----------
    a : int or float
        Starting point of grid in all dimensions.
    b : int or float
        Ending point of grid in all dimensions.
    N : int
        Number of grid-points.
    dim : int > 0
        The spatial dimension of the grid.

    Returns
    -------
    grid_points : np.ndarray of shape (n,)
        The Gaussian function evaulated at all points X.
    """
    grid = np.meshgrid(*dim * [np.linspace(a, b, N)])
    grid_points = np.vstack([x.flatten() for x in grid]).T
    return grid_points


def gaussian(X, mu=None, var=None):
    """
    Gaussian function

    Parameters
    ----------
    X : int, float, or np.ndarray of shape (n, dim)
        Point(s) where the Gaussian function should be evaluated.
        Format: array([[x1, y1, ...], [x2, y2, ...], ...])
    mu : int, float, or np.ndarray of shape (dim,)
        Multivariate mean vector of the Gaussian function. If None is given, the
        zero vector is taken. If int or float are given, they are extended to
        the constant mean vector of appropriate dimension (given by X).
    var : int, float, or np.ndarray of shape (dim,) or (dim, dim)
        Multivariate variance of the Gaussian function. If None is given, the
        identity matrix is taken. If int or float are given, they are extended
        to the constant variance matrix of appropriate dimension (given by X).

    Returns
    -------
    g(X) : np.ndarray of shape (n, dim)
        The Gaussian function evaulated at all points X.

    [1] Simon J.D. Prince. Computer Vision: Models, Learning, and Inference.
        Cambridge University Press. (2012)
    """
    # Convert X to a numpy array  of shape (n, d) to make computations easier
    if not isinstance(X, np.ndarray):
        X = np.array(X).reshape(-1, 1)
    if len(X.shape) < 2:
        X = X.reshape(1, -1)  # Need to break tie between (n, 1) and (1, dim) vectors
    dim = X.shape[1]

    # Parse the mean vector
    if mu is None:
        mu = np.zeros(dim)
    elif not isinstance(mu, np.ndarray):
        mu = mu * np.ones(dim)

    # Parse the variance matrix
    if var is None:
        var = np.ones(dim)
    elif not isinstance(var, np.ndarray):
        var = var * np.ones(dim)
    if len(var.shape) < 2:
        var = np.diag(var)

    normalization = np.sqrt((2 * np.pi) ** dim * np.linalg.det(var))
    diff = X - mu
    exponent = -0.5 * np.sum(diff * np.dot(diff, np.linalg.inv(var)), axis=1)
    return np.exp(exponent) / normalization


def periodic_gaussian(X, n=1, L=6, var=None):
    """
    Potential constructed using periodic repetitions of a Gaussian unit cell.

    Parameters
    ----------
    X : int, float, or np.ndarray of shape (n, dim)
        Point at which the potential function should be evaluated.
    n : int
        Number of repeated unit cells in each dimension.
    L : int or float
        Length of the unit cells.
    var : int or float
        Variance of the Gaussians.

    Returns
    -------
    potential : np.ndarray of shape (n,)
        The periodic Gaussian potential function evaulated at all points X.

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017).
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    if np.min(X) < 0 or np.max(X) > n * L:
        raise ValueError("Point x={} is outside of specified domain.".format(X))

    if not isinstance(X, np.ndarray):
        X = np.array(X).reshape(-1, 1)
    if len(X.shape) < 2:
        X = X.reshape(1, -1)
    dim = X.shape[1]

    # Generate the indices of all cells (e.g. 1st (0, 0, 0), 2nd (0, 0, 1), ...)
    cell_indices = itertools.product(*dim * [range(n)])

    # Add up the contributions from all cells
    potential = np.zeros(X.shape[0])
    for cell_index in cell_indices:
        mu = L / 2 * np.ones(dim) + L * np.array(cell_index)
        potential += gaussian(X, mu=mu, var=var)

    return potential


def spectral_transformation(A, min_ev=None, max_ev=None, return_ev=False):
    """
    Perform a spectral transformation of a matrix, i.e. transform a spectrum
    contained in (a, b) to (-1, 1).

    Parameters
    ----------
    A : np.ndarray of shape (N, N) or (N,)
        The matrix or vector to be spectrally transformed.
    min_ev : int, float or None
        The starting point of the spectrum. If None is specified, the smallest
        eigenvalue of A is computed and used for min_ev.
    max_ev : int, float or None
        The ending point of the spectrum. If None is specified, the largest
        eigenvalue of A is computed and used for max_ev.
    return_ev : bool
        Whether to return the computed minimum and maximum eigenvalue of A.
    """
    if min_ev is None or max_ev is None:
        eigenvalues = np.linalg.eigvalsh(A if isinstance(A, np.ndarray) else A.toarray())
        if min_ev is None:
            min_ev = np.min(eigenvalues)
        if max_ev is None:
            max_ev = np.max(eigenvalues)
    I = 1
    if isinstance(A, np.ndarray) or isinstance(A, sp.sparse.spmatrix):
        if len(A.shape) == 2:
            I = sp.sparse.eye(*A.shape)
    A_transformed = (2 * A - (min_ev + max_ev) * I) / (max_ev - min_ev)
    
    if return_ev:
        return A_transformed, min_ev, max_ev
    return A_transformed


def inverse_spectral_transformation(A, min_ev, max_ev):
    """
    Perform an inverse spectral transformation of a matrix, i.e. transform a
    spectrum contained in (-1, 1) to (a, b).

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        A matrix which was spectrally transformed from (a, b) to (-1, 1).
    min_ev : int, float or None
        The starting point of the spectrum.
    max_ev : int, float or None
        The ending point of the spectrum.
    """
    I = 1
    if isinstance(A, np.ndarray) or isinstance(A, sp.sparse.spmatrix):
        if len(A.shape) == 2:
            I = sp.sparse.eye(*A.shape)
    return (max_ev - min_ev) / 2 * A + (min_ev + max_ev) / 2 * I


def continued_fraction(z, a, b):
    """
    Recursively compute the continued fraction of the form

        1 / (z - a[0] - b[0]^2 * 1 / (z - a[1] - b[1] * 1 / (...))).

    Parameters
    ----------
    z : float or complex
        The point at which it is evaluated.
    a : np.ndarray
        Coefficients (diagonal of tridiagonal matrix).
    b : np.ndarray
        Coefficients (off-diagonals of tridiagonal matrix).

    Returns
    -------
    float or complex
        The result of the recursion.
    """
    if len(a) == 1:
        return 1 / (z - a[-1])
    # Here it's - b[0] and not + b[0] like in Lin/Saad/Yang 2016
    return 1 / (z - a[0] - b[0]**2 * continued_fraction(z, a[1:], b[1:]))


def theoretical_numerical_rank(N, sigma, epsilon=1e-16):
    """
    Determine the theoretical numerical rank.

    Parameters
    ----------
    N : int > 0
        The size of the matrix, i.e. the number of eigenvalues.
    sigma : int or float > 0
        The smearing-parameters, i.e. the width of the Gaussians.
    epsilon : float > 0
        The value below which singular values are considered equal to zero.
    """
    return N * sigma * np.sqrt(- 2 * np.log(sigma * epsilon * np.sqrt(2 * np.pi)))


def theoretical_chebyshev_degree(sigma, epsilon=1e-16):
    """
    Determine the theoretical degree needed to achieve an error of epsilon.

    Parameters
    ----------
    sigma : int or float > 0
        The smearing-parameters, i.e. the width of the Gaussians.
    epsilon : float > 0
        The tolerated error of the Chebyshev interpolation.
    """
    return - np.log(epsilon * sigma**2) / np.log(1 + sigma)


def verify_parameters(N, sigma, M, N_v, N_v_tilde=None, epsilon=1e-16):
    M_min = theoretical_chebyshev_degree(sigma, epsilon)
    N_v_min = theoretical_numerical_rank(N, sigma, epsilon)

    if M < M_min:
        print("Degree of Chebyshev polynomial too low.")
    if N_v < N_v_min:
        print("Number of random vectors for low-rank approximation too low.")
