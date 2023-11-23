"""
Matrices
--------

Assembly routines for the matrices used to test the algorithms.
"""

import functools

import numpy as np
import scipy as sp

from src.utils import regular_grid, periodic_gaussian


def second_derivative_finite_difference(N, h=1, bc="dirichlet"):
    """
    Matrix which applies the fininte difference second derivative to a vector.

    Parameters
    ----------
    N : int > 0
        Number of grid points.
    h : int or float > 0
        Spacing between the grid points.
    bc : "dirichlet" or "periodic"
        Nature of the boundary conditions.

    Returns
    -------
    A : np.ndarray of shape (N, N)
        The finite difference matrix corresponding to the problem.
    """
    A = sp.sparse.diags(
        diagonals=np.multiply.outer([1, -2, 1], np.ones(N)),
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="lil",
    )

    if bc == "dirichlet":
        pass
    elif bc == "periodic":
        A[-1, 0] = 1
        A[0, -1] = 1
    else:
        raise ValueError("Boundary condition bc='{}' is unknown.".format(bc))

    return A / h**2


def laplace_finite_difference(N, h=1, dim=3, bc="dirichlet"):
    """
    Matrix which applies the fininte difference second derivative to a vector.

    Parameters
    ----------
    N : int > 0
        Number of grid points.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.
    bc : str {"dirichlet", "periodic"}
        Nature of the boundary conditions.

    Returns
    -------
    L : np.ndarray of shape (N, N)
        The finite difference matrix corresponding to the Laplace operator.

    References
    ----------
    [3] D. Kressner. Low Rank Approximation Techniques. Lecture Notes. (2022)
        Lecture 7, Slide 24.
    """
    A = -second_derivative_finite_difference(N=N, h=h, bc=bc)

    # Generate Laplace matrix using Kronecker products as seen in [3]
    L = sp.sparse.csr_matrix((N**dim, N**dim))
    for i in range(dim):
        L += functools.reduce(
            sp.sparse.kron,
            i * [sp.sparse.eye(N)] + [A] + (dim - i - 1) * [sp.sparse.eye(N)],
        )

    return L


def ModES3D(n=1, L=6, h=0.6, dim=3, bc="periodic", var=1, prefactor=1, shift=0):
    """
    Generate the example matrices 'ModES3D_X' from [2].

    Parameters
    ----------
    n : int > 0
        Number of unit cells of Gaussians in each dimension (X = n**dim).
    L : int or float > 0
        Length of the unit cells.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.
    bc : str {"dirichlet", "periodic"}
        Nature of the boundary conditions.
    var : int or float > 0
        The variance of the Gaussians.
    prefactor :  int or float
        Scaling factor of the Gaussians.
    shift : int or float
        Shift of the Gaussians.

    Returns
    -------
    np.ndarray of shape (N, N)
        The matrix corresponding to the operator.

    Remarks
    -------
    The Gaussians are constructed using the following formula:
 
        g(r) = prefactor / √(2π * var) * exp(- r^2 / (2 * var)) + shift

    References
    ----------
    [2] Lin, L. Randomized estimation of spectral densities of large matrices
        made accurate. Numer. Math. 136, 203-204 (2017).
        DOI: https://doi.org/10.1007/s00211-016-0837-7
    """
    N = n * round(L / h)
    A = laplace_finite_difference(N=N, h=h, dim=dim, bc=bc)
    grid_points = regular_grid(a=0, b=L * n, N=N, dim=dim)
    V = sp.sparse.diags(
        prefactor * periodic_gaussian(grid_points, L=L, n=n, var=var) + shift
    )
    return A + V


def laplace_finite_difference_eigvals(n=1, L=6, h=0.6, dim=3):
    """
    Eigenvalues of periodic finite difference discretization of the Laplacian.

    Parameters
    ----------
    n : int > 0
        Number of unit cells of Gaussians in each dimension (X = n**dim).
    L : int or float > 0
        Length of the unit cells.
    h : int or float > 0
        Spacing between the grid points.
    dim : int > 0
        The spatial dimension of the grid.

    Returns
    -------
    eigvals : np.ndarray of shape (N,)
        The eigenvalues corresponding to the problem.

    References
    ----------
    [4] Wikipedia. Eigenvalues and eigenvectors of the second derivative. 2023.
        https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors_of_the_second_derivative
    """
    n = L / h
    j = np.arange(n) + np.arange(n) % 2
    eigvals1d = 4 / h**2 * np.sin(np.pi * j / (2 * n)) ** 2
    eigvals = functools.reduce(np.add.outer, dim * [eigvals1d]).flatten()
    return eigvals


class WikiVoteGraph(object):
    def __init__(self, edges_filename="matrices/WikiVote.npz"):
        self.edges = np.load(edges_filename)["edges"]
        self.num_nodes = np.max(self.edges) + 1
        self.cliques = []

    def add_clique(self, clique_nodes):
        self.cliques.append(clique_nodes)

    def add_random_clique(self, k_min=10, k_max=150):
        k = np.random.randint(k_min, k_max)
        nodes = np.random.choice(np.arange(0, self.num_nodes), size=k, replace=False)
        self.add_clique(nodes)

    def delete_clique(self, k):
        if k >= len(self.cliques):
            print("Clique {} does not exist.".format(k))
        self.cliques.pop(k)

    def delete_random_clique(self):
        k = np.random.randint(0, len(self.cliques))
        self.cliques.pop(k)

    def assemble_adjacency_matrix(self):
        A_directed = sp.sparse.coo_matrix(
            (np.ones(self.edges.shape[1]), (self.edges[0], self.edges[1])),
            shape=2*(self.num_nodes,),
            dtype=bool
        )
        A = (A_directed + A_directed.T).astype(int)
        for clique in self.cliques:
            A_clique = np.ones(2*(len(clique),))
            np.fill_diagonal(A_clique, 0)
            A[np.ix_(clique, clique)] = A_clique
        return A
