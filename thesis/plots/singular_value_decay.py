import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.utils import spectral_transformation, theoretical_numerical_rank, form_spectral_density
from src.kernel import gaussian_kernel

import matplotlib
colors = matplotlib.colormaps["magma_r"]
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
min_ev = sp.sparse.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
max_ev = sp.sparse.linalg.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
A = spectral_transformation(A, min_ev, max_ev)

eigenvalues = np.sort(np.linalg.eigvalsh(A.toarray()))

def g_sigma_eigenvalues(A, t, sigma):
    A_eigenvalues = np.linalg.eigvalsh(A.toarray())
    t_minus_A_eigenvalues = np.subtract.outer(t, A_eigenvalues)
    eigenvalues = gaussian_kernel(t_minus_A_eigenvalues, n=A.shape[0], sigma=sigma)
    return np.sort(eigenvalues, axis=1)[:, ::-1]

sigma = 0.05
epsilon = 1e-16 
n_t = 100
t = np.linspace(0, 1, n_t)
eigenvalues = g_sigma_eigenvalues(A, t=t, sigma=sigma / ((max_ev - min_ev) / 2))

plt.figure(figsize=(4, 3))
for i in range(n_t):
    plt.plot(eigenvalues[i], color=colors(i / n_t), label="$t = {:.2f}$".format(t[i]))
#plt.axvline(theoretical_numerical_rank(A.shape[0], sigma / ((max_ev - min_ev) / 2), epsilon=epsilon), linewidth=3, color="white")
plt.axvline(theoretical_numerical_rank(A.shape[0], sigma / ((max_ev - min_ev) / 2), epsilon=epsilon), linewidth=1, linestyle="dashed", color="black")
plt.text(theoretical_numerical_rank(A.shape[0], sigma / ((max_ev - min_ev) / 2), epsilon=epsilon), 8e-1, "$r_{\\varepsilon, 2}$", va="top", rotation=270)
plt.yscale("log")
plt.ylim([1e-16, 1e0])
plt.xlim([0, A.shape[0] / 8])
plt.ylabel("$g_{\sigma}(t - \lambda_{(i)})$")
plt.xlabel("$i$")
plt.savefig("thesis/plots/singular_value_decay.pgf", bbox_inches="tight")

plt.figure(figsize=(1.5, 0.2))
plt.ylim([0, 1])
plt.xlim([0, 100])
for i in range(100):
    plt.axvspan(i, i+1, color=colors(abs(100 - 2*i)/100))
plt.xticks([0, 25, 50, 75, 100], [-1.0, -0.5, 0.0, 0.5, 1.0])
plt.xlabel("$t$")
plt.yticks([])
plt.savefig("thesis/plots/singular_value_decay_colormap.pgf", bbox_inches="tight", transparent=True)

spectral_density = form_spectral_density(np.linalg.eigvalsh(A.toarray()), gaussian_kernel, sigma=0.05 * 2 / (max_ev - min_ev))
plt.figure(figsize=(1.5, 0.5))
plt.xlim([-1, 1])
plt.axis("off")
plt.plot(np.linspace(-1, 1, len(spectral_density)), spectral_density, color="black")
plt.savefig("thesis/plots/singular_value_decay_spectral_density.pgf", bbox_inches="tight", transparent=True)
