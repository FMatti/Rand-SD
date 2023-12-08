import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms import chebyshev_coefficients
from src.kernel import gaussian_kernel
from src.metrics import p_norm

import matplotlib
colors = matplotlib.colormaps["magma_r"]
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

t = 0.0
s = np.linspace(-1, 1, 100)
n = 1

plt.figure(figsize=(4, 3))
m_list = np.logspace(1, 4, 10).astype(int)
sigmas = np.linspace(0.01, 1, 10)
for i, sigma in enumerate(sigmas):
    g = lambda x: gaussian_kernel(x, n=n, sigma=sigma)
    g_exact = gaussian_kernel(t - s, n=n, sigma=sigma)

    errors = []
    for m in m_list:
        mu = chebyshev_coefficients(t, m, function=g)
        g_interp = np.polynomial.chebyshev.Chebyshev(mu[0])(s)
        errors.append(p_norm(g_exact, g_interp))

    plt.plot(m_list, errors, color=colors((i + 1) / len(sigmas)))

plt.yscale("log")
plt.xscale("log")
plt.ylabel("$L^1$ relative error")
plt.xlabel("$m$")
plt.savefig("thesis/plots/chebyshev_convergence.pgf".format(sigma), bbox_inches="tight")

plt.figure(figsize=(1.5, 0.2))
plt.ylim([0, 1])
plt.xlim([0, 100])
for i in range(100):
    plt.axvspan(i, i+1, color=colors(i/100))
plt.xticks([0, 50, 100], [0.01, 0.5, 1])
plt.xlabel("t")
plt.yticks([])
plt.savefig("thesis/plots/chebyshev_convergence_colormap.pgf", bbox_inches="tight", transparent=True)
