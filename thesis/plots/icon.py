import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.utils import form_spectral_density
from src.matrices import gaussian_orthogonal_ensemble

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

# Define sample points
n_t = 500
x = np.linspace(-10, 10, n_t)

# Generate random eigenvalues
n_ev = 10
sigma = 0.5

fig, ax = plt.subplots(figsize=(2.25, 2), facecolor="#2F455C")
ax.set_xlim([-10, 10])
ax.set_ylim(-0.1, 2)
ax.axis("off")
ax.set_facecolor("#2F455C")

# Plot spectral density for different values of sigma
for i in range(15):
    ev = np.linalg.eigvalsh(gaussian_orthogonal_ensemble(n=7, seed=i))
    phi = i/8 + form_spectral_density(ev, a=-10, b=10, n_t=n_t, n=7, sigma=sigma)
    plt.plot(x, phi, linewidth=1.0, color="white", zorder=101-2*i)
    plt.fill_between(x, 0, phi, color="#2F455C", zorder=100-2*i)

plt.savefig("thesis/plots/icon.pgf".format(sigma), bbox_inches="tight")

fig, ax = plt.subplots(figsize=(7, 2), facecolor="#0d1117")
ax.set_xlim([-11, 54])
ax.set_ylim(-0.1, 2)
ax.axis("off")
ax.set_facecolor("#0d1117")

# Plot spectral density for different values of sigma
for i in range(15):
    ev = np.linalg.eigvalsh(gaussian_orthogonal_ensemble(n=7, seed=i))
    phi = i/8 + form_spectral_density(ev, a=-10, b=10, n_t=n_t, n=7, sigma=sigma)
    plt.plot(x, phi, linewidth=1.0, color="white", zorder=101-2*i)
    plt.fill_between(x, 0, phi, color="#0d1117", zorder=100-2*i)

font = {'color' : 'white', 'weight' : 'light', 'size' : 38}
ax.text(13, 1.2, r"\underline{Rand}omized estimation", fontdict=font, ha="left")
ax.text(13, 0.2, r"of \underline{S}pectral \underline{D}ensities", fontdict=font, ha="left")
plt.savefig("thesis/plots/icon_github.png".format(sigma), bbox_inches="tight")
