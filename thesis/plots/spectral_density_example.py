import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.utils import form_spectral_density

# Define sample points
n_t = 500
x = np.linspace(0, 1, n_t)

# Generate random eigenvalues
n_ev = 10
np.random.seed(42)
ev = np.random.rand(n_ev)

# Plot spectral density for different values of sigma
sigmas = [0.01, 0.02, 0.05]
for sigma in sigmas:
    fig, ax = plt.subplots(figsize=(2.25, 2))
    ax.set_xlim([0, 1])
    ax.set_ylim(-11, 80)
    ax.axis("off")
    phi = form_spectral_density(ev, a=0, b=1, n_t=n_t, n=1, sigma=sigma)
    plt.plot(x, phi, linewidth=1.0, color="#2F455C")
    plt.savefig("thesis/plots/spectral_density_example_{:.2f}.pgf".format(sigma), bbox_inches="tight")
