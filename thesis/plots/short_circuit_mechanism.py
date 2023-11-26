import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms import FastNyCheb
from src.matrices import ModES3D
from src.plots import compute_spectral_densities, plot_spectral_densities

np.random.seed(0)

methods = FastNyCheb
labels = ["no short-circuit", "short-circuit"]
parameters = [{"m": 2000, "sigma": 0.05, "n_v": 80, "delta": -1},
              {"m": 2000, "sigma": 0.05, "n_v": 80, "delta": 1e-5}]

A = ModES3D()
colors = ["#1f1f1f", "#7ab3f0", "#2F455C"]
spectral_densities = compute_spectral_densities(A, methods, labels, parameters, add_baseline=True, n_t=500)

fig, ax = plt.subplots(figsize=(7, 3))
plot_spectral_densities(spectral_densities, parameters, variable_parameter="delta", ignored_parameters=["m", "sigma", "n_v", "delta"], colors=colors, ax=ax)

plt.savefig("thesis/plots/short_circuit_mechanism.pgf", bbox_inches="tight")