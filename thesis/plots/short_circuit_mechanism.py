import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import FastNyCheb
from src.plots import compute_spectral_densities, plot_spectral_densities

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

methods = FastNyCheb
labels = ["no short-circuit", "short-circuit"]
parameters = [{"m": 2000, "sigma": 0.05, "n_v": 80, "kappa": -1},
              {"m": 2000, "sigma": 0.05, "n_v": 80, "kappa": 1e-5}]

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
colors = ["#2F455C", "#F98125"]
spectral_densities = compute_spectral_densities(A, methods, labels, parameters, add_baseline=False, n_t=500)

fig, ax = plt.subplots(figsize=(6, 3))
plot_spectral_densities(spectral_densities, parameters, variable_parameter="kappa", ignored_parameters=["m", "sigma", "n_v", "kappa"], colors=colors, ax=ax)

plt.savefig("thesis/plots/short_circuit_mechanism.pgf", bbox_inches="tight")
