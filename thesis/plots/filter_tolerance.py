import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import NC
from src.plots import compute_spectral_densities, plot_spectral_densities
from src.utils import spectral_transformation

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

methods = NC
labels = ["no filter tolerance", "filter tolerance"]
parameters = [{"m": 2000, "sigma": 0.05, "n_v": 80, "eta": 0},
              {"m": 2000, "sigma": 0.05, "n_v": 80, "eta": 1e-3}]

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
colors = ["#2F455C", "#F98125"]

eigenvalues = np.linalg.eigvalsh(A.todense())
t = np.sort(np.append(np.linspace(-1, 1, 100), spectral_transformation(eigenvalues, eigenvalues[0], eigenvalues[-1])))
spectral_densities = compute_spectral_densities(A, methods, labels, parameters, add_baseline=False, t=t)

fig, ax = plt.subplots(figsize=(6, 3))
plot_spectral_densities(spectral_densities, parameters, variable_parameter="eta", ignored_parameters=["m", "sigma", "n_v", "eta"], colors=colors, ax=ax, t=t)

plt.savefig("thesis/plots/filter_tolerance.pgf", bbox_inches="tight")
