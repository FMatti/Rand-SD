import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import FastNyCheb
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

methods = FastNyCheb
labels = ["raw algorithm", "improved algorithm"]
fixed_parameters = [{"n_v": 80, "sigma": 0.05, "eigenproblem": "pinv", "eta": 0, "kappa": 0},
                    {"n_v": 80, "sigma": 0.05, "eigenproblem": "standard", "eta": 1e-3, "kappa": 1e-5}]
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.5, 7).astype(int) // 2) * 2

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
colors = ["#2F455C", "#F98125"]
spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

fig, ax = plt.subplots(figsize=(4, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["eigenproblem", "n_v", "sigma", "eta", "kappa"], colors=colors, error_metric_name="$L^1$ relative error", x_label="$m$", ax=ax)

plt.savefig("thesis/plots/algorithmic_improvements.pgf", bbox_inches="tight")