import __context__

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import NC
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

methods = NC
labels = [r"$\rho=0$", r"$\rho=10^{-7}$"]
fixed_parameters = [{"m": 2400, "sigma": 0.05, "rho": 0.0},
                    {"m": 2400, "sigma": 0.05, "rho": 1e-7}]
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")
colors = ["#2F455C", "#F98125"]
spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

fig, ax = plt.subplots(figsize=(4, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["rho", "m", "sigma"], colors=colors, error_metric_name="$L^1$ relative error", x_label="$n_{\Omega}$", ax=ax)

plt.savefig("thesis/plots/kernel_shift.pgf", bbox_inches="tight")