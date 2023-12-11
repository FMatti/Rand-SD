import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms import FastNyCheb
from src.matrices import ModES3D
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

np.random.seed(0)

methods = FastNyCheb
labels = ["interpolation", "squaring"]
fixed_parameters = [{"n_v": 80, "sigma": 0.05, "square_coefficients": "interpolation"},
                    {"n_v": 80, "sigma": 0.05, "square_coefficients": "transformation"}]
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.5, 7).astype(int) // 2) * 2

A = ModES3D(dim=2)
colors = ["#7ab3f0", "#2F455C"]
spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

fig, ax = plt.subplots(figsize=(4, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["square_coefficients", "n_v", "sigma"], colors=colors, error_metric_name="$L^1$ relative error", x_label="$m$", ax=ax)

plt.savefig("thesis/plots/interpolation_issue.pgf", bbox_inches="tight")