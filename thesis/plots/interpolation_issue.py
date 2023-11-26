import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.algorithms import FastNyCheb
from src.matrices import ModES3D
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

np.random.seed(0)

methods = FastNyCheb
labels = ["interpolation", "squaring"]
fixed_parameters = [{"N_v": 80, "sigma": 0.05, "square_coefficients": "interpolation"},
                    {"N_v": 80, "sigma": 0.05, "square_coefficients": "transformation"}]
variable_parameters = "M"
variable_parameters_values = (np.logspace(2.3, 3.5, 7).astype(int) // 2) * 2

A = ModES3D(dim=2)
colors = ["#7ab3f0", "#2F455C"]
spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, N_t=100)

fig, ax = plt.subplots(figsize=(4, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["square_coefficients", "N_v", "sigma"], colors=colors, error_metric_name="$L^1$ error", ax=ax)

plt.savefig("thesis/plots/interpolation_issue.pgf", bbox_inches="tight")