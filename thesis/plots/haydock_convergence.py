import __context__

import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import Haydock, NC, NCPP
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors
from src.kernel import lorentzian_kernel

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")

methods = [Haydock, NC, NCPP]
labels = ["Haydock", "NC", "NC++"]
colors = ["#89A5C2", "#2F455C", "#F98125"]

################################################################################

fixed_parameters = {"m": 800, "sigma": 0.05, "kernel": lorentzian_kernel, "eta": None}
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, kernel=lorentzian_kernel)

with open("thesis/plots/haydock_convergence_nv_m800.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/haydock_convergence_nv_m800.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma", "kernel", "eta"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
plt.savefig("thesis/plots/haydock_convergence_nv_m800.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"m": 2400, "sigma": 0.05, "kernel": lorentzian_kernel, "eta": None}
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, kernel=lorentzian_kernel)

with open("thesis/plots/haydock_convergence_nv_m2400.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/haydock_convergence_nv_m2400.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.plot(variable_parameters_values[1:-1], 0.65/variable_parameters_values[1:-1], linestyle="dashed", color="#7a7a7a", alpha=0.5)
ax.text(8e+1, 9e-3, r"$\mathcal{O}(\varepsilon^{-1})$", color="#7a7a7a")
ax.plot(variable_parameters_values[1:-1], 0.3/variable_parameters_values[1:-1]**(0.5), linestyle="dashed", color="#7a7a7a", alpha=0.5)
ax.text(2.6e+1, 6.5e-2, r"$\mathcal{O}(\varepsilon^{-2})$", color="#7a7a7a")
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma", "kernel", "eta"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
plt.savefig("thesis/plots/haydock_convergence_nv_m2400.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"n_v": 40, "sigma": 0.05, "kernel": lorentzian_kernel, "eta": None}
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.6, 6).astype(int) // 2) * 2

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, kernel=lorentzian_kernel)

with open("thesis/plots/haydock_convergence_m_nv40.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/haydock_convergence_m_nv40.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma", "kernel", "eta"], error_metric_name="$L^1$ relative error", x_label="$m$", colors=colors, ax=ax)
plt.savefig("thesis/plots/haydock_convergence_m_nv40.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"n_v": 160, "sigma": 0.05, "kernel": lorentzian_kernel, "eta": None}
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.6, 6).astype(int) // 2) * 2

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, kernel=lorentzian_kernel)

with open("thesis/plots/haydock_convergence_m_nv160.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/haydock_convergence_m_nv160.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma", "kernel", "eta"], error_metric_name="$L^1$ relative error", x_label="$m$", colors=colors, ax=ax)
plt.savefig("thesis/plots/haydock_convergence_m_nv160.pgf", bbox_inches="tight")
