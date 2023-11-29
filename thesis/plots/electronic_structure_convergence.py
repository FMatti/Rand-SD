import __context__

import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import DGC, FastNyCheb, FastNyChebPP
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")

methods = [DGC, FastNyCheb, FastNyChebPP]
labels = ["DGC", "NC", "NC++"]

################################################################################

fixed_parameters = {"m": 800, "sigma": 0.05}
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

with open("thesis/plots/electronic_structure_convergence_nv_m800.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/electronic_structure_convergence_nv_m800.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ error", ax=ax)
plt.savefig("thesis/plots/electronic_structure_convergence_nv_m800.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"m": 2400, "sigma": 0.05}
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

with open("thesis/plots/electronic_structure_convergence_nv_m2400.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/electronic_structure_convergence_nv_m2400.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ error", ax=ax)
plt.savefig("thesis/plots/electronic_structure_convergence_nv_m2400.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"n_v": 40, "sigma": 0.05}
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.6, 6).astype(int) // 2) * 2

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

with open("thesis/plots/electronic_structure_convergence_m_nv40.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/electronic_structure_convergence_nv_m2400.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ error", ax=ax)
plt.savefig("thesis/plots/electronic_structure_convergence_m_nv40.pgf", bbox_inches="tight")

################################################################################

fixed_parameters = {"n_v": 160, "sigma": 0.05}
variable_parameters = "m"
variable_parameters_values = (np.logspace(2.3, 3.6, 6).astype(int) // 2) * 2

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

with open("thesis/plots/electronic_structure_convergence_m_nv160.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/electronic_structure_convergence_nv_m2400.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ error", ax=ax)
plt.savefig("thesis/plots/electronic_structure_convergence_m_nv160.pgf", bbox_inches="tight")
