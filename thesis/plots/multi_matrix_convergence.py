import __context__

import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import DGC, FastNyCheb, FastNyChebPP
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

fixed_parameters = {"m": 2400, "sigma": 0.05}
variable_parameters = "n_v"
variable_parameters_values = np.logspace(1.3, 2.6, 6).astype(int)

methods = [DGC, FastNyCheb, FastNyChebPP]
labels = ["DGC", "NC", "NC++"]
colors = ["#89A5C2", "#2F455C", "#F98125"]

###############################################################################

#A = sp.sparse.load_npz("matrices/ModES3D_8.npz")
#
#spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)
#
#with open("thesis/plots/multi_matrix_convergence_ModES3D_8.pkl", "wb") as handle:
#    pickle.dump(spectral_density_errors, handle)
#
##with open("thesis/plots/multi_matrix_convergence_ModES3D_8.pkl", "rb") as handle:
##    spectral_density_errors = pickle.load(handle)
#
#fig, ax = plt.subplots(figsize=(2.5, 2.5))
#plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
#plt.savefig("thesis/plots/multi_matrix_convergence_ModES3D_8.pgf", bbox_inches="tight")

################################################################################

#A = sp.sparse.load_npz("matrices/Erdos992.npz")
#
#spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)
#
#with open("thesis/plots/multi_matrix_convergence_Erdos992.pkl", "wb") as handle:
#    pickle.dump(spectral_density_errors, handle)
#
##with open("thesis/plots/multi_matrix_convergence_Erdos992.pkl", "rb") as handle:
##    spectral_density_errors = pickle.load(handle)
#
#fig, ax = plt.subplots(figsize=(2.5, 2.5))
#plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
#plt.savefig("thesis/plots/multi_matrix_convergence_Erdos992.pgf", bbox_inches="tight")

################################################################################

#A = sp.sparse.load_npz("matrices/nd3k.npz")
#
#spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)
#
#with open("thesis/plots/multi_matrix_convergence_nd3k.pkl", "wb") as handle:
#    pickle.dump(spectral_density_errors, handle)
#
##with open("thesis/plots/multi_matrix_convergence_nd3k.pkl", "rb") as handle:
##    spectral_density_errors = pickle.load(handle)
#
#fig, ax = plt.subplots(figsize=(2.5, 2.5))
#plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
#plt.savefig("thesis/plots/multi_matrix_convergence_nd3k.pgf", bbox_inches="tight")

################################################################################

#A = sp.sparse.load_npz("matrices/uniform.npz")
#eigenvalues = np.linspace(-1, 1, A.shape[0])
#
#spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, eigenvalues=eigenvalues)
#
#with open("thesis/plots/multi_matrix_convergence_uniform.pkl", "wb") as handle:
#    pickle.dump(spectral_density_errors, handle)
#
##with open("thesis/plots/multi_matrix_convergence_uniform.pkl", "rb") as handle:
##    spectral_density_errors = pickle.load(handle)
#
#fig, ax = plt.subplots(figsize=(2.5, 2.5))
#plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
#plt.savefig("thesis/plots/multi_matrix_convergence_uniform.pgf", bbox_inches="tight")

################################################################################

#A = sp.sparse.load_npz("matrices/California.npz")
#
##spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)
##
##with open("thesis/plots/multi_matrix_convergence_California.pkl", "wb") as handle:
##    pickle.dump(spectral_density_errors, handle)
#
#with open("thesis/plots/multi_matrix_convergence_California.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)
#
#fig, ax = plt.subplots(figsize=(2.5, 2.5))
#plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
#plt.savefig("thesis/plots/multi_matrix_convergence_California.pgf", bbox_inches="tight")

################################################################################

A = np.load("matrices/goe.npz")
A = A.f.arr_0

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100)

with open("thesis/plots/multi_matrix_convergence_goe.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/multi_matrix_convergence_goe.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["m", "n_v", "sigma"], error_metric_name="$L^1$ relative error", x_label="$n_{\Omega} + n_{\Psi}$", colors=colors, ax=ax)
plt.savefig("thesis/plots/multi_matrix_convergence_goe.pgf", bbox_inches="tight")
