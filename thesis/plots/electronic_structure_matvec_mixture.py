import __context__

import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.algorithms import FastNyChebPP
from src.plots import compute_spectral_density_errors, plot_spectral_density_errors

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

A = sp.sparse.load_npz("matrices/ModES3D_1.npz")

methods = [FastNyChebPP]
labels = ["$n_{\Omega}=0$, $n_{\Psi}=80$ (DGC)", "$n_{\Omega}=20$, $n_{\Psi}=60$ (NC++)", "$n_{\Omega}=40$, $n_{\Psi}=40$ (NC++)", "$n_{\Omega}=60$, $n_{\Psi}=20$ (NC++)", "$n_{\Omega}=80$, $n_{\Psi}=0$ (NC)"]
fixed_parameters = [{"n_v": 0, "n_v_tilde": 80},
                    {"n_v": 20, "n_v_tilde": 60},
                    {"n_v": 40, "n_v_tilde": 40},
                    {"n_v": 60, "n_v_tilde": 20},
                    {"n_v": 80, "n_v_tilde": 0}]
variable_parameters = "sigma"
variable_parameters_values = np.logspace(-2, 0, 10)

spectral_density_errors = compute_spectral_density_errors(A, methods, labels, variable_parameters, variable_parameters_values, fixed_parameters, n_t=100, correlated_parameter="m", correlated_parameter_values=lambda x: int(120 / x))

with open("thesis/plots/electronic_structure_matvec_mixture.pkl", "wb") as handle:
    pickle.dump(spectral_density_errors, handle)

#with open("thesis/plots/electronic_structure_matvec_mixture.pkl", "rb") as handle:
#    spectral_density_errors = pickle.load(handle)

fig, ax = plt.subplots(figsize=(6, 3))
plot_spectral_density_errors(spectral_density_errors, fixed_parameters, variable_parameters, variable_parameters_values, ignored_parameters=["n_v", "n_v_tilde", "m"], error_metric_name="$L^1$ relative error", x_label="$\sigma$", ax=ax)
plt.savefig("thesis/plots/electronic_structure_matvec_mixture.pgf", bbox_inches="tight")