import __context__

import numpy as np
import scipy as sp

from src.algorithms import Haydock, NC, NCPP
from src.utils import time_method, generate_tex_tabular, spectral_transformation
from src.kernel import lorentzian_kernel

A = spectral_transformation(sp.sparse.load_npz("matrices/ModES3D_1.npz"))

methods = [Haydock, NC, NCPP]
labels = ["Haydock", "NC", "NC++"]

n_t = 100
t = np.arange(-1, 1, n_t)
parameters = [{"A": A, "t": t, "m": 800, "n_v": 40, "sigma": 0.05, "kernel": lorentzian_kernel},
              {"A": A, "t": t, "m": 2400, "n_v": 40, "sigma": 0.05, "kernel": lorentzian_kernel},
              {"A": A, "t": t, "m": 800, "n_v": 160, "sigma": 0.05, "kernel": lorentzian_kernel},
              {"A": A, "t": t, "m": 2400, "n_v": 160, "sigma": 0.05, "kernel": lorentzian_kernel}]

means = np.empty((len(methods), len(parameters)))
errors = np.empty((len(methods), len(parameters)))

for i in range(len(methods)):
    for j in range(len(parameters)):
        mean, error = time_method(methods[i], parameters[j], num_times=1, num_repeats=7)
        means[i, j] = mean
        errors[i, j] = error

headline = ["", r"\shortstack[c]{$m=800$ \\ $n_{\Omega} + n_{\Psi}=40$}", r"\shortstack[c]{$m=2400$ \\ $n_{\Omega} + n_{\Psi}=40$}", r"\shortstack[c]{$m=800$ \\ $n_{\Omega} + n_{\Psi}=160$}", r"\shortstack[c]{$m=2400$ \\ $n_{\Omega} + n_{\Psi}=160$}"]

generate_tex_tabular(means, "thesis/tables/timing_haydock.tex", headline, labels, errors)
