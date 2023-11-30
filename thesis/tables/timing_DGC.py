import __context__

import numpy as np
import scipy as sp

from src.algorithms import DGC, FastNyCheb, FastNyChebPP
from src.utils import time_method, generate_tex_tabular, spectral_transformation

A = spectral_transformation(sp.sparse.load_npz("matrices/ModES3D_1.npz"))

methods = [DGC, FastNyCheb, FastNyChebPP]
labels = ["DGC", "NC", "NC++"]

n_t = 100
t = np.arange(-1, 1, n_t)
parameters = [{"A": A, "t": t, "m": 800, "n_v": 40, "sigma": 0.05},
              {"A": A, "t": t, "m": 2400, "n_v": 40, "sigma": 0.05},
              {"A": A, "t": t, "m": 800, "n_v": 160, "sigma": 0.05},
              {"A": A, "t": t, "m": 2400, "n_v": 160, "sigma": 0.05}]

means = np.empty((len(methods), len(parameters)))
errors = np.empty((len(methods), len(parameters)))

for i in range(len(methods)):
    for j in range(len(parameters)):
        mean, error = time_method(methods[i], parameters[j], num_times=1, num_repeats=7)
        means[i, j] = mean
        errors[i, j] = error

headline = ["", "$m=800, n_v=40$", "$m=2400, n_v=40$", "$m=800,n_v=160$", "$m=2400,n_v=160$"]

generate_tex_tabular(means, "thesis/tables/timing_DGC.tex", headline, labels, errors)
