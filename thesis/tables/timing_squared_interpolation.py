import __context__

import numpy as np

from src.interpolation import chebyshev_coefficients, exponentiate_chebyshev_coefficients_cosine_transform, _chebyshev_coefficients_quadrature
from src.kernel import gaussian_kernel
from src.utils import time_method, generate_tex_tabular

methods = [_chebyshev_coefficients_quadrature, chebyshev_coefficients, exponentiate_chebyshev_coefficients_cosine_transform]
labels = ["quadrature", "DCT", "squaring"]

n_t = 1000
t = np.arange(-1, 1, n_t)
sigma = 0.05
n = 1000
g_squared = lambda x: gaussian_kernel(x, n=n, sigma=sigma) ** 2
parameters = [{"t": t, "m": 2 * 800, "function": g_squared},
              {"t": t, "m": 2 * 1600, "function": g_squared},
              {"t": t, "m": 2 * 2400, "function": g_squared},
              {"t": t, "m": 2 * 3200, "function": g_squared}]

means = np.empty((len(methods), len(parameters)))
errors = np.empty((len(methods), len(parameters)))

for i in range(len(methods)):
    for j in range(len(parameters)):
        if methods[i] == exponentiate_chebyshev_coefficients_cosine_transform:
            param = {"mu": chebyshev_coefficients(t, parameters[j]["m"] // 2, function=g_squared)}
        else:
            param = parameters[j]
        mean, error = time_method(methods[i], param, num_times=100, num_repeats=7)
        means[i, j] = mean
        errors[i, j] = error

headline = ["", r"$m=800$", r"$m=1600$", r"$m=2400$", r"$m=3200$"]

generate_tex_tabular(means, "thesis/tables/timing_squared_interpolation.tex", headline, labels, errors)
