"""
Plots
-----

Helper functions for quickly creating convergence plots.
"""

from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import spectral_transformation, form_spectral_density
from src.metrics import p_norm
from src.kernel import gaussian_kernel


def compute_spectral_densities(A, methods, labels, parameters, kernel=gaussian_kernel, N_t=1000, add_baseline=False):
    parameters = deepcopy(parameters)

    eigenvalues = np.linalg.eigvalsh(A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, np.min(eigenvalues), np.max(eigenvalues))

    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(labels, list):
        if labels is None:
            labels = methods.copy()
        else:
            labels = [labels]

    t = np.linspace(-1, 1, N_t)
    for parameter in parameters:
        parameter["sigma"] /= (max_ev - min_ev) / 2

    l = max(len(methods), len(parameters), len(labels))
    methods *= l if len(methods) < l else 1
    parameters *= l if len(parameters) < l else 1
    labels *= l if len(labels) < l else 1

    spectral_densities = {}
    if add_baseline:
        spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, N=A_transformed.shape[0], a=-1, b=1, N_t=N_t, sigma=parameters[0]["sigma"])
        spectral_densities["baseline"] = spectral_density_baseline

    for method, parameter, label in zip(methods, parameters, labels):
        spectral_density = method(A=A_transformed, t=t, **parameter)
        spectral_densities[label] = spectral_density

    return spectral_densities


def plot_spectral_densities(spectral_densities, parameters, variable_parameter=None, ignored_parameters=[], ax=None, colors=None):
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key == variable_parameter or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(title)
    ax.set_xlim([-1, 1])
    ax.set_ylabel("$\phi_{\sigma}$")
    ax.set_xlabel("$t$")

    if colors is None:
        colors = [matplotlib.colormaps["magma"](i / len(spectral_densities)) for i in range(len(spectral_densities))]

    for i, (label, spectral_density) in enumerate(spectral_densities.items()):
        t = np.linspace(-1, 1, len(spectral_density))
        plt.plot(t, spectral_density, linewidth=1, color=colors[i], label=label)
    plt.legend()
    return ax


def compute_spectral_density_errors(A, methods, labels, variable_parameter, variable_parameter_values, parameters, kernel=gaussian_kernel, N_t=1000, error_metric=p_norm, correlated_parameter=None, correlated_parameter_values=None):
    parameters = deepcopy(parameters)

    # Spectral transform of matrix
    eigenvalues = np.linalg.eigvalsh(A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, min_ev, max_ev)

    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(labels, list):
        if labels is None:
            labels = methods.copy()
        else:
            labels = [labels]
    
    for parameter in parameters:
        try:
            parameter["sigma"] /= (max_ev - min_ev) / 2
        except:
            pass

    l = max(len(methods), len(parameters), len(labels))
    methods *= l if len(methods) < l else 1
    parameters *= l if len(parameters) < l else 1
    labels *= l if len(labels) < l else 1

    t = np.linspace(-1, 1, N_t)

    dos_errors = {}
    for label, method, parameter in zip(labels, methods, parameters):
        dos_errors[label] = []
        for param in variable_parameter_values:
            if correlated_parameter is not None:
                parameter[correlated_parameter] = correlated_parameter_values(param)
                if correlated_parameter == "sigma":
                    parameter[correlated_parameter] /= (max_ev - min_ev) / 2
            if variable_parameter == "sigma":
                param /= (max_ev - min_ev) / 2
                #parameter["M"] = int(120 / param)
            try:
                kernel = parameter["kernel"]
            except:
                pass
            parameter[variable_parameter] = param
            spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, N=A_transformed.shape[0], N_t=N_t, sigma=parameter["sigma"])
            spectral_density = method(A=A_transformed, t=t, **parameter)
            dos_errors[label].append(error_metric(spectral_density_baseline, spectral_density))

    return dos_errors


def plot_spectral_density_errors(spectral_density_errors, parameters, variable_parameter, variable_parameter_values, error_metric_name="Error", ignored_parameters=[], ax=None, colors=None):
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key == variable_parameter or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("{}".format(error_metric_name))
    ax.set_xlabel("${}$".format(variable_parameter))

    if colors is None:
        colors = [matplotlib.colormaps["magma"](i / len(spectral_density_errors)) for i in range(len(spectral_density_errors))]

    for i, (label, spectral_density_error) in enumerate(spectral_density_errors.items()):
        ax.plot(variable_parameter_values, spectral_density_error, linewidth=1, color=colors[i], label=label, marker=".")
    ax.legend()
    return ax


def compute_spectral_density_errors_heatmap(A, method, variable_parameters, parameters, kernel=gaussian_kernel, N_t=1000, error_metric=p_norm):
    parameters = deepcopy(parameters)

    # Spectral transform of matrix
    eigenvalues = np.linalg.eigvalsh(A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, min_ev, max_ev)

    try:
        parameters["sigma"] /= (max_ev - min_ev) / 2
    except:
        pass

    param_1, param_2 = variable_parameters.keys()
    values_1, values_2 = variable_parameters.values()

    dos_errors = np.empty((len(values_1), len(values_2)))

    t = np.linspace(-1, 1, N_t)
    spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, N=A_transformed.shape[0], N_t=N_t, sigma=parameters["sigma"])
    for i, value_1 in enumerate(values_1):
        for j, value_2 in enumerate(values_2):
            spectral_density = method(A=A_transformed, t=t, **{param_1: value_1, param_2: value_2}, **parameters)
            dos_errors[i, j] = error_metric(spectral_density_baseline, spectral_density)

    return dos_errors


def plot_spectral_density_errors_heatmap(spectral_density_errors, variable_parameters, parameters, ignored_parameters=[], ax=None):
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key in variable_parameters.keys() or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    param_1, param_2 = variable_parameters.keys()
    values_1, values_2 = variable_parameters.values()

    ax.set_title(title)
    ax.set_ylabel("${}$".format(param_1))
    ax.set_xlabel("${}$".format(param_2))
    ax.set_yticks(range(len(values_1)), values_1)
    ax.set_xticks(range(len(values_2)), values_2)

    plt.imshow(spectral_density_errors, cmap="magma", norm=matplotlib.colors.LogNorm())
    plt.colorbar()

    return ax
