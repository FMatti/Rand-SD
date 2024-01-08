import __context__

import numpy as np
import matplotlib.pyplot as plt

from src.kernel import lorentzian_kernel, gaussian_kernel

import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams["font.family"] = r"serif"
matplotlib.rcParams["font.serif"] = r"Palatino"
matplotlib.rcParams["font.size"] = 12

s = np.linspace(-0.5, 0.5, 100)
sigma = 0.05
n = 1
g = lambda x: gaussian_kernel(x, n=n, sigma=sigma)
h = lambda x: lorentzian_kernel(x, n=n, sigma=sigma)

plt.figure(figsize=(4, 1.5))

plt.plot(s, g(s), label="Gaussian", color="#2F455C")
plt.plot(s, h(s), label="Lorentzian", color="#F98125")

plt.ylim([0, 8.5])
plt.xlim([-0.5, 0.5])
plt.ylabel("$g_{\sigma}(s)$")
plt.xlabel("$s$")
plt.legend()

plt.savefig("thesis/plots/haydock_kernel.pgf", bbox_inches="tight", transparent=True)
