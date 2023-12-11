import __context__

import os

import scipy as sp

from src.utils import download_matrix
from src.matrices import ModES3D

matrix_dir = "matrices"

if not os.path.exists(matrix_dir):
    os.makedirs(matrix_dir)

matrix_urls = [
    "https://suitesparse-collection-website.herokuapp.com/MM/ND/nd3k.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/MaxPlanck/shallow_water1.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/Erdos992.tar.gz"
]

print("\nDownloading matrices")
for i, matrix_url in enumerate(matrix_urls):
    download_matrix(matrix_url, save_path=matrix_dir)
    print(u"\u2713 Downloaded matrix ({}/{})".format(i + 1, len(matrix_urls)))

ns = [1, 2, 3, 4]

print("\nGenerating matrices")
for i, n in enumerate(ns):
    matrix = ModES3D(n=n, L=6, h=0.6, dim=3, bc="periodic", var=4.0, scaling_factor=-4.0)
    sp.sparse.save_npz(os.path.join(matrix_dir, "ModES3D_{}".format(n**3)), matrix)
    print("\u2713 Generated matrix ({}/{})".format(i + 1, len(ns)))
