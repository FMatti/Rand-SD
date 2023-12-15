# Rand-SD

Randomized estimation of spectral densities.

![](https://img.shields.io/badge/-Compatibility-gray?style=flat-square) &ensp;
![](https://img.shields.io/badge/Python_3.8+-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/Latex_3-white?style=flat-square&logo=latex&color=white&logoColor=white&labelColor=gray)

![](https://img.shields.io/badge/-Dependencies-gray?style=flat-square)&ensp;
![](https://img.shields.io/badge/NumPy-white?style=flat-square&logo=numpy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/SciPy-white?style=flat-square&logo=scipy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/Matplotlib-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)

## Theoretical background

We concern ourselves with the approximation of the smoothened spectral density 

$$
\phi_{\sigma}(t) = \sum_{i=1}^n g_{\sigma}(t - \lambda_i) = \mathrm{Tr}(g_{\sigma}(t\boldsymbol{I} - \boldsymbol{A}))
$$

of a large symmeric matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$.

### Chebyshev expansion

In all methods, we first compute the Chebyshev expansion

$$
g_{\sigma}(t\boldsymbol{I} - \boldsymbol{A}) \approx g_{\sigma}^{(m)}(t\boldsymbol{I} - \boldsymbol{A}) = \sum_{l=0}^{m} \mu_l(t) T_l(\boldsymbol{A}).
$$

### Delta-Gauss-Chebyshev

We directly estimate the trace using the Hutchinson's trace estimator with a standard Gaussian random matrix $\boldsymbol{\Psi} \in \mathbb{R}^{n \times n_{\Psi}}$ to obtain

$$
\phi_{\sigma}(t) \approx \widetilde \phi_{\sigma}^{(m)}(t) = \frac{1}{n_{\Psi}} \sum_{l=0}^{m} \mu_l(t) \mathrm{Tr}(\boldsymbol{\Psi}^{\top} T_l(\boldsymbol{A}) \boldsymbol{\Psi}).
$$

### Nyström-Chebyshev

We compute the Nyström approximation with a standard Gaussian sketching matrix $\boldsymbol{\Omega} \in \mathbb{R}^{n \times n_{\Omega}}$

$$
g_{\sigma}(t\boldsymbol{I}- \boldsymbol{A}) \approx \widehat g_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A}) =
(g_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A}) \boldsymbol{\Omega})(\boldsymbol{\Omega}^{\top} g_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A}) \boldsymbol{\Omega})(g_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A}) \boldsymbol{\Omega})^{\top}
$$

and compute its trace

$$
\phi_{\sigma}(t) \approx \widehat \phi_{\sigma}^{(m)}(t) = \widehat \phi_{\sigma}^{(m)}(t) = \mathrm{Tr}(\widehat{g}_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A})).
$$

### Nyström-Chebyshev++

We compute the Nyström approximation and apply the Hutchinson's to the residual of the approximation to get the trace 

$$
\phi_{\sigma}(t) \approx \breve \phi_{\sigma}^{(m)}(t) = \mathrm{Tr}(\widehat g_{\sigma}^{(m)}(t\boldsymbol{I} - \boldsymbol{A})) + \frac{1}{n_{\Psi}} \mathrm{Tr}(\boldsymbol{\Psi}^{\top} (g_{\sigma}^{(m)}(t\boldsymbol{I}- \boldsymbol{A}) - \widehat g_{\sigma}^{(m)}(t\boldsymbol{I} - \boldsymbol{A})) \boldsymbol{\Psi}).
$$

## Quick start

### Prerequisites

To reproduce our results, you will need

- a [Git](https://git-scm.com/downloads) installation to clone the repository;
- a recent version of [Python](https://www.python.org/downloads) to run the experiments;
- and (optionally) a [LaTeX](https://www.latex-project.org/get/#tex-distributions) distribution to build the thesis.

> [!NOTE]
> The commands `git` and `python`have to be discoverable by your terminal. If you want to build the thesis, additionally `pdflatex`, `bibtex`, and `makeglossaries` need to work. To verify this, use `[command] --version`.

### Setup

Clone this repository using
```[shell]
git clone https://github.com/FMatti/Rand-SD
cd Rand-SD
```

Install all the requirements with
```[shell]
python -m pip install --upgrade pip
python -m install -r requirements.txt
```

Reproduce the whole project with the following command
```[shell]
python -m setup.py -a
```
> [!NOTE]
> Reproducing the whole project might take a few hours!

## Contact

In case of questions and unclarities, feel free to contact us through one of the following channels:

[![Gmail](https://img.shields.io/badge/Mail-D14836?logo=gmail&logoColor=white)](mailto:somecallmefabio@gmail.ch)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/FMatti)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fmatti/)
