# Welcome to microlux

`microlux` is a <a href='https://github.com/jax-ml/jax'>Jax</a>-based package that can calculate the binary lensing light curve and its derivatives both efficiently and accurately.  We use the modified adaptive contour integratoin in <a href='https://github.com/valboz/VBBinaryLensing'>`VBBinaryLensing`</a> to maximize the performance. 
With the access to the gradient, we can use more advanced algorithms for microlensing modeling, such as Hamiltonian Monte Carlo (HMC) in <a href='https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS'>`numpyro`</a>.

## Features

- Adaptive contour integration with error control to calculate the binary lens microlensing light curve with finite source effect.
- Robust and accurate calculation: Widely test over the broad parameter space compared with `VBBinaryLensing`.
- Fast speed: Fully compatible with JIT compilation to speed up calculation. 
- Accurate gradient: Automatic differentiation with specially designed error estimator to ensure the convergence of gradient.
- <a href='https://en.wikipedia.org/wiki/Aberth_method'>Aberth–Ehrlich</a> method to find the roots of the polynomial and <a href='https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.optimize.linear_sum_assignment.html'>Liner sum assignment </a> algorithm to match the images.
- [Application](example/KB0371.ipynb) on the real event modeling.


## Quick Start
Please make sure that you have known the basic usage of `Jax`. Check the [Jax documentation](https://jax.readthedocs.io/en/latest/quickstart.html) for more details.
```python
import jax.numpy as jnp
from microlux import binary_mag

b = 0.1
t_0 = 0.
t_E = 1.
alphadeg = 270.
q = 0.2
s = 0.9
rho = 10 ** (-2)
trajectory_n = 1000
times = jnp.linspace(t_0 - 1.0 * t_E, t_0 + 1.0 * t_E, trajectory_n)
#calculate the binary lensing magnification
mag = binary_mag(t_0, b, t_E, rho, q, s, alphadeg, times, tol=1e-3, retol=1e-3)
```

Detailed example usages are documented in the [Quick Start Guide](quick-start.md).
Additional examples can be found in the [`test`](https://github.com/CoastEgo/microlux/tree/master/test) folder.

## Citation

`microlux` is open-source software licensed under the MIT license. If you use this package for your research, please cite our paper:

- A differentiable binary microlensing model using adaptive contour integration method: <a href='https://arxiv.org/abs/2501.07268'>in arXiv</a> and <a href= 'https://iopscience.iop.org/article/10.3847/1538-3881/adb1b2'>in AJ </a>.

``` bibtex
@article{ren2025microlux,
       author = {{Ren}, Haibin and {Zhu}, Wei},
        title = "{A Differentiable Binary Microlensing Model Using Adaptive Contour Integration Method}",
      journal = {The Astronomical Journal},
         year = 2025,
       volume = {169},
       number = {3},
          eid = {170},
        pages = {170},
          doi = {10.3847/1538-3881/adb1b2},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025AJ....169..170R},
}
```