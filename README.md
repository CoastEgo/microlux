## microlux: Microlensing using Jax

[![Test Status](https://github.com/coastego/microlux/actions/workflows/run_test.yml/badge.svg)](https://github.com/CoastEgo/microlux/actions/workflows/run_test.yml)
[![Documentation Status](https://github.com/coastego/microlux/actions/workflows/build_docs.yml/badge.svg)](https://coastego.github.io/microlux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
---

`microlux` is a <a href='https://github.com/jax-ml/jax'>Jax</a>-based package that can calculate the binary lensing light curve and its derivatives both efficiently and accurately.  We use the modified adaptive contour integratoin in <a href='https://github.com/valboz/VBBinaryLensing'>`VBBinaryLensing`</a> to maximize the performance. 
With the access to the gradient, we can use more advanced algorithms for microlensing modeling, such as Hamiltonian Monte Carlo (HMC) in <a href='https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS'>`numpyro`</a>.


## Installation

``` bash
pip install microlux
```
or you can install this package from source for development. 
``` bash 
git clone https://github.com/CoastEgo/microlux.git
cd microlux
pip install -e .
```
## Documentation
The documentation is available at <a href='https://coastego.github.io/microlux/'>here</a>. See this for more details.


  
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
