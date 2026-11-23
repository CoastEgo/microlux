# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modeling KB-19-0371 with microlux
#
# Fit KB-19-0371 using SciPy and automatic differentiation, then sample with
# Fisher-reparameterized HMC and compare with an optional emcee chain.

# %%
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path


N_PMAP = 10
HMC_NUM_CHAINS = 1
RUN_HMC = True
HMC_NUM_WARMUP = 500
HMC_NUM_SAMPLES = 1000
TARGET_ACCEPT = 0.8
RUN_EMCEE = False
EMCEE_NWALKERS = 20
EMCEE_NUM_WARMUP = 500
EMCEE_NUM_SAMPLES = 1000
BOUNDARY_STEEPNESS = 100.0
PENALTY_STRENGTH = 1000.0

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={N_PMAP * HMC_NUM_CHAINS}"
)
if "__file__" in globals():
    EXAMPLE_DIR = Path(__file__).resolve().parent
elif Path("event_utils.py").exists():
    EXAMPLE_DIR = Path.cwd()
else:
    EXAMPLE_DIR = Path.cwd() / "example"
sys.path.insert(0, str(EXAMPLE_DIR))
FIGURE_DIR = EXAMPLE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

import corner
import emcee
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from event_utils import (
    align_function,
    flux_to_mag,
    light_curve_Jax,
    light_curve_VBBL,
    mag_to_flux,
    PARAMETER_NAMES,
    plot_covariance,
)
from IPython.display import display
from jax.nn import softplus
from microlux import fisher_information, fit_grouped_fluxes
from MulensModel import CausticsBinary
from numpyro.diagnostics import print_summary
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS
from scipy.optimize import least_squares


def save_figure(fig, filename):
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")


# %% [markdown]
# ## Photometric data preparation
#
# Filter the photometry and rescale the reported uncertainties. Alignment
# to the OGLE flux system is used only for plotting.

# %%
if jax.local_device_count() < N_PMAP * HMC_NUM_CHAINS:
    raise RuntimeError(
        f"Restart the Python kernel so XLA_FLAGS can set {N_PMAP * HMC_NUM_CHAINS} CPU devices."
    )

data = pd.read_csv(EXAMPLE_DIR / "data" / "KB_19_0371.csv")

cond = (data["e_mag"] < 0.4) & (data["HJD"] > 8500)
data = data[cond].copy()

error_frac = {
    "OGLE": 1.59,
    "KMTC01": 1.41,
    "KMTC41": 1.38,
    "KMTA01": 1.35,
    "KMTA41": 1.57,
    "KMTS01": 1.19,
    "KMTS41": 1.41,
}
data["e_mag"] = np.hypot(0.003, data["e_mag"] * data["Tel"].map(error_frac))

# %% [markdown]
# ## Close binary-lens solution
#
# Start from the close solution and fit the source and blend fluxes for each
# data set. Use the fitted OGLE fluxes to align the light-curve plot.

# %%

parms_close = {
    "t0": 8592.388619,
    "u0": 0.140631,
    "tE": 6.655161,
    "logrho": -2.231148,
    "alpha": 271.695690,
    "logs": -0.079158,
    "logq": -1.141006,
}
# parms_wide = {'t0': 8592.391925, 'u0': 0.144696, 'tE': 6.640740, 'logrho': -2.187052, 'alpha': 271.325666, 'logs': 0.188680, 'logq': -0.957499}

observation_times = data["HJD"].values
optimization_start = np.array([parms_close[name] for name in PARAMETER_NAMES])
group_ids, telescope_names = pd.factorize(data["Tel"], sort=True)
fit_times = jnp.asarray(observation_times)
fit_flux, fit_error = mag_to_flux(data["mag"].values, data["e_mag"].values)
fit_flux, fit_error = jnp.asarray(fit_flux), jnp.asarray(fit_error)
group_ids = jnp.asarray(group_ids)


@jax.jit
def grouped_flux_fit(params):
    magnification = light_curve_Jax(params, fit_times)
    return fit_grouped_fluxes(
        magnification,
        fit_flux,
        fit_error,
        group_ids,
        n_groups=len(telescope_names),
    )


initial_flux_fit = grouped_flux_fit(jnp.asarray(optimization_start))
fs = np.asarray(initial_flux_fit.source_flux)
fb = np.asarray(initial_flux_fit.blend_flux)
ogle_group = telescope_names.get_loc("OGLE")
data["mag_aligned"], data["e_mag_aligned"] = align_function(
    data["mag"].values,
    data["e_mag"].values,
    fs[group_ids],
    fb[group_ids],
    fs[ogle_group],
    fb[ogle_group],
)

plot_times = np.linspace(8500, 8800, 2000)
mag_close = light_curve_VBBL(plot_times, parms_close)
flux_close = mag_close * fs[ogle_group] + fb[ogle_group]
mag_close = flux_to_mag(flux_close)

fig, ax = plt.subplots(figsize=(10, 6))
all_tel = data["Tel"].unique()
for i in all_tel:
    tel_data = data[data["Tel"] == i]
    ax.errorbar(
        tel_data["HJD"],
        tel_data["mag_aligned"],
        yerr=tel_data["e_mag_aligned"],
        fmt="o",
        label=i,
    )
ax.plot(plot_times, mag_close, label="Close solution")
ax.legend(loc="upper right")
ax.set_xlim(8580, 8600)
ax.set_xlabel("HJD")
ax.set_ylabel("Magnitude")
ax.invert_yaxis()

ax_traj = fig.add_axes([0.2, 0.6, 0.25, 0.25])
tau = (plot_times - parms_close["t0"]) / parms_close["tE"]
alpha = parms_close["alpha"] / 180 * np.pi
y1 = -parms_close["u0"] * np.sin(alpha) + tau * np.cos(alpha)
y2 = parms_close["u0"] * np.cos(alpha) + tau * np.sin(alpha)
ax_traj.plot(y1, y2, c="black")
ax_traj.set_aspect("equal")
ax_traj.set_xlim(-1.0, 1.0)
ax_traj.set_ylim(-1.0, 1.0)
caustics_instance = CausticsBinary(
    s=10 ** parms_close["logs"], q=10 ** parms_close["logq"]
)
caustics_x, caustics_y = caustics_instance.get_caustics()
ax_traj.scatter(caustics_x, caustics_y, c="r", s=1)

save_figure(fig, "KB0371_close_solution.png")
display(fig)

# %% [markdown]
# ![Close solution light curve and trajectory](figures/KB0371_close_solution.png)

# %% [markdown]
# ## Gradient optimization with SciPy and JAX
#
# Optimize the seven lens parameters with `scipy.optimize.least_squares`,
# using the residual Jacobian supplied by JAX automatic differentiation.

# %%
parameter_bounds = np.array(
    [
        [optimization_start[0] - 50.0, optimization_start[0] + 50.0],
        [-2.0, 2.0],
        [0.1, 200.0],
        [-4.0, -2.0],
        [0.0, 360.0],
        [-1.5, 1.5],
        [-4.0, 3.0],
    ]
)


def profiled_residuals(params):
    return grouped_flux_fit(params).residuals


residual_jacobian = jax.jacfwd(profiled_residuals)

# %%
optimization = least_squares(
    profiled_residuals,
    optimization_start,
    jac=residual_jacobian,
    bounds=parameter_bounds.T,
    x_scale="jac",
    max_nfev=100,
    verbose=1,
)
optimized_params = optimization.x
optimized_flux_fit = grouped_flux_fit(jnp.asarray(optimized_params))
print("Optimizer:", optimization.message)
print("Initial chi2:", float(jnp.sum(initial_flux_fit.chi2_by_group)))
print("Optimized chi2:", float(jnp.sum(optimized_flux_fit.chi2_by_group)))
print(
    "Degrees of freedom:",
    len(fit_times) - len(optimization_start) - 2 * len(telescope_names),
)
display(
    pd.DataFrame(
        {"initial": optimization_start, "optimized": optimized_params},
        index=PARAMETER_NAMES,
    )
)
display(
    pd.DataFrame(
        {
            "source_flux": np.asarray(optimized_flux_fit.source_flux),
            "blend_flux": np.asarray(optimized_flux_fit.blend_flux),
            "chi2": np.asarray(optimized_flux_fit.chi2_by_group),
        },
        index=telescope_names,
    )
)

# %% [markdown]
# ## Local Fisher approximation
#
# HMC can become inefficient when the posterior is strongly correlated or has
# very different scales along different directions. Following the discussion
# in the NumPyro tutorial
# [Bad posterior geometry and how to deal with it](https://num.pyro.ai/en/stable/tutorials/bad_posterior_geometry.html),
# we change the coordinate system before running HMC.
#
# For Gaussian observational errors, we approximate the local Fisher matrix
# from the Jacobian of the normalized residuals at `optimized_params`:
#
# $$
# r_i(\boldsymbol{\theta})
# = \frac{f_i - f_{\mathrm{model},i}(\boldsymbol{\theta})}{\sigma_i},
# \qquad
# J_{ij} = \frac{\partial r_i}{\partial \theta_j},
# $$
#
# $$
# F = J^\mathsf{T} J,
# \qquad
# C \approx F^{-1}.
# $$
#
# `profiled_residuals` refits the source and blend fluxes at each lens
# parameter vector, so their parameter dependence is included in the
# Jacobian. Here, $C$ approximates the local covariance of the seven lens
# parameters and supplies the scales and correlations for HMC below.

# %%

fisher_matrix = fisher_information(profiled_residuals, jnp.asarray(optimized_params))
fisher_cov = jnp.linalg.inv(fisher_matrix)
display(
    pd.Series(
        np.sqrt(np.diag(np.asarray(fisher_cov))),
        index=PARAMETER_NAMES,
        name="Local parameter uncertainty",
    )
)

# %% [markdown]
# ## Fisher reparameterization for HMC
#
# Set $\boldsymbol{\theta}_0$ to `optimized_params` and compute the lower
# triangular Cholesky factor $L$ of `fisher_cov`. NUTS samples the latent
# coordinates $\boldsymbol{z}$ and maps them to the lens parameters:
#
# $$
# C = L L^\mathsf{T},
# \qquad
# \boldsymbol{\theta}
# =
# \boldsymbol{\theta}_0 + L \boldsymbol{z}.
# $$
#
# Under the local Gaussian approximation, the covariance of these new
# coordinates is
#
# $$
# \operatorname{Cov}(\boldsymbol{z})
# \approx L^{-1} C L^{-\mathsf{T}} = I.
# $$
#
# This reduces scale differences and correlations near the fitted solution.
# We compute $L$ once and keep it fixed during sampling; `dense_mass=True`
# lets NumPyro adapt the remaining correlations during warmup.
#
# In the model, `param_base` stores $\boldsymbol{z}$ and `param` records
# $\boldsymbol{\theta}$. Initializing `param_base` at zero starts the chain
# at `optimized_params`. Every likelihood evaluation uses the full light
# curve at the transformed parameters.
#
# `dist.ImproperUniform` keeps the latent prior flat. Since $L$ is fixed,
# the change-of-variables Jacobian determinant is constant: this
# reparameterization does not add a Gaussian prior. The boundary penalties
# are still evaluated in the original lens coordinates.
#
# Pad only the times for parallel evaluation, then trim the predictions
# before evaluating the likelihood and chi2.

# %%
padding = (-fit_times.shape[0]) % N_PMAP
time_chunks = jnp.pad(fit_times, (0, padding), mode="edge").reshape(-1, N_PMAP).T
parallel_light_curve = jax.pmap(light_curve_Jax, in_axes=(None, 0))

# %%

param_lowers = jnp.array(parameter_bounds[:, 0])
param_uppers = jnp.array(parameter_bounds[:, 1])

initial_param = jnp.asarray(optimized_params)
if bool(jnp.any(initial_param <= param_lowers)) or bool(
    jnp.any(initial_param >= param_uppers)
):
    raise ValueError("Initial parameters must be strictly inside soft boundaries.")

cholesky_transform = jnp.linalg.cholesky(fisher_cov)

# %% [markdown]
# ### HMC model
#
# Fit the source and blend fluxes at each likelihood evaluation. Clip lens
# parameters to the numerical domain and apply a soft boundary penalty.


# %%
def model_HMC_reparameterized(
    data,
    group_ids,
    n_groups,
    init_val,
    transform_matrix,
    param_lowers,
    param_uppers,
    boundary_steepness,
    penalty_strength,
):
    time_chunks, flux, ferr = data
    param_base = numpyro.sample(
        "param_base",
        dist.ImproperUniform(constraints.real, (), event_shape=(len(init_val),)),
    )
    parmsample = jnp.dot(transform_matrix, param_base) + jnp.asarray(init_val)
    numpyro.deterministic("param", parmsample)

    lower_penalty = jnp.sum(softplus(boundary_steepness * (param_lowers - parmsample)))
    upper_penalty = jnp.sum(softplus(boundary_steepness * (parmsample - param_uppers)))
    penalty = lower_penalty + upper_penalty
    safe_params = jnp.clip(parmsample, param_lowers + 1e-5, param_uppers - 1e-5)

    mag_mod = parallel_light_curve(safe_params, time_chunks).T.reshape(-1)
    mag_mod = mag_mod[: flux.shape[0]]
    flux_fit = fit_grouped_fluxes(mag_mod, flux, ferr, group_ids, n_groups=n_groups)
    numpyro.sample("obs", dist.Normal(flux_fit.model_flux, ferr), obs=flux)
    numpyro.factor("boundary_penalty", -penalty_strength * penalty)
    chi2 = jnp.sum(flux_fit.chi2_by_group)
    numpyro.deterministic("source_flux", flux_fit.source_flux)
    numpyro.deterministic("blend_flux", flux_fit.blend_flux)
    numpyro.deterministic("chi2", chi2)
    numpyro.deterministic("penalty", penalty)


# %%
if RUN_HMC:
    init_strategy = numpyro.infer.init_to_value(
        values={"param_base": jnp.zeros(len(initial_param))}
    )
    nuts_kernel = NUTS(
        model_HMC_reparameterized,
        step_size=1.0 / 5.0,
        target_accept_prob=TARGET_ACCEPT,
        init_strategy=init_strategy,
        forward_mode_differentiation=True,
        dense_mass=True,
        adapt_mass_matrix=True,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=HMC_NUM_WARMUP,
        num_samples=HMC_NUM_SAMPLES,
        num_chains=HMC_NUM_CHAINS,
        progress_bar=True,
    )

    mcmc.run(
        jax.random.PRNGKey(0),
        data=[time_chunks, fit_flux, fit_error],
        group_ids=group_ids,
        n_groups=len(telescope_names),
        init_val=initial_param,
        transform_matrix=cholesky_transform,
        param_lowers=param_lowers,
        param_uppers=param_uppers,
        boundary_steepness=BOUNDARY_STEEPNESS,
        penalty_strength=PENALTY_STRENGTH,
    )
    mcmc.print_summary(exclude_deterministic=False)
else:
    print("Skipping HMC; set RUN_HMC = True to run it.")

# %% [markdown]
# ## HMC posterior

# %%
if RUN_HMC:
    hmc_samples = np.asarray(mcmc.get_samples()["param"])
    fig = corner.corner(
        hmc_samples,
        labels=PARAMETER_NAMES,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    save_figure(fig, "KB0371_hmc_corner.png")

# %% [markdown]
# ![HMC posterior](figures/KB0371_hmc_corner.png)

# %% [markdown]
# ## emcee reference chain
#
# Start emcee from `optimized_params`, using VBM with the same grouped fit
# and boundary penalties as HMC.


# %%
def log_probability(params):
    lower, upper = parameter_bounds.T
    penalty = np.sum(np.logaddexp(0.0, BOUNDARY_STEEPNESS * (lower - params)))
    penalty += np.sum(np.logaddexp(0.0, BOUNDARY_STEEPNESS * (params - upper)))
    safe_params = np.clip(params, lower + 1e-5, upper - 1e-5)
    magnification = light_curve_VBBL(
        observation_times, dict(zip(PARAMETER_NAMES, safe_params))
    )
    flux_fit = fit_grouped_fluxes(
        magnification,
        fit_flux,
        fit_error,
        group_ids,
        n_groups=len(telescope_names),
    )
    return -0.5 * float(jnp.sum(flux_fit.chi2_by_group)) - PENALTY_STRENGTH * penalty


# %%

n_dim = len(optimized_params)
step_size = 0.001 * np.ones_like(optimized_params)
if RUN_EMCEE:
    pos = [
        optimized_params + step_size * np.random.randn(n_dim)
        for i in range(EMCEE_NWALKERS)
    ]
    with ThreadPool(EMCEE_NWALKERS) as pool:
        sampler = emcee.EnsembleSampler(
            EMCEE_NWALKERS,
            n_dim,
            log_probability,
            pool=pool,
        )
        pos, _, _ = sampler.run_mcmc(pos, EMCEE_NUM_WARMUP, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, EMCEE_NUM_SAMPLES, progress=True)
else:
    print("Skipping emcee; set RUN_EMCEE = True to run it.")

# %%

if RUN_EMCEE:
    emcee_samples = sampler.get_chain()
    print_summary(np.transpose(emcee_samples, (1, 0, 2)))

# %%
if RUN_EMCEE:
    emcee_chain = sampler.get_chain(flat=True)
    fig = corner.corner(
        emcee_chain,
        labels=PARAMETER_NAMES,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        truths=np.median(emcee_chain, axis=0),
    )
    save_figure(fig, "KB0371_emcee_corner.png")
    plt.show()

# %% [markdown]
# ![emcee posterior](figures/KB0371_emcee_corner.png)

# %% [markdown]
# Compare the local Fisher approximation (red) with the emcee posterior
# (blue).

# %%
if RUN_EMCEE:
    fig, axes = plot_covariance(
        optimized_params, PARAMETER_NAMES, np.asarray(fisher_cov), emcee_chain
    )
    save_figure(fig, "KB0371_fisher_covariance.png")
    plt.show()

# %% [markdown]
# ![Fisher approximation and emcee posterior](figures/KB0371_fisher_covariance.png)
