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
# This notebook shows the application of our code to the real event analysis including NUTS, Fisher matrix and Basin-hopping optimization.

# %%
import os
from multiprocessing import Pool

import corner
import emcee
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from event_utils import (
    align_function,
    flux_to_mag,
    light_curve_Jax,
    light_curve_VBBL,
    mag_to_flux,
    model_HMC,
    objective_func,
    PARAMETER_NAMES,
    plot_covariance,
)
from IPython.display import display
from MulensModel import caustics
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS


# # %matplotlib ipympl
global numofchains
global N_pmap
N_pmap = 10
numofchains = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_pmap*numofchains}'

print(os.getcwd())
data = pd.read_csv('microlensing/example/data/KB_19_0371.csv') ## remind to change the path to the data file

cond = (data['e_mag'] < 0.4) & (data['HJD'] > 8500)
data = data[cond]

error_frac = {'OGLE':1.59, 'KMTC01':1.41, 'KMTC41':1.38, 'KMTA01':1.35, 'KMTA41':1.57, 'KMTS01':1.19, 'KMTS41':1.41}
data['e_mag'] = data.apply(lambda x: np.sqrt(0.003**2+x['e_mag']**2*error_frac[x['Tel']]**2), axis=1)

fs_dict = {'OGLE':0.1865329, 'KMTC01':0.15551681, 'KMTC41':0.16063666, 'KMTA01':0.1964294, 'KMTA41':0.12068191, 'KMTS01':0.22724801, 'KMTS41':0.16661919}
fb_dict = {'OGLE':0.07354933, 'KMTC01':0.10144077, 'KMTC41':0.10602545, 'KMTA01':0.04612094, 'KMTA41':0.144712, 'KMTS01':0.00623068, 'KMTS41':0.09311172}
data['mag_aligned'], data['e_mag_aligned'] = zip(*data.apply(lambda x: align_function(x['mag'], x['e_mag'], fs_dict[x['Tel']], fb_dict[x['Tel']], fs_dict['OGLE'], fb_dict['OGLE']), axis=1))

data

# %%
cond = (data['e_mag_aligned'] < 0.4) & (data['HJD'] > 8500)
data = data[cond]
# error_frac = {'OGLE':1.59, 'KMTC01':1.41, 'KMTC41':1.38, 'KMTA01':1.35, 'KMTA41':1.57, 'KMTS01':1.19, 'KMTS41':1.41}
# data['e_mag_aligned'] = data.apply(lambda x: np.sqrt(0.003**2+x['e_mag_aligned']**2*error_frac[x['Tel']]**2), axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
all_tel = data['Tel'].unique()
for i in all_tel:
    tel_data = data[data['Tel'] == i]
    ax.errorbar(tel_data['HJD'], tel_data['mag_aligned'], yerr=tel_data['e_mag_aligned'], fmt='o', label=i)
ax.set_xlabel('HJD')
ax.set_ylabel('Magnitude')
ax.legend()

fig.gca().invert_yaxis()
plt.show()

# %% [markdown]
# There are two degenrate solutions for this event. The wide/close degeneracy light curve is plotted in below. We use the close solution as the example.

# %%

parms_close = {'t0': 8592.388619, 'u0': 0.140631, 'tE': 6.655161, 'logrho': -2.231148, 'alpha': 271.695690, 'logs': -0.079158, 'logq': -1.141006}
# parms_wide = {'t0': 8592.391925, 'u0': 0.144696, 'tE': 6.640740, 'logrho': -2.187052, 'alpha': 271.325666, 'logs': 0.188680, 'logq': -0.957499}

flux,ferr = mag_to_flux(data['mag_aligned'].values, data['e_mag_aligned'].values)
HJD = data['HJD'].values
fs,fb = 0.18893952,0.07114746

times = np.linspace(8500, 8800, 2000)
mag_close = light_curve_VBBL(times, parms_close)
flux_close = mag_close*fs + fb
mag_close = flux_to_mag(flux_close)

ax.plot(times, mag_close,label='Close solution')
ax.legend()
ax.set_xlim(8580, 8600)

ax_traj = fig.add_axes([0.2, 0.6, 0.25, 0.25])
tau = (times-parms_close['t0'])/parms_close['tE']
alpha = parms_close['alpha']/180*np.pi
y1 = -parms_close['u0']*np.sin(alpha) + tau*np.cos(alpha)
y2 = parms_close['u0']*np.cos(alpha) + tau*np.sin(alpha)
ax_traj.plot(y1, y2,c='black')
ax_traj.set_aspect('equal')
ax_traj.set_xlim(-1., 1.)
ax_traj.set_ylim(-1., 1.)
caustics_instance = caustics.Caustics(s=10**parms_close['logs'], q=10**parms_close['logq'])
caustics_x, caustics_y = caustics_instance.get_caustics()
ax_traj.scatter(caustics_x, caustics_y, c='r', s=1)

display(fig)

# %%
initial_guess = [8.59238794e+03, 1.42915228e-01, 6.61567944e+00, -2.23131913e+00, 2.71714918e+02, -7.73128397e-02, -1.14229367e+00]
print(objective_func(initial_guess, [HJD,flux,ferr], fs, fb))
print('tot dof = ', len(HJD)-len(parms_close))

# %%
# import scipy.optimize as op
# res = op.minimize(objective_func, x0=initial_guess, args=([HJD,flux,ferr], fs, fb), method='Nelder-Mead')
# print(res.x)
# print(res.fun)

# %%

if __name__ == '__main__':
    n_dim = len(initial_guess)
    nwalkers = 20
    step_size = 0.001*np.ones_like(initial_guess)
    pos = [initial_guess+step_size*np.random.randn(n_dim) for i in range(nwalkers)] 
    with Pool(nwalkers) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, objective_func, args=([HJD,flux,ferr], fs, fb, False), pool=pool)
        pos, prob, state = sampler.run_mcmc(pos, 500, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, 1000, progress=True)

# %%

sample_chain = sampler.get_chain()
print(sample_chain.shape)

sample_chain_reshape = jnp.transpose(sample_chain, (1, 0, 2))

print(sample_chain_reshape.shape)
print_summary(sample_chain_reshape)

# %%
parm_name = PARAMETER_NAMES
chain = sampler.get_chain(flat=True)
fig = corner.corner(chain,labels=parm_name,quantiles=[0.16, 0.5, 0.84],show_titles=True,truths=np.median(chain,axis=0))
plt.show()
for i in range(len(parm_name)):
    print(parm_name[i], np.median(chain[:,i]), np.std(chain[:,i]))

# %%
## fisher information matrix

initial_guess = [8.59238794e+03, 1.42915228e-01, 6.61567944e+00, -2.23131913e+00, 2.71714918e+02, -7.73128397e-02, -1.14229367e+00]
times,flux,ferr = HJD,flux,ferr
weight_light_curve = lambda x: (light_curve_Jax(x, times)*fs+fb)/ferr
jacobian_fun = jax.jacfwd(weight_light_curve)
jacobian = jacobian_fun(jnp.array(initial_guess))
fisher_matrix = jnp.dot(jacobian.T, jacobian)
fisher_cov = jnp.linalg.inv(fisher_matrix)
fig,axes= plot_covariance(initial_guess,parm_name,fisher_cov,chain)
plt.show()

# %%
print(HJD.shape)
HJD_pad = jnp.pad(HJD, (0, 10170-HJD.shape[0]), 'constant', constant_values=HJD[-1])
print(HJD_pad.shape)
flux_pad = jnp.pad(flux, (0, 10170-flux.shape[0]), 'constant', constant_values=flux[-1])
ferr_pad = jnp.pad(ferr, (0, 10170-ferr.shape[0]), 'constant', constant_values=ferr[-1])

# %%

L = jnp.linalg.cholesky(fisher_cov)

init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(initial_guess))})
nuts_kernel = NUTS(model_HMC,step_size=1e-2,target_accept_prob=0.8,init_strategy=init_strategy,forward_mode_differentiation=True)
mcmc = MCMC(nuts_kernel,num_warmup=500,num_samples=1000,num_chains=1,progress_bar=True)

mcmc.run(jax.random.PRNGKey(0),data=[HJD_pad,flux_pad,ferr_pad],fs=fs,fb=fb,init_val=initial_guess,L=L)
mcmc.print_summary(exclude_deterministic=False)

# %%
hmc_sample = mcmc.get_samples()['param']
print(hmc_sample.shape)
fig = corner.corner(np.array(hmc_sample),quantiles=[0.16, 0.5, 0.84],show_titles=True)
