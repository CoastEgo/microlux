# %%
import numpy as np

# %matplotlib ipympl
import matplotlib.pyplot as plt
import pandas as pd

import os
global numofchains
global N_pmap
N_pmap = 10
numofchains = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_pmap*numofchains}'
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import print_summary
from microlux import binary_mag

import VBBinaryLensing
from MulensModel import caustics
from IPython.display import display, clear_output

import emcee
import corner
from multiprocessing import Pool

print(os.getcwd())
data = pd.read_csv('microlensing/example/data/KB_19_0371.csv') ## remind to change the path to the data file

cond = (data['e_mag'] < 0.4) & (data['HJD'] > 8500)
data = data[cond]

error_frac = {'OGLE':1.59, 'KMTC01':1.41, 'KMTC41':1.38, 'KMTA01':1.35, 'KMTA41':1.57, 'KMTS01':1.19, 'KMTS41':1.41}
data['e_mag'] = data.apply(lambda x: np.sqrt(0.003**2+x['e_mag']**2*error_frac[x['Tel']]**2), axis=1)

fs_dict = {'OGLE':0.1865329, 'KMTC01':0.15551681, 'KMTC41':0.16063666, 'KMTA01':0.1964294, 'KMTA41':0.12068191, 'KMTS01':0.22724801, 'KMTS41':0.16661919}
fb_dict = {'OGLE':0.07354933, 'KMTC01':0.10144077, 'KMTC41':0.10602545, 'KMTA01':0.04612094, 'KMTA41':0.144712, 'KMTS01':0.00623068, 'KMTS41':0.09311172}
def align_function(mag, mag_err, fs, fb, fs_ogle, fb_ogle):
    flux = 10.**(0.4*(18.-mag))
    ferr = mag_err*flux*np.log(10.)/2.5

    flux_ogle = (flux-fb)/fs*fs_ogle+fb_ogle
    ferr_ogle = ferr/fs*fs_ogle

    mag_ogle = 18.-2.5*np.log10(flux_ogle)
    mag_err_ogle = ferr_ogle/flux_ogle*2.5/np.log(10.)
    return mag_ogle, mag_err_ogle
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


# %%

def mag_to_flux(mag, e_mag):
    flux = 10.**(0.4*(18.-mag))
    ferr = e_mag*flux*np.log(10.)/2.5
    return flux, ferr
def flux_to_mag(flux):
    mag = 18.-2.5*np.log10(flux)
    return mag
def light_curve_VBBL(times,parms):
    t0 = parms['t0']
    u0 = parms['u0']
    tE = parms['tE']
    rho = 10.**parms['logrho']
    alpha_deg = parms['alpha']
    s = 10.**parms['logs']
    q = 10.**parms['logq']
    tau = (times-t0)/tE
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=alpha_deg/180*np.pi+np.pi
    VBBL.Tol=1e-2
    VBBL.RelTol=1e-3
    VBBL.BinaryLightCurve
    y1 = -u0*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = u0*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), u0, alpha_VBBL, np.log(rho), np.log(tE), t0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return np.array(VBBL_mag)

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
def objective_func(parms, data, fs, fb, return_chi2=True):
    parm_name = ['t0', 'u0', 'tE', 'logrho', 'alpha', 'logs', 'logq']
    parm_dict = dict(zip(parm_name, parms))
    times,flux,ferr = data
    model_flux = light_curve_VBBL(times, parm_dict)*fs + fb
    chi2 = np.sum(((model_flux-flux)/ferr)**2)
    if return_chi2:
        return chi2
    else:
        return -0.5*chi2
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
parm_name = ['t0', 'u0', 'tE', 'logrho', 'alpha', 'logs', 'logq']
chain = sampler.get_chain(flat=True)
fig = corner.corner(chain,labels=parm_name,quantiles=[0.16, 0.5, 0.84],show_titles=True,truths=np.median(chain,axis=0))
plt.show()
for i in range(len(parm_name)):
    print(parm_name[i], np.median(chain[:,i]), np.std(chain[:,i]))


# %%
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", plt.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 30)
    color = kwargs.pop("color", "b")
    linewidths = kwargs.pop("linewidths", None)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap=plt.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                 weights=kwargs.get('weights', None))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    V = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        if plot_contours:
            ax.contourf(X1, Y1, H.T, [V[-1], H.max()],
                        cmap=LinearSegmentedColormap.from_list("cmap",
                                                               ([1] * 3,
                                                                [1] * 3),
                        N=2),antialiased=False)

    if plot_contours:
#        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
        V = [V[-1],V[-2],V[-3]]
        ax.contour(X1, Y1, H.T, V, colors=color, alpha=0.5,linewidths=linewidths)
#        ax.contourf(X1, Y1, H.T, [V[-1], H.max()], cmap=LinearSegmentedColormap.from_list("cmap",([1] * 3,[1] * 3),N=2), antialiased=False)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    return


def plot_covariance(params,labels,cov_mat,chain):
    ''' plot covariance matrix: both theoretical & mcmc chain. '''
    ## set up axes ##
    K = len(params)
    factor = 2.0           # size of one side of one panel
    lbdim = 1.2 * factor   # size of left/bottom margin
    trdim = 0.15 * factor  # size of top/right margin
    whspace = 0.1         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig,axes = plt.subplots(K,K,figsize=(10,10))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    ## set up axex extent ##
    extents = [[x.min(), x.max()] for x in chain.T]
    ##
    for i in range(K):
        ax = axes[i,i]
        mu_x,sigma_x = params[i],np.sqrt(cov_mat[i,i])
        x = np.linspace(extents[i][0],extents[i][1],100)
        p = 1/np.sqrt(2*np.pi)/sigma_x * np.exp(-(x-mu_x)**2/2./sigma_x**2)
        ax.plot(x,p,'r',alpha=0.5)
        ax.hist(chain[:,i],histtype='step',density=1)
        ax.set_xlim(extents[i])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if i < K-1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.7)
        for j in range(K):
            ax = axes[i,j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue
            ## plot error ellipse from given covariance matrix ##
            mu_y,sigma_y = params[j],np.sqrt(cov_mat[j,j])
            sigx2,sigy2,sigxy = cov_mat[i,i],cov_mat[j,j],cov_mat[i,j]
            ## find principle axes ##
            sig12 = 0.5*(sigx2+sigy2) + np.sqrt((sigx2-sigy2)**2*0.25+sigxy**2)
            sig22 = 0.5*(sigx2+sigy2) - np.sqrt((sigx2-sigy2)**2*0.25+sigxy**2)
            sig1 = np.sqrt(sig12)
            sig2 = np.sqrt(sig22)
            alpha = 0.5*np.arctan(2*sigxy/(sigx2-sigy2))
            if sigy2 > sigx2:
                alpha += np.pi/2.
            ## plot ellipse ##
            t = np.linspace(0,2*np.pi,300)
            x = mu_x + sig1*np.cos(t)*np.cos(alpha) - sig2*np.sin(t)*np.sin(alpha)
            y = mu_y + sig1*np.cos(t)*np.sin(alpha) + sig2*np.sin(t)*np.cos(alpha)
            ax.plot(y,x,'r',alpha=0.5)
            ## plot error ellipse from mcmc chain ##
            hist2d(chain[:,j],chain[:,i],ax=ax,extent=[extents[j],extents[i]],plot_contours=True,plot_datapoints=False)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i < K-1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5,-0.7)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.6,0.5)
    return fig,axes


# %%
## fisher information matrix

def light_curve_Jax(parms,times):
    t0 = parms[0]
    u0 = parms[1]
    tE = parms[2]
    rho = 10.**parms[3]
    alpha_deg = parms[4]
    s = 10.**parms[5]
    q = 10.**parms[6]
    mag_Jax = binary_mag(t0, u0, tE, rho, q, s, alpha_deg, times)
    return mag_Jax
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

def light_curve_Jax(times,parms):
    t0 = parms[0]
    u0 = parms[1]
    tE = parms[2]
    rho = 10.**parms[3]
    alpha_deg = parms[4]
    s = 10.**parms[5]
    q = 10.**parms[6]
    mag_Jax = binary_mag(t0, u0, tE, rho, q, s, alpha_deg, times)
    return mag_Jax
def light_curve_Jax_pmap(times,parms,i):
    times = jnp.reshape(times,(-1,N_pmap),order='C')
    times_i = times[:,i]
    return light_curve_Jax(times_i,parms)
# def objective_func(parms, data, fs, fb, return_chi2=True):
#     times,flux,ferr = data
#     model_flux = light_curve_Jax(times, parms)*fs + fb
#     chi2 = np.sum(((model_flux-flux)/ferr)**2)
#     if return_chi2:
#         return chi2
#     else:
#         return -0.5*chi2
# print(objective_func(initial_guess, [HJD,flux,ferr], fs, fb))
# print('tot dof = ', len(HJD)-len(parms_close))
def model_HMC(data, fs, fb, init_val, L):
    times,flux,ferr = data
    parmsample=numpyro.sample('param_base',dist.Uniform(-1*jnp.ones(len(init_val)),1*jnp.ones(len(init_val))))
    parmsample=jnp.dot(L*10,parmsample)+jnp.array(init_val)
    numpyro.deterministic('param',parmsample)
    mag_mod = jax.pmap(light_curve_Jax_pmap,in_axes=(None,None,0))(times,parmsample,jnp.arange(10))
    mag_mod = jnp.reshape(mag_mod,(flux.shape[0],),order='F')
    flux_mod = mag_mod*fs + fb
    # flux_mod = light_curve_Jax(times, parmsample)*fs + fb
    numpyro.sample('obs', dist.Normal(flux_mod, ferr), obs=flux)
    chi2 = jnp.sum(((flux_mod-flux)/ferr)**2)
    numpyro.deterministic('chi2',chi2)

L = jnp.linalg.cholesky(fisher_cov)

init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(initial_guess))})
nuts_kernel = NUTS(model_HMC,step_size=1e-2,target_accept_prob=0.8,init_strategy=init_strategy,forward_mode_differentiation=True)
mcmc = MCMC(nuts_kernel,num_warmup=500,num_samples=1000,num_chains=1,progress_bar=True)

mcmc.run(jax.random.PRNGKey(0),data=[HJD_pad,flux_pad,ferr_pad],fs=fs,fb=fb,init_val=initial_guess,L=L)
mcmc.print_summary(exclude_deterministic=False)


# %%
import corner
hmc_sample = mcmc.get_samples()['param']
print(hmc_sample.shape)
fig = corner.corner(np.array(hmc_sample),quantiles=[0.16, 0.5, 0.84],show_titles=True)
