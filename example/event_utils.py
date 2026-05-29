"""Helpers for the KB-19-0371 event modeling example."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from microlux import binary_mag


PARAMETER_NAMES = ["t0", "u0", "tE", "logrho", "alpha", "logs", "logq"]


def align_function(mag, mag_err, fs, fb, fs_ogle, fb_ogle):
    flux = 10.0 ** (0.4 * (18.0 - mag))
    ferr = mag_err * flux * np.log(10.0) / 2.5

    flux_ogle = (flux - fb) / fs * fs_ogle + fb_ogle
    ferr_ogle = ferr / fs * fs_ogle

    mag_ogle = 18.0 - 2.5 * np.log10(flux_ogle)
    mag_err_ogle = ferr_ogle / flux_ogle * 2.5 / np.log(10.0)
    return mag_ogle, mag_err_ogle


def mag_to_flux(mag, e_mag):
    flux = 10.0 ** (0.4 * (18.0 - mag))
    ferr = e_mag * flux * np.log(10.0) / 2.5
    return flux, ferr


def flux_to_mag(flux):
    return 18.0 - 2.5 * np.log10(flux)


def light_curve_VBBL(times, parms):
    import VBBinaryLensing

    t0 = parms["t0"]
    u0 = parms["u0"]
    tE = parms["tE"]
    rho = 10.0**parms["logrho"]
    alpha_deg = parms["alpha"]
    s = 10.0**parms["logs"]
    q = 10.0**parms["logq"]
    tau = (times - t0) / tE
    vbbl = VBBinaryLensing.VBBinaryLensing()
    alpha_vbbl = alpha_deg / 180.0 * np.pi + np.pi
    vbbl.Tol = 1e-2
    vbbl.RelTol = 1e-3
    y1 = -u0 * np.sin(alpha_vbbl) + tau * np.cos(alpha_vbbl)
    y2 = u0 * np.cos(alpha_vbbl) + tau * np.sin(alpha_vbbl)
    params = [np.log(s), np.log(q), u0, alpha_vbbl, np.log(rho), np.log(tE), t0]
    return np.array(vbbl.BinaryLightCurve(params, times, y1, y2))


def objective_func(parms, data, fs, fb, return_chi2=True):
    parm_dict = dict(zip(PARAMETER_NAMES, parms))
    times, flux, ferr = data
    model_flux = light_curve_VBBL(times, parm_dict) * fs + fb
    chi2 = np.sum(((model_flux - flux) / ferr) ** 2)
    if return_chi2:
        return chi2
    else:
        return -0.5 * chi2


def hist2d(x, y, *args, **kwargs):
    """Plot a 2-D histogram of samples."""
    ax = kwargs.pop("ax", plt.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 30)
    color = kwargs.pop("color", "b")
    linewidths = kwargs.pop("linewidths", None)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap = plt.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.0
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(
            x.flatten(), y.flatten(), bins=(X, Y), weights=kwargs.get("weights", None)
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "`extent` argument."
        )

    V = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except Exception:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1, rasterized=True)
        if plot_contours:
            ax.contourf(
                X1,
                Y1,
                H.T,
                [V[-1], H.max()],
                cmap=LinearSegmentedColormap.from_list("cmap", ([1] * 3, [1] * 3), N=2),
                antialiased=False,
            )

    if plot_contours:
        V = [V[-1], V[-2], V[-3]]
        ax.contour(X1, Y1, H.T, V, colors=color, alpha=0.5, linewidths=linewidths)

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
    return


def plot_covariance(params, labels, cov_mat, chain):
    """Plot covariance matrix from both theoretical covariance and MCMC chain."""
    K = len(params)
    factor = 2.0
    lbdim = 1.2 * factor
    trdim = 0.15 * factor
    whspace = 0.1
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(K, K, figsize=(10, 10))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    extents = [[x.min(), x.max()] for x in chain.T]

    for i in range(K):
        ax = axes[i, i]
        mu_x, sigma_x = params[i], np.sqrt(cov_mat[i, i])
        x = np.linspace(extents[i][0], extents[i][1], 100)
        p = 1 / np.sqrt(2 * np.pi) / sigma_x * np.exp(-((x - mu_x) ** 2) / 2.0 / sigma_x**2)
        ax.plot(x, p, "r", alpha=0.5)
        ax.hist(chain[:, i], histtype="step", density=1)
        ax.set_xlim(extents[i])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [label.set_rotation(45) for label in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.7)
        for j in range(K):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            mu_y = params[j]
            sigx2, sigy2, sigxy = cov_mat[i, i], cov_mat[j, j], cov_mat[i, j]
            sig12 = 0.5 * (sigx2 + sigy2) + np.sqrt((sigx2 - sigy2) ** 2 * 0.25 + sigxy**2)
            sig22 = 0.5 * (sigx2 + sigy2) - np.sqrt((sigx2 - sigy2) ** 2 * 0.25 + sigxy**2)
            sig1 = np.sqrt(sig12)
            sig2 = np.sqrt(sig22)
            alpha = 0.5 * np.arctan(2 * sigxy / (sigx2 - sigy2))
            if sigy2 > sigx2:
                alpha += np.pi / 2.0

            t = np.linspace(0, 2 * np.pi, 300)
            x = mu_x + sig1 * np.cos(t) * np.cos(alpha) - sig2 * np.sin(t) * np.sin(alpha)
            y = mu_y + sig1 * np.cos(t) * np.sin(alpha) + sig2 * np.sin(t) * np.cos(alpha)
            ax.plot(y, x, "r", alpha=0.5)

            hist2d(
                chain[:, j],
                chain[:, i],
                ax=ax,
                extent=[extents[j], extents[i]],
                plot_contours=True,
                plot_datapoints=False,
            )
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [label.set_rotation(45) for label in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.7)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [label.set_rotation(45) for label in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.6, 0.5)
    return fig, axes


def light_curve_Jax(parms, times):
    t0 = parms[0]
    u0 = parms[1]
    tE = parms[2]
    rho = 10.0**parms[3]
    alpha_deg = parms[4]
    s = 10.0**parms[5]
    q = 10.0**parms[6]
    return binary_mag(t0, u0, tE, rho, q, s, alpha_deg, times)


def light_curve_Jax_pmap(times, parms, i, n_pmap):
    times = jnp.reshape(times, (-1, n_pmap), order="C")
    times_i = times[:, i]
    return light_curve_Jax(parms, times_i)


def model_HMC(data, fs, fb, init_val, L, n_pmap=10):
    times, flux, ferr = data
    parmsample = numpyro.sample(
        "param_base", dist.Uniform(-1 * jnp.ones(len(init_val)), 1 * jnp.ones(len(init_val)))
    )
    parmsample = jnp.dot(L * 10, parmsample) + jnp.array(init_val)
    numpyro.deterministic("param", parmsample)
    pmap_light_curve = lambda curve_times, params, index: light_curve_Jax_pmap(
        curve_times, params, index, n_pmap
    )
    mag_mod = jax.pmap(pmap_light_curve, in_axes=(None, None, 0))(
        times, parmsample, jnp.arange(n_pmap)
    )
    mag_mod = jnp.reshape(mag_mod, (flux.shape[0],), order="F")
    flux_mod = mag_mod * fs + fb
    numpyro.sample("obs", dist.Normal(flux_mod, ferr), obs=flux)
    chi2 = jnp.sum(((flux_mod - flux) / ferr) ** 2)
    numpyro.deterministic("chi2", chi2)
