"""Differentiable grouped-flux fitting and Fisher information."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class GroupedFluxFit(NamedTuple):
    """Analytic source/blend-flux fit for independent photometric groups.

    Attributes:
        source_flux (jax.Array): Source flux for each group, shape (G,).
        blend_flux (jax.Array): Blend flux for each group, shape (G,).
        covariance (jax.Array): Source/blend covariance per group, shape
            (G, 2, 2), conditional on the supplied magnifications and errors.
        model_flux (jax.Array): Fitted fluxes in observation order, shape (N,).
        residuals (jax.Array): (flux - model_flux) / flux_err, shape (N,).
        chi2_by_group (jax.Array): Sum of squared residuals per group, shape (G,).
    """

    source_flux: jax.Array
    blend_flux: jax.Array
    covariance: jax.Array
    model_flux: jax.Array
    residuals: jax.Array
    chi2_by_group: jax.Array


def fisher_information(
    residual_fn: Callable[[jax.Array], jax.Array], params: jax.Array
) -> jax.Array:
    """Return the Gauss--Newton Fisher matrix of normalized residuals.

    Args:
        residual_fn (Callable): Function mapping params to one-dimensional
            normalized residuals of shape (N,).
        params (jax.Array): Parameter vector of shape (P,).

    Returns:
        fisher_matrix (jax.Array): J.T @ J of shape (P, P), where J is the
            residual Jacobian. Supports JIT compilation.
    """
    jacobian = jax.jacfwd(residual_fn)(params)
    if jacobian.ndim != 2:
        raise ValueError("params and normalized residuals must be one-dimensional.")
    return jacobian.T @ jacobian


def fit_grouped_fluxes(
    magnification: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    group_ids: jax.Array,
    *,
    n_groups: int,
) -> GroupedFluxFit:
    """Fit source and blend fluxes for each group with JAX.

    Args:
        magnification (jax.Array): Finite magnifications of shape (N,).
        flux (jax.Array): Finite observed fluxes of shape (N,).
        flux_err (jax.Array): Positive uncertainties of shape (N,). Use
            infinity for zero-weight padding with finite flux and magnification.
        group_ids (jax.Array): Integer labels of shape (N,), contiguous from
            zero to n_groups - 1. Observations may be interleaved.
        n_groups (int): Number of groups, fixed when tracing a JIT function.

    Returns:
        result (GroupedFluxFit): Weighted least-squares solution of
            `flux = source_flux[group_ids] * magnification + blend_flux[group_ids]`.
            Differentiation includes the dependence of the fitted fluxes
            on the supplied magnifications.

    Notes:
        Accepts NumPy or JAX arrays and supports JIT and automatic differentiation.
        Each group needs a full-rank weighted source/blend design. Validate
        data values and handle degenerate fits in the caller. Only shapes
        and label dtype are checked here.
    """
    magnification = jnp.asarray(magnification)
    flux = jnp.asarray(flux)
    flux_err = jnp.asarray(flux_err)
    group_ids = jnp.asarray(group_ids)
    arrays = (magnification, flux, flux_err, group_ids)
    if any(
        values.ndim != 1 or values.shape != magnification.shape for values in arrays
    ):
        raise ValueError(
            "Grouped flux fitting requires matching one-dimensional arrays."
        )
    if not jnp.issubdtype(group_ids.dtype, jnp.integer):
        raise ValueError("group_ids must contain integers.")
    weight = 1.0 / flux_err**2

    def grouped_sum(values):
        """Sum per-observation values within each photometric group."""
        return jax.ops.segment_sum(values, group_ids, num_segments=n_groups)

    sum_aa = grouped_sum(weight * magnification**2)
    sum_a = grouped_sum(weight * magnification)
    sum_one = grouped_sum(weight)
    sum_ay = grouped_sum(weight * magnification * flux)
    sum_y = grouped_sum(weight * flux)
    determinant = sum_aa * sum_one - sum_a**2

    source_flux = (sum_ay * sum_one - sum_y * sum_a) / determinant
    blend_flux = (sum_aa * sum_y - sum_a * sum_ay) / determinant
    covariance = (
        jnp.stack(
            [
                jnp.stack([sum_one, -sum_a], axis=-1),
                jnp.stack([-sum_a, sum_aa], axis=-1),
            ],
            axis=-2,
        )
        / determinant[:, None, None]
    )

    model_flux = source_flux[group_ids] * magnification + blend_flux[group_ids]
    residuals = (flux - model_flux) / flux_err
    chi2_by_group = grouped_sum(residuals**2)
    return GroupedFluxFit(
        source_flux=source_flux,
        blend_flux=blend_flux,
        covariance=covariance,
        model_flux=model_flux,
        residuals=residuals,
        chi2_by_group=chi2_by_group,
    )
