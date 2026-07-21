"""Differentiable residual, linear-flux, and Fisher utilities."""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class FisherInformation(NamedTuple):
    """JAX-compatible residual Jacobian and Fisher matrix."""

    jacobian: jax.Array
    fisher_matrix: jax.Array


class FisherResult(NamedTuple):
    """Eager Fisher result with covariance and reliability diagnostics."""

    jacobian: np.ndarray
    fisher_matrix: np.ndarray
    covariance: np.ndarray
    errors: np.ndarray
    correlation: np.ndarray
    rank: int
    condition_number: float


class GroupedFluxFit(NamedTuple):
    """Analytic source/blend-flux fit for independent photometric groups."""

    source_flux: jax.Array
    blend_flux: jax.Array
    covariance: jax.Array
    model_flux: jax.Array
    residuals: jax.Array
    chi2_by_group: jax.Array


def fisher_information(
    residual_fn: Callable[[jax.Array], jax.Array], params: jax.Array
) -> FisherInformation:
    """Return the residual Jacobian and Gauss--Newton Fisher matrix."""
    jacobian = jax.jacfwd(residual_fn)(params)
    if jacobian.ndim != 2:
        raise ValueError("params and normalized residuals must be one-dimensional.")
    fisher_matrix = jacobian.T @ jacobian
    return FisherInformation(jacobian, fisher_matrix)


def fisher_from_residuals(
    residual_fn: Callable[[jax.Array], jax.Array], params: jax.Array
) -> FisherResult:
    """Compute an eager, validated Fisher covariance from normalized residuals."""
    information = fisher_information(residual_fn, params)
    jacobian = np.asarray(information.jacobian)
    fisher_matrix = np.asarray(information.fisher_matrix)

    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(fisher_matrix)):
        raise ValueError("Fisher calculation produced non-finite values.")

    rank = int(np.linalg.matrix_rank(fisher_matrix))
    condition_number = float(np.linalg.cond(fisher_matrix))
    if rank < fisher_matrix.shape[0]:
        raise np.linalg.LinAlgError("Fisher matrix is singular.")

    covariance = np.linalg.inv(fisher_matrix)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Fisher covariance contains non-finite values.")

    variance = np.diag(covariance)
    if np.any(variance < 0):
        raise ValueError("Fisher covariance has a negative diagonal entry.")

    errors = np.sqrt(variance)
    correlation = covariance / np.outer(errors, errors)
    return FisherResult(
        jacobian=jacobian,
        fisher_matrix=fisher_matrix,
        covariance=covariance,
        errors=errors,
        correlation=correlation,
        rank=rank,
        condition_number=condition_number,
    )


def normalized_residuals(
    params: jax.Array,
    times: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    model_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Return Gaussian residuals normalized by their flux uncertainties."""
    model_flux = model_fn(params, times)
    return (flux - model_flux) / flux_err


def _validate_grouped_inputs(
    magnification,
    flux,
    flux_err,
    group_ids,
    n_groups: Optional[int],
) -> int:
    """Validate eager grouped-fit inputs and return the group count."""
    arrays = tuple(
        np.asarray(values) for values in (magnification, flux, flux_err, group_ids)
    )
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("Grouped flux fitting requires one-dimensional arrays.")
    if len(arrays[0]) == 0:
        raise ValueError("Grouped flux fitting requires non-empty arrays.")
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("Grouped flux fitting inputs must have the same shape.")
    if not all(np.all(np.isfinite(values)) for values in arrays[:3]):
        raise ValueError("Grouped flux fitting inputs must be finite.")
    if np.any(arrays[2] <= 0):
        raise ValueError("Flux uncertainties must be positive.")
    if not np.issubdtype(arrays[3].dtype, np.integer):
        raise ValueError("group_ids must contain integers.")

    unique_groups = np.unique(arrays[3])
    inferred_groups = len(unique_groups)
    if n_groups is None:
        n_groups = inferred_groups
    if n_groups < 1 or not np.array_equal(unique_groups, np.arange(n_groups)):
        raise ValueError("group_ids must be contiguous from 0 to n_groups - 1.")

    for group_id in range(n_groups):
        mask = arrays[3] == group_id
        design = np.column_stack([arrays[0][mask], np.ones(np.sum(mask))])
        if np.linalg.matrix_rank(design) < 2:
            raise np.linalg.LinAlgError(
                f"Source/blend flux design is singular for group {group_id}."
            )
    return n_groups


def _fit_grouped_fluxes(
    magnification: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    group_ids: jax.Array,
    n_groups: int,
) -> GroupedFluxFit:
    """Solve all grouped source/blend flux fits with JAX segment sums."""
    magnification = jnp.asarray(magnification)
    flux = jnp.asarray(flux)
    flux_err = jnp.asarray(flux_err)
    group_ids = jnp.asarray(group_ids)
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


def fit_grouped_fluxes(
    magnification,
    flux,
    flux_err,
    group_ids,
    *,
    n_groups: Optional[int] = None,
) -> GroupedFluxFit:
    """Fit independent source and blend fluxes for contiguous groups."""
    n_groups = _validate_grouped_inputs(
        magnification, flux, flux_err, group_ids, n_groups
    )
    result = _fit_grouped_fluxes(magnification, flux, flux_err, group_ids, n_groups)
    if not all(np.all(np.isfinite(np.asarray(values))) for values in result):
        raise ValueError("Grouped flux fitting produced non-finite values.")
    return result


def profiled_grouped_fluxes(
    params: jax.Array,
    times: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    group_ids: jax.Array,
    magnification_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    n_groups: int,
) -> GroupedFluxFit:
    """Return a differentiable per-group source and blend flux fit."""
    magnification = magnification_fn(params, times)
    return _fit_grouped_fluxes(magnification, flux, flux_err, group_ids, n_groups)


def profiled_grouped_residuals(
    params: jax.Array,
    times: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    group_ids: jax.Array,
    magnification_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    n_groups: int,
) -> jax.Array:
    """Return residuals after profiling per-group source and blend fluxes."""
    return profiled_grouped_fluxes(
        params,
        times,
        flux,
        flux_err,
        group_ids,
        magnification_fn,
        n_groups=n_groups,
    ).residuals


def profiled_grouped_fisher(
    params: jax.Array,
    times: jax.Array,
    flux: jax.Array,
    flux_err: jax.Array,
    group_ids: jax.Array,
    magnification_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    n_groups: Optional[int] = None,
) -> FisherResult:
    """Compute a validated Fisher result with per-group fluxes profiled out."""
    magnification = magnification_fn(params, times)
    n_groups = _validate_grouped_inputs(
        magnification, flux, flux_err, group_ids, n_groups
    )

    def residual_fn(values):
        """Evaluate normalized residuals after profiling grouped fluxes."""
        return profiled_grouped_residuals(
            values,
            times,
            flux,
            flux_err,
            group_ids,
            magnification_fn,
            n_groups=n_groups,
        )

    return fisher_from_residuals(residual_fn, params)
