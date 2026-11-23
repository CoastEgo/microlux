import jax
import jax.numpy as jnp
import numpy as np
import pytest
from microlux.fitting import (
    fisher_information,
    fit_grouped_fluxes,
)


@pytest.mark.fast
def test_fisher_information_can_be_jitted():
    design = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    residual_fn = lambda params: design @ params
    fisher_matrix = jax.jit(lambda params: fisher_information(residual_fn, params))(
        jnp.array([0.2, -0.1])
    )

    np.testing.assert_allclose(fisher_matrix, [[35.0, 49.0], [49.0, 69.0]])


@pytest.mark.fast
def test_fit_grouped_fluxes_matches_weighted_least_squares():
    magnification = np.array([1.0, 1.8, 2.7, 1.2, 2.2, 3.1])
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    flux_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2, 0.25])
    source_flux = np.array([2.0, 3.5])
    blend_flux = np.array([0.4, -0.7])
    noise = np.array([0.01, -0.03, 0.02, -0.04, 0.05, -0.01])
    flux = source_flux[group_ids] * magnification + blend_flux[group_ids] + noise

    result = fit_grouped_fluxes(magnification, flux, flux_err, group_ids, n_groups=2)

    compiled = jax.jit(fit_grouped_fluxes, static_argnames=("n_groups",))(
        magnification, flux, flux_err, group_ids, n_groups=2
    )
    for actual, expected in zip(compiled, result):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    for group_id in range(2):
        mask = group_ids == group_id
        design = np.column_stack([magnification[mask], np.ones(np.sum(mask))])
        weighted_design = design / flux_err[mask, None]
        weighted_flux = flux[mask] / flux_err[mask]
        expected_coefficients = np.linalg.solve(
            weighted_design.T @ weighted_design,
            weighted_design.T @ weighted_flux,
        )
        expected_covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        np.testing.assert_allclose(
            [result.source_flux[group_id], result.blend_flux[group_id]],
            expected_coefficients,
        )
        np.testing.assert_allclose(result.covariance[group_id], expected_covariance)
        expected_model_flux = design @ expected_coefficients
        expected_residuals = (flux[mask] - expected_model_flux) / flux_err[mask]
        np.testing.assert_allclose(result.model_flux[mask], expected_model_flux)
        np.testing.assert_allclose(result.residuals[mask], expected_residuals)
        np.testing.assert_allclose(
            result.chi2_by_group[group_id], np.sum(expected_residuals**2)
        )


@pytest.mark.fast
def test_profiled_grouped_residual_jacobian_matches_finite_difference():
    times = jnp.linspace(-1.5, 1.5, 10)
    group_ids = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    flux_err = jnp.linspace(0.08, 0.16, 10)

    def magnification_fn(params, model_times):
        return (
            1.5
            + jnp.exp(params[0] * model_times)
            + params[1] * jnp.sin(2.0 * model_times)
        )

    injected_params = jnp.array([0.25, 0.12])
    magnification = magnification_fn(injected_params, times)
    source_flux = jnp.array([2.0, 3.0])
    blend_flux = jnp.array([0.5, -0.4])
    noise = jnp.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02, -0.02, 0.01])
    flux = source_flux[group_ids] * magnification + blend_flux[group_ids] + noise
    params = jnp.array([0.22, 0.1])

    def residual_fn(values):
        return fit_grouped_fluxes(
            magnification_fn(values, times),
            flux,
            flux_err,
            group_ids,
            n_groups=2,
        ).residuals

    profiled_fit = fit_grouped_fluxes(
        magnification_fn(params, times), flux, flux_err, group_ids, n_groups=2
    )
    jacobian = np.asarray(jax.jit(jax.jacfwd(residual_fn))(params))
    epsilon = 1e-5
    finite_difference = np.column_stack(
        [
            (
                np.asarray(residual_fn(params.at[index].add(epsilon)))
                - np.asarray(residual_fn(params.at[index].add(-epsilon)))
            )
            / (2.0 * epsilon)
            for index in range(2)
        ]
    )

    np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(residual_fn))(params),
        finite_difference,
        rtol=2e-5,
        atol=2e-6,
    )

    # Padded observations must not change the solved fluxes or their derivatives.
    padded_times = jnp.pad(times, (0, 2), mode="edge")
    padded_flux = jnp.pad(flux, (0, 2), mode="edge")
    padded_errors = jnp.pad(flux_err, (0, 2), constant_values=jnp.inf)
    padded_groups = jnp.pad(group_ids, (0, 2))

    def padded_fit(values):
        return fit_grouped_fluxes(
            magnification_fn(values, padded_times),
            padded_flux,
            padded_errors,
            padded_groups,
            n_groups=2,
        )

    padded = jax.jit(padded_fit)(params)
    for name in ("source_flux", "blend_flux", "covariance", "chi2_by_group"):
        np.testing.assert_allclose(getattr(padded, name), getattr(profiled_fit, name))
    np.testing.assert_allclose(padded.residuals[:-2], profiled_fit.residuals)
    np.testing.assert_array_equal(padded.residuals[-2:], 0)
    expected_padded_jacobian = np.pad(jacobian, ((0, 2), (0, 0)))
    padded_residuals = lambda p: padded_fit(p).residuals
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(padded_residuals))(params), expected_padded_jacobian
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(padded_residuals))(params), expected_padded_jacobian
    )
