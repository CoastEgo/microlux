import jax
import jax.numpy as jnp
import numpy as np
import pytest
from microlux.fitting import (
    fisher_from_residuals,
    fisher_information,
    fit_grouped_fluxes,
    normalized_residuals,
    profiled_grouped_fisher,
    profiled_grouped_fluxes,
    profiled_grouped_residuals,
)


@pytest.mark.fast
def test_fisher_from_linear_residuals():
    design = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )
    offset = jnp.array([0.2, -0.3, 0.5])

    def residual_fn(params):
        return design @ params - offset

    result = fisher_from_residuals(residual_fn, jnp.array([0.1, -0.2]))
    expected_fisher = np.asarray(design.T @ design)
    expected_covariance = np.linalg.inv(expected_fisher)

    np.testing.assert_allclose(result.jacobian, design)
    np.testing.assert_allclose(result.fisher_matrix, expected_fisher)
    np.testing.assert_allclose(result.covariance, expected_covariance)
    np.testing.assert_allclose(result.errors, np.sqrt(np.diag(expected_covariance)))
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert result.rank == 2
    assert np.isclose(result.condition_number, np.linalg.cond(expected_fisher))


@pytest.mark.fast
def test_fisher_information_can_be_jitted():
    design = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    residual_fn = lambda params: design @ params
    jacobian, fisher_matrix = jax.jit(
        lambda params: fisher_information(residual_fn, params)
    )(jnp.array([0.2, -0.1]))

    np.testing.assert_allclose(jacobian, design)
    np.testing.assert_allclose(fisher_matrix, design.T @ design)


@pytest.mark.fast
def test_fisher_from_residuals_rejects_singular_matrix():
    def residual_fn(params):
        return jnp.array([params[0] + params[1], 2.0 * (params[0] + params[1])])

    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fisher_from_residuals(residual_fn, jnp.array([1.0, 2.0]))


@pytest.mark.fast
def test_fisher_from_residuals_rejects_nonfinite_values():
    def residual_fn(params):
        return jnp.array([params[0], jnp.nan * params[1]])

    with pytest.raises(ValueError, match="non-finite"):
        fisher_from_residuals(residual_fn, jnp.array([1.0, 2.0]))


@pytest.mark.fast
def test_normalized_residuals_uses_flux_model():
    times = jnp.array([0.0, 1.0, 2.0])
    flux = jnp.array([1.2, 2.9, 5.1])
    flux_err = jnp.array([0.1, 0.2, 0.5])
    model_fn = lambda params, model_times: params[0] * model_times + params[1]

    residuals = normalized_residuals(
        jnp.array([2.0, 1.0]), times, flux, flux_err, model_fn
    )

    np.testing.assert_allclose(residuals, [2.0, -0.5, 0.2])


@pytest.mark.fast
def test_fit_grouped_fluxes_matches_weighted_least_squares():
    magnification = np.array([1.0, 1.8, 2.7, 1.2, 2.2, 3.1])
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    flux_err = np.array([0.1, 0.2, 0.15, 0.3, 0.2, 0.25])
    source_flux = np.array([2.0, 3.5])
    blend_flux = np.array([0.4, -0.7])
    noise = np.array([0.01, -0.03, 0.02, -0.04, 0.05, -0.01])
    flux = source_flux[group_ids] * magnification + blend_flux[group_ids] + noise

    result = fit_grouped_fluxes(magnification, flux, flux_err, group_ids)

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

    np.testing.assert_allclose(
        result.model_flux,
        result.source_flux[group_ids] * magnification + result.blend_flux[group_ids],
    )
    np.testing.assert_allclose(
        result.chi2_by_group,
        [
            np.sum(np.asarray(result.residuals)[group_ids == group_id] ** 2)
            for group_id in range(2)
        ],
    )


@pytest.mark.fast
def test_fit_grouped_fluxes_rejects_noncontiguous_groups():
    with pytest.raises(ValueError, match="contiguous"):
        fit_grouped_fluxes(
            np.array([1.0, 2.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 1.0, 2.0]),
            np.ones(4),
            np.array([0, 0, 2, 2]),
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
        return profiled_grouped_residuals(
            values,
            times,
            flux,
            flux_err,
            group_ids,
            magnification_fn,
            n_groups=2,
        )

    profiled_fit = jax.jit(
        lambda values: profiled_grouped_fluxes(
            values,
            times,
            flux,
            flux_err,
            group_ids,
            magnification_fn,
            n_groups=2,
        )
    )(params)
    np.testing.assert_allclose(profiled_fit.residuals, residual_fn(params))
    assert profiled_fit.source_flux.shape == (2,)
    assert profiled_fit.blend_flux.shape == (2,)

    jacobian = np.asarray(jax.jacfwd(residual_fn)(params))
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
    np.testing.assert_allclose(jax.jit(residual_fn)(params), residual_fn(params))

    fisher_result = profiled_grouped_fisher(
        params,
        times,
        flux,
        flux_err,
        group_ids,
        magnification_fn,
    )
    np.testing.assert_allclose(fisher_result.jacobian, jacobian)
