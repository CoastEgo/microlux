from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from microlux.basic_function import get_poly_coff, to_lowmass, verify
from microlux.polynomial_solver import (
    Aberth_Ehrlich,
    AE_roots0,
    get_point_roots,
    get_roots,
)
from test_util import get_caustic_permutation


rho_values = [1e-2, 1e-3, 1e-4]
q_values = [1e-1, 1e-2, 1e-3]
s_values = [0.6, 1.0, 1.4]


@pytest.mark.fast
def test_complex_root_tangents():
    def roots_of_quintic(c):
        coeffs = jnp.array([1, 0, 0, 0, 0, -c])
        return Aberth_Ehrlich(coeffs, AE_roots0(coeffs))

    c = 1.0 + 0.5j
    direction = 0.3 - 0.7j
    roots, tangents = jax.jvp(roots_of_quintic, (c,), (direction,))
    # Differentiating z**5 = c gives 5 * z**4 * dz = dc for each root.
    np.testing.assert_allclose(5 * roots**4 * tangents, direction, rtol=1e-12)


@pytest.mark.fast
@pytest.mark.parametrize(
    "padded", [False, True], ids=["point_source", "contour_padding"]
)
def test_root_derivatives_across_epochs(padded):
    constants = jnp.array([1 + 0.5j, 0.9 - 0.1j, 1.1 + 0.3j])
    directions = jnp.array([0.3 - 0.7j, -0.2 + 0.1j, 0.5 + 0.2j])

    def coefficients(t):
        rows = jnp.zeros((3, 6), dtype=complex).at[:, 0].set(1)
        rows = rows.at[:, -1].set(-(constants + t * directions))
        return rows.at[1].set(0) if padded else rows

    def solve(t):
        rows = coefficients(t)
        return get_roots(rows.shape[0], rows) if padded else get_point_roots(rows)

    roots, dz = jax.jvp(solve, (0.0,), (1.0,))
    if padded:
        np.testing.assert_array_equal(roots[1], roots[0])
    else:
        np.testing.assert_array_equal(roots, get_roots(3, coefficients(0.0)))
    dc = directions.at[1].set(directions[0]) if padded else directions
    expected_dz = dc[:, None] / (5 * roots**4)
    np.testing.assert_allclose(dz, expected_dz, rtol=1e-12)

    # For z**5 = c(t) with linear c, differentiating twice gives z'' = -4 z'²/z.
    objective = lambda t: solve(t)[:, 0].real.sum()
    first, second = jax.jvp(jax.grad(objective), (0.0,), (1.0,))
    np.testing.assert_allclose(first, expected_dz[:, 0].real.sum(), rtol=1e-12)
    np.testing.assert_allclose(
        second, (-4 * expected_dz[:, 0] ** 2 / roots[:, 0]).real.sum(), rtol=1e-12
    )


@pytest.mark.fast
@pytest.mark.parametrize("lens_position", [0.0, 0.9], ids=["secondary", "primary"])
def test_polynomial_near_lens_cold_start(lens_position):
    q = 0.2
    s = 0.9
    m1 = 1 / (1 + q)
    m2 = q / (1 + q)
    offset = 1e-14 * (1 + 1j) / jnp.sqrt(2)
    zeta_l = jnp.array([[lens_position + offset]])
    coff = get_poly_coff(zeta_l, s, m2)[0]

    roots = Aberth_Ehrlich(coff, AE_roots0(coff), MAX_ITER=50)
    lens_errors = jnp.sort(verify(zeta_l, roots[None, :], s, m1, m2)[0])

    assert lens_errors[2] < 1e-10


@pytest.mark.fast
@pytest.mark.parametrize("rho, q, s", product(rho_values, q_values, s_values))
def test_polynomial_caustic(rho, q, s):
    trajectory_c = get_caustic_permutation(rho, q, s, n_points=100)
    theta_sample = jnp.linspace(0, 2 * jnp.pi, 100)
    contours = (trajectory_c + rho * jnp.exp(1j * theta_sample)[:, None]).ravel()

    z_lowmass = to_lowmass(s, q, contours)

    coff = get_poly_coff(z_lowmass[:, None], s, q / (1 + q))

    get_AE_roots = lambda x: Aberth_Ehrlich(x, AE_roots0(x), MAX_ITER=50).sort()
    AE_roots = jax.jit(jax.vmap(get_AE_roots))(coff)

    get_numpy_roots = lambda x: jnp.roots(x, strip_zeros=False).sort()
    numpy_roots = jax.jit(jax.vmap(get_numpy_roots))(coff)

    error = jnp.abs(AE_roots - numpy_roots)

    print("max absolute error is", jnp.max(error))
    assert jnp.allclose(AE_roots, numpy_roots, atol=1e-10)


@pytest.mark.fast
@pytest.mark.parametrize("q, s", product(q_values, s_values))
def test_polynomial_uniform(q, s):
    x, y = jax.random.uniform(jax.random.PRNGKey(0), (2, 100000), minval=-2, maxval=2)

    trajectory_c = x + 1j * y
    z_lowmass = to_lowmass(s, q, trajectory_c)

    coff = get_poly_coff(z_lowmass[:, None], s, q / (1 + q))

    get_AE_roots = lambda x: Aberth_Ehrlich(x, AE_roots0(x), MAX_ITER=50).sort()
    AE_roots = jax.jit(jax.vmap(get_AE_roots))(coff)

    get_numpy_roots = lambda x: jnp.roots(x, strip_zeros=False).sort()
    numpy_roots = jax.jit(jax.vmap(get_numpy_roots))(coff)

    error = jnp.abs(AE_roots - numpy_roots)
    print("max absolute error is", jnp.max(error))

    assert jnp.allclose(AE_roots, numpy_roots, atol=1e-10)


def get_poly_coff_old(zeta_l, s, m2):
    """
    get the polynomial cofficients of the polynomial equation of the lens equation. The low mass object is at the origin and the primary is at s.
    The input zeta_l should have the shape of (n,1) for broadcasting.
    """
    zeta_conj = jnp.conj(zeta_l)
    c0 = s**2 * zeta_l * m2**2
    c1 = -s * m2 * (2 * zeta_l + s * (-1 + s * zeta_l - 2 * zeta_l * zeta_conj + m2))
    c2 = (
        zeta_l
        - s**3 * zeta_l * zeta_conj
        + s * (-1 + m2 - 2 * zeta_conj * zeta_l * (1 + m2))
        + s**2 * (zeta_conj - 2 * zeta_conj * m2 + zeta_l * (1 + zeta_conj**2 + m2))
    )
    c3 = (
        s**3 * zeta_conj
        + 2 * zeta_l * zeta_conj
        + s**2 * (-1 + 2 * zeta_conj * zeta_l - zeta_conj**2 + m2)
        - s * (zeta_l + 2 * zeta_l * zeta_conj**2 - 2 * zeta_conj * m2)
    )
    c4 = zeta_conj * (-1 + 2 * s * zeta_conj + zeta_conj * zeta_l) - s * (
        -1 + 2 * s * zeta_conj + zeta_conj * zeta_l + m2
    )
    c5 = (s - zeta_conj) * zeta_conj
    coff = jnp.concatenate((c5, c4, c3, c2, c1, c0), axis=1)
    return coff


# Define parameter ranges for pytest use list type. See : Known limitations in pytest-xdist https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html
s_values_coeff = [0.5, 1.0, 1.5]
m2_values = [0.01, 0.1, 0.5]


@pytest.mark.fast
@pytest.mark.parametrize("s, m2", product(s_values_coeff, m2_values))
def test_poly_coefficients_consistency(s, m2):
    """
    Tests if get_poly_coff and get_poly_coff_old produce the same coefficients
    for a range of random inputs.
    """
    # 1. Generate random test data for zeta_l
    n_samples = 10
    np.random.seed(42)
    zeta_l_real = np.random.uniform(-2, 2, (n_samples, 1))
    zeta_l_imag = np.random.uniform(-2, 2, (n_samples, 1))
    zeta_l = jnp.array(zeta_l_real + 1j * zeta_l_imag)

    # 2. Call both functions
    coeffs_original = get_poly_coff_old(zeta_l, s, m2)
    coeffs_refactored = get_poly_coff(zeta_l, s, m2)

    # 3. Compare results using an assertion
    assert np.allclose(
        coeffs_original, coeffs_refactored
    ), f"Coefficients do not match for s={s:.4f} and m2={m2:.4f}"


if __name__ == "__main__":
    # test_polynomial_caustic(1e-2, 0.2, 0.9)
    # test_polynomial_uniform(0.2, 0.9)
    test_poly_coefficients_consistency(s=1.0, m2=0.01)  # Example call for quick check
