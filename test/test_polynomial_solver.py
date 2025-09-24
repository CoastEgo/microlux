from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from microlux.basic_function import get_poly_coff, to_lowmass
from microlux.polynomial_solver import Aberth_Ehrlich, AE_roots0
from test_util import get_caustic_permutation


rho_values = [1e-2, 1e-3, 1e-4]
q_values = [1e-1, 1e-2, 1e-3]
s_values = [0.6, 1.0, 1.4]


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


# Define parameter ranges for pytest
s_values_coeff = np.random.uniform(0.5, 1.5, 3)
m2_values = np.random.uniform(0.001, 0.1, 3)


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
    assert np.allclose(coeffs_original, coeffs_refactored), (
        f"Coefficients do not match for s={s:.4f} and m2={m2:.4f}"
    )


if __name__ == "__main__":
    test_polynomial_caustic(1e-2, 0.2, 0.9)
    test_polynomial_uniform(0.2, 0.9)
    test_poly_coefficients_consistency(s=1.0, m2=0.01)  # Example call for quick check
