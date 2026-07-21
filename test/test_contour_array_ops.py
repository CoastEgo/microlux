import jax.numpy as jnp
import numpy as np
import pytest
from microlux.countour import build_add_theta
from microlux.utils import (
    apply_insert,
    compact_delete,
    custom_delete,
    custom_insert,
    get_delete_source_indices,
    get_insert_source_indices,
)


@pytest.mark.fast
@pytest.mark.parametrize(
    "add_idx",
    [
        [2, -1, -1],
        [2, 2, -1],
        [1, 1, 5, -1],
        [3, 6, 6, 9],
        [10, -1],
    ],
)
def test_batch_insert_matches_custom_insert(add_idx):
    array = jnp.arange(60, dtype=float).reshape(12, 5)
    add_array = 100 + jnp.arange(5 * len(add_idx), dtype=float).reshape(-1, 5)
    add_idx = jnp.asarray(add_idx)

    source_indices = get_insert_source_indices(array.shape[0], add_idx)
    result = apply_insert(array, add_array, source_indices)
    expected = custom_insert(array, add_idx, add_array)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast
def test_batch_insert_preserves_padding_values():
    array = jnp.array([[False], [False], [False], [True], [True], [True]])
    add_array = jnp.zeros((3, 1), dtype=bool)
    add_idx = jnp.array([1, 2, -1])

    source_indices = get_insert_source_indices(array.shape[0], add_idx)
    result = apply_insert(array, add_array, source_indices)
    expected = custom_insert(array, add_idx, add_array)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast
@pytest.mark.parametrize(
    "delidx",
    [
        [10000, 10000],
        [2, 10000],
        [1, 3, 10000],
        [1, 2, 3],
        [0, 7, 10000],
    ],
)
def test_batch_delete_matches_custom_delete(delidx):
    array = jnp.arange(40, dtype=float).reshape(8, 5)
    delidx = jnp.asarray(delidx)

    source_indices = get_delete_source_indices(array.shape[0], delidx)
    result = compact_delete(array, source_indices)
    expected = custom_delete(array, delidx)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast
def test_build_add_theta_preserves_interval_order():
    theta = jnp.concatenate(
        [jnp.linspace(0, 2 * jnp.pi, 8)[:, None], jnp.full((4, 1), jnp.nan)]
    )
    idx = jnp.array([2, 5, 7, -1])
    add_number = jnp.array([[2], [1], [3], [0]])

    add_theta, add_idx = build_add_theta(
        theta, idx, add_number, max_add=4, max_total_num=8
    )

    expected_theta = jnp.array(
        [
            theta[1, 0] + (theta[2, 0] - theta[1, 0]) / 3,
            theta[1, 0] + 2 * (theta[2, 0] - theta[1, 0]) / 3,
            theta[4, 0] + (theta[5, 0] - theta[4, 0]) / 2,
            theta[6, 0] + (theta[7, 0] - theta[6, 0]) / 4,
            theta[6, 0] + 2 * (theta[7, 0] - theta[6, 0]) / 4,
            theta[6, 0] + 3 * (theta[7, 0] - theta[6, 0]) / 4,
            jnp.nan,
            jnp.nan,
        ]
    )

    np.testing.assert_allclose(add_theta[:, 0], expected_theta, equal_nan=True)
    np.testing.assert_array_equal(add_idx, [2, 2, 5, 7, 7, 7, -1, -1])
