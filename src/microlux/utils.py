import warnings
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import lax


MAX_CAUSTIC_INTERSECT_NUM = 15


class Iterative_State(NamedTuple):
    """
    A NamedTuple representing the state of an iterative process in the microlux module.

    Attributes:
        sample_num (int): The number of samples.
        theta (jax.Array): The sampling angles.
        roots (jax.Array): The roots array.
        parity (jax.Array): The parity array.
        ghost_roots_distant (jax.Array): The ghost roots distant array, used to detect the buried images (hidden cusps)
        sort_flag (Union[bool, jax.Array]): A boolean flag indicating whether the roots are sorted and matched.
        Is_create (jax.Array): The Is_create array. A boolean flag indicating whether the image is created or destroyed.
    """

    sample_num: int
    theta: jax.Array
    roots: jax.Array
    parity: jax.Array
    ghost_roots_distant: jax.Array
    sort_flag: Union[bool, jax.Array]
    Is_create: jax.Array = jnp.zeros((4, MAX_CAUSTIC_INTERSECT_NUM), dtype=int)


class Error_State(NamedTuple):
    """
    Error_State is a NamedTuple that holds various attributes related to the error state in the adaptive sampling.

    Attributes:
        mag (jax.Array): Current magnification values.
        mag_no_diff (int): The number of magnification values without sufficient difference.
        outloop (int): A integer flag indicating whether the iteration should be terminated.
        error_hist (jax.Array): The current error estimated in each sampling interval.
        epsilon (float): The absolute tolerance value.
        epsilon_rel (float): The relative tolerance value.
        exceed_flag (bool): The flag indicating whether the current sampling number exceeds the length of the array.
    """

    mag: jax.Array
    mag_no_diff: int
    outloop: int
    error_hist: jax.Array
    epsilon: float
    epsilon_rel: float
    exceed_flag: bool = False


def get_default_state(total_length: int) -> tuple[Iterative_State, Error_State]:
    """
    used to get the default state of Iterative_State and Error_State. This acts as the placeholder for point source approximation when user wants to return the state in light curve.
    """

    pad_value = [jnp.nan, 0.0, jnp.nan + 1j * jnp.nan, jnp.nan, jnp.nan, 0.0, True]
    shape = [1, 1, 5, 5, 1, 1, 1]
    init_fun = lambda x, y: jnp.full((total_length, y), x)
    theta, error_hist, roots, parity, ghost_roots_dis, buried_error, sort_flag = (
        jax.tree.map(init_fun, pad_value, shape)
    )
    sample_n = 0
    roots_state = Iterative_State(
        sample_n, theta, roots, parity, ghost_roots_dis, sort_flag
    )
    error_state = Error_State(jnp.array([0.0]), 0, 0, error_hist, 1e-3, 1e-3)

    return roots_state, error_state


def insert_body(carry, k):
    array, add_array, idx, add_number = carry
    ite = jnp.arange(array.shape[0])
    mask = ite < idx[k]
    array = jnp.where(mask[:, None], array, jnp.roll(array, add_number[k], axis=0))
    mask2 = (ite >= idx[k]) & (ite < idx[k] + add_number[k])
    add_array = jnp.roll(add_array, idx[k], axis=0)
    array = jnp.where(mask2[:, None], add_array, array)
    add_array = jnp.roll(add_array, -1 * add_number[k] - idx[k], axis=0)
    idx += add_number[k]
    return (array, add_array, idx, add_number), k


def custom_insert(array, idx, add_array):
    """
    custom defined insert function to insert the elements in the array without changing the shape of the array
    """
    final_array = jnp.insert(array, idx, add_array, axis=0)
    final_array = final_array[: array.shape[0]]
    return final_array


def delete_body(carry, k):
    array, ite2, delidx = carry
    mask = ite2 < delidx[k]
    array = jnp.where(mask[:, None], array, jnp.roll(array, -1, axis=0))
    delidx -= (~mask).any()
    return (array, ite2, delidx), k


def custom_delete(array, delidx):
    """
    custom defined delete function to delete the elements in the array without changing the shape of the array
    """
    fill_value = array[-1]
    ite = jnp.arange(array.shape[0])
    carry, _ = lax.scan(delete_body, (array, ite, delidx), jnp.arange(delidx.shape[0]))
    array, _, _ = carry
    array = jnp.where(
        (ite < ite.size - (delidx < array.shape[0]).sum())[:, None], array, fill_value
    )
    return array


def stop_grad_wrapper(func):
    def wrapper(*args, **kwargs):
        args = jax.lax.stop_gradient(args)
        kwargs = jax.lax.stop_gradient(kwargs)
        return jax.lax.stop_gradient(func(*args, **kwargs))

    return wrapper


def warn_length_not_enough(required_length, Max_length):
    warnings.warn(
        "No enough space to insert new samplings, which may cause the error larger than the tolerance. Current length vs max length: {} vs {}. Consider incresing default_strategy parameters.".format(
            required_length, Max_length - 2
        )
    )
