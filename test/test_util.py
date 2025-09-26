import time

import jax
import jax.numpy as jnp
import numpy as np
import VBBinaryLensing
from MulensModel import CausticsBinary


def timeit(f, iters=10, verbose=True):
    """
    A decorator to measure the execution time of a function, especially for JAX functions.

    This decorator calculates the mean and standard deviation of the execution time
    of the decorated function over a specified number of iterations.

    Args:
        f (callable): The function to be timed.
        iters (int, optional): The number of iterations to run the function for timing. Default is 10.

    Returns:
        callable: A wrapped function that, when called, prints the compile time and the mean
                    and standard deviation of the execution time over the specified iterations.
    """

    def timed(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        if verbose:
            print(f"{f.__name__} compile time={te - ts}")
        alltime = []
        for i in range(iters):
            ts = time.perf_counter()
            result = f(*args, **kw)
            jax.block_until_ready(result)
            te = time.perf_counter()
            alltime.append(te - ts)
        alltime = np.array(alltime)
        if verbose:
            print(f"{f.__name__} time={np.mean(alltime)}+/-{np.std(alltime)}")
        return result, np.mean(alltime)

    return timed


def VBBL_light_curve(
    t_0,
    u_0,
    t_E,
    rho,
    q,
    s,
    alpha_deg,
    times,
    retol=0.0,
    tol=1e-2,
    limb_darkening: None | float = None,
):
    """
    Calculate the light curve of a binary lensing event using the VBBL model. Modified to the same coordinate system as the JAX model.

    Args:
        t_0 (float): The closest approach time.
        u_0 (float): The impact parameter of the event.
        t_E (float): The Einstein crossing time.
        rho (float): The angular source size in the unit of the Einstein radius.
        q (float): The mass ratio of the binary lens.
        s (float): The separation of the binary lens in the unit of the Einstein radius.
        alpha_deg (float): The angle of the source trajectory in degrees.
        times (array): The times at which to calculate the light curve.
        retol (float): The relative tolerance. Default is 0.
        tol (float, optional): The tolerance. Default is 1e-2.
        limb_darkening (float, optional): The limb darkening coefficient for linear limb darkening. Default is None.

    Returns:
        array: The magnification of this parameter set.
    """
    VBBL = VBBinaryLensing.VBBinaryLensing()
    if limb_darkening is not None:
        VBBL.a1 = limb_darkening
    alpha_VBBL = np.pi + alpha_deg / 180 * np.pi
    VBBL.Tol = tol
    VBBL.RelTol = retol
    VBBL.BinaryLightCurve
    times = np.array(times)
    tau = (times - t_0) / t_E
    y1 = -u_0 * np.sin(alpha_VBBL) + tau * np.cos(alpha_VBBL)
    y2 = u_0 * np.cos(alpha_VBBL) + tau * np.sin(alpha_VBBL)
    VBBL_mag = []
    # for i in range(len(times)):
    #     VBBL_mag.append(VBBL.BinaryMag2(s,q, y1[i], y2[i], rho))
    params = [np.log(s), np.log(q), u_0, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    VBBL_mag = np.array(VBBL_mag)
    return VBBL_mag


def get_trajectory(tau, u_0, alpha_deg):
    """
    Get the trajectory of the source star in the complex plane.
    """
    alpha = alpha_deg / 180 * np.pi
    trajectory = tau * np.exp(1j * alpha) + 1j * u_0 * np.exp(1j * alpha)
    return trajectory


def get_caustic_permutation(rho, q, s, n_points=1000):
    """
    Test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py

    **returns**:

    - return the permutation of the caustic in the central of mass coordinate system
    """
    caustic = CausticsBinary(q, s)
    x, y = caustic.get_caustics(n_points)
    z_centeral = jnp.array(jnp.array(x) + 1j * jnp.array(y))
    ## random change the position of the source
    key = jax.random.key(42)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    phi = jax.random.uniform(subkey1, z_centeral.shape, minval=-np.pi, maxval=np.pi)
    r = jax.random.uniform(subkey2, z_centeral.shape, minval=0.0, maxval=2 * rho)
    z_centeral = z_centeral + r * jnp.exp(1j * phi)
    return z_centeral
