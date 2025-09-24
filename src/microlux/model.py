from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .basic_function import (
    Quadrupole_test,
    to_lowmass,
    verify,
)
from .countour import contour_integral
from .limb_darkening import AbstractLimbDarkening, LinearLimbDarkening
from .solution import (
    get_poly_coff,
    get_roots,
)
from .utils import (
    get_default_state,
)


jax.config.update("jax_enable_x64", True)


# @partial(jax.jit,static_argnames=['return_num'])
def point_light_curve(trajectory_l, s, q, rho, tol, return_num: bool = False):
    """
    Calculate the point source light curve.

    **Parameters**

    - `trajectory_l`: The trajectory of the lensing event.
    - `s`: The projected separation between the lens and the source.
    - `q` : The mass ratio between the lens and the source.
    - `rho`: The source radius in units of the Einstein radius.
    - `tol`: The absolute tolerance for the quadrupole test.
    - `return_num`: Whether to return the number of real roots. Defaults to False.

    **Returns**

    - `result`: A tuple containing:
        - The magnification array.
        - A boolean array indicating the validity of the calculation. If the quadrupole test is passed, the corresponding element in the boolean array is `True`.
        - If `return_num` is `True`, the tuple will also contain the number of real roots.
    - `cond`: A boolean array indicating whether the quadrupole test is passed. `True` means the quadrupole test is passed.
    - `mask`: An integer array indicating the number of real roots.
    """

    m1 = 1 / (1 + q)
    m2 = q / (1 + q)
    zeta_l = trajectory_l[:, None]
    coff = get_poly_coff(zeta_l, s, m2)
    z_l = get_roots(trajectory_l.shape[0], coff)
    error = verify(zeta_l, z_l, s, m1, m2)

    iterator = jnp.arange(
        zeta_l.shape[0]
    )  # criterion to select the roots, same as the VBBL
    dlmin = 1.0e-4
    sort_idx = jnp.argsort(error, axis=1)
    third_error = error[iterator, sort_idx[:, 2]]
    forth_error = error[iterator, sort_idx[:, 3]]
    three_roots_cond = (forth_error * dlmin) > (
        third_error + 1e-12
    )  # three roots criterion
    # bad_roots_cond = (~three_roots_cond) & ((forth_error*dlmax) > (third_error+1e-12)) # bad roots criterion
    mask = jnp.ones_like(z_l, dtype=bool)
    full_value = jnp.where(three_roots_cond, False, True)
    mask = mask.at[iterator[:, None], sort_idx[:, 3:]].set(full_value[:, None])

    cond, mag = Quadrupole_test(rho, s, q, zeta_l, z_l, mask, tol)
    if return_num:
        return mag, cond, mask.sum(axis=1)
    else:
        return mag, cond


@partial(
    jax.jit,
    static_argnames=[
        "default_strategy",
        "analytic",
        "return_info",
        "limb_darkening_coeff",
    ],
)
def binary_mag(
    t_0,
    u_0,
    t_E,
    rho,
    q,
    s,
    alpha_deg,
    times: jnp.ndarray,
    tol=1e-2,
    retol=0.001,
    default_strategy: Tuple[int] = (30, 30, 60, 120, 240),
    analytic: bool = True,
    return_info: bool = False,
    limb_darkening_coeff: float | None = None,
):
    # write the docstring with same format as the point_light_curve
    """
    Compute the light curve of a binary lens system with finite source effects.
    This function will dynamically choose full contour integration or point source approximation based on the quadrupole test.

    !!! note
        The coordinate system is consistent with the MulensModel(Center of mass).

    !!! warning
        Currently, to deal with limb-darkening effect, we only use 10 annuli which are uniformly distributed in terms of the area,
        which is **not** in an adaptive scheme. So the tolerance of the limb darkening effect is not guaranteed.

    **Parameters**

    - `t_0`: The time of the peak of the microlensing event.
    - `u_0`: The impact parameter of the source trajectory.
    - `t_E`: The Einstein crossing time.
    - `rho`: The source radius normalized to the Einstein radius.
    - `q`: The planet to host mass ratio of the binary lens system.
    - `s`: The projected separation of the binary lens system normalized to the Einstein radius.
    - `alpha_deg`: The angle between the source trajectory and the binary axis in degrees.
    - `times`: The times at which to compute the model.
    - `tol`: The tolerance for the adaptive contour integration. Defaults to 1e-2.
    - `retol`: The relative tolerance for the adaptive contour integration. Defaults to 0.001.
    - `default_strategy`: The default strategy for the contour integration. Defaults to (30, 30, 60, 120, 240). more details can be found in the [`microlux.contour_integral`][].
    - `analytic`: Whether to use the analytic chain rule to simplify the computation graph. Set this to True will accelerate the computation of the gradient and will support the reverse mode differentiation containing the while loop. But set this to True will slow down if only calculate the model without differentiation. Defaults to True.
    - `return_info`: Whether to return additional information about the computation. Defaults to False.
    - `limb_darkening_coeff`: The limb darkening coefficient for the source star. Defaults to None. currently only support linear limb darkening.


    **Returns**

    - `magnification`: The magnification of the source at the given times.
    - `info`: Additional information about the computation used for debugging if return_info is True.
    """
    # Here the parameterization is consistent with Mulensmodel and VBBinaryLensing
    ### initialize parameters
    alpha_rad = alpha_deg * 2 * jnp.pi / 360
    tau = (times - t_0) / t_E
    ## switch the coordinate system to the lowmass
    trajectory = tau * jnp.exp(1j * alpha_rad) + 1j * u_0 * jnp.exp(1j * alpha_rad)
    trajectory_l = to_lowmass(s, q, trajectory)

    limb_darkening_instance = (
        LinearLimbDarkening(limb_darkening_coeff)
        if limb_darkening_coeff is not None
        else None
    )
    mag = extended_light_curve(
        trajectory_l,
        s,
        q,
        rho,
        tol,
        retol,
        default_strategy=default_strategy,
        analytic=analytic,
        return_info=return_info,
        limb_darkening=limb_darkening_instance,
    )
    return mag


@partial(
    jax.jit,
    static_argnames=[
        "default_strategy",
        "analytic",
        "return_info",
        "limb_darkening",
        "n_annuli",
    ],
)
def extended_light_curve(
    trajectory_l,
    s,
    q,
    rho,
    tol=1e-2,
    retol=0.001,
    default_strategy: Tuple[int] = (30, 30, 60, 120, 240),
    analytic: bool = True,
    return_info: bool = False,
    limb_darkening: AbstractLimbDarkening | None = None,
    n_annuli: int = 10,
):
    """
    compute the light curve of a binary lens system with finite source effects.

    **Parameters**

    - `trajectory_l`: The trajectory in the low mass coordinate system.
    - `n_annuli`: The number of annuli for the limb darkening calculation.
    - for the definition of the other parameters, please see [`microlux.binary_mag`][].

    """

    mag, cond = point_light_curve(trajectory_l, s, q, rho, tol)

    if limb_darkening is not None:
        # uniform in radius
        # rho_frac = jnp.arange(1, n_annuli + 1) / n_annuli  # 1/n, 2/n, ..., 1
        # uniform in area
        rho_frac = jnp.sqrt(
            jnp.arange(1, n_annuli + 1) / n_annuli
        )  # sqrt(1/n), sqrt(2/n), ..., 1

        rho_frac2 = rho_frac**2
        cumulative_profile = limb_darkening.cumulative_profile(rho_frac)

        surface_brightness = jnp.diff(cumulative_profile, prepend=0) / jnp.diff(
            rho_frac2, prepend=0
        )

        rho_array = rho * rho_frac

        @jax.jit
        def mag_limbdarkening(trajectory_l):
            mag_contour_rho_fun = lambda rho: contour_integral(
                trajectory_l, tol, retol, rho, s, q, default_strategy, analytic
            )
            mag_concentric, mag_con_res = lax.map(
                mag_contour_rho_fun, rho_array
            )  # exclude the first element which is zero

            mag_annuli = jnp.diff(mag_concentric * rho_frac2, prepend=0)
            total_mag = jnp.sum(mag_annuli * surface_brightness)
            return total_mag, mag_con_res[-1]

        mag_contour = mag_limbdarkening
    else:
        mag_contour = lambda trajectory_l: contour_integral(
            trajectory_l, tol, retol, rho, s, q, default_strategy, analytic
        )

    if return_info:
        default_roots_state, default_error_state = get_default_state(
            np.sum(default_strategy)
        )

        result = lax.map(
            lambda x: lax.cond(
                x[0],
                lambda _: (
                    x[1],
                    (
                        x[2],
                        rho,
                        s,
                        q,
                        default_roots_state,
                        default_error_state,
                    ),
                ),
                jax.jit(mag_contour),
                x[2],
            ),
            [cond, mag, trajectory_l],
        )
        return result

    else:
        mag_contour_return_mag = lambda x: mag_contour(x)[0]
        mag_final = lax.map(
            lambda x: lax.cond(
                x[0], lambda _: x[1], jax.jit(mag_contour_return_mag), x[2]
            ),
            [cond, mag, trajectory_l],
        )

        return mag_final
