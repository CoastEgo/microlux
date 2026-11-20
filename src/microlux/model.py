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
from .polynomial_solver import get_point_roots
from .solution import get_poly_coff
from .trajectory import TrajectoryModel, TrajectoryParameters
from .utils import (
    get_default_state,
)


jax.config.update("jax_enable_x64", True)


# @partial(jax.jit,static_argnames=['return_num'])
def point_light_curve(trajectory, s, q, rho, tol, return_num: bool = False):
    """
    Calculate the point source light curve.

    !!! note
        The trajectory parameter should be in the center-of-mass coordinate system
        (consistent with MulensModel). The function internally transforms to
        low-mass coordinates for the calculation.

    **Parameters**

    - `trajectory`: The trajectory of the lensing event in center-of-mass coordinates.
    - `s`: Binary lens separation, scalar or shape `(N,)` matching trajectory.
    - `q`: The mass ratio of the secondary lens to the primary lens.
    - `rho`: The source radius in units of the Einstein radius.
    - `tol`: The absolute tolerance for the point-source validity checks.
    - `return_num`: Whether to return the number of physical images. Defaults to False.

    **Returns**

    A tuple `(magnification, cond)`, or `(magnification, cond, image_count)` if `return_num=True`.

    - `magnification`: The magnification array.
    - `cond`: A boolean array. `True` means the point-source approximation passes the validity checks.
    - `image_count`: The number of physical images at each epoch, included only when `return_num=True`.
    """

    # Transform from center-of-mass to low-mass coordinate system
    trajectory_l = to_lowmass(s, q, trajectory)

    m1 = 1 / (1 + q)
    m2 = q / (1 + q)
    zeta_l = trajectory_l[:, None]
    # The image axis needs a singleton dimension for per-epoch separation.
    s = jnp.broadcast_to(jnp.asarray(s), trajectory.shape)[:, None]
    coff = get_poly_coff(zeta_l, s, m2)
    z_l = get_point_roots(coff)
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
    This function will dynamically choose full contour integration or point source approximation based on the point-source validity checks.

    !!! note
        The coordinate system is consistent with the MulensModel(Center of mass).

    !!! warning
        Currently, to deal with limb-darkening effect, we only use 10 annuli which are uniformly distributed in terms of the area,
        which is **not** in an adaptive scheme. So the tolerance of the limb darkening effect is not guaranteed.

    **Parameters**

    - `t_0`: The time of closest approach to the center of mass.
    - `u_0`: The impact parameter of the source trajectory.
    - `t_E`: The Einstein crossing time.
    - `rho`: The source radius normalized to the Einstein radius.
    - `q`: The mass ratio of the secondary lens to the primary lens.
    - `s`: The projected separation of the binary lens system normalized to the Einstein radius.
    - `alpha_deg`: The angle between the source trajectory and the binary axis in degrees.
    - `times`: The times at which to compute the model.
    - `tol`: The tolerance for the adaptive contour integration. Defaults to 1e-2.
    - `retol`: The relative tolerance for the adaptive contour integration. Defaults to 0.001.
    - `default_strategy`: The default strategy for the contour integration. Defaults to (30, 30, 60, 120, 240). more details can be found in the [`microlux.contour_integral`][].
    - `analytic`: Use implicit root derivatives on the final contour mesh to support forward- and reverse-mode differentiation. Defaults to True.
        If False, differentiate through the adaptive iterations in forward mode.
    - `return_info`: Whether to return additional information about the computation. Defaults to False. when set to True, the function will return a tuple containing (magnification, info), where info is a tuple containing additional information about the computation used for debugging.
    - `limb_darkening_coeff`: The limb darkening coefficient for the source star. Defaults to None for a uniformly bright source. Currently only supports linear limb darkening.


    **Returns**

    - `magnification`: The magnification of the source at the given times.
    - `info`: Additional information about the computation used for debugging if return_info is True.
        The structure is `(trajectory_l, rho, s, q, roots_state, error_state)`, with an epoch axis on the arrays.
        `trajectory_l` uses the low-mass lens coordinate system. Point-source epochs contain placeholder
        contour states; use the returned `magnification` array for the light curve.
        See [`microlux.utils.Iterative_State`][] and [`microlux.utils.Error_State`][] for state fields.
    """
    # Here the parameterization is consistent with Mulensmodel and VBBinaryLensing
    ### initialize parameters
    alpha_rad = alpha_deg * 2 * jnp.pi / 360
    traj_model = TrajectoryModel(times=times, delta_s=None)
    traj_params = TrajectoryParameters(
        t0=t_0,
        u0=u_0,
        tE=t_E,
        rho=rho,
        alpha_rad=alpha_rad,
        s=s,
        q=q,
    )
    trajectory, separation = traj_model.calculate_trajectory(traj_params)

    limb_darkening_instance = (
        LinearLimbDarkening(limb_darkening_coeff)
        if limb_darkening_coeff is not None
        else None
    )
    mag = extended_light_curve(
        trajectory,
        separation,
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
    trajectory,
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
    Compute the light curve of a binary lens system with finite source effects.

    !!! note
        The trajectory parameter should be in the center-of-mass coordinate system
        (consistent with MulensModel). The function internally transforms to
        low-mass coordinates for the calculation.

    **Parameters**

    - `trajectory`: The trajectory in the center-of-mass coordinate system.
    - `s`: Binary lens separation, scalar or shape `(N,)` matching trajectory.
    - `limb_darkening`: An `AbstractLimbDarkening` instance providing `profile` and `cumulative_profile`,
        or None for a uniformly bright source. Defaults to None.
    - `n_annuli`: The number of equal-area annuli for the limb darkening calculation. Defaults to 10.
        Increase this value to check convergence.
    - For the definition of the other parameters, please see [`microlux.binary_mag`][].

    **Returns**

    - `magnification`: The magnification of the source at the given trajectory points.
    - `info`: Additional information about the computation used for debugging if return_info is True.
        See [`microlux.binary_mag`][] for the layout and point-source placeholders. For epochs using
        contour integration with limb darkening, the states are from the outermost source radius.
    """

    s = jnp.broadcast_to(jnp.asarray(s), trajectory.shape)

    # Transform from center-of-mass to low-mass coordinate system
    trajectory_l = to_lowmass(s, q, trajectory)

    mag, cond = point_light_curve(trajectory, s, q, rho, tol)

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
        def mag_limbdarkening(geometry):
            trajectory_l, separation = geometry
            mag_contour_rho_fun = lambda rho: contour_integral(
                trajectory_l, tol, retol, rho, separation, q, default_strategy, analytic
            )
            mag_concentric, info = lax.map(
                mag_contour_rho_fun, rho_array
            )  # exclude the first element which is zero

            mag_annuli = jnp.diff(mag_concentric * rho_frac2, prepend=0)
            total_mag = jnp.sum(mag_annuli * surface_brightness)
            # return the info of the last annulus
            info_last = jax.tree.map(lambda x: x[-1], info)
            return total_mag, info_last

        mag_contour = mag_limbdarkening
    else:
        mag_contour = lambda geometry: contour_integral(
            geometry[0], tol, retol, rho, geometry[1], q, default_strategy, analytic
        )

    if return_info:
        default_roots_state, default_error_state = get_default_state(
            np.sum(default_strategy)
        )

        def evaluate_epoch(epoch):
            use_point_source, point_mag, source_position, separation = epoch

            def point_source_result(geometry):
                position, separation = geometry
                info = (
                    position,
                    rho,
                    separation,
                    q,
                    default_roots_state,
                    default_error_state,
                )
                return point_mag, info

            return lax.cond(
                use_point_source,
                point_source_result,
                jax.jit(mag_contour),
                (source_position, separation),
            )

        result = lax.map(
            evaluate_epoch,
            (cond, mag, trajectory_l, s),
        )
        return result

    else:
        mag_contour_return_mag = lambda x: mag_contour(x)[0]
        mag_final = lax.map(
            lambda x: lax.cond(
                x[0], lambda _: x[1], jax.jit(mag_contour_return_mag), (x[2], x[3])
            ),
            [cond, mag, trajectory_l, s],
        )

        return mag_final
