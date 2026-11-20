from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from .coordinates import annual_parallax_shift, Coordinates, normalize_jd_for_ephemeris


class TrajectoryParameters(NamedTuple):
    """
    A NamedTuple containing microlensing trajectory parameters.

    Orbital rates follow VBMicrolensing and are defined at the center-of-mass
    t0. They are ignored unless orbital motion is enabled.

    Attributes:
        t0 (jnp.ndarray): Time of closest approach, in days.
        u0 (jnp.ndarray): Impact parameter at `t0`.
        tE (jnp.ndarray): Einstein timescale in days.
        rho (jnp.ndarray): Source radius in units of Einstein radius.
        alpha_rad (jnp.ndarray): Trajectory angle in radians.
        s (jnp.ndarray): Projected lens separation in Einstein-radius units.
        q (jnp.ndarray): Lens mass ratio.
        pi_E_N (jnp.ndarray): North component of microlensing parallax vector.
            Only used when parallax is enabled.
        pi_E_E (jnp.ndarray): East component of microlensing parallax vector.
            Only used when parallax is enabled.
        gamma1 (jnp.ndarray): Fractional projected-separation rate
            `(ds/dt)/s`, in day^-1.
        gamma2 (jnp.ndarray): Angular rate in rad day^-1, with the sign
            following the VBM trajectory convention.
        gamma3 (jnp.ndarray): Fractional line-of-sight separation rate
            `(ds_z/dt)/s`, in day^-1.
    """

    t0: jnp.ndarray
    u0: jnp.ndarray
    tE: jnp.ndarray
    rho: jnp.ndarray
    alpha_rad: jnp.ndarray
    s: jnp.ndarray
    q: jnp.ndarray
    pi_E_N: jnp.ndarray = jnp.asarray(0.0)
    pi_E_E: jnp.ndarray = jnp.asarray(0.0)
    gamma1: jnp.ndarray = jnp.asarray(0.0)
    gamma2: jnp.ndarray = jnp.asarray(0.0)
    gamma3: jnp.ndarray = jnp.asarray(0.0)


class TrajectoryModel(NamedTuple):
    """
    A trajectory model with independent parallax and circular-orbit effects.

    Attributes:
        times (jnp.ndarray): Observation times in Julian Date.
        delta_s (jnp.ndarray | None): Observer displacement in AU projected on
            the sky plane in (North, East), shape `(N, 2)`. If `None`, parallax
            correction is disabled.
        photo_center (bool): Whether input parameters use the magnification
            center for `s > 1`. Output trajectories always use the center of mass.
        orbital_motion (str | None): None or "circular". Independent of delta_s.
    """

    times: jnp.ndarray
    delta_s: jnp.ndarray | None
    photo_center: bool = False
    orbital_motion: str | None = None

    def calculate_trajectory(
        self, params: TrajectoryParameters | np.ndarray | jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (source trajectory, lens separation), each with shape `(N,)`.

        The trajectory uses center-of-mass coordinates in the instantaneous
        binary-axis frame. Without orbital motion, separation is constant.

        Parameters may be a TrajectoryParameters instance or a 1D array:
        [t0, u0, tE, rho, alpha_rad, s, q], followed by [pi_E_N, pi_E_E]
        when parallax is enabled. Circular motion appends [gamma1, gamma2,
        gamma3] (10/12 entries without/with parallax).

        The returned trajectory is complex and separation is real; pass both
        to extended_light_curve. To plot only the source, unpack trajectory, _.

        Reference frame:
            With photo_center=False, the orbital reference epoch is params.t0.
            With photo_center=True and s > 1, input parameters refer to the
            magnification center and are converted to the center of mass:

                shift = q / (1 + q) * (s - 1 / s)
                t0_com = t0 + tE * shift * cos(alpha_rad)
                u0_com = u0 + shift * sin(alpha_rad)

            The shift uses the reference separation s and stays fixed along the
            trajectory; it is not recomputed from s(t). For s <= 1 it is zero.
            The reference s, alpha_rad, and orbital rates refer to t0_com.
            The parallax reference epoch t0_par is independent of this epoch.

        Circular orbit:
            Away from the numerical boundary handling below, the circular
            constraint fixes the line-of-sight separation s_z/s = -gamma1/gamma3
            at the reference epoch. The projected separation can therefore vary
            even though the three-dimensional orbit is circular.

            Following VBMicrolensing, hypot(gamma1, gamma3) <= 1e-8 selects a
            face-on orbit with constant separation and angular rate gamma2.
            All-zero rates recover the static/parallax model. Otherwise
            abs(gamma3) is floored at 1e-8 day^-1, preserving its sign and
            taking zero as positive. Opposite signs give the same projected
            orbit. This numerical convention does not uniquely continue the
            physical orbit through the degenerate parameters.

        Fitting:
            Start a full orbital fit near static with (gamma1, gamma2, gamma3)
            = (0, 0, 1e-6), rather than all zeros. RTModel uses a gamma3 lower
            bound of 1e-7 day^-1 in its fitter; configure such a bound in your
            optimizer or sampler, since this forward model does not impose it.
            Derivatives in the floored region or across the branch boundary
            follow the numerical guard; avoid those regions for inference and
            Fisher calculations.
        """
        if self.orbital_motion not in (None, "circular"):
            raise ValueError('orbital_motion must be None or "circular".')
        has_parallax = self.delta_s is not None
        parsed = _parse_trajectory_parameters(
            params,
            has_parallax=has_parallax,
            has_orbital_motion=self.orbital_motion is not None,
        )
        if self.delta_s is None:
            delta_s = jnp.zeros((self.times.shape[0], 2), dtype=self.times.dtype)
        else:
            delta_s = self.delta_s

        vector_tau = (self.times - parsed.t0) / parsed.tE
        vector_beta = parsed.u0 * jnp.ones_like(self.times)

        if has_parallax:
            pi_e_vector = jnp.array(
                [parsed.pi_E_N, parsed.pi_E_E], dtype=self.times.dtype
            )
            delta_tau, delta_beta = _project_delta_ne_to_tau_beta(delta_s, pi_e_vector)
            vector_tau = vector_tau + delta_tau
            vector_beta = vector_beta + delta_beta

        trajectory = vector_tau * jnp.exp(
            1j * parsed.alpha_rad
        ) + 1j * vector_beta * jnp.exp(1j * parsed.alpha_rad)

        t0_orb = parsed.t0
        if self.photo_center:
            masscent_to_magcent = (
                parsed.q / (1.0 + parsed.q) * (parsed.s - 1.0 / parsed.s)
            )
            shift = jnp.where(parsed.s > 1.0, masscent_to_magcent, 0.0)
            trajectory = trajectory - shift
            t0_orb = parsed.t0 + parsed.tE * shift * jnp.cos(parsed.alpha_rad)

        separation = jnp.broadcast_to(jnp.asarray(parsed.s), self.times.shape)
        if self.orbital_motion == "circular":
            dt = self.times - t0_orb
            g1, g2, g3 = jnp.asarray([parsed.gamma1, parsed.gamma2, parsed.gamma3])

            def face_on(_):
                return trajectory * jnp.exp(1j * g2 * dt), separation

            def inclined(_):
                # VBM evaluates the norms before flooring the line-of-sight rate.
                rate_norm = jnp.sqrt(g1**2 + g2**2 + g3**2)
                projected_rate_norm = jnp.hypot(g1, g3)
                g3_safe = jnp.where(
                    jnp.abs(g3) > 1e-8, g3, jnp.where(g3 < 0, -1e-8, 1e-8)
                )
                omega = g3_safe * rate_norm / projected_rate_norm
                cos_inclination = g2 * g3_safe / (projected_rate_norm * rate_norm)
                phase0 = jnp.arctan2(-g1 * rate_norm, g3_safe * projected_rate_norm)
                phase = phase0 + omega * dt

                # Project the circle onto the sky, then divide by its reference
                # position to recover relative separation and lens-axis rotation.
                position0 = jnp.cos(phase0) + 1j * cos_inclination * jnp.sin(phase0)
                position = jnp.cos(phase) + 1j * cos_inclination * jnp.sin(phase)
                relative_position = position / position0
                radius = jnp.abs(relative_position)
                # Positive phase follows the existing VBM alpha convention.
                return trajectory * relative_position / radius, separation * radius

            trajectory, separation = lax.cond(
                jnp.hypot(g1, g3) <= 1e-8, face_on, inclined, operand=None
            )

        return trajectory, separation


def _parse_trajectory_parameters(
    params: TrajectoryParameters | np.ndarray | jnp.ndarray,
    has_parallax: bool,
    has_orbital_motion: bool = False,
) -> TrajectoryParameters:
    """
    Parse array-like inputs into `TrajectoryParameters`.

    Args:
        params: Input parameters as a `TrajectoryParameters` instance or 1D array.
        has_parallax: Whether the model includes parallax.
        has_orbital_motion: Whether arrays include the three orbital rates.

    Returns:
        parsed_params: Parsed `TrajectoryParameters` object.
    """
    if isinstance(params, TrajectoryParameters):
        return params

    arr = jnp.asarray(params)
    if arr.ndim != 1:
        raise ValueError("params must be a 1D array or TrajectoryParameters.")

    base_size = 9 if has_parallax else 7
    expected = base_size + (3 if has_orbital_motion else 0)
    if arr.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} parameters (has_parallax={has_parallax}), got {arr.shape[0]}."
        )

    parsed = TrajectoryParameters(*arr[:base_size])
    if has_orbital_motion:
        parsed = parsed._replace(
            gamma1=arr[base_size],
            gamma2=arr[base_size + 1],
            gamma3=arr[base_size + 2],
        )
    return parsed


def _project_delta_ne_to_tau_beta(
    delta_ne: jnp.ndarray, projection_vec_ne: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Project North/East displacement onto `(tau, beta)` shifts.

    Args:
        delta_ne: Displacement in North/East with shape `(N, 2)`.
        projection_vec_ne: Projection vector in North/East with shape `(2,)`.

    Returns:
        delta_tau: Shift along trajectory direction.
        delta_beta: Shift perpendicular to trajectory direction.
    """
    delta_tau = jnp.dot(delta_ne, projection_vec_ne)
    delta_beta = (
        delta_ne[..., 1] * projection_vec_ne[0]
        - delta_ne[..., 0] * projection_vec_ne[1]
    )
    return delta_tau, delta_beta


def get_trajectory_model(
    times: np.ndarray,
    coords: Coordinates | None = None,
    t0_par: float | None = None,
    photo_center: bool = False,
    *,
    orbital_motion: str | None = None,
) -> TrajectoryModel:
    """
    Build a trajectory model with optional annual parallax and circular orbit.

    Args:
        times: Observation epochs in full JD or reduced JD (JD - 2450000).
        coords: Source sky coordinates; required only for annual parallax.
        t0_par: Parallax reference epoch. If `None`, parallax is disabled.
        photo_center: Whether input parameters use the magnification center
            for `s > 1`; output is always in the center-of-mass frame.
        orbital_motion: None or "circular", independently of parallax.

    Returns:
        trajectory_model: Trajectory model instance.

    Notes:
        - Use the same full or reduced JD convention for times, t0, and t0_par.
        - Annual ephemerides are precomputed here, outside JAX differentiation.
          The returned model stores the observer displacement in delta_s, in AU
          projected onto North/East. This factory computes annual parallax only.
        - Named `TrajectoryParameters` work with every combination of effects.
          See `TrajectoryModel.calculate_trajectory` for array layouts, reference
          frames, and orbital boundary handling.
    """
    if orbital_motion not in (None, "circular"):
        raise ValueError('orbital_motion must be None or "circular".')
    times_arr = jnp.asarray(times)

    if t0_par is None:
        delta_s_projected = None
    else:
        if coords is None:
            raise ValueError("coords is required when t0_par is provided.")
        times_jpl, t0_par_jpl = normalize_jd_for_ephemeris(
            np.asarray(times, dtype=float), float(t0_par)
        )
        delta_s_projected = jnp.asarray(
            annual_parallax_shift(
                times=times_jpl,
                time_ref=t0_par_jpl,
                coords=coords,
            )
        )

    return TrajectoryModel(
        times=times_arr,
        delta_s=delta_s_projected,
        photo_center=photo_center,
        orbital_motion=orbital_motion,
    )
