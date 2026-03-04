from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .coordinates import annual_parallax_shift, Coordinates


class TrajectoryModel(NamedTuple):
    """
    Parallax trajectory model

    - times: times of the observations in Julian days
    - delta_s: s the projected offset of the Earth-to-Sun vector in AU. see (Gould 2004) for details.
    """

    times: np.ndarray
    delta_s: np.ndarray | None
    photo_center: bool = False

    def calculate_trajectory(self, params: np.ndarray) -> np.ndarray:
        """
        Calculate the microlensing trajectory for the given parameters.

        Args:
            params (np.ndarray): Array containing [t0, u0, te, rho, alpha_rad, s, q, pi_E_N, pi_E_E].
        Returns:
            np.ndarray: Complex trajectory at each time point.
        """
        if self.delta_s is not None:
            assert len(params) == 9, "Expected 9 parameters when delta_s is provided."
            t0, u0, te, rho, alpha_rad, s, q, pi_E_N, pi_E_E = params
        else:
            assert (
                len(params) == 7
            ), "Expected 7 parameters when delta_s is not provided."
            t0, u0, te, rho, alpha_rad, s, q = params
            pi_E_N = 0.0
            pi_E_E = 0.0

        times = self.times

        vector_tau = (times - t0) / te
        vector_beta = u0 * jnp.ones_like(times)

        pi_E_vector = jnp.array([pi_E_N, pi_E_E])

        if self.delta_s is not None:
            delta_tau_vector = jnp.dot(self.delta_s, pi_E_vector)
            delta_beta_vector = -jnp.cross(self.delta_s, pi_E_vector)

            vector_tau += delta_tau_vector
            vector_beta += delta_beta_vector

        trajectory = vector_tau * jnp.exp(1j * alpha_rad) + 1j * vector_beta * jnp.exp(
            1j * alpha_rad
        )

        if self.photo_center:
            # convert to photocenter coordinates for s>1 case
            masscent_to_magcent = q / (1 + q) * (s - 1 / s)
            cond = s > 1
            shift = jnp.where(cond, masscent_to_magcent, 0)
            trajectory -= shift

        return trajectory


def get_trajectory_model(
    times: np.ndarray,
    coords: Coordinates,
    t0_par: float = None,
    photo_center: bool = False,
) -> TrajectoryModel:
    """
    Create a trajectory model based on the provided times and coordinates.

    Args:
        times (np.ndarray): Array of observation times in Julian days.
        coords (Coordinates): Sky coordinates of the target.
        t0_par (float): Reference time for parallax shift in Julian days.

    Returns:
        TrajectoryModel: A trajectory model with calculated delta_s.
    """
    # Calculate the annual parallax shift projected onto the sky plane
    if t0_par is None:
        return TrajectoryModel(
            times=times,
            delta_s=None,
        )

    # Handle times in reduced JD format (times < 2450000)
    if times[0] < 2450000:
        # print('adding 2450000 to times for jpl ephemeris compatibility')
        times_jpl = times + 2450000
        t0_par_jpl = t0_par + 2450000
    else:
        times_jpl = times
        t0_par_jpl = t0_par

    delta_s_projected = annual_parallax_shift(
        times=times_jpl,
        time_ref=t0_par_jpl,  # t_0_par = t0
        coords=coords,
    )

    return TrajectoryModel(
        times=times, delta_s=delta_s_projected, photo_center=photo_center
    )
