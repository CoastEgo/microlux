from typing import NamedTuple

import numpy as np


class TrajectoryModel(NamedTuple):
    """
    Parallax trajectory model

    - times: times of the observations in Julian days
    - delta_s: s the projected offset of the Earth-to-Sun vector in AU. see (Gould 2004) for details.
    """

    times: np.ndarray
    delta_s: np.ndarray | None

    def calculate_trajectory(self, params: np.ndarray) -> np.ndarray:
        t0, u0, te, alpha_rad, pi_E_N, pi_E_E = params

        vector_tau = (self.times - t0) / te
        vector_beta = u0 * np.ones_like(self.times)

        pi_E_vector = np.array([pi_E_N, pi_E_E])

        if self.delta_s is not None:
            delta_tau_vector = np.dot(self.delta_s, pi_E_vector)
            delta_beta_vector = -np.cross(self.delta_s, pi_E_vector)

            vector_tau += delta_tau_vector
            vector_beta += delta_beta_vector

        trajectory = vector_tau * np.exp(1j * alpha_rad) + 1j * vector_beta * np.exp(
            1j * alpha_rad
        )
        return trajectory
