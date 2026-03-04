from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .coordinates import annual_parallax_shift, Coordinates, normalize_jd_for_ephemeris


class TrajectoryParameters(NamedTuple):
    """
    Microlensing trajectory parameters.

    **Notes**

    - The first 7 fields are base geometric parameters.
    - `pi_E_N` and `pi_E_E` are only used when annual parallax is enabled.
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


class TrajectoryModel(NamedTuple):
    """
    Trajectory model in center-of-mass/magnification coordinates, depends on the photo_center flag.

    **Attributes**

    - `times`: Observation epochs in JD.
    - `delta_s`: Projected annual-parallax displacement in `(North, East)` with shape `(N, 2)`.
      Set to `None` to disable parallax.
    - `photo_center`: If `True`, apply photocenter correction for `s > 1`.
    """

    times: jnp.ndarray
    delta_s: jnp.ndarray | None
    photo_center: bool = False

    def calculate_trajectory(
        self, params: TrajectoryParameters | np.ndarray | jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate complex source trajectory at each epoch.

        **Parameters**

        - `params`: One of:
          - `TrajectoryParameters`
          - Length-9 array: `[t0, u0, tE, rho, alpha_rad, s, q, pi_E_N, pi_E_E]`
          - Length-7 array: `[t0, u0, tE, rho, alpha_rad, s, q]` (parallax disabled)

        **Returns**

        - `trajectory`: Complex trajectory array with shape `(N,)`.
        """
        has_parallax = self.delta_s is not None
        parsed = _parse_trajectory_parameters(params, has_parallax=has_parallax)
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

        if self.photo_center:
            masscent_to_magcent = (
                parsed.q / (1.0 + parsed.q) * (parsed.s - 1.0 / parsed.s)
            )
            shift = jnp.where(parsed.s > 1.0, masscent_to_magcent, 0.0)
            trajectory = trajectory - shift

        return trajectory


def _parse_trajectory_parameters(
    params: TrajectoryParameters | np.ndarray | jnp.ndarray,
    has_parallax: bool,
) -> TrajectoryParameters:
    """
    Parse array-like inputs into `TrajectoryParameters`.

    **Parameters**

    - `params`: Input parameters as a `TrajectoryParameters` instance or 1D array.
    - `has_parallax`: Whether the model includes annual parallax.

    **Returns**

    - `parsed_params`: Parsed `TrajectoryParameters` object.
    """
    if isinstance(params, TrajectoryParameters):
        return params

    arr = jnp.asarray(params)
    if arr.ndim != 1:
        raise ValueError("params must be a 1D array or TrajectoryParameters.")

    expected = 9 if has_parallax else 7
    if arr.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} parameters (has_parallax={has_parallax}), got {arr.shape[0]}."
        )

    if has_parallax:
        return TrajectoryParameters(
            t0=arr[0],
            u0=arr[1],
            tE=arr[2],
            rho=arr[3],
            alpha_rad=arr[4],
            s=arr[5],
            q=arr[6],
            pi_E_N=arr[7],
            pi_E_E=arr[8],
        )

    return TrajectoryParameters(
        t0=arr[0],
        u0=arr[1],
        tE=arr[2],
        rho=arr[3],
        alpha_rad=arr[4],
        s=arr[5],
        q=arr[6],
    )


def _project_delta_ne_to_tau_beta(
    delta_ne: jnp.ndarray, projection_vec_ne: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Project North/East displacement onto `(tau, beta)` shifts.

    **Parameters**

    - `delta_ne`: Displacement in North/East with shape `(N, 2)`.
    - `projection_vec_ne`: Projection vector in North/East with shape `(2,)`.

    **Returns**

    - `delta_tau`: Shift along trajectory direction.
    - `delta_beta`: Shift perpendicular to trajectory direction.
    """
    delta_tau = jnp.dot(delta_ne, projection_vec_ne)
    delta_beta = -jnp.cross(delta_ne, projection_vec_ne)
    return delta_tau, delta_beta


def get_trajectory_model(
    times: np.ndarray,
    coords: Coordinates,
    t0_par: float | None = None,
    photo_center: bool = False,
) -> TrajectoryModel:
    """
    Build a `TrajectoryModel` with optional annual parallax precomputation.

    **Parameters**

    - `times`: Observation epochs in JD.
    - `coords`: Source sky coordinates used for annual parallax projection.
    - `t0_par`: Parallax reference epoch. If `None`, parallax is disabled.
    - `photo_center`: Whether to apply photocenter correction for `s > 1`.

    **Returns**

    - `trajectory_model`: Trajectory model instance.

    **Notes**

    - When `t0_par` is `None`, the model expects 7 trajectory parameters.
    - When `t0_par` is set, annual parallax is enabled and the model expects 9 trajectory parameters.
    """
    times_arr = jnp.asarray(times)

    if t0_par is None:
        delta_s_projected = None
    else:
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
    )
