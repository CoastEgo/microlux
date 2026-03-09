"""
Coordinate utilities used by microlux trajectory models.

!!! note
    Parts of this module are inspired by MulensModel's coordinate/parallax
    implementation:
    https://github.com/rpoleski/MulensModel
"""

from typing import NamedTuple

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.time import Time


def velocity_of_Earth(full_BJD):
    """
    Calculate Earth's barycentric velocity at a given epoch.

    **Parameters**

    - `full_BJD`: Barycentric Julian Date in full-JD form (e.g., `245xxxx` or `246xxxx`).

    **Returns**

    - `velocity`: Earth barycentric velocity vector in `AU/day`, shape `(3,)`.
    """
    # The 4 lines below, that calculate velocity for given epoch,
    # are based on astropy 1.3 code:
    # https://github.com/astropy/astropy/blob/master/astropy/
    # coordinates/solar_system.py
    time = Time(full_BJD, format="jd", scale="tdb")
    (jd1, jd2) = get_jd12(time, "tdb")
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2)
    # factor = 1731.45683  # This scales AU/day to km/s.
    velocity = np.asarray(earth_pv_bary[1])  # AU/day
    return velocity


def normalize_jd_for_ephemeris(
    times,
    time_ref,
    reduced_jd_cutoff: float = 2450000.0,
):
    """
    Normalize reduced Julian dates for ephemeris calls.

    **Parameters**

    - `times`: Observation epochs as a 1D array.
    - `time_ref`: Reference epoch.
    - `reduced_jd_cutoff`: Threshold used to detect reduced-JD input. Defaults to `2450000.0`.

    **Returns**

    - `times_jd`: Full-JD epochs for ephemeris lookup.
    - `time_ref_jd`: Full-JD reference epoch.

    If `times[0] < reduced_jd_cutoff`, both outputs are shifted by `reduced_jd_cutoff`.
    """
    times_arr = np.asarray(times, dtype=float)
    time_ref_val = float(time_ref)
    if times_arr[0] < reduced_jd_cutoff:
        return times_arr + reduced_jd_cutoff, time_ref_val + reduced_jd_cutoff
    return times_arr, time_ref_val


class Coordinates(NamedTuple):
    """
    A NamedTuple storing sky coordinates of the target source.

    Attributes:
        ra (str): Right ascension string in sexagesimal format (`HH:MM:SS`).
        dec (str): Declination string in sexagesimal format (`DD:MM:SS`).
    """

    ra: str
    dec: str

    def get_degrees(self):
        """
        Convert RA/Dec from sexagesimal strings to decimal degrees.

        **Returns**

        - `alpha`: Right ascension in degrees.
        - `delta`: Declination in degrees.
        """
        ra_sep = np.array(self.ra.split(":")).astype(float)
        alpha = (ra_sep[0] + ra_sep[1] / 60.0 + ra_sep[2] / 3600.0) * 15.0
        dec_sep = np.array(self.dec.split(":")).astype(float)
        if dec_sep[0] < 0:
            delta = dec_sep[0] - dec_sep[1] / 60.0 - dec_sep[2] / 3600.0
        else:
            delta = dec_sep[0] + dec_sep[1] / 60.0 + dec_sep[2] / 3600.0

        return alpha, delta

    def get_EN_vector(self):
        """
        Calculate East/North unit vectors on the sky plane.

        **Returns**

        - `north_projected`: Unit vector toward North on the tangent plane.
        - `east_projected`: Unit vector toward East on the tangent plane.
        """
        alpha_deg, delta_deg = self.get_degrees()
        alpha_rad = np.deg2rad(alpha_deg)
        delta_rad = np.deg2rad(delta_deg)

        target = np.array(
            [
                np.cos(delta_rad) * np.cos(alpha_rad),  # x
                np.cos(delta_rad) * np.sin(alpha_rad),  # y
                np.sin(delta_rad),
            ]
        )  # z
        # z vector
        z = np.array([0.0, 0.0, 1.0])

        east_projected = np.cross(z, target)
        east_projected /= np.linalg.norm(east_projected)

        north_projected = np.cross(target, east_projected)

        return north_projected, east_projected


def annual_parallax_shift(times, time_ref, coords: Coordinates):
    """
    Compute annual parallax displacement projected to North/East.

    **Parameters**

    - `times`: Observation epochs in JD.
    - `time_ref`: Reference epoch in JD (typically `t0_par`).
    - `coords`: Source sky coordinates.

    **Returns**

    - `delta_s_projected`: Array with shape `(N, 2)` where columns are `[North, East]` in `AU`.
    """
    times_arr = np.asarray(times, dtype=float)
    time_ref_val = float(time_ref)
    north_projected, east_projected = coords.get_EN_vector()

    earth_pos_ref = get_body_barycentric(
        body="earth", time=Time(time_ref_val, format="jd", scale="tdb")
    )
    earth_pos = get_body_barycentric(
        body="earth", time=Time(times_arr, format="jd", scale="tdb")
    )
    velocity = velocity_of_Earth(time_ref_val)

    delta_s = (earth_pos_ref.xyz.T - earth_pos.xyz.T).to(u.au).value
    delta_s += np.outer(times_arr - time_ref_val, velocity)

    delta_s_projected_n = np.dot(delta_s, north_projected)
    delta_s_projected_e = np.dot(delta_s, east_projected)
    delta_s_projected = np.array([delta_s_projected_n, delta_s_projected_e]).T
    return delta_s_projected
