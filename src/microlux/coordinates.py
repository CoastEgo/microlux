from typing import NamedTuple

import astropy.units as u
import erfa
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.time import Time


def velocity_of_Earth(full_BJD):
    """
    Calculate 3D velocity of Earth for given epoch.

    If you need velocity projected on the plane of the sky, then use
    :py:func:`~MulensModel.coordinates.Coordinates.v_Earth_projected`

    Parameters :
        full_BJD: *float*
            Barycentric Julian Data. Full means it should begin
            with 245... or 246...

    Returns :
        velocity: *np.ndarray* (*float*, size of (3,))
            3D velocity in km/s. The frame follows *Astropy* conventions.
    """
    # The 4 lines below, that calculate velocity for given epoch,
    # are based on astropy 1.3 code:
    # https://github.com/astropy/astropy/blob/master/astropy/
    # coordinates/solar_system.py
    time = Time(full_BJD, format="jd", scale="tdb")
    (jd1, jd2) = get_jd12(time, "tdb")
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2)
    # factor = 1731.45683  # This scales AU/day to km/s.
    velocity = jnp.asarray(earth_pv_bary[1])  # AU/day
    return velocity


class Coordinates(NamedTuple):
    """
    Class for coordinates of the source and the observer
    """

    ra: str
    dec: str

    def get_degrees(self):
        """
        Convert RA and Dec from sexagesimal (HH:MM:SS, DD:MM:SS) to decimal degrees.
        
        Returns:
            tuple: (alpha, delta) where:
                - alpha: Right Ascension in degrees (0 to 360)
                - delta: Declination in degrees (-90 to 90)
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
        Calculate the East-North unit vectors projected on the plane of the sky.
        
        Returns:
            tuple: (north_projected, east_projected) where:
                - north_projected: Unit vector pointing North in the plane of the sky
                - east_projected: Unit vector pointing East in the plane of the sky
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
    Calculate the annual parallax shift projected onto the sky plane.
    
    This function computes the Earth's position relative to the reference time
    and projects this displacement onto the plane of the sky in the North and East 
    directions.
    
    Parameters:
        times: array_like
            Times in Julian Date for which to calculate the parallax shift
        time_ref: float
            Reference time in Julian Date
        coords: Coordinates
            Sky coordinates (RA, Dec) of the target
            
    Returns:
        array_like:
            Projected parallax shift in AU as an array of shape (N, 2),
            where N is the length of times. Each row contains [North, East] components.
    """
    north_projected, east_projected = coords.get_EN_vector()

    earth_pos_ref = get_body_barycentric(
        body="earth", time=Time(time_ref, format="jd", scale="tdb")
    )
    earth_pos = get_body_barycentric(
        body="earth", time=Time(times, format="jd", scale="tdb")
    )
    velocity = velocity_of_Earth(time_ref)

    delta_s = (earth_pos_ref.xyz.T - earth_pos.xyz.T).to(u.au).value
    delta_s += np.outer(times - time_ref, velocity)

    delta_s_projected_n = np.dot(delta_s, north_projected)
    delta_s_projected_e = np.dot(delta_s, east_projected)
    delta_s_projected = np.array([delta_s_projected_n, delta_s_projected_e]).T

    return delta_s_projected
