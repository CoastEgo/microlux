from itertools import product

import numpy as np
import pytest
import VBBinaryLensing
from microlux import contour_integral, extended_light_curve, to_lowmass
from microlux.limb_darkening import LinearLimbDarkening
from test_util import get_caustic_permutation


rho_values = [1e-3]
q_values = [1e-1, 1e-2, 1e-3]
s_values = [0.6, 1.0, 1.4]
limb_a_values = [0.5]


@pytest.mark.fast
@pytest.mark.parametrize("rho, q, s", product(rho_values, q_values, s_values))
def test_extend_sorce(rho, q, s, retol=1e-3):
    """
    Test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py
    """

    z_centeral = get_caustic_permutation(rho, q, s)

    ### change the coordinate system
    z_lowmass = to_lowmass(s, q, z_centeral)
    trajectory_n = z_centeral.shape[0]

    ### change the coordinate system
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol = retol
    VBBL_mag = []
    for i in range(trajectory_n):
        VBBL_mag.append(
            VBBL.BinaryMag2(s, q, z_centeral.real[i], z_centeral.imag[i], rho)
        )
    VBBL_mag = np.array(VBBL_mag)

    Jax_mag = []

    ## real time
    for i in range(trajectory_n):
        mag = contour_integral(
            z_lowmass[i],
            retol,
            retol,
            rho,
            s,
            q,
            default_strategy=(30, 30, 60, 120, 240, 480, 2000),
        )[0]
        Jax_mag.append(mag)

    Jaxmag = np.array(Jax_mag)

    rel_error = np.abs(Jaxmag - VBBL_mag) / VBBL_mag
    abs_error = np.abs(Jaxmag - VBBL_mag)
    print(
        "max relative error is {}, max absolute error is {}".format(
            np.max(rel_error), np.max(abs_error)
        )
    )
    assert np.allclose(Jaxmag, VBBL_mag, rtol=retol * 3)


@pytest.mark.fast
@pytest.mark.parametrize("limb_a", limb_a_values)
def test_limb_darkening(limb_a, rho=1e-2, q=0.2, s=0.9, retol=1e-3):
    """
    Test the limb darkening effect
    """

    z_centeral = get_caustic_permutation(rho, q, s, n_points=1000)

    trajectory_n = z_centeral.shape[0]

    ### change the coordinate system
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.a1 = limb_a
    VBBL.RelTol = retol
    VBBL_mag = []
    for i in range(trajectory_n):
        VBBL_mag.append(
            VBBL.BinaryMag2(s, q, z_centeral.real[i], z_centeral.imag[i], rho)
        )
    VBBL_mag = np.array(VBBL_mag)

    limb_darkening_instance = LinearLimbDarkening(limb_a)
    ## real time
    Jaxmag = extended_light_curve(
        z_centeral,  # Pass center-of-mass coordinates, extended_light_curve will handle to_lowmass transform
        s,
        q,
        rho,
        tol=1e-2,
        retol=1e-3,
        default_strategy=(30, 30, 60, 120, 240, 480, 2000),
        limb_darkening=limb_darkening_instance,
        n_annuli=20,
    )
    rel_error = np.abs(Jaxmag - VBBL_mag) / VBBL_mag
    abs_error = np.abs(Jaxmag - VBBL_mag)
    print(
        "max relative error is {}, max absolute error is {}".format(
            np.max(rel_error), np.max(abs_error)
        )
    )
    assert np.allclose(
        Jaxmag, VBBL_mag, rtol=0.05
    )  # since the limb darkening relization currently is not adaptive, the error is larger than the tolerance, this will be fixed in the future.


if __name__ == "__main__":
    test_extend_sorce(1e-2, 0.2, 0.9)
    test_limb_darkening(rho=1e-3, q=0.2, s=0.9, limb_a=0.5)
