from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import VBMicrolensing
from microlux import contour_integral, extended_light_curve, to_lowmass
from microlux.limb_darkening import LinearLimbDarkening
from test_util import get_caustic_permutation


matplotlib.use("Agg")
import matplotlib.pyplot as plt


rho_values = [1e-3]
q_values = [1e-1, 1e-2, 1e-3]
s_values = [0.6, 1.0, 1.4]
limb_a_values = [0.5]
DEFAULT_STRATEGY = (30, 30, 60, 120, 240, 480, 2000)
DEBUG_DIR = Path("picture/caustic_mag_debug")


def _save_local_light_curve_debug(
    z0: complex,
    rho: float,
    q: float,
    s: float,
    retol: float,
    case_tag: str,
):
    """
    Save a local 1D trajectory comparison (JAX vs VBMicrolensing) passing through z0.
    """
    alpha_rad = np.deg2rad(37.0)
    tau = np.linspace(-10.0 * rho, 10.0 * rho, 600)
    trajectory = z0 + tau * np.exp(1j * alpha_rad)

    jax_mag = np.asarray(
        extended_light_curve(
            trajectory,
            s,
            q,
            rho,
            tol=1e-2,
            retol=retol,
            default_strategy=DEFAULT_STRATEGY,
        )
    )

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.RelTol = retol
    vbm_mag = np.array(
        [vbm.BinaryMag2(s, q, z.real, z.imag, rho) for z in trajectory], dtype=float
    )
    rel = np.abs(jax_mag - vbm_mag) / np.maximum(np.abs(vbm_mag), 1e-15)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = DEBUG_DIR / f"{case_tag}_local_light_curve.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(tau, jax_mag, label="JAX", linewidth=1.4)
    axes[0].plot(tau, vbm_mag, "--", label="VBMicrolensing", linewidth=1.2)
    axes[0].set_ylabel("Magnification")
    axes[0].legend(loc="best")
    axes[0].set_title(f"rho={rho}, q={q}, s={s}, z0=({z0.real:.6e}, {z0.imag:.6e})")
    axes[1].plot(tau, rel, color="tab:red", linewidth=1.1)
    axes[1].set_xlabel("tau (local offset)")
    axes[1].set_ylabel("Relative error")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path, float(np.max(rel))


@pytest.mark.fast
@pytest.mark.parametrize("rho, q, s", product(rho_values, q_values, s_values))
def test_extend_sorce(rho, q, s, retol=1e-3):
    """
    Test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py
    """

    z_centeral = np.asarray(get_caustic_permutation(rho, q, s))

    ### change the coordinate system
    z_lowmass = to_lowmass(s, q, z_centeral)
    trajectory_n = z_centeral.shape[0]

    ### change the coordinate system
    VBBL = VBMicrolensing.VBMicrolensing()
    VBBL.RelTol = retol
    VBBL_mag = np.array(
        [VBBL.BinaryMag2(s, q, z.real, z.imag, rho) for z in z_centeral], dtype=float
    )

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
            default_strategy=DEFAULT_STRATEGY,
        )[0]
        Jax_mag.append(float(mag))

    Jaxmag = np.array(Jax_mag, dtype=float)

    rel_error = np.abs(Jaxmag - VBBL_mag) / VBBL_mag
    abs_error = np.abs(Jaxmag - VBBL_mag)
    print(
        "max relative error is {}, max absolute error is {}".format(
            np.max(rel_error), np.max(abs_error)
        )
    )

    mismatch_mask = ~np.isclose(Jaxmag, VBBL_mag, rtol=retol * 3, atol=1e-8)
    if np.any(mismatch_mask):
        mismatch_idx = np.where(mismatch_mask)[0]
        print(
            f"[caustic_debug] mismatch_count={mismatch_idx.size} "
            f"(rtol={retol * 3}, atol=1e-8)"
        )
        for idx in mismatch_idx[:20]:
            z = z_centeral[idx]
            print(
                f"[caustic_debug] idx={idx}, z=({z.real:.10e}, {z.imag:.10e}), "
                f"jax={Jaxmag[idx]:.10e}, vbm={VBBL_mag[idx]:.10e}, "
                f"rel={rel_error[idx]:.10e}, abs={abs_error[idx]:.10e}"
            )

        worst_idx = int(np.argmax(rel_error))
        worst_z = complex(z_centeral[worst_idx])
        case_tag = f"rho{rho:.0e}_q{q:.0e}_s{s:.1f}_idx{worst_idx}".replace("+", "")
        figure_path, local_max_rel = _save_local_light_curve_debug(
            worst_z, rho, q, s, retol, case_tag
        )
        print(
            f"[caustic_debug] worst_idx={worst_idx}, "
            f"worst_z=({worst_z.real:.10e}, {worst_z.imag:.10e}), "
            f"local_curve_max_rel={local_max_rel:.10e}"
        )
        print(f"[caustic_debug] saved_local_curve={figure_path}")

    assert np.allclose(Jaxmag, VBBL_mag, rtol=retol * 3.5)


@pytest.mark.fast
@pytest.mark.parametrize("limb_a", limb_a_values)
def test_limb_darkening(limb_a, rho=1e-2, q=0.2, s=0.9, retol=1e-3):
    """
    Test the limb darkening effect
    """

    z_centeral = get_caustic_permutation(rho, q, s, n_points=1000)

    trajectory_n = z_centeral.shape[0]

    ### change the coordinate system
    VBBL = VBMicrolensing.VBMicrolensing()
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
        default_strategy=DEFAULT_STRATEGY,
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
