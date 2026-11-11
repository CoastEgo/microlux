"""Circular-orbit geometry and magnification against VBMicrolensing.

Run this file directly to save model and forward-gradient comparison plots.
Numerical gradients are diagnostic only; no agreement tolerance is imposed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import VBMicrolensing
from microlux import extended_light_curve, point_light_curve
from microlux.coordinates import Coordinates
from microlux.trajectory import get_trajectory_model, TrajectoryParameters


PARAMS = TrajectoryParameters(
    t0=10000.0,
    u0=0.1,
    tE=100.0,
    rho=0.01,
    alpha_rad=1.2,
    s=1.3,
    q=0.2,
    pi_E_N=0.1,
    pi_E_E=-0.1,
    gamma1=0.001,
    gamma2=0.002,
    gamma3=0.003,
)
COORDS = Coordinates(ra="17:59:02.3", dec="-29:04:15.2")
T0_PAR = 9983.0  # Deliberately different from both input and COM t0.
PICTURE_DIR = Path(__file__).parent / "picture" / "orbital"


def _vbm_orbit(params, times, parallax, photo_center):
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.SetObjectCoordinates("17:59:02.3 -29:04:15.2")
    vbm.t_in_HJD = True
    vbm.parallaxsystem = 1
    vbm.t0_par_fixed = 1
    vbm.t0_par = T0_PAR
    vbm.Tol = 1e-3
    vbm.RelTol = 1e-4
    p = params
    shift = p.q / (1 + p.q) * (p.s - 1 / p.s) if photo_center and p.s > 1 else 0.0
    t0 = p.t0 + p.tE * shift * np.cos(p.alpha_rad)
    u0 = p.u0 + shift * np.sin(p.alpha_rad)
    values = [
        np.log(p.s),
        np.log(p.q),
        u0,
        p.alpha_rad - np.pi,
        np.log(p.rho),
        np.log(p.tE),
        t0,
        p.pi_E_N if parallax else 0.0,
        p.pi_E_E if parallax else 0.0,
        p.gamma1,
        p.gamma2,
        p.gamma3,
    ]
    return np.asarray(vbm.BinaryLightCurveOrbital(values, np.asarray(times)))


@pytest.mark.fast
@pytest.mark.parametrize(
    "parallax, photo_center, rates",
    [
        (False, False, None),
        (True, True, None),
        (False, False, (0.0, 0.0, 0.0)),
        (False, False, (3e-9, 0.002, 4e-9)),
        (False, False, (0.001, 0.002, 0.0)),
        (False, False, (2e-8, 0.002, 1e-9)),
    ],
)
def test_orbital_accuracy(parallax, photo_center, rates, plot=False):
    p = PARAMS._replace(gamma2=-0.002 if parallax else 0.002)
    if rates is not None:
        p = p._replace(**dict(zip(("gamma1", "gamma2", "gamma3"), rates)))
    times = jnp.linspace(p.t0 - p.tE, p.t0 + p.tE, 161)
    model = get_trajectory_model(
        times,
        COORDS,
        t0_par=T0_PAR if parallax else None,
        photo_center=photo_center,
        orbital_motion="circular",
    )
    trajectory, separation = jax.jit(model.calculate_trajectory)(p)
    mag_vbm, x_vbm, y_vbm, s_vbm = _vbm_orbit(p, times, parallax, photo_center)
    np.testing.assert_allclose(separation, s_vbm, rtol=1e-12, atol=1e-12)
    # Annual ephemerides differ between Astropy and VBM. Without parallax,
    # the two analytic circular-orbit trajectories agree to roundoff.
    np.testing.assert_allclose(
        trajectory,
        x_vbm + 1j * y_vbm,
        rtol=0,
        atol=1e-4 if parallax else 1e-12,
    )
    # Current VBM preserves gamma3's sign; the projected mirror orbit is identical.
    mirrored_trajectory, mirrored_separation = model.calculate_trajectory(
        p._replace(gamma3=-p.gamma3)
    )
    np.testing.assert_allclose(mirrored_trajectory, trajectory, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mirrored_separation, separation, rtol=1e-12, atol=1e-12)
    mag = np.asarray(extended_light_curve(trajectory, separation, p.q, p.rho))
    rel_err = np.abs(mag - mag_vbm) / mag_vbm
    print(
        f"orbital parallax={parallax}: max={rel_err.max():.4e}, mean={rel_err.mean():.4e}"
    )
    assert rel_err.max() < 0.01
    assert rel_err.mean() < 0.001
    # Ensure the comparison exercises finite-source contour integration.
    _, point_ok = point_light_curve(trajectory, separation, p.q, p.rho, 1e-2)
    assert np.any(~np.asarray(point_ok))

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].plot(times, mag, label="microlux")
        axes[0, 0].plot(times, mag_vbm, "--", label="VBMicrolensing")
        axes[0, 0].set_ylabel("Magnification")
        axes[0, 0].legend()
        axes[1, 0].semilogy(times, rel_err)
        axes[1, 0].set(xlabel="HJD - 2450000", ylabel="Relative difference")
        axes[0, 1].plot(trajectory.real, trajectory.imag)
        axes[0, 1].plot(x_vbm, y_vbm, "--")
        axes[0, 1].set(xlabel="x", ylabel="y", aspect="equal")
        axes[1, 1].plot(times, separation)
        axes[1, 1].plot(times, s_vbm, "--")
        axes[1, 1].set(xlabel="HJD - 2450000", ylabel="s(t)")
        fig.tight_layout()
        PICTURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PICTURE_DIR / f"orbit_parallax_{parallax}.png", dpi=150)
        plt.close(fig)


@pytest.mark.fast
def test_orbital_forward_gradient(plot=False):
    times = jnp.linspace(PARAMS.t0 - 100, PARAMS.t0 + 100, 81)
    model = get_trajectory_model(
        times, COORDS, T0_PAR, photo_center=True, orbital_motion="circular"
    )
    names = ("pi_E_N", "pi_E_E", "gamma1", "gamma2", "gamma3")
    values = jnp.array([getattr(PARAMS, name) for name in names])

    def magnification(values):
        p = PARAMS._replace(**dict(zip(names, values)))
        z, s = model.calculate_trajectory(p)
        return extended_light_curve(z, s, p.q, p.rho)

    jacobian = np.asarray(jax.jit(jax.jacfwd(magnification))(values))
    assert jacobian.shape == (len(times), len(names))
    assert np.isfinite(jacobian).all()
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=(10, 10))
        for i, name in enumerate(names):
            step = 1e-5 if name.startswith("pi") else 1e-6
            plus = PARAMS._replace(**{name: getattr(PARAMS, name) + step})
            minus = PARAMS._replace(**{name: getattr(PARAMS, name) - step})
            numerical = (
                _vbm_orbit(plus, times, True, True)[0]
                - _vbm_orbit(minus, times, True, True)[0]
            ) / (2 * step)
            axes[i].plot(times, jacobian[:, i], label="microlux jacfwd")
            axes[i].plot(
                times, numerical, "--", label="VBM finite difference (reference)"
            )
            axes[i].set_ylabel(f"dA/d{name}")
            axes[i].set_yscale("symlog")
        axes[0].legend()
        axes[-1].set_xlabel("HJD - 2450000")
        fig.tight_layout()
        PICTURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PICTURE_DIR / "forward_gradients.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    test_orbital_accuracy(False, False, None, plot=True)
    test_orbital_accuracy(True, True, None, plot=True)
    test_orbital_forward_gradient(plot=True)
    print(f"Plots saved to {PICTURE_DIR}")
