"""Test parallax magnification against VBMicrolensing."""

import math
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import VBMicrolensing
from microlux import extended_light_curve
from microlux.coordinates import Coordinates
from microlux.trajectory import get_trajectory_model


jax.config.update("jax_enable_x64", True)


# Parallax parameter combinations for testing
piEN_values = [0.0, 0.1, 0.2]
piEE_values = [0.0, 0.1, 0.2]


@pytest.mark.fast
@pytest.mark.parametrize("piEN, piEE", product(piEN_values, piEE_values))
def test_parallax_accuracy(piEN, piEE, plot=False):
    """
    Test that microlux parallax calculation matches VBMicrolensing.

    This test verifies that the parallax implementation produces
    magnification values consistent with VBMicrolensing to within
    acceptable tolerance for various parallax parameter combinations.

    Args:
        piEN: North component of the microlens parallax vector.
        piEE: East component of the microlens parallax vector.
        plot: If True, save plots to disk. Default is False for pytest.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Fixed parameters
    t0 = 2460000.0
    tE = 100.0
    rho = 1e-3
    u0 = 0.1
    q = 1e-3
    s = 0.9
    alpha_deg = 120.0

    # Coordinates
    ra_dec_str = "17:59:02.3 -29:04:15.2"
    ra_str = "17:59:02.3"
    dec_str = "-29:04:15.2"

    # Time grid (reduced for faster test execution)
    n_times = 500
    times = jnp.linspace(t0 - 2.0 * tE, t0 + 2.0 * tE, n_times)

    # microlux model with parallax
    coords = Coordinates(ra=ra_str, dec=dec_str)
    trajectory_model = get_trajectory_model(times=times, coords=coords, t0_par=t0)

    alpha_rad = alpha_deg * 2 * np.pi / 360
    params = jnp.array(
        [t0, u0, tE, rho, alpha_rad, s, q, piEN, piEE], dtype=jnp.float64
    )

    # Calculate trajectory in center-of-mass coordinates
    trajectory = trajectory_model.calculate_trajectory(params)

    # Calculate magnification using the trajectory
    mag_lux = extended_light_curve(trajectory, s, q, rho)

    # VBMicrolensing model
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.SetObjectCoordinates(ra_dec_str)

    # VBMicrolensing parameter vector:
    # [log(s), log(q), u0, alpha(rad)-pi, log(rho), log(tE), t0-2450000, piEN, piEE]
    pr_vbm = [
        math.log(s),
        math.log(q),
        u0,
        alpha_rad - np.pi,
        math.log(rho),
        math.log(tE),
        t0 - 2450000.0,
        piEN,
        piEE,
    ]

    times_np = np.asarray(times - 2450000.0, dtype=float)
    mag_vbm = np.asarray(VBM.BinaryLightCurveParallax(pr_vbm, times_np)[0], dtype=float)

    # Check accuracy
    mag_lux_np = np.asarray(mag_lux, dtype=float)
    denom = np.maximum(np.abs(mag_vbm), 1e-15)
    rel_err = np.abs(mag_vbm - mag_lux_np) / denom

    max_err = np.max(rel_err)
    mean_err = np.mean(rel_err)

    print(f"piEN={piEN}, piEE={piEE}: max_err={max_err:.4e}, mean_err={mean_err:.4e}")

    # Assert that error is acceptable
    assert (
        max_err < 0.01
    ), f"Max relative error {max_err:.4e} exceeds 0.01 for piEN={piEN}, piEE={piEE}"
    assert (
        mean_err < 0.001
    ), f"Mean relative error {mean_err:.4e} exceeds 0.001 for piEN={piEN}, piEE={piEE}"

    # Plot and save if requested
    if plot:
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        # Top panel: light curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(np.asarray(times), mag_lux_np, label="microlux (new)", linewidth=1.8)
        ax1.plot(
            np.asarray(times), mag_vbm, label="VBMicrolensing", linewidth=1.2, alpha=0.9
        )
        ax1.set_title("Binary microlensing with parallax: magnification vs time")
        ax1.set_ylabel("Magnification")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="best")

        # Bottom panel: relative difference
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(np.asarray(times), rel_err, linewidth=1.4)
        ax2.set_yscale("log")
        ax2.set_title("Relative difference |VBM - microlux| / |VBM|")
        ax2.set_xlabel("Time (JD)")
        ax2.set_ylabel("Relative error")
        ax2.grid(True, which="both", alpha=0.3)

        # Save plot
        filename = f"parallax_piEN{piEN}_piEE{piEE}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {filename}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Running parallax test with VBMicrolensing comparison")
    print("=" * 60)

    # Run tests with plotting enabled for visualization
    test_parallax_accuracy(piEN=0.1, piEE=0.1, plot=True)

    print("\n" + "=" * 60)
    print("Test complete! Plots saved to disk.")
    print("=" * 60)
