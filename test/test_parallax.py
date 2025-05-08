import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm
import pytest

from microlux.coordinates import annual_parallax_shift, Coordinates
from microlux.trajectory import TrajectoryModel
from microlux import to_lowmass, extended_light_curve


@pytest.mark.fast
def test_parallax(pi_E_N=0.2, pi_E_E=0.1, params: dict = None, coords: Coordinates = None, times: np.ndarray = None, test: bool = True):
    """
    Compare magnification from MulensModel and microlux to ensure consistency.
    """

    if params is None and coords is None and times is None:
        coords = Coordinates(ra="18:04:45.71", dec="-26:59:15.2")
        params = dict()
        params['t_0'] = 2453628.3
        params['t_0_par'] = 2453628.
        params['u_0'] = 0.1
        params['t_E'] = 100.
        params['rho'] = 0.01
        params['s'] = 1.0
        params['q'] = 0.1
        params['alpha'] = 60
        times = np.linspace(2453628.3 - 100, 2453628.3 + 100, 100)

    # MulensModel calculation
    params['pi_E_N'] = pi_E_N
    params['pi_E_E'] = pi_E_E

    my_model = mm.Model(params, ra= coords.ra, dec=coords.dec)
    my_model.set_magnification_methods([2453628.3-10000., 'VBBL', 2453628.3+10000])
    mag_mulensmodel = my_model.get_magnification(time=times)

    # Microlux calculation
    time_ref = params['t_0_par']
    delta_s_projected = annual_parallax_shift(times=times, time_ref=time_ref, coords=coords)
    trajectory = TrajectoryModel(times=times, delta_s=delta_s_projected).calculate_trajectory(
        params=[params['t_0'], params['u_0'], params['t_E'], np.deg2rad(params['alpha'])+np.pi, # the difference convention
                params['pi_E_N'], params['pi_E_E']]
    )
    trajectory_l = to_lowmass(s=params['s'], q=params['q'], x=trajectory)
    mag_microlux = extended_light_curve(trajectory_l, s=params['s'], q=params['q'], rho=params['rho'])

    # Compare results
    max_rel_error = np.max(np.abs(mag_microlux - mag_mulensmodel)/mag_mulensmodel)
    print(f"max relative error: {max_rel_error}")

    assert max_rel_error < 1e-3, f"Max relative error {max_rel_error} exceeds threshold"
    
    if not test:
        return my_model, trajectory, mag_mulensmodel, mag_microlux, times, max_rel_error


def plot_trajectories(my_model, trajectory, mag_mulensmodel, mag_microlux, times, 
                     traj_plot_path='fig/test_parallax_trajectory.png', 
                     mag_plot_path='fig/test_parallax.png'):
    """
    Plot trajectories and magnification curves for both MulensModel and microlux.
    """
    # Plot trajectories
    plt.figure()
    my_model.plot_trajectory(times=times, label='mulensmodel', alpha=0.5)
    my_model.plot_caustics()
    plt.plot(trajectory.real, trajectory.imag, label='microlux', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.savefig(traj_plot_path)
    
    # Plot magnification curves
    plt.figure()
    plt.plot(times, mag_mulensmodel, label='mulensmodel', alpha=0.5)
    plt.plot(times, mag_microlux, label='microlux', alpha=0.5)
    plt.legend()
    plt.savefig(mag_plot_path)


if __name__ == "__main__":
    coords = Coordinates(ra="18:04:45.71", dec="-26:59:15.2")
    params = dict()
    params['t_0'] = 2453628.3
    params['t_0_par'] = 2453628.
    params['u_0'] = 0.1
    params['t_E'] = 100.
    params['rho'] = 0.01
    params['s'] = 1.0
    params['q'] = 0.1
    params['alpha'] = 60
    times = np.linspace(2453628.3 - 100, 2453628.3 + 100, 100)
    
    model, trajectory, mag_mm, mag_ml, times, error = test_parallax(params=params, coords=coords, times=times)
    plot_trajectories(model, trajectory, mag_mm, mag_ml, times)