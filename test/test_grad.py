import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import gridspec
from microlux import binary_mag, contour_integral
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import approx_fprime
from test_util import VBM_light_curve


@pytest.mark.fast
def test_analytic_contour_source_curvature():
    def magnification(x):
        # Loose tolerances keep the contour mesh fixed across the difference step.
        mag, _ = contour_integral(
            x + 0.2j,
            tol=1.0,
            retol=1.0,
            rho=0.01,
            s=0.9,
            q=0.2,
            default_strategy=(60,),
            analytic=True,
        )
        return jnp.stack((mag, jnp.log(mag)))

    x = 0.62
    step = 1e-5
    first = jax.jit(jax.jacfwd(magnification))
    second = jax.jit(jax.jacfwd(jax.jacfwd(magnification)))
    expected = (first(x + step) - first(x - step)) / (2 * step)
    np.testing.assert_allclose(second(x), expected, rtol=1e-5)


@pytest.mark.fast
@pytest.mark.parametrize("strategy, exceeded", [((30, 30, 60), False), ((30,), True)])
def test_analytic_gradient_uses_selected_contour(strategy, exceeded):
    # A source crossing a caustic needs the second layer at this tolerance.
    params = jnp.array([0.7145553, 0.2613920, 0.01, 0.9, 0.2])

    def evaluate(params, analytic):
        x, y, rho, s, q = params
        mag, info = contour_integral(
            x + 1j * y, 1e-3, 1e-3, rho, s, q, strategy, analytic
        )
        return mag, (info[-2].sample_num, info[-1].exceed_flag, info[-1].mag[0])

    (mag, (count, flag, info_mag)), gradient = jax.value_and_grad(
        evaluate, has_aux=True
    )(params, True)
    expected, (raw_count, raw_flag, raw_mag) = jax.jacfwd(evaluate, has_aux=True)(
        params, False
    )
    assert bool(flag) == bool(raw_flag) == exceeded
    assert count == raw_count
    if not exceeded:
        assert strategy[0] < count < sum(strategy[:2]) - 2
    np.testing.assert_allclose([mag, info_mag], raw_mag, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-7)


def grad_test(t_0, u_0, t_E, rho, q, s, alpha_deg, times, retol, tol):
    vbbl_fun = lambda x: VBM_light_curve(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], times, retol, tol
    )
    grad_scipy = approx_fprime(
        [t_0, u_0, t_E, rho, q, s, alpha_deg],
        vbbl_fun,
    )

    grad_fun = jax.jacfwd(binary_mag, argnums=(0, 1, 2, 3, 4, 5, 6))
    jacobian = jnp.array(
        grad_fun(t_0, u_0, t_E, rho, q, s, alpha_deg, times, tol, retol)
    )
    name = ["t_0", "u_0", "t_E", r"\rho", "q", "s", r"\alpha"]

    # grad_caustic = jax.jit(jax.jacfwd(caustic_fun,argnums=0))
    # jacobian_caustic = jnp.array(grad_caustic([t_0,u_0,t_E,rho,q,s,alpha_deg],times))

    gc = gridspec.GridSpec(8, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1])
    plt.figure(figsize=(10, 8))
    # fig = plt.figure(figsize=(8,6))
    # gc = gridspec.GridSpec(2, 1,height_ratios=[1,1])
    mag = binary_mag(t_0, u_0, t_E, rho, q, s, alpha_deg, times, retol, retol)
    mag_vbl = np.array(
        VBM_light_curve(t_0, u_0, t_E, rho, q, s, alpha_deg, times, retol)
    )
    # mag_caustic = caustic_fun([t_0,u_0,t_E,rho,q,s,alpha_deg],times)

    ax = plt.subplot(gc[0])
    ax_traj = inset_axes(
        ax,
        width="20%",
        height="50%",
        loc="upper left",
        bbox_to_anchor=(0.1, 0, 1, 1),
        bbox_transform=ax.transAxes,
    )

    from test_util import get_caustics

    caustic_x, caustic_y = get_caustics(q, s)
    ax_traj.scatter(caustic_x, caustic_y, s=0.5, c="black")
    ax_traj.axis("equal")
    trajectory = (times - t_0) / t_E * np.exp(1j * alpha) + 1j * u_0 * np.exp(
        1j * alpha
    )
    ax_traj.plot(trajectory.real, trajectory.imag, c="black")

    ax.plot(mag_vbl, label="finite diff")
    ax.plot(mag, label="auto diff")
    # ax.plot(mag_caustic,label='caustic')

    ax.set_ylabel("A(t)", rotation=0, labelpad=10)
    ax.legend()
    # ax.set_yscale('symlog')
    print("max error microlux=", np.max(np.abs(mag - mag_vbl)))
    # print('max error caustic=',np.max(np.abs(mag_vbl-mag_caustic)))

    def format_func(value, tick_number):
        if np.abs(value) > 1e-2:
            # 只在10的偶数次方显示刻度标签
            if int(np.log10(np.abs(value))) % 2 == 0:
                if value > 0:
                    sign = ""
                else:
                    sign = "-"
                return f"${sign}10^{{{int(np.log10(np.abs(value)))}}}$"
        else:
            return ""

    for i in range(jacobian.shape[0]):
        ax_i = plt.subplot(gc[i + 1], sharex=ax)
        ax_i.plot(grad_scipy[:, i])
        ax_i.plot(jacobian[i, :])
        # ax_i.plot(jacobian_caustic[i,:])
        # ax_i.plot(grad_scipy[:,i]-jacobian[i,:])
        ax_i.set_ylabel(
            rf"$ \frac{{\partial A(t) }} {{\partial {name[i]}}} $",
            rotation=0,
            labelpad=10,
        )
        # res = np.sqrt(np.sum((grad_scipy[:,i]-jacobian[i,:])**2))
        # print(f'{name[i]} erro jax and vbblr={res}')

        # res_caustic = np.sqrt(np.sum((grad_scipy[:,i]-jacobian_caustic[i,:])**2))
        # print(f'{name[i]} error caustic and vbbl={res_caustic}')

        # res_caustic = np.sqrt(np.sum((jacobian[i,:]-jacobian_caustic[i,:])**2))
        # print(f'{name[i]} error caustic and microlux={res_caustic}')
        ax_i.set_yscale("symlog", base=10)
    plt.savefig("test_grad.png")


if __name__ == "__main__":
    b = 0.1
    t_0 = 8280.0
    t_E = 40.0
    alphadeg = 60.0
    q = 0.2
    s = 0.9
    rho = 10 ** (-2)
    tol = 1e-3
    retol = 1e-3
    trajectory_n = 2000
    alpha = alphadeg * 2 * jnp.pi / 360
    # times=jnp.linspace(t_0-1.*t_E,t_0+1*t_E,trajectory_n)
    times = jnp.linspace(8260, 8320, trajectory_n)[250:1000]

    grad_fun = jax.jacfwd(binary_mag, argnums=(0, 1, 2, 3, 4, 5, 6))
    grad_fun = jax.jit(grad_fun)

    grad_test(t_0, b, t_E, rho, q, s, alphadeg, times, retol, tol)
