import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from microlux import binary_mag
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import approx_fprime
from test_util import VBM_light_curve


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
