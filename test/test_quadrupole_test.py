import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Patch
from microlux import point_light_curve
from test_util import VBM_light_curve


np.seterr(divide="ignore", invalid="ignore")


@jax.jit
def point_mag_jax(t_0, u_0, t_E, rho, q, s, alphadeg, times, tol):
    alpha_rad = alphadeg * 2 * jnp.pi / 360
    tau = (times - t_0) / t_E
    ## trajectory in center-of-mass coordinates
    trajectory = tau * jnp.exp(1j * alpha_rad) + 1j * u_0 * jnp.exp(1j * alpha_rad)
    mag, cond = point_light_curve(trajectory, s, q, rho, tol)
    return mag, cond


def mag_gen_jax(i):  # t_0,b_map,t_E,rho,q,s,alphadeg,times_jax,tol,
    # parms=[t_0,b_map[i],t_E,rho,q,s,alphadeg,times_jax,tol]
    uniform_mag, cond = point_mag_jax(
        t_0, b_map[i], t_E, rho, q, s, alphadeg, times_jax, tol
    )
    return uniform_mag, cond, i


if __name__ == "__main__":
    trajectory_n = 1000
    sample_n = 1000

    tol = 1e-3
    retol = 0.0
    t_0 = 8283.594505
    t_E = 42.824343
    times = np.linspace(t_0 - 1.0 * t_E, t_0 + 1.0 * t_E, trajectory_n)
    alphadeg = 270
    times_jax = jnp.array(times)
    tau = (times - t_0) / t_E
    alpha_VBBL = np.pi + alphadeg / 180 * np.pi
    q = 0.2
    s = 0.9
    rho = 10 ** (-2)

    b_map = np.linspace(-1.0, 1.0, sample_n)
    VBBL_mag_map = np.zeros((sample_n, trajectory_n))
    jax_map = np.zeros((sample_n, trajectory_n))
    jax_test = np.zeros((sample_n, trajectory_n))

    start = time.perf_counter()

    # vbbl time
    start = time.perf_counter()
    mag_gen_vbbl = lambda i: VBM_light_curve(
        t_0, b_map[i], t_E, rho, q, s, alphadeg, times, retol, tol
    )
    for i in range(sample_n):
        uniform = mag_gen_vbbl(i)
        VBBL_mag_map[i, :] = uniform
    print(f"vbbl time took: {time.perf_counter() - start:.4f}")

    # compile time
    start = time.perf_counter()
    jax.block_until_ready(mag_gen_jax(0))
    print(f"compile time took: {time.perf_counter() - start:.4f}")

    # jax time
    start = time.perf_counter()
    for i in range(sample_n):
        uniform, cond, i = mag_gen_jax(i)
        jax_map[i, :] = uniform
        jax_test[i, :] = cond

    jax.block_until_ready(jax_map)
    print(f"jax time took: {time.perf_counter() - start:.4f}")

    residual = np.abs(VBBL_mag_map - jax_map)

    classifiers = np.zeros_like(residual)

    classifiers[residual < 1e-5] = 0  # point
    classifiers[(residual > 1e-5) & (residual < tol)] = 1  # VBBL safe region
    classifiers[jax_test == 0] = 2  # Jax safe region
    classifiers[residual > tol] = 3  # real safe region

    # check if the classifiers are correct

    jax_safe = jax_test == 0
    real_safe = residual > tol

    print("real region that is not captured by Jax:", np.sum(real_safe & (~jax_safe)))

    # plot the map
    fig, ax = plt.subplots(figsize=(8, 6))

    colors_list = cm.viridis(np.linspace(0, 1, 4))
    cmap = colors.ListedColormap(colors_list)

    labels = ["point", "VBBL safe region", "Jax safe region", "real safe region"]
    patches = [Patch(color=colors_list[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, loc="upper right")

    surf = ax.pcolormesh(b_map, tau, classifiers.T, shading="nearest")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\theta_E$")
    ax.set_ylabel(r"$y/\theta_E$")

    plt.savefig("picture/quadrupole_test.png")
