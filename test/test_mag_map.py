import os


process_number = 100
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d" % process_number

import time
from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp
import numpy as np
from test_util import VBM_light_curve


np.seterr(divide="ignore", invalid="ignore")


def mag_map_vbbl(i, all_params, fix_params):
    rho, q, s = all_params[i]
    t_0, b_map, t_E, alphadeg, times, tol = fix_params
    VBBL_mag_map = np.zeros((sample_n, trajectory_n))
    mag_vbbl = lambda i: VBM_light_curve(
        t_0, b_map[i], t_E, rho, q, s, alphadeg, times, 1e-4, 1e-3
    )
    for i in range(sample_n):
        VBBL_mag_map[i, :] = mag_vbbl(i)
    return VBBL_mag_map, i


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    trajectory_n = 1000
    sample_n = 1000
    tol = 1e-3
    t_0 = 8000
    t_E = 50
    times = np.linspace(t_0 - 0.0 * t_E, t_0 + 2.0 * t_E, trajectory_n)
    alphadeg = 270
    tau = (times - t_0) / t_E
    alpha_VBBL = np.pi + alphadeg / 180 * np.pi

    b_map = np.linspace(-4.0, 3.0, sample_n)

    np.random.seed(42)

    N_Runs = 1000
    N_save = 100

    parameter_space = np.zeros((N_Runs, 3))
    VBBL_mag_map_list, jax_map_list = [], []
    sample_num_list, exceed_flag_list = [], []

    jax_time = 0

    for k in range(N_Runs):
        q = 10 ** (np.random.uniform(-6.0, 0.0))
        s = 10 ** np.random.uniform(-0.5, 0.5)
        rho = 10 ** (np.random.uniform(-3.0, -1.0))
        parameter_space[k] = [rho, q, s]

    # vbbl mag map test

    from tqdm import tqdm

    start = time.perf_counter()
    mag_vbbl_warp = partial(
        mag_map_vbbl,
        all_params=parameter_space,
        fix_params=[t_0, b_map, t_E, alphadeg, times, tol],
    )
    with Pool(processes=process_number) as pool:
        for VBBL_mag, i in tqdm(
            pool.imap(mag_vbbl_warp, range(N_Runs)), total=N_Runs, mininterval=1
        ):
            VBBL_mag = VBBL_mag.astype(np.float32)
            VBBL_mag_map_list.append(VBBL_mag)

    print(f"vbbl time took: {time.perf_counter() - start:.4f}")
    np.savez("test_result/vbbl_map_time_test.npz", VBBL_mag_map_list=VBBL_mag_map_list)

    # jax mag map test

    from microlux import binary_mag

    def mag_jax(i, rho, q, s, parm):
        t_0, b_map, t_E, alphadeg, times_jax, tol = parm
        uniform_mag, info = binary_mag(
            t_0,
            b_map[i],
            t_E,
            rho,
            q,
            s,
            alphadeg,
            times_jax,
            tol=1e-3,
            retol=1e-3,
            analytic=False,
            return_info=True,
            default_strategy=(30, 30, 60, 120, 240, 480),
        )
        return uniform_mag, info[-2].sample_num, info[-1].exceed_flag

    for k in range(N_Runs):
        rho, q, s = parameter_space[k]
        jax_map = np.zeros((sample_n, trajectory_n))
        all_nodes = jnp.arange(sample_n)
        outputs = []
        sample_num = []
        exceed_flag = []

        parm = [t_0, jnp.array(b_map), t_E, alphadeg, jnp.array(times), tol]
        if k == 0:
            start = time.monotonic()
            for i in range(sample_n // process_number):
                slic = all_nodes[process_number * i : process_number * (i + 1)]
                mag_i, sample_num_i, exceed_flag_i = jax.pmap(
                    mag_jax, in_axes=(0, None, None, None, None)
                )(slic, rho, q, s, parm)
            print("jax compile time took: {:.4f}".format(time.monotonic() - start))

        start = time.monotonic()
        for i in range(sample_n // process_number):
            slic = all_nodes[process_number * i : process_number * (i + 1)]
            mag_i, sample_num_i, exceed_flag_i = jax.pmap(
                mag_jax, in_axes=(0, None, None, None, None)
            )(slic, rho, q, s, parm)
            outputs.append(mag_i)
            sample_num.append(sample_num_i)
            exceed_flag.append(exceed_flag_i)
        jax_map = jnp.concatenate(outputs, axis=0)
        sample_num = jnp.concatenate(sample_num, axis=0)
        exceed_flag = jnp.concatenate(exceed_flag, axis=0)

        jax_map = jax_map.astype(np.float32)
        sample_num = sample_num.astype(np.int32)

        jax_time += time.monotonic() - start
        jax_map_list.append(jax_map)
        sample_num_list.append(sample_num)
        exceed_flag_list.append(exceed_flag)

        if (k + 1) % N_save == 0:
            np.savez(
                "test_result/jax_map_test.npz",
                jax_map_list=jax_map_list,
                sample_num_list=sample_num_list,
                exceed_flag_list=exceed_flag_list,
            )
            print("{}/{} run saved".format(k + 1, N_Runs))
    print("jax time took: {:.4f}".format(jax_time))
