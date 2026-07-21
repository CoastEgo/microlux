from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .basic_function import get_parity, get_parity_error, get_poly_coff, verify
from .linear_sum_assignment import find_nearest
from .polynomial_solver import get_roots
from .utils import (
    apply_insert,
    compact_delete,
    get_delete_source_indices,
    get_insert_source_indices,
    Iterative_State,
    MAX_CAUSTIC_INTERSECT_NUM,
)


jax.config.update("jax_enable_x64", True)


def add_points(add_idx, add_zeta, add_theta, roots_State, s, m1, m2):
    """
    add the new points in the adaptive sampling scheme

    """
    sample_n, theta, roots, parity, ghost_roots_dis, sort_flag, Is_create = roots_State
    add_coff = get_poly_coff(add_zeta, s, m2)
    (
        add_roots,
        add_parity,
        add_ghost_roots,
        outloop,
        add_coff,
        add_zeta,
        add_theta,
        add_idx,
    ) = get_real_roots(
        add_coff, add_zeta, add_theta, s, m1, m2, add_idx
    )  # 可能删掉不合适的根

    sample_n += (add_idx != -1).sum()

    ## insert the new samplings
    source_indices = get_insert_source_indices(theta.shape[0], add_idx)
    insert_fun = lambda x, y: apply_insert(x, y, source_indices)
    original_list = [theta, roots, parity, ghost_roots_dis, sort_flag]
    add_list = [
        add_theta,
        add_roots,
        add_parity,
        add_ghost_roots,
        jnp.full([add_roots.shape[0], 1], False),
    ]
    theta, unsorted_roots, unsorted_parity, ghost_roots_dis, sort_flag = jax.tree.map(
        insert_fun, original_list, add_list
    )

    buried_error = get_buried_error(ghost_roots_dis, sample_n)

    # reorder the whole roots and parity
    indices_update, sort_flag = get_sorted_roots(
        unsorted_roots, unsorted_parity, sort_flag, add_theta.shape[0]
    )

    roots = unsorted_roots[jnp.arange(unsorted_roots.shape[0])[:, None], indices_update]
    parity = unsorted_parity[
        jnp.arange(unsorted_parity.shape[0])[:, None], indices_update
    ]

    Is_create = find_create_points(roots, parity, sample_n)
    return (
        Iterative_State(
            sample_n, theta, roots, parity, ghost_roots_dis, sort_flag, Is_create
        ),
        buried_error,
        outloop,
    )


def get_buried_error(ghost_roots_dis, sample_n):
    """
    get the error to avoid the buried images. proposed by the Bozza 2010. We modify the criterion to a more conservative one to avoid the burried images.

    """
    n_ite = ghost_roots_dis.shape[0]
    error_buried = jnp.zeros((n_ite, 1))

    idx_j = jnp.arange(1, n_ite)
    idx_i = jnp.roll(idx_j, shift=1)
    idx_k = jnp.roll(idx_j, shift=-1)

    idx_k = jnp.where(
        idx_k == sample_n, 1, idx_k
    )  # because the last point is the same as the first point 0=2pi
    idx_i = jnp.where(idx_i == n_ite - 1, sample_n - 1, idx_i)

    Ghost_i = ghost_roots_dis[idx_i]
    Ghost_j = ghost_roots_dis[idx_j]
    Ghost_k = ghost_roots_dis[idx_k]
    KInCaustic = jnp.isnan(Ghost_k)
    IInCaustic = jnp.isnan(Ghost_i)

    # only add points in one side same as the VBBL
    add_item = jnp.where(
        (
            # ( (Ghost_i>Ghost_j)&(Ghost_k>Ghost_j)
            # ((((Ghost_i-Ghost_j)/(theta[idx_i]-theta[idx_j])*(theta[idx_j]-theta[idx_k]))>Ghost_j)
            (
                (Ghost_i > 2 * Ghost_j)
                | (
                    (Ghost_i > 1.5 * Ghost_j) & (Ghost_k > Ghost_j)
                )  # supplementary condition to avoid the burried images, 1.5 is a tunable parameter
            )
            & (~KInCaustic)
        ),
        (Ghost_i - Ghost_j) ** 2,
        0,
    )

    error_buried = error_buried.at[idx_k].add(add_item)

    add_item = jnp.where(
        (
            # ( (Ghost_i>Ghost_j)&(Ghost_k>Ghost_j)
            # ((Ghost_j<((Ghost_k-Ghost_j)/(theta[idx_k]-theta[idx_j])*(theta[idx_j]-theta[idx_i])))
            (
                (2 * Ghost_j < Ghost_k)
                | ((1.5 * Ghost_j < Ghost_k) & (Ghost_i > Ghost_j))  # same as above
            )
            & (~IInCaustic)
        ),
        (Ghost_k - Ghost_j) ** 2,
        0,
    )
    error_buried = error_buried.at[idx_j].add(add_item)

    return error_buried


@partial(jax.jit, static_argnames=("max_unsorted_num",))
def get_sorted_roots(roots, parity, sort_flag, max_unsorted_num):
    """

    sort the roots and parity to keep the minimum distance between the adjacent points and same parity for the adjacent points
    """

    indices = jnp.tile(jnp.arange(roots.shape[1]), (roots.shape[0], 1))

    def sort_body1(
        indices, k
    ):  # sort the roots and parity for adjacent points olde-new and new-new pairs
        def False_fun_sort1(indices):
            sort_indices = find_nearest(
                roots[k - 1, indices[k - 1]],
                parity[k - 1, indices[k - 1]],
                roots[k, :],
                parity[k, :],
            )
            indices = indices.at[k].set(sort_indices)
            return indices

        indices = lax.cond(k == -1, lambda x: x, False_fun_sort1, indices)
        return indices, k

    flase_i = jnp.where(~sort_flag, size=max_unsorted_num, fill_value=-1)[0]
    indices_update, _ = lax.scan(sort_body1, indices, flase_i)

    def sort_body2(indices_temp, i):  # sort the roots and parity for new-old pairs
        def False_fun(indices_temp):
            sort_indices = find_nearest(
                roots[i, indices_temp[i]],
                parity[i, indices_temp[i]],
                roots[i + 1, indices_temp[i + 1]],
                parity[i + 1, indices_temp[i + 1]],
            )

            cond = jnp.arange(roots.shape[0])[:, None] < i + 1
            indices_temp = jnp.where(cond, indices_temp, indices_temp[:, sort_indices])

            return indices_temp

        indices_temp = lax.cond(i == -2, lambda x: x, False_fun, indices_temp)

        return indices_temp, i

    resort_i = jnp.where(
        (~sort_flag[0:-1]) & (sort_flag[1:]), size=max_unsorted_num, fill_value=-2
    )[0]
    indices_update2, _ = lax.scan(sort_body2, indices_update, resort_i)

    roots = roots[jnp.arange(roots.shape[0])[:, None], indices_update2]
    parity = parity[jnp.arange(parity.shape[0])[:, None], indices_update2]

    sort_flag = sort_flag.at[:].set(True)
    return indices_update2, sort_flag


def get_real_roots(coff, zeta_l, theta, s, m1, m2, add_idx):
    """
    get the real roots and parity for the new points. This function contains the following steps:
        1. get the roots of the polynomial
        2. get the parity of the roots
        3. verify the roots by putting the roots into the lens equation
        4. select the roots by the relative criterion same as the VBBinaryLensing
        5. find the wrong parity and fix it
        6. delete the remaining wrong roots/parity


    """

    n_ite = zeta_l.shape[0]
    sample_n = (~jnp.isnan(zeta_l)).any(axis=1).sum()
    mask = jnp.arange(n_ite) < sample_n
    roots = get_roots(n_ite, jnp.where(mask[:, None], coff, 0.0))  # 求有效的根
    roots = jnp.where(mask[:, None], roots, jnp.nan)
    parity = get_parity(roots, s, m1, m2)
    error = verify(zeta_l, roots, s, m1, m2)

    iterator = jnp.arange(n_ite)  # new criterion to select the roots, same as the VBBL
    dlmin = 1.0e-4
    dlmax = 1.0e-3
    sort_idx = jnp.argsort(error, axis=1)
    third_error = error[iterator, sort_idx[:, 2]]
    forth_error = error[iterator, sort_idx[:, 3]]
    three_roots_cond = (forth_error * dlmin) > (
        third_error + 1e-12
    )  # three roots criterion
    bad_roots_cond = (~three_roots_cond) & (
        (forth_error * dlmax) > (third_error + 1e-12)
    )  # bad roots criterion
    cond = jnp.zeros_like(roots, dtype=bool)
    full_value = jnp.where(three_roots_cond, True, False)
    cond = cond.at[iterator[:, None], sort_idx[:, 3:]].set(full_value[:, None])

    ghost_roots_dis = jnp.abs(
        roots[iterator, sort_idx[:, 3]] - roots[iterator, sort_idx[:, 4]]
    )
    ghost_roots_dis = jnp.where(three_roots_cond, ghost_roots_dis, jnp.nan)[:, None]

    # find the wrong parity and fix it
    nan_num = cond.sum(axis=1)  ##对于没有采样到的位置也是0
    real_roots = jnp.where(cond, jnp.nan + jnp.nan * 1j, roots)
    real_parity = jnp.where(cond, jnp.nan, parity)
    parity_sum = jnp.nansum(real_parity, axis=1)
    idx_parity_wrong = jnp.where((parity_sum != -1) & mask, size=n_ite, fill_value=-1)[
        0
    ]  # parity计算出现错误的根的索引
    real_parity = lax.cond(
        (idx_parity_wrong != -1).any(),
        update_parity,
        lambda x: x[-1],
        (
            zeta_l,
            real_roots,
            nan_num,
            sample_n,
            idx_parity_wrong,
            cond,
            s,
            m1,
            m2,
            real_parity,
        ),
    )
    parity_sum = jnp.nansum(real_parity, axis=1)
    bad_parities_cond = parity_sum != -1
    outloop = 0

    # delete the remaining wrong roots/parity
    carry = lax.cond(
        (bad_parities_cond & bad_roots_cond & mask).any(),
        theta_remove_fun,
        lambda x: x,
        (
            sample_n,
            theta,
            real_parity,
            real_roots,
            ghost_roots_dis,
            outloop,
            parity_sum,
            mask,
            add_idx,
        ),
    )
    (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        _,
        add_idx,
    ) = carry

    return (
        real_roots,
        real_parity,
        ghost_roots_dis,
        outloop,
        coff,
        zeta_l,
        theta,
        add_idx,
    )


def update_parity(carry):
    (
        zeta_l,
        real_roots,
        nan_num,
        sample_n,
        idx_parity_wrong,
        cond,
        s,
        m1,
        m2,
        real_parity,
    ) = carry

    def loop_parity_body(carry, i):  ##循环体
        zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2 = carry
        temp = real_roots[i]
        parity_process_fun = lambda x: lax.cond(
            (nan_num[i] == 0) & (i < sample_n),
            parity_5_roots_fun,
            parity_3_roots_fun,
            x,
        )
        real_parity = lax.cond(
            i < sample_n,
            parity_process_fun,
            lambda x: x[2],
            (temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2),
        )
        # real_parity=lax.cond((nan_num[i]==0)&(i<sample_n),parity_true1_fun,parity_false1_fun,(temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2))
        return (zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2), i

    carry, _ = lax.scan(
        loop_parity_body,
        (zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2),
        idx_parity_wrong,
    )
    zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2 = carry
    real_parity = real_parity.at[-1].set(jnp.nan)
    return real_parity


def parity_5_roots_fun(carry):  ##对于5个根怎么判断其parity更加合理
    """

    Determine the parity of the roots for ambiguous cases with 5 roots
    We first select principal and fifth roots, whose parity are 1 and -1, respectively.
    The the rest three roots are sorted by their real parts (x-axis) and the middle one is assigned parity 1, the other two are assigned parity -1.
    """
    ##对于parity计算错误的点，分为fifth principal left center right，其中left center right 的parity为-1，1，-1
    temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2 = carry
    prin_idx = jnp.where(
        jnp.sign(temp.imag) == jnp.sign(zeta_l.imag[i]), size=1, fill_value=0
    )[0]  # 主图像的索引
    prin_root = temp[prin_idx][jnp.newaxis][0]
    prin_root = jnp.concatenate(
        [prin_root, temp[jnp.argmax(get_parity_error(temp, s, m1, m2))][jnp.newaxis]]
    )
    other = jnp.setdiff1d(temp, prin_root, size=3)
    x_sort = jnp.argsort(other.real)
    real_parity = real_parity.at[
        i,
        jnp.where((temp == other[x_sort[0]]) | (temp == other[x_sort[-1]]), size=2)[0],
    ].set(-1)
    real_parity = real_parity.at[
        i, jnp.where((temp == other[x_sort[1]]), size=1)[0]
    ].set(1)
    return real_parity


def parity_3_roots_fun(carry):  ##对于3个根怎么判断其parity更加合理
    """

    Determine the parity of the roots for ambiguous cases with 3 roots:
    The principal image is set to positive parity, and the other two images are set to negative parity.
    This will only work for the case where the source is not very close to the x-axis where the criterion for judging the principal image is not valid.
    """
    temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2 = carry

    def parity_true_fun(carry):  ##通过主图像判断，与zeta位于y轴同一侧的为1
        real_parity = carry
        real_parity = real_parity.at[i, jnp.where(~cond[i], size=3)].set(-1)
        real_parity = real_parity.at[
            i, jnp.where(jnp.sign(temp.imag) == jnp.sign(zeta_l.imag[i]), size=1)[0]
        ].set(1)
        return real_parity

    real_parity = lax.cond(
        (nan_num[i] != 0) & ((jnp.abs(zeta_l.imag[i]) > 1e-5)[0]),
        parity_true_fun,
        lambda x: x,
        real_parity,
    )
    return real_parity


def find_create_points(roots, parity, sample_n):
    """
    find the image are created or destroyed and return the index of the created or destroyed points,
    this index is used to determine the order for trapzoidal integration:
        for image creation, the integration should be  (z.imag_- + z.imag_+)*(z.real_-  - z.real_+)
        for image destruction, the integration should be (z.imag_- + z.imag_+)*(z.real_+  - z.real_-)
    the create represents the flag for image creation or destruction :
        if create=1, it means image creation, if create=-1, it means image destruction
        and for image creation, the error should be added to the previous row, for image destruction, the error should be added to the next row

    Parameters
    ----------
    roots : jnp.ndarray
        the roots of the polynomial
    parity : jnp.ndarray
        the parity of the roots
    sample_n : int
        the number of the sample points

    Returns
    -------
    Is_create : jnp.ndarray
        the row and column index of the created or destroyed points

    """
    cond = jnp.isnan(roots)
    Num_change_cond = jnp.diff(
        cond, axis=0
    )  # the roots at i is nan but the roots at i+1 is not nan/ the roots at i is not nan but the roots at i+1 is nan
    idx_x, idx_y = jnp.where(
        Num_change_cond & (jnp.arange(roots.shape[0] - 1) < (sample_n - 1))[:, None],
        size=MAX_CAUSTIC_INTERSECT_NUM * 2,
        fill_value=-2,
    )  ## the index i can't be the last index
    shift = jnp.where(cond[idx_x, idx_y], 1, 0)
    idx_x_create = idx_x + shift
    Create_Destory = jnp.where(cond[idx_x, idx_y], 1, -1)
    Create_Destory = jnp.where(idx_x < 0, 0, Create_Destory)
    critical_idx = idx_x_create[0::2]
    critical_idy1 = idx_y[0::2]
    critical_idy2 = idx_y[1::2]

    critical_idx = jnp.where(critical_idx < 0, 0, critical_idx)
    critical_idy1 = jnp.where(critical_idy1 < 0, 0, critical_idy1)
    critical_idy2 = jnp.where(critical_idy2 < 0, 0, critical_idy2)

    critical_pos_idy = jnp.where(
        Create_Destory[0::2] == -1 * parity[critical_idx, critical_idy1],
        critical_idy1,
        critical_idy2,
    )
    critical_neg_idy = jnp.where(
        Create_Destory[0::2] == 1 * parity[critical_idx, critical_idy1],
        critical_idy1,
        critical_idy2,
    )

    return jnp.stack(
        [critical_idx, critical_pos_idy, critical_neg_idy, Create_Destory[0::2]], axis=0
    )


def theta_remove_fun(carry):
    """
    remove the points with wrong parity
    """
    (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        mask,
        add_idx,
    ) = carry
    n_ite = theta.shape[0]
    cond = (parity_sum != -1) & mask
    delidx = jnp.where(cond, size=n_ite, fill_value=10000)[0]

    sample_n -= cond.sum()

    delete_tree = [theta, real_parity, real_roots, ghost_roots_dis, add_idx[:, None]]
    source_indices = get_delete_source_indices(n_ite, delidx)
    theta, real_parity, real_roots, ghost_roots_dis, add_idx = jax.tree.map(
        lambda x: compact_delete(x, source_indices), delete_tree
    )
    add_idx = add_idx[:, 0]

    outloop += cond.sum()
    # if the parity is still wrong, then delete the point
    return (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        mask,
        add_idx,
    )
