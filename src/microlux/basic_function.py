import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def to_centroid(s, q, x):
    """
    Transforms the coordinate system to the centroid.

    Parameters:
    s (float): The projected separation between the two objects.
    q (float): The planet to host mass ratio.
    x (complex): The original coordinate.

    Returns:
    complex: The transformed coordinate in the centroid system.
    """
    delta_x = s / (1 + q)
    return -(jnp.conj(x) - delta_x)


def to_lowmass(s, q, x):
    """
    Transforms the coordinate system to the system where the lower mass object is at the origin.

    Parameters:
    s (float): The separation between the two components.
    q (float): The mass ratio of the two components.
    x (complex): The original centroid coordinate.

    Returns:
    complex: The transformed coordinate in the low mass component coordinate system.
    """
    delta_x = s / (1 + q)
    return -jnp.conj(x) + delta_x


def Quadrupole_test(rho, s, q, zeta, z, cond, tol=1e-2):
    """
    The quadrupole test, ghost image test, and planetary caustic test proposed by Bozza 2010 to check the validity of the point source approximation.
    The coefficients are fine-tuned in our implementation.

    """
    m1 = 1 / (1 + q)
    m2 = q / (1 + q)
    cQ = 2
    cG = 3
    cP = 4  # tunable parameters vbbl 2018 + version=3.6.2 choose cQ=3,cG=miu_G (vbbl typo) ,cP=4

    # basic derivatives
    fz0 = lambda z: -m1 / (z - s) - m2 / z
    fz1 = lambda z: m1 / (z - s) ** 2 + m2 / z**2
    fz2 = lambda z: -2 * m1 / (z - s) ** 3 - 2 * m2 / z**3
    fz3 = lambda z: 6 * m1 / (z - s) ** 4 + 6 * m2 / z**4
    J = lambda z: 1 - fz1(z) * jnp.conj(fz1(z))

    ####Quadrupole test
    miu_Q = jnp.abs(
        -2
        * jnp.real(
            3 * jnp.conj(fz1(z)) ** 3 * fz2(z) ** 2
            - (3 - 3 * J(z) + J(z) ** 2 / 2) * jnp.abs(fz2(z)) ** 2
            + J(z) * jnp.conj(fz1(z)) ** 2 * fz3(z)
        )
        / (J(z) ** 5)
    )

    # cusp test
    miu_C = jnp.abs(jnp.imag(3 * jnp.conj(fz1(z)) ** 3 * fz2(z) ** 2) / (J(z) ** 5))
    mag = jnp.sum(jnp.where(cond, jnp.abs(1 / J(z)), 0), axis=1)
    cond1 = (
        jnp.sum(jnp.where(cond, (miu_Q + miu_C), 0), axis=1)
        * cQ
        * (rho**2 + 1e-4 * tol)
        < tol
    )

    ####ghost image test
    zwave = jnp.conj(zeta) - fz0(z)
    J_wave = 1 - fz1(z) * fz1(zwave)
    J3 = J_wave * fz2(jnp.conj(z)) * fz1(z)
    miu_G = jnp.abs((J3 - jnp.conj(J3) * fz1(zwave)) / (J(z) * J_wave**2))
    miu_G = jnp.where(cond, 0, miu_G)
    cond2 = ((rho + 1e-3) * miu_G * cG < 1).all(axis=1)  # all() is same with VBBL code

    #####planet test # in our frame primary is at s, the planet is at 0, so the position of the planetary caustic is 1/s
    cond3 = (
        (q > 1e-2)
        | (
            jnp.abs(zeta - 1 / s) ** 2
            > cP * (rho**2 + 9 * q / s**2)  # rho**2*s**2<q comment out in vbbl 3.6.2
        )
    )[:, 0]

    return cond1 & cond2 & cond3, mag


def get_poly_coff(zeta_l, s, m2):
    """
    get the polynomial cofficients of the polynomial equation of the lens equation.
    The low mass object is at the origin and the primary is at s.
    The input zeta_l should have the shape of (n,1) for broadcasting.
    This calculation is based on the Wang et al. 2025 paper. (https://arxiv.org/pdf/2501.03322) to avoid the cancellation error, but the coordinates convention is different (z1 = (-s,0) in Wang et al. 2025 and z1 = (s,0) here).
    So we need to change the sign of s in the calculation.
    """
    zeta_conj = jnp.conj(zeta_l)
    y = zeta_l
    s = -s
    y_bar = zeta_conj

    # 定义中间变量
    v_c = y_bar + s  # v_c ≡ \bar{y} + s
    v_p = 1 + s * y_bar  # v_p ≡ 1 + s\bar{y}

    # 计算实部和虚部
    Im_y = jnp.imag(y)
    Re_v_p = jnp.real(v_p)
    Re_v_c = jnp.real(v_c)

    # 计算各个系数
    c5 = -y_bar * v_c  # c5 = -\bar{y}v_c

    c4 = (
        y_bar * y - 1 - 2 * s * y_bar
    ) * v_c + m2 * s  # c4 = (\bar{y}y - 1 - 2s\bar{y})v_c + m_2s

    # c3 = (2y - s)[v_p*v_c + 2Im(y)i - m_2s] + 4(m_2s - y)Im(y)i
    c3 = (2 * y - s) * (v_p * v_c + 2 * Im_y * 1j - m2 * s) + 4 * (
        m2 * s - y
    ) * Im_y * 1j

    # c2 = Re(v_p)^2 Re(v_c) + Im(y)Re(v_p)Re(1 - sy)i + sIm(y)^2(2 + sv_c) + m_2s[2\bar{y}(y - s) - (1 - sy)]
    term1 = Re_v_p**2 * Re_v_c
    term2 = Im_y * Re_v_p * jnp.real(1 - s * y) * 1j
    term3 = s * Im_y**2 * (2 + s * v_c)
    term4 = m2 * s * (2 * y_bar * (y - s) - (1 - s * y))
    c2 = term1 + term2 + term3 + term4

    # c1 = m_2s[(s + 2y)(v_p) + s(2sIm(y)i - m_2)]
    c1 = m2 * s * ((s + 2 * y) * v_p + s * (2 * s * Im_y * 1j - m2))

    c0 = s**2 * m2**2 * y  # c0 = s^2 m_2^2 y

    # 将系数组合成数组
    coff = jnp.concatenate((c5, c4, c3, c2, c1, c0), axis=1)
    return coff


def get_zeta_l(rho, trajectory_centroid_l, theta):  # 获得等高线采样的zeta
    zeta_l = trajectory_centroid_l + rho * jnp.exp(1j * theta)
    return zeta_l


def verify(zeta_l, z_l, s, m1, m2):  # verify whether the root is right
    return jnp.abs(z_l - m1 / (jnp.conj(z_l) - s) - m2 / jnp.conj(z_l) - zeta_l)


def get_parity(z, s, m1, m2):  # get the parity of roots
    de_conjzeta_z1 = m1 / (jnp.conj(z) - s) ** 2 + m2 / jnp.conj(z) ** 2
    return jnp.sign((1 - jnp.abs(de_conjzeta_z1) ** 2))


def get_parity_error(z, s, m1, m2):
    de_conjzeta_z1 = m1 / (jnp.conj(z) - s) ** 2 + m2 / jnp.conj(z) ** 2
    return jnp.abs((1 - jnp.abs(de_conjzeta_z1) ** 2))


def dot_product(a, b):
    return jnp.real(a) * jnp.real(b) + jnp.imag(a) * jnp.imag(b)


def basic_partial(z, theta, rho, q, s, caustic_crossing):
    """

    basic partial derivatives of the lens equation with respect to zeta, z, and theta used in the error estimation.

    """
    z_c = jnp.conj(z)
    parZetaConZ = 1 / (1 + q) * (1 / (z_c - s) ** 2 + q / z_c**2)
    par2ConZetaZ = -2 / (1 + q) * (1 / (z - s) ** 3 + q / (z) ** 3)
    de_zeta = 1j * rho * jnp.exp(1j * theta)
    detJ = 1 - jnp.abs(parZetaConZ) ** 2
    de_z = (de_zeta - parZetaConZ * jnp.conj(de_zeta)) / detJ
    deXProde2X = (rho**2 + jnp.imag(de_z**2 * de_zeta * par2ConZetaZ)) / detJ

    def get_de_deXPro_de2X(carry):
        # now only calculate the derivative of x'^x'' with respect to \theta if caustic_crossing is True which is used in e4 calculation
        # still need to test weather this is robust enough for the case that source is very close to the caustic but not crossing it

        de2_zeta = -rho * jnp.exp(1j * theta)
        de2_zetaConj = -rho * jnp.exp(-1j * theta)
        par3ConZetaZ = 6 / (1 + q) * (1 / (z - s) ** 4 + q / (z) ** 4)
        de2_z = (
            de2_zeta
            - jnp.conj(par2ConZetaZ) * jnp.conj(de_z) ** 2
            - parZetaConZ * (de2_zetaConj - par2ConZetaZ * de_z**2)
        ) / detJ
        # deXProde2X_test = 1/(2*1j)*(de2_z*jnp.conj(de_z)-de_z*jnp.conj(de2_z))
        # jax.debug.print('deXProde2X_test error is {}',jnp.nansum(jnp.abs(deXProde2X_test-deXProde2X)))
        de_deXPro_de2X = (
            1
            / detJ**2
            * jnp.imag(
                detJ
                * (
                    de2_zeta * par2ConZetaZ * de_z**2
                    + de_zeta * par3ConZetaZ * de_z**3
                    + de_zeta * par2ConZetaZ * 2 * de_z * de2_z
                )
                + (
                    jnp.conj(par2ConZetaZ) * jnp.conj(de_z) * jnp.conj(parZetaConZ)
                    + parZetaConZ * par2ConZetaZ * de_z
                )
                * de_zeta
                * par2ConZetaZ
                * de_z**2
            )
        )
        return de_deXPro_de2X

    de_deXPro_de2X = jax.lax.cond(
        caustic_crossing, get_de_deXPro_de2X, lambda x: jnp.zeros_like(deXProde2X), None
    )
    # deXProde2X = jax.lax.stop_gradient(deXProde2X)
    return deXProde2X, de_z, de_deXPro_de2X


@jax.custom_jvp
def refine_gradient(zeta_l, q, s, z):
    return z


@refine_gradient.defjvp
def refine_gradient_jvp(primals, tangents):
    """
    use the custom jvp to refine the gradient of roots respect to zeta_l, based on the equation on V.Bozza 2010 eq 20 and also see our paper for the details
    This will simplify the computational graph and accelerate the gradient calculation
    """
    zeta, q, s, z = primals
    tangent_zeta, tangent_q, tangent_s, tangent_z = tangents

    z_c = jnp.conj(z)
    parZetaConZ = 1 / (1 + q) * (1 / (z_c - s) ** 2 + q / z_c**2)
    detJ = 1 - jnp.abs(parZetaConZ) ** 2

    parZetaq = 1 / (1 + q) ** 2 * (1 / (z_c - s) - 1 / z_c)
    add_item_q = tangent_q * (parZetaq - jnp.conj(parZetaq) * parZetaConZ)

    parZetas = -1 / (1 + q) / (z_c - s) ** 2
    add_item_s = tangent_s * (parZetas - jnp.conj(parZetas) * parZetaConZ)

    tangent_z2 = (
        tangent_zeta - parZetaConZ * jnp.conj(tangent_zeta) - add_item_q - add_item_s
    ) / detJ
    # tangent_z2 = jnp.where(jnp.isnan(tangent_z2),0.,tangent_z2)
    # jax.debug.print('{}',(tangent_z2-tangent_z).sum())
    return z, tangent_z2
