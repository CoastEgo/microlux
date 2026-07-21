import jax


jax.config.update("jax_enable_x64", True)
from functools import partial

from jax import lax, numpy as jnp


def loop_body(roots0, coff):  # 采用判断来减少浪费
    def False_fun(carry):
        coff, roots0 = carry
        roots_new = Aberth_Ehrlich(coff, roots0)
        return roots_new

    roots_new = lax.cond((coff == 0).all(), lambda x: x[1], False_fun, (coff, roots0))
    return roots_new, roots_new


@partial(jax.jit, static_argnums=0)
def get_roots(sample_n, coff):
    roots0 = AE_roots0(coff[0])
    _, roots = lax.scan(loop_body, roots0, coff)  # scan循环，但是没有浪费
    return roots


@partial(jax.jit, static_argnums=0)
def get_roots_vmap(sample_n, coff):
    ## used when solving the coff without zero coffes
    roots_solver = lambda x: Aberth_Ehrlich(x, AE_roots0(x))
    roots = jax.vmap(roots_solver, in_axes=(0))(coff)
    return roots


# @jax.jit
# def laguerre_method(coff,x0,n, epsilon= 1e-10):
#     der1=jnp.polyder(coff,1)
#     der2=jnp.polyder(coff,2)
#     xk = x0
#     @jax.jit
#     def loop_body(carry):
#         xk,n,epsilon,a=carry
#         G = jnp.polyval(der1,xk) / jnp.polyval(coff,xk)
#         H = G ** 2 - jnp.polyval(der2,xk) / jnp.polyval(coff,xk)
#         root = jnp.sqrt((n - 1) * (n * H - G ** 2))
#         temp=jnp.stack([G + root, G - root])
#         d = temp[jnp.argmax(jnp.abs(temp))]
#         a = n / d
#         xk -= a
#         return (xk,n,epsilon,a)
#     def cond_fun(carry):
#         xk,n,epsilon,a=carry
#         return ((jnp.abs(jnp.polyval(coff,xk)) > epsilon).all()&(jnp.abs(a) > epsilon).all())
#     carry=lax.while_loop(cond_fun,loop_body,(xk,n,epsilon,jnp.ones_like(x0)))
#     return carry[0]
# @jax.jit
# def zroots(coff,roots):
#     ##roots is the initial guess
#     #roots=AE_roots0(coff)
#     #polydiv_array = jnp.array([1, -1*roots[0]])  # 创建一个 jnp.array
#     def body_fun(carry, k):
#         coff, roots = carry
#         root_k = laguerre_method(coff, roots[k], 5 - k)  # order of polynomial
#         roots = roots.at[k].set(root_k)
#         polydiv_array = jnp.array([1, -1 * root_k])  # 创建一个 jnp.array
#         coff, _ = jnp.polydiv(coff, polydiv_array)
#         coff = jnp.concatenate([jnp.zeros(1),coff])  # number of coffs
#         return (coff, roots), None
#     carry,_=lax.scan(body_fun,(coff,roots),jnp.arange(coff.shape[0]-1))
#     '''for i in range(coff.shape[0]-1):
#         carry,_=body_fun((coff,roots),i)
#         coff,roots=carry'''
#     _,roots=carry
#     roots=newton_polish(coff,roots,10)
#     return roots
# @jax.jit
# def implict_zroots(coff,x0):
#     #x0=jnp.empty(coff.shape[0]-1,dtype=jnp.complex128)
#     f=lambda x:jnp.polyval(coff,x)
#     solution=lambda f,x0: zroots(coff,x0)
#     #solution=lambda f,x0: Aberth_Ehrlich(coff)
#     sclar=lambda g, y: jnp.linalg.solve(jax.jacobian(g,holomorphic=True)(y), y)
#     return lax.custom_root(f,x0,solve=solution,tangent_solve=sclar)
# # closed-form roots for quadratic, cubic, and quartic polynomials
# # multi_quadratic and multi_quartic adapted from https://github.com/NKrvavica/fqs
# # fast_cubic rewritten for complex polynomials
# # https://arxiv.org/abs/2207.12412
# # author: Keming Zhang
# @jax.jit
# def multi_quadratic(a0, b0, c0):
#     ''' Analytical solver for multiple quadratic equations
#     (2nd order polynomial), based on `numpy` functions.
#     Parameters
#     ----------
#     a0, b0, c0: array_like
#         Input data are coefficients of the Quadratic polynomial::
#             a0*x^2 + b0*x + c0 = 0
#     Returns
#     -------
#     r1, r2: ndarray
#         Output data is an array of two roots of given polynomials.
#     '''
#     ''' Reduce the quadratic equation to to form:
#         x^2 + ax + b = 0'''
#     a, b = b0 / a0, c0 / a0

#     # Some repating variables
#     a0 = -0.5*a
#     delta = a0*a0 - b
#     sqrt_delta = jnp.sqrt(delta + 0j)

#     # Roots
#     r1 = a0 - sqrt_delta
#     r2 = a0 + sqrt_delta

#     return jnp.concatenate((r1, r2))
# @jax.jit
# def fast_cubic(a, b, c, d):
#     d0 = b**2-3*a*c
#     d1 = 2*b**3-9*a*b*c+27*a**2*d
#     d2 = jnp.sqrt(d1**2-4*d0**3+0j)

#     d3 = d1-d2
#     mask = d2 == d1
#     d3=jnp.where(mask,d3+d2,d3)[0]
#     #d3[mask] += 2*d2[mask]

#     C = (d3/2)**(1/3)
#     d4 = d0/C
#     #d4[(C == 0)*(d0 == 0)] = 0
#     d4=jnp.where((C==0)&(d0==0),0,d4)[0]
#     pcru = (-1-(-3)**0.5)/2
#     #roots
#     x0 = -1/3/a*(b+C+d4)
#     x1 = -1/3/a*(b+C*pcru+d4/pcru)
#     x2 = -1/3/a*(b+C*pcru**2+d4/pcru**2)
#     return jnp.concatenate((x0, x1, x2))
# @jax.jit
# def multi_quartic(a0, b0, c0, d0, e0):
#     ''' Analytical closed-form solver for multiple quartic equations
#     (4th order polynomial), based on `numpy` functions. Calls
#     `multi_cubic` and `multi_quadratic`.
#     Parameters
#     ----------
#     a0, b0, c0, d0, e0: array_like
#         Input data are coefficients of the Quartic polynomial::
#             a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0
#     Returns
#     -------
#     r1, r2, r3, r4: ndarray
#         Output data is an array of four roots of given polynomials.
#     '''

#     ''' Reduce the quartic equation to to form:
#         x^4 ax^3 + bx^2 + cx + d = 0'''
#     a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

#     # Some repeating variables
#     a0 = 0.25*a
#     a02 = a0*a0

#     # Coefficients of subsidiary cubic euqtion
#     p = 3*a02 - 0.5*b
#     q = a*a02 - b*a0 + 0.5*c
#     r = 3*a02*a02 - b*a02 + c*a0 - d

#     # One root of the cubic equation
#     z0 = fast_cubic(1, p, r, p*r - 0.5*q*q)[0]

#     # Additional variables
#     s = jnp.sqrt(2*p + 2*z0 + 0j)
#     t = jnp.zeros_like(s)
#     mask = (jnp.abs(s) < 1e-8)
#     t=jnp.where(mask,z0*z0+r,-q/s)[0]
#     #t[mask] = z0[mask]*z0[mask] + r[mask]
#     #t[~mask] = -q[~mask] / s[~mask]

#     # Compute roots by quadratic equations
#     r01 = multi_quadratic(1, s, z0 + t) - a0
#     r23 = multi_quadratic(1, -s, z0 - t) - a0

#     return jnp.concatenate((r01, r23))
# @jax.jit
# def halfanalytical(coff):
#     roots=jnp.empty(coff.shape[0]-1,dtype=jnp.complex128)
#     roots=roots.at[0].set(laguerre_method(coff,0.,5))#order of polynomial
#     coff_4,_=jnp.polydiv(coff,jnp.array([1,-1*roots[0]]))
#     roots=roots.at[1:].set(multi_quartic(coff_4[0:1],coff_4[1:2],coff_4[2:3],coff_4[3:4],coff_4[4:]))
#     roots=newton_polish(coff,roots,10)
#     return roots
# @partial(jax.jit,static_argnums=2)
# def newton_polish(coff,roots,n=10):
#     derp=jnp.polyder(coff)
#     def loop_body(carry,k):
#         roots,coff=carry
#         val=jnp.polyval(coff,roots)
#         roots=roots-val/jnp.polyval(derp,roots)
#         return (roots,coff),k
#     carry,_=lax.scan(loop_body,(roots,coff),jnp.arange(n))
#     roots,_=carry
#     return roots


@jax.jit
def AE_roots0(coff: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the initial guesses using the Aberth-Ehrlich method. This code is adapted from [https://github.com/afoures/aberth-method](https://github.com/afoures/aberth-method)
    **Args**:

    - `coff` (ndarray): Coefficients of the polynomial.

    **Returns**:

    - `initial_guess`: Initial guesses for the roots of the polynomial.
    """

    def UV(coff):
        scale = jnp.max(jnp.abs(coff[:-1]))
        leading_coff = jnp.maximum(
            jnp.abs(coff[0]), jnp.sqrt(jnp.finfo(coff.real.dtype).eps) * scale
        )
        U = 1 + scale / leading_coff
        V = jnp.abs(coff[-1]) / (jnp.abs(coff[-1]) + scale)
        return U, V

    def Roots0(coff):
        U, V = UV(coff)
        r = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(coff.shape[0] - 1,), minval=V, maxval=U
        )
        phi = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(coff.shape[0] - 1,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        return r * jnp.exp(1j * phi)

    roots = Roots0(coff)
    return roots


@jax.jit
def Aberth_Ehrlich(
    coff: jnp.ndarray, roots: jnp.ndarray, MAX_ITER: int = 50
) -> jnp.ndarray:
    """
    Solves a polynomial equation using the Aberth-Ehrlich method. Adapted from [https://github.com/afoures/aberth-method](https://github.com/afoures/aberth-method).
    Use `jax.lax.custom_root` to get precise derivative in automatic differentiation.

    **Parameters**:

    - `coff`: Coefficients of the polynomial equation.
    - `roots`: Initial guesses for the roots of the polynomial equation.
    - `MAX_ITER`: Maximum number of iterations. Defaults to 100.

    **Returns**:

    - `roots`: The roots of the polynomial equation.

    """
    derp = jnp.polyder(coff)
    mask = 1 - jnp.eye(roots.shape[0])
    # alpha = jnp.abs(coff)*((2*jnp.sqrt(2))*1j+1)

    def loop_body(carry):
        roots, coff, cond, ratio_old, n_iter = carry
        # h = jnp.polyval(coff, roots)
        # b = jnp.polyval(alpha, jnp.abs(roots))
        ratio = jnp.polyval(coff, roots) / jnp.polyval(derp, roots)

        sum_term = jnp.nansum(mask * 1 / (roots - roots[:, None]), axis=0)
        w = ratio / (1 - (ratio * sum_term))
        cond = jnp.abs(w) > 2e-14
        # cond = jnp.abs(h) > 1e-15*b
        roots -= w
        return (roots, coff, cond, ratio, n_iter + 1)

    def cond_fun(carry):
        roots, coff, cond, ratio, n_iter = carry
        return cond.any() & (n_iter < MAX_ITER)

    f = lambda x: jnp.polyval(coff, x)
    solution = lambda f, x0: lax.while_loop(
        cond_fun, loop_body, (x0, coff, jnp.ones_like(x0, dtype=bool), x0, 0)
    )[0]
    sclar = lambda g, y: jnp.linalg.solve(jax.jacobian(g, holomorphic=True)(y), y)

    return lax.custom_root(f, roots, solve=solution, tangent_solve=sclar)


"""
if __name__=='__main__':
    inite=30;n_ite=400
    sample_n=120
    b_map =jnp.linspace(-4.0,3.0,sample_n)
    b=b_map[51]
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q=1e-3;s=1;rho=0.001
    times=jnp.linspace(t_0-0.*t_E,t_0+1.5*t_E,1)
    times=(times-t_0)/t_E
    alpha_rad=alphadeg*2*jnp.pi/360
    trajectory_l=get_trajectory_l(s,q,alpha_rad,b,times)
    m1=1/(1+q)
    m2=q/(1+q)
    sample_n=jnp.array([inite])
    theta=jnp.where(jnp.arange(n_ite)<inite,jnp.resize(jnp.linspace(0,2*jnp.pi,inite),n_ite),jnp.nan)[:,None]#shape(500,1)
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    coff=get_poly_coff(zeta_l,s,m2)[0]
    np_roots=jnp.roots(coff)
    print(np.poly1d(coff))
    lague_roots=implict_zroots(coff)
    print('numpy roots',np_roots)
    print('numpy error',jnp.abs(jnp.polyval(coff,np_roots)))
    print('laguerre roots',lague_roots)
    print('laguerre error',jnp.abs(jnp.polyval(coff,lague_roots)))
    print('numpy der',jax.jacfwd(jnp.roots,holomorphic=True)(coff))
    print('numpy error',jnp.abs(jnp.polyval(coff,jnp.sort(np_roots))))
    half=halfanalytical(coff)
    print('half roots',half)
    print('half error',jnp.abs(jnp.polyval(coff,half)))
    print('laguerre roots',lague_roots)
    print('laguerre der',jax.jacfwd(implict_zroots,holomorphic=True)(coff))
    print('laguerre error',jnp.abs(jnp.polyval(coff,jnp.sort(lague_roots))))#"""
