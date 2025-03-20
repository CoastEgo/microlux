import matplotlib.pyplot as plt
import numpy as np

# from microlux import model_numpy
from microlux import binary_mag, to_centroid, to_lowmass
from MulensModel import CausticsBinary


# deprecated function

# def contour_plot_numpy(parms):
#     ## function of numpy version , you can use it to plot the image contour
#     model_instance=model_numpy(parms)

#     mag=model_instance.get_magnifaction2(parms['retol'],retol=parms['retol'])
#     image_contour=model_instance.image_contour_all[0]

#     theta=np.linspace(0,2*np.pi,100)
#     trajectory_l=model_instance.trajectory_l
#     zeta=model_instance.to_centroid(model_instance.get_zeta_l(trajectory_l,theta))
#     plt.figure(figsize=(6,6))
#     plt.plot(zeta.real,zeta.imag,color='r',linewidth=0.15)
#     caustic_1=caustics.Caustics(q,s)
#     caustic_1.plot(5000,s=0.5)
#     x,y=caustic_1.get_caustics()
#     x=caustic_1.critical_curve.x
#     y=caustic_1.critical_curve.y
#     plt.scatter(x,y,s=0.005)
#     plt.axis('equal')

#     for k in range(len(image_contour)):
#         cur=model_instance.to_centroid(image_contour[k])
#         plt.plot(cur.real,cur.imag)

#     plt.savefig('picture/image_contours.png')


def contour_plot(t_0, b, t_E, rho, q, s, alphadeg, times, retol=1e-3, tol=1e-3):
    alpha = alphadeg * 2 * np.pi / 360

    mag, info = binary_mag(
        t_0,
        b,
        t_E,
        rho,
        q,
        s,
        alphadeg,
        times,
        retol=retol,
        tol=tol,
        return_info=True,
        analytic=False,
    )
    theta = info[-2].theta[0]

    trajectory_c = (times - t_0) / t_E * np.exp(1j * alpha) + 1j * b * np.exp(
        1j * alpha
    )
    source_l = to_lowmass(s, q, trajectory_c) + rho * np.exp(1j * theta)
    source_c = to_centroid(s, q, source_l)

    plt.figure(figsize=(6, 6))
    plt.scatter(source_c.real, source_c.imag, color="r", s=0.5)
    caustic_1 = CausticsBinary(q, s)
    caustic_1.plot(5000, s=0.5)
    x, y = caustic_1.get_caustics()
    x = caustic_1.critical_curve.x
    y = caustic_1.critical_curve.y
    plt.scatter(x, y, s=0.005)
    plt.axis("equal")

    roots_l = info[-2].roots
    roots_c = to_centroid(s, q, roots_l)
    plt.scatter(roots_c.real, roots_c.imag, s=0.5)
    plt.savefig("picture/image_contours2.png")


if __name__ == "__main__":
    b = 0.1
    t_0 = 8280.094505
    t_E = 39.824343
    alphadeg = 270
    q = 0.2
    s = 0.9
    rho = 10 ** (-1)
    trajectory_n = 1000
    idy = 500
    times = np.linspace(t_0 - 1.0 * t_E, t_0 + 1.0 * t_E, trajectory_n)[idy : idy + 1]

    parms = {
        "t_0": t_0,
        "u_0": 0.1,
        "t_E": t_E,
        "rho": rho,
        "q": q,
        "s": s,
        "alpha_deg": alphadeg,
        "times": times,
        "retol": 1e-3,
    }
    # contour_plot_numpy(parms)
    contour_plot(t_0, b, t_E, rho, q, s, alphadeg, times, retol=1e-3, tol=1e-3)
