# Quick Start Guide

Calculate binary-lens magnification, then add annual parallax or orbital motion.

## Example 1: Basic Binary Magnification

```python
import jax.numpy as jnp
from microlux import binary_mag

t_0 = 0.0
u_0 = 0.1
t_E = 1.0
rho = 1e-2
q = 0.2
s = 0.9
alpha_deg = 270.0
times = jnp.linspace(t_0 - 1.0 * t_E, t_0 + 1.0 * t_E, 1000)

mag = binary_mag(t_0, u_0, t_E, rho, q, s, alpha_deg, times, tol=1e-3, retol=1e-3)
```

## Example 2: Trajectory + Annual Parallax

```python
import jax.numpy as jnp
import numpy as np
from microlux.coordinates import Coordinates
from microlux.trajectory import TrajectoryParameters, get_trajectory_model
from microlux import extended_light_curve

t0 = 2460000.0
tE = 100.0
u0 = 0.1
rho = 1e-3
q = 1e-3
s = 0.9
alpha_deg = 120.0
piEN = 0.1
piEE = 0.1

times = jnp.linspace(t0 - 2.0 * tE, t0 + 2.0 * tE, 500)
coords = Coordinates(ra="17:59:02.3", dec="-29:04:15.2")
traj_model = get_trajectory_model(times=times, coords=coords, t0_par=t0)

params = TrajectoryParameters(
    t0=t0,
    u0=u0,
    tE=tE,
    rho=rho,
    alpha_rad=alpha_deg * 2.0 * np.pi / 360.0,
    s=s,
    q=q,
    pi_E_N=piEN,
    pi_E_E=piEE,
)

trajectory, separation = traj_model.calculate_trajectory(params)
mag = extended_light_curve(trajectory, separation, q, rho)
```

## Example 3: Circular Orbital Motion, with Optional Parallax

The same trajectory model supports the effects independently:

| Model | `t0_par` | `orbital_motion` |
| --- | --- | --- |
| Static | `None` | `None` |
| Annual parallax | Reference epoch | `None` |
| Circular orbit | `None` | `"circular"` |
| Annual parallax + circular orbit | Reference epoch | `"circular"` |

```python
import jax.numpy as jnp
from microlux import (
    Coordinates, TrajectoryParameters, get_trajectory_model, extended_light_curve,
)

params = TrajectoryParameters(
    t0=2460000.0, u0=0.1, tE=100.0, rho=0.01,
    alpha_rad=1.2, s=1.3, q=0.2,
    pi_E_N=0.1, pi_E_E=-0.1,
    gamma1=0.001, gamma2=0.002, gamma3=0.003,
)
times = jnp.linspace(params.t0 - 100.0, params.t0 + 100.0, 161)
model = get_trajectory_model(
    times,
    coords=Coordinates(ra="17:59:02.3", dec="-29:04:15.2"),
    t0_par=2459983.0,
    orbital_motion="circular",
    photo_center=True,
)
trajectory, separation = model.calculate_trajectory(params)
mag = extended_light_curve(trajectory, separation, params.q, params.rho)
```

`trajectory` and `separation` are arrays of shape `(N,)`, giving the source
position and lens separation at each observation. Pass both to
`extended_light_curve`; without orbital motion, separation is constant.
