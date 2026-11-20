# Quick Start Guide

Calculate binary-lens magnification, then add annual parallax or orbital motion.

## Installation

Install microlux:

```bash
pip install microlux
```

Or install from source:

```bash
git clone https://github.com/CoastEgo/microlux.git
cd microlux
pip install -e .
```

## Example 1: Basic Binary Magnification

```python
import jax.numpy as jnp
from microlux import binary_mag

# t0, u0, tE, rho, q, s, alpha in degrees
params = jnp.array([0.0, 0.1, 1.0, 0.01, 0.2, 0.9, 270.0])
times = jnp.linspace(-1.0, 1.0, 1000)

mag = binary_mag(*params, times, tol=1e-3, retol=1e-3)
```

## Example 2: Trajectory + Annual Parallax

```python
import jax.numpy as jnp
from microlux.coordinates import Coordinates
from microlux.trajectory import TrajectoryParameters, get_trajectory_model
from microlux import extended_light_curve

# t0, u0, tE, rho, alpha in radians, s, q, pi_E_N, pi_E_E
params = jnp.array([
    2460000.0, 0.1, 100.0, 1e-3, jnp.deg2rad(120.0), 0.9, 1e-3, 0.1, 0.1,
])
params = TrajectoryParameters(*params)

times = jnp.linspace(params.t0 - 2.0 * params.tE, params.t0 + 2.0 * params.tE, 500)
coords = Coordinates(ra="17:59:02.3", dec="-29:04:15.2")
traj_model = get_trajectory_model(times=times, coords=coords, t0_par=params.t0)

trajectory, separation = traj_model.calculate_trajectory(params)
mag = extended_light_curve(trajectory, separation, params.q, params.rho)
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

# t0, u0, tE, rho, alpha in radians, s, q, pi_E_N, pi_E_E, gamma1, gamma2, gamma3
params = jnp.array([
    2460000.0, 0.1, 100.0, 0.01, 1.2, 1.3, 0.2,
    0.1, -0.1, 0.001, 0.002, 0.003,
])
params = TrajectoryParameters(*params)
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
