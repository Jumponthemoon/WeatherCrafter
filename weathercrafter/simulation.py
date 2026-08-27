"""Physics-informed particle simulation (paper Sec. 3.2).

Anisotropic Gaussian particles advected under gravity + wind + curl-noise
turbulence. The simulation is seeded (see ``constants.SEED``); the seeded RNG
draw order is deliberately preserved (including a few no-op draws) so the
particle field is stable across runs.
"""
import math
import random

import numpy as np
from noise import pnoise3   # pip install noise
from PIL import Image

from .constants import DEFAULT_GRAVITY, DEFAULT_VEL_RANGES, DEFAULT_WIND_STRENGTH, Z_MAX, Z_MIN


# ==========================
#   NOISE HELPERS
# ==========================
def perlin3(x: float, y: float, z: float, scale: float) -> float:
    return pnoise3(x * scale, y * scale, z * scale)


def _vector_potential(x: float, y: float, z: float, scale: float) -> tuple:
    """Vector potential psi from a 3D noise field (paper Eq. 6):
    psi = (o(x,y,z), o(y,z,x), o(z,x,y)). Permuting the coordinates yields three
    decorrelated components, so the resulting curl is an isotropic, divergence-free field."""
    return (
        perlin3(x, y, z, scale),
        perlin3(y, z, x, scale),
        perlin3(z, x, y, scale),
    )


def curl_noise(x: float, y: float, z: float, _t: float, scale: float = 0.05, eps: float = 1e-3) -> np.ndarray:
    """Divergence-free turbulence n = curl(psi) with psi from Eq. 6 (paper Eq. 5).

    Note: ``_t`` is accepted for parity with the paper's time-varying field n(x, t)
    but is currently unused - the implemented field is static in time.
    """
    # Central finite differences of each psi component along x, y, z.
    px = _vector_potential(x + eps, y, z, scale)
    mx = _vector_potential(x - eps, y, z, scale)
    py = _vector_potential(x, y + eps, z, scale)
    my = _vector_potential(x, y - eps, z, scale)
    pz = _vector_potential(x, y, z + eps, scale)
    mz = _vector_potential(x, y, z - eps, scale)

    dpsi_dx = [(px[i] - mx[i]) / (2 * eps) for i in range(3)]
    dpsi_dy = [(py[i] - my[i]) / (2 * eps) for i in range(3)]
    dpsi_dz = [(pz[i] - mz[i]) / (2 * eps) for i in range(3)]

    # n = curl(psi) = (dpsi3/dy - dpsi2/dz, dpsi1/dz - dpsi3/dx, dpsi2/dx - dpsi1/dy)
    return np.array([
        dpsi_dy[2] - dpsi_dz[1],
        dpsi_dz[0] - dpsi_dx[2],
        dpsi_dx[1] - dpsi_dy[0],
    ])


# ==========================
#   PARTICLE KERNELS
# ==========================
# Per-weather stamp parameters. ``prenormalize`` reproduces rain's extra
# normalize of the raw Gaussian before the noise is applied (snow skips it).
KERNEL_PARAMS = {
    "snow": {"opacity": 255, "scaling_factor": (0.5, 3.0), "prenormalize": False},
    "rain": {"opacity": 110, "scaling_factor": (0.15, 10.0), "prenormalize": True},
}


def gaussian_kernel(size, sigma, angle, sigma_x, sigma_y, noise_seed,
                    opacity, scaling_factor, prenormalize) -> np.ndarray:
    """Anisotropic Gaussian particle stamp modulated by fixed per-flake noise.

    snow -> soft and roundish; rain -> elongated streak. The two differ only in
    ``opacity`` / ``scaling_factor`` and whether the raw Gaussian is normalized
    before the noise is applied (``prenormalize``); see ``KERNEL_PARAMS``.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    sx, sy = scaling_factor

    kernel = np.exp(-(xx**2 / (2 * (sigma_x**2) * sigma * sx) +
                      yy**2 / (2 * (sigma_y**2) * sigma * sy)))
    if prenormalize:
        kernel /= (kernel.max() + 1e-8)

    rng = np.random.RandomState(noise_seed)   # fixed per-flake noise
    noise = rng.rand(size, size)
    kernel = kernel * (0.8 + 0.4 * noise)
    kernel /= (kernel.max() + 1e-8)

    kernel_img = Image.fromarray((kernel * opacity).astype(np.uint8))
    kernel_img = kernel_img.rotate(angle, resample=Image.BICUBIC)

    return np.array(kernel_img).astype(np.float32) / 255.0


# ==========================
#   PARTICLE STATE + DYNAMICS
# ==========================
class Snowflake:
    """State of a single particle (position, velocity, and per-flake stamp parameters)."""

    def __init__(self, X, Y, Z, vx, vy, vz, angle_offset, sigma_x, sigma_y, noise_seed):
        self.X, self.Y, self.Z = X, Y, Z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.angle_offset = angle_offset
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.noise_seed = noise_seed
        self.turb_phase = random.uniform(100, 100)
        self.prev_u = None
        self.prev_v = None


def spawn_flake(fx, fy, W, H, R_align, Z_max=Z_MAX, vel_ranges=DEFAULT_VEL_RANGES) -> Snowflake:
    """Sample a particle uniformly within the view frustum, with gravity-aligned velocity."""
    fov_x = 2 * math.atan(W / (2 * fx))
    fov_y = 2 * math.atan(H / (2 * fy))

    maxX = Z_max * math.tan(fov_x / 2)
    maxY = Z_max * math.tan(fov_y / 2)

    X = random.uniform(-maxX, maxX)
    Y = random.uniform(-maxY, maxY)
    Z = random.uniform(Z_MIN, Z_max)

    v = np.array([random.uniform(*vel_ranges["vx"]),
                  random.uniform(*vel_ranges["vy"]),
                  random.uniform(*vel_ranges["vz"])])
    # Rotate the initial velocity into the gravity-aligned frame.
    v = R_align @ v

    # sigma / angle_offset are kept fixed; the random.uniform(0, 0) draw is retained
    # so the seeded RNG stream (and thus the particle field) stays reproducible.
    sigma_x = 1
    sigma_y = 1
    noise_seed = random.randint(0, 2**31 - 1)
    angle_offset = random.uniform(0, 0)

    return Snowflake(X, Y, Z, v[0], v[1], v[2], angle_offset, sigma_x, sigma_y, noise_seed)


def update_flakes(flakes, dt, fov_x, fov_y, cam_movement, turb_strength,
                  wind_strength=DEFAULT_WIND_STRENGTH, wind_dir=None,
                  g=DEFAULT_GRAVITY, Zmin=Z_MIN, Z_max=Z_MAX) -> None:
    """Advance every particle one timestep under turbulence + wind + gravity, then
    wrap at the frustum bounds and compensate for camera motion."""
    R_cam, t_cam = cam_movement

    # ===== physics parameters (minimal, explicit) =====
    m = 1.0                                # particle mass
    if wind_dir is None:
        wind_dir = np.array([1.0, 0.0, 0.0])   # default wind along +x
    gravity_dir = np.array([0.0, 1, 0.0])  # gravity acceleration direction (fixed canonical axis)
    for f in flakes:

        turb = curl_noise(
            f.X, f.Y, f.Z,
            f.turb_phase,
            scale=0.3
        )
        F_turb = turb_strength * turb

        v = np.array([f.vx, f.vy, f.vz])
        F_wind = wind_strength * wind_dir

        F_g = m * g * gravity_dir
        F = F_turb + F_wind + F_g
        a = F / m

        # velocity update
        v = v + a * dt
        f.vx, f.vy, f.vz = v

        # position update
        f.X += f.vx * dt
        f.Y += f.vy * dt
        f.Z += f.vz * dt

        f.turb_phase += dt

        # 4. Wrap-around at the frustum bounds.
        maxX = Z_max * math.tan(fov_x / 2)
        maxY = Z_max * math.tan(fov_y / 2)

        if f.X > maxX: f.X = -maxX
        if f.X < -maxX: f.X = maxX
        if f.Y > maxY: f.Y = -maxY
        if f.Y < -maxY: f.Y = maxY
        if f.Z > Z_max: f.Z = Zmin
        if f.Z < Zmin: f.Z = Z_max

        # 5. Camera-motion compensation (3x translation gain kept as tuned).
        f.X, f.Y, f.Z = R_cam @ np.array([f.X, f.Y, f.Z]) + 3 * t_cam
