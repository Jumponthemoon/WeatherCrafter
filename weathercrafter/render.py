"""Particle projection, compositing, and video assembly.

Projects the simulated particle field into each frame and assembles the RGB /
depth output videos (geometry grounding, paper Sec. 3.3).

Note on occlusion: the particle depth is blended into the scene depth as a
modulation term rather than resolved with a z-buffer test, so occlusion is left
for the video model to interpret. The two depth maps are not on a comparable
scale anyway -- the scene depth is per-frame percentile-normalised *inverse*
depth from DA3, while particle depth is linear over [Z_MIN, Z_MAX].
"""
import glob
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .constants import DEFAULT_DEPTH_SCALE, DEFAULT_FPS, DEFAULT_GRAVITY, DEFAULT_SNOW_COUNT, \
    DEFAULT_VEL_RANGES, DEFAULT_WIND_STRENGTH, Z_MAX, Z_MIN
from .geometry import align_direction, compute_camera_movement, extract_camera_from_npz, \
    extract_gravity_ransac
from .simulation import KERNEL_PARAMS, Snowflake, gaussian_kernel, spawn_flake, update_flakes


def velocity_to_image_angle(f: Snowflake, fx: float, fy: float) -> float:
    """Estimate the image-plane motion direction (degrees) from a 3D velocity."""
    # First-order approximation under perspective projection.
    du = fx * (f.vx / f.Z)
    dv = fy * (f.vy / f.Z)

    # Image coordinates: x to the right, y downwards.
    angle = np.degrees(np.arctan2(du, dv))
    return angle


def render_snow_frame(base_img, flakes, K, weather_type, depth_scale=DEFAULT_DEPTH_SCALE,
                      Z_max=Z_MAX) -> tuple:
    """Project and alpha-composite the particles over ``base_img``.

    Returns the composited RGBA image and a per-pixel particle depth map.
    """
    img = base_img.copy()
    W, H = img.size

    snow_layer = Image.new("RGBA", (W, H), (255, 255, 255, 0))

    # Per-pixel particle depth, initialised to the far plane.
    depth_map = np.ones((H, W), dtype=np.float32) * Z_max

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for f in flakes:

        u = int(fx * (f.X / f.Z) + cx)
        v = int(fy * (f.Y / f.Z) + cy)

        if f.prev_u is None:
            angle = velocity_to_image_angle(f, fx, fy)
        else:
            du = u - f.prev_u
            dv = v - f.prev_v
            angle = np.degrees(np.arctan2(du, dv))

        f.prev_u = u
        f.prev_v = v

        if not (0 <= u < W and 0 <= v < H):
            continue

        # Closer flakes (smaller Z) get a larger kernel.
        sigma = int(200 / f.Z)**1.5 + 1
        size = int(sigma * 10)

        kernel = gaussian_kernel(size, sigma, angle, sigma_x=f.sigma_x, sigma_y=f.sigma_y,
                                 noise_seed=f.noise_seed, **KERNEL_PARAMS[weather_type])

        depth_alpha = np.clip((1 - f.Z / Z_max), 0.1, 1.0)
        alpha = (kernel * 255 * depth_alpha).astype(np.uint8)

        rgb = np.ones((size, size, 3), dtype=np.uint8) * 255
        rgba = np.dstack([rgb, alpha])
        snow_img = Image.fromarray(rgba, mode="RGBA")

        u0 = u - size // 2
        v0 = v - size // 2

        # Composite this flake into the particle depth map (vectorized over the
        # stamp's visible region; identical alpha-blend to the per-pixel loop).
        y0, y1 = max(v0, 0), min(v0 + size, H)
        x0, x1 = max(u0, 0), min(u0 + size, W)
        if y1 > y0 and x1 > x0:
            a = kernel[y0 - v0:y1 - v0, x0 - u0:x1 - u0] * depth_scale  # Gaussian alpha
            region = depth_map[y0:y1, x0:x1]
            depth_map[y0:y1, x0:x1] = region * (1 - a) + f.Z * a

        snow_layer.alpha_composite(snow_img, (u0, v0))

    return Image.alpha_composite(img, snow_layer), depth_map


# ==========================
#   SIMULATION SETUP (shared)
# ==========================
@dataclass
class SimContext:
    """Everything the renderers need after loading geometry and spawning the field."""
    K: np.ndarray
    R_t: np.ndarray
    W: int
    H: int
    fov_x: float
    fov_y: float
    R_align: np.ndarray
    flakes: list
    images_folder: str
    images_list: list
    dt: float


def _setup_simulation(data_root, snow_count, vel_ranges, fps) -> SimContext:
    """Load camera/geometry, compute FOV, align gravity, and spawn the particle field.

    Shared by the video and edited-first-frame renderers so the (seeded) spawn
    order is identical between them.
    """
    time.sleep(3)
    dt = 1.0 / fps
    geometry = os.path.join(data_root, "geometry")
    npz_path = os.path.join(geometry, "exports/mini_npz/results.npz")
    print(npz_path)
    K, R_t = extract_camera_from_npz(npz_path)

    images_folder = os.path.join(geometry, "images")
    images_list = sorted([f for f in os.listdir(images_folder)
                          if f.endswith('.png') or f.endswith('.jpg')])
    first_image = Image.open(images_folder + "/" + images_list[0]).convert("RGBA")
    W, H = first_image.size

    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2 * math.atan(W / (2 * fx))
    fov_y = 2 * math.atan(H / (2 * fy))

    gravity_direction = extract_gravity_ransac(f"{geometry}/scene.glb")
    R_align = align_direction(gravity_direction)

    flakes = [spawn_flake(fx, fy, W, H, R_align, vel_ranges=vel_ranges) for _ in range(snow_count)]

    return SimContext(K, R_t, W, H, fov_x, fov_y, R_align, flakes, images_folder, images_list, dt)


# ==========================
#   RENDER VIDEO FOLDER (with DEPTH)
# ==========================
def render_snow_video_folder(data_root, output_folder, target_weather, weather_type,
                             turb_strength, wind_strength=DEFAULT_WIND_STRENGTH,
                             wind_dir=None, gravity=DEFAULT_GRAVITY,
                             snow_count=DEFAULT_SNOW_COUNT, vel_ranges=DEFAULT_VEL_RANGES,
                             depth_scale=DEFAULT_DEPTH_SCALE, fps=DEFAULT_FPS,
                             Zmin=0, Zmax=Z_MAX) -> None:
    """Simulate and render the particle field over every frame, writing RGB and
    particle-depth image folders plus an output video."""
    ctx = _setup_simulation(data_root, snow_count, vel_ranges, fps)

    # Create separate folders for RGB and depth images.
    rgb_folder = os.path.join(output_folder, "rgb_frames")
    depth_folder = os.path.join(output_folder, f"depth_particles_{target_weather}")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    print("Video resolution:", ctx.W, ctx.H)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(output_folder, f'output_video_{weather_type}.mp4')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (ctx.W, ctx.H))

    frames = len(ctx.images_list)
    for frame in range(frames):
        print("Rendering frame:", frame)
        image = Image.open(ctx.images_folder + "/" + ctx.images_list[frame]).convert("RGBA")
        cam_movement = compute_camera_movement(ctx.R_t, frame) if frame > 0 else (np.eye(3), np.zeros(3))
        update_flakes(ctx.flakes, ctx.dt, ctx.fov_x, ctx.fov_y, cam_movement, turb_strength,
                      wind_strength=wind_strength, wind_dir=wind_dir, g=gravity)
        out, depth_map = render_snow_frame(image, ctx.flakes, ctx.K, weather_type, depth_scale)

        # Save RGB image to the RGB folder.
        rgb_path = os.path.join(rgb_folder, f"{frame:04d}.png")
        out.convert("RGB").save(rgb_path)

        # Save depth image to the depth folder.
        depth_map = (depth_map - Zmin) / (Zmax - Zmin)
        depth_map = 1.0 - depth_map  # reverse: near=white, far=black
        depth_map = np.clip(depth_map, 0, 1)
        depth_path = os.path.join(depth_folder, f"{frame:04d}.png")
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(depth_path, (depth_map * 255).astype(np.uint8))

        # Write RGB frame to the video.
        frame_cv = cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGR)
        video_writer.write(frame_cv)

    video_writer.release()
    print(f"Video saved as {video_path}")
    print(f"RGB images saved in: {rgb_folder}")
    print(f"Depth images saved in: {depth_folder}")


def render_snow_edited_first_frame(data_root, target_weather, weather_type, turb_strength,
                                   wind_strength=DEFAULT_WIND_STRENGTH, wind_dir=None,
                                   gravity=DEFAULT_GRAVITY, snow_count=DEFAULT_SNOW_COUNT,
                                   vel_ranges=DEFAULT_VEL_RANGES,
                                   depth_scale=DEFAULT_DEPTH_SCALE, fps=DEFAULT_FPS) -> None:
    """Render the particle field onto the weather-edited first frame (extrinsics unused;
    the first frame uses an identity camera to match the video renderer's frame 0)."""
    ctx = _setup_simulation(data_root, snow_count, vel_ranges, fps)

    # Render the particles onto the weather-edited first frame.
    images_folder = os.path.join(data_root, "edited_first_frame")
    path = images_folder + f"/0000_{target_weather}.png"
    image = Image.open(path).convert("RGBA")

    # Identity camera step to match the video renderer's frame 0, so the
    # first-frame particles (VACE reference) align with the control video's frame 0.
    cam_movement = (np.eye(3), np.zeros(3))
    update_flakes(ctx.flakes, ctx.dt, ctx.fov_x, ctx.fov_y, cam_movement, turb_strength,
                  wind_strength=wind_strength, wind_dir=wind_dir, g=gravity)

    out, _ = render_snow_frame(image, ctx.flakes, ctx.K, weather_type, depth_scale)
    save_path = path.replace('edited_first_frame', 'edited_first_frame_with_particles')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(save_path)
    out.convert("RGB").save(save_path)


# ==========================
#   DEPTH MERGE + VIDEO ASSEMBLY
# ==========================
def merge_depth_and_assemble_videos(dataset_name, target_weather, fps=DEFAULT_FPS) -> None:
    """Blend particle depth over scene depth and assemble the depth/mask videos."""
    background_dir = f"output/{dataset_name}/geometry/depth_vis"
    foreground_dir = f"output/{dataset_name}/depth_particles_{target_weather}"
    output_dir = f"output/{dataset_name}/merged_depth_{target_weather}"
    num_frames = len(glob.glob(os.path.join(foreground_dir, "*.png")))
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_frames):
        filename = f"{i:04d}.png"

        fg_path = os.path.join(foreground_dir, filename)
        bg_path = os.path.join(background_dir, filename)
        out_path = os.path.join(output_dir, filename)

        D_fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        D_bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        if D_fg is None or D_bg is None:
            print(f"Missing: {filename}, skipping.")
            continue

        if D_fg.shape != D_bg.shape:
            print(f"Size mismatch: {filename}")
            continue

        # Blend particle depth over the scene depth (small weight keeps the scene dominant).
        D_out = D_fg * 0.12 + D_bg
        cv2.imwrite(out_path, D_out)

    print("Done!")

    video_merged_depth_images = sorted(glob.glob(f'output/{dataset_name}/merged_depth_{target_weather}/*.png'))
    video_pure_depth_images = sorted(glob.glob(f'output/{dataset_name}/geometry/depth_vis/*.png'))
    video_particle_depth_images = sorted(glob.glob(f'output/{dataset_name}/depth_particles_{target_weather}/*.png'))

    if not video_merged_depth_images:
        print("No merged depth frames to assemble; skipping video writing.")
        return

    frame = cv2.imread(video_merged_depth_images[0])
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_pure_depth = cv2.VideoWriter(f'output/{dataset_name}/{dataset_name}_pure_depth.mp4', fourcc, fps, (width, height))
    video_particles_depth = cv2.VideoWriter(f'output/{dataset_name}/{dataset_name}_particles_depth.mp4', fourcc, fps, (width, height))
    video_merged_depth = cv2.VideoWriter(f'output/{dataset_name}/{dataset_name}_merged_depth_{target_weather}.mp4', fourcc, fps, (width, height))
    video_mask = cv2.VideoWriter(f'output/{dataset_name}/{dataset_name}_mask.mp4', fourcc, fps, (width, height))

    for img in video_merged_depth_images:
        video_merged_depth.write(cv2.imread(img))
        video_mask.write(np.ones((height, width, 3), dtype=np.uint8)*255)

    for img in video_pure_depth_images:
        video_pure_depth.write(cv2.imread(img))

    for img in video_particle_depth_images:
        video_particles_depth.write(cv2.imread(img))

    video_pure_depth.release()
    video_particles_depth.release()
    video_merged_depth.release()
    video_mask.release()
