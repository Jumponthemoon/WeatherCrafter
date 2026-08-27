"""Geometry estimation and grounding (paper Sec. 3.3).

Depth Anything 3 depth/camera estimation, plus the gravity-direction and
camera-motion helpers used to ground the particle field in scene geometry.
"""
import glob
import os
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .constants import DEFAULT_DURATION, DEFAULT_FPS


# ==========================
#   GRAVITY / CAMERA HELPERS
# ==========================
def extract_gravity_ransac(path: str) -> Optional[np.ndarray]:
    """Extract the gravity direction from a 3D scene via RANSAC plane fitting.

    Args:
        path: Path to the 3D scene file (.glb).

    Returns:
        Normalized gravity direction vector (plane normal), or None if no plane is found.
    """
    scene = trimesh.load(path)

    points_list = []
    for g in scene.geometry.values():
        if isinstance(g, trimesh.Trimesh):
            points_list.append(g.vertices)
        elif isinstance(g, trimesh.points.PointCloud):
            points_list.append(g.vertices)

    points = np.vstack(points_list)

    # RANSAC plane fitting (keep the best plane over several attempts).
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    best_inliers = []
    best_normal = None

    for _ in range(30):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.5,
            ransac_n=3,
            num_iterations=300
        )

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])

    if best_normal is None:
        print("No valid RANSAC plane found")
        return None

    print("Best RANSAC plane normal (gravity direction):", best_normal)
    return best_normal


def extract_camera_from_npz(npz_path: str) -> tuple:
    """Load the (mean) camera intrinsics K and per-frame extrinsics from an export npz."""
    loaded = np.load(npz_path, allow_pickle=False)
    K = np.mean(loaded["intrinsics"], axis=0)
    R_t = loaded["extrinsics"]
    return K, R_t


def align_direction(target_direction: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the canonical up axis [0,1,0] onto ``target_direction``."""
    rotation_axis = np.cross([0, 1, 0], target_direction)
    cos_theta = np.clip(np.dot([0, 1, 0], target_direction), -1, 1)
    theta = np.arccos(cos_theta)

    axis = rotation_axis / np.linalg.norm(rotation_axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def compute_camera_movement(R_t, frame_idx) -> tuple:
    """Relative camera rotation/translation between consecutive frames."""
    R_curr = R_t[frame_idx-1][:3, :3]
    t_curr = R_t[frame_idx-1][:3, 3]

    R_prev = R_t[frame_idx][:3, :3]
    t_prev = R_t[frame_idx][:3, 3]

    R_rel = R_curr @ R_prev.T
    t_rel = t_curr - R_rel @ t_prev

    return R_rel, t_rel


# ==========================
#   DEPTH INFERENCE (DA3)
# ==========================
def infer_depth(dataset_name, fps=DEFAULT_FPS, duration=DEFAULT_DURATION) -> None:
    """Run Depth Anything 3 over the dataset frames and export depth / camera / point cloud."""
    import torch
    from depth_anything_3.api import DepthAnything3

    frames = fps * duration  # process `duration` seconds of video

    # -------------------------------------------------------
    # Load images
    # -------------------------------------------------------
    image_dir = f"data/{dataset_name}/images"

    image_list = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )

    image_list = image_list[:frames]

    if len(image_list) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large").to(device)

    # -------------------------------------------------------
    # Run inference
    # -------------------------------------------------------
    prediction = model.inference(
        image=image_list,
        export_dir=f"./output/{dataset_name}/geometry",
        export_format="mini_npz-glb",
        conf_thresh_percentile=30.0,
        show_cameras=True,
        process_res_method="lower_bound_resize")

    # -------------------------------------------------------
    # Save processed images
    # -------------------------------------------------------
    processed_images_path = f'output/{dataset_name}/geometry/images'
    os.makedirs(processed_images_path, exist_ok=True)

    for i in range(prediction.processed_images.shape[0]):
        img = prediction.processed_images[i]

        # Convert to 0-255 uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(processed_images_path, f'{i:04d}.png'), img)
        print(f"Saved processed image: {i:04d}.png")

    # -------------------------------------------------------
    # Save depth maps
    # -------------------------------------------------------
    depth = prediction.depth  # (N, H, W) float32 numpy array

    save_path = f'output/{dataset_name}/geometry/depths'
    os.makedirs(save_path, exist_ok=True)

    for i in range(depth.shape[0]):
        d = depth[i]

        # Normalize depth to 0-1 then to 0-255
        depth_pt = d.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_pt = 1.0 - depth_pt  # invert so closer = brighter

        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        # Convert to 3-channel RGB
        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)

        # Save colored visualization
        cv2.imwrite(os.path.join(save_path, f'{i:04d}.png'), depth_image)

        print(f"Saved depth: {i:04d}.png")
