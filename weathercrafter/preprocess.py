"""Preprocessing: extract frames from a dataset video and resize them.

Reads   data/<dataset_name>/<dataset_name>.mp4
Writes  data/<dataset_name>/images/%04d.png   (resized in place)

If the images directory is already populated, extraction is skipped and the
existing frames are simply (re)resized.
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def resize_shortest_side(img: Image.Image, target_size: int) -> Image.Image:
    """Resize so the shortest side equals ``target_size``, preserving aspect ratio."""
    w, h = img.size
    shortest = min(w, h)
    if shortest == target_size:
        return img

    scale = target_size / float(shortest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interp)
    return Image.fromarray(arr)


def resize_directory(dir_path: str, target_size: int) -> None:
    """Resize every image in ``dir_path`` in place."""
    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        path = os.path.join(dir_path, name)
        resize_shortest_side(Image.open(path), target_size).save(path)


def extract_frames(input_video: str, output_dir: str, fps: int, max_frames: int) -> None:
    """Extract up to ``max_frames`` frames from ``input_video`` at ``fps`` via ffmpeg."""
    subprocess.run([
        "ffmpeg", "-i", input_video,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        f"{output_dir}/%04d.png",
    ], check=True)


def run(dataset_name: str, fps: int = 15, sec: int = 5, target_size: int = 504) -> None:
    """Extract (if needed) and resize the frames for ``dataset_name``."""
    input_video = f"data/{dataset_name}/{dataset_name}.mp4"
    output_dir = f"data/{dataset_name}/images"
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if existing:
        print(f"Found {len(existing)} existing images - skipping extraction")
    else:
        print("No images found - extracting frames from video...")
        extract_frames(input_video, output_dir, fps, max_frames=fps * sec)
        print("Frame extraction complete")

    resize_directory(output_dir, target_size)
    print("All images resized")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second to extract from the video.")
    parser.add_argument("--sec", type=int, default=5, help="Duration in seconds to process.")
    parser.add_argument("--target_size", type=int, default=504,
                        help="Target size for the shortest side of the resized images.")


def main(args: argparse.Namespace) -> None:
    run(args.dataset_name, fps=args.fps, sec=args.sec, target_size=args.target_size)
