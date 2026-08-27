"""Assemble a per-run folder for the video-editing (VACE) stage.

Reorganizes the scattered stage outputs under ``output/<dataset>/`` into:
    output/<dataset>/<weather>_a_<appStage>_p_<partSeverity>/
        intermediate_results/   - the raw stage outputs, incl. geometry/ (moved here)
        preprocessed_results/   - the VACE-ready canonical files (control_depth.mp4,
                                  reference.png, mask.mp4, prompt.txt, info.json, ...)

Geometry (depth/camera) is moved into the run's intermediate_results/ too, so each
run folder is self-contained. Missing inputs are skipped, so a partial pipeline
still packages what exists.
"""
import json
import os
import shutil

from .constants import WEATHER_TYPE


def run_dir_for(dataset_name: str, target_weather: str,
                appearance_stage: str, particle_severity: str) -> str:
    """Path of the run folder for a (weather, appearance stage, particle severity) combo.

    The scene is already the parent directory, so the folder is named
    ``<weather>_a_<appearance_stage>_p_<particle_severity>`` (``a`` = appearance,
    ``p`` = particle), e.g. ``snowy_a_long_p_moderate``.
    """
    return f"output/{dataset_name}/{target_weather}_a_{appearance_stage}_p_{particle_severity}"


def _move(src: str, dst_dir: str) -> bool:
    """Move ``src`` (file or dir) into ``dst_dir`` if it exists; return whether it moved."""
    if not os.path.lexists(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    # islink must be checked first: os.path.isdir() follows symlinks, and rmtree
    # refuses to delete one. Re-packaging an existing run hits exactly this, since
    # geometry/ is typically a symlink.
    if os.path.islink(dst) or os.path.isfile(dst):
        os.remove(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print(f"  intermediate_results/{os.path.basename(src)}")
    return True


def _copy(src: str, dst: str) -> bool:
    """Copy ``src`` to ``dst`` if it exists; return whether it copied."""
    if not os.path.exists(src):
        print(f"  skip (missing): {os.path.basename(src)}")
        return False
    shutil.copy2(src, dst)
    print(f"  preprocessed_results/{os.path.basename(dst)} <- {os.path.basename(src)}")
    return True


def package_run(dataset_name: str, target_weather: str,
                appearance_stage: str, particle_severity: str) -> str:
    """Reorganize the scattered stage outputs into the run folder and return its path."""
    out = f"output/{dataset_name}"
    weather_type = WEATHER_TYPE.get(target_weather, target_weather)
    run_dir = run_dir_for(dataset_name, target_weather, appearance_stage, particle_severity)
    inter = os.path.join(run_dir, "intermediate_results")
    prep = os.path.join(run_dir, "preprocessed_results")
    os.makedirs(inter, exist_ok=True)
    os.makedirs(prep, exist_ok=True)
    print(f"Packaging run: {run_dir}")

    # 1. Move the scattered stage outputs into intermediate_results/.
    #    Geometry is included, so each run folder is fully self-contained.
    scattered = [
        f"{out}/geometry",
        f"{out}/rgb_frames",
        f"{out}/depth_particles_{target_weather}",
        f"{out}/merged_depth_{target_weather}",
        f"{out}/edited_first_frame",
        f"{out}/edited_first_frame_with_particles",
        f"{out}/output_video_{weather_type}.mp4",
        f"{out}/{dataset_name}_merged_depth_{target_weather}.mp4",
        f"{out}/{dataset_name}_mask.mp4",
        f"{out}/{dataset_name}_pure_depth.mp4",
        f"{out}/{dataset_name}_particles_depth.mp4",
        f"{out}/{target_weather}.json",
    ]
    for src in scattered:
        _move(src, inter)

    # 2. Copy the canonical VACE-ready subset into preprocessed_results/.
    _copy(f"{inter}/{dataset_name}_merged_depth_{target_weather}.mp4", f"{prep}/control_depth.mp4")
    _copy(f"{inter}/{dataset_name}_mask.mp4", f"{prep}/mask.mp4")
    _copy(f"{inter}/output_video_{weather_type}.mp4", f"{prep}/particle_video.mp4")
    _copy(f"{inter}/edited_first_frame_with_particles/0000_{target_weather}.png", f"{prep}/reference.png")
    _copy(f"{inter}/edited_first_frame/0000_{target_weather}.png", f"{prep}/edited_image.png")
    _copy(f"{inter}/merged_depth_{target_weather}/0000.png", f"{prep}/merged_depth_0000.png")
    _copy(f"{inter}/depth_particles_{target_weather}/0000.png", f"{prep}/particle_depth_0000.png")

    info_src = f"{inter}/{target_weather}.json"
    if _copy(info_src, f"{prep}/info.json"):
        with open(info_src) as f:
            info = json.load(f)
        with open(f"{prep}/prompt.txt", "w") as f:
            f.write(info.get("edited_scene_summary", "").strip() + "\n")
        print("  preprocessed_results/prompt.txt <- info.json:edited_scene_summary")

    print(f"Run ready: {run_dir}")
    return run_dir
