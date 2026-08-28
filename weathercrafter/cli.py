"""Command-line entry point: ``python -m weathercrafter <command> [options]``.

Stages follow the paper:
  preprocess   extract frames + estimate depth/camera geometry (Depth Anything 3)
  anchoring    Sec. 3.1  semantic-aware appearance anchoring (OpenAI + FLUX)
  simulation   Sec. 3.2  particle simulation + projection -> preprocessed_results/
  synthesis    Sec. 3.3  geometry-grounded video synthesis (VACE) -> synthesis_results/

  pipeline     run several stages in order (the end-to-end pipeline)
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import appearance, package, preprocess, synthesis
from .constants import (DEFAULT_DEPTH_SCALE, DEFAULT_DURATION, DEFAULT_FPS, DEFAULT_GRAVITY,
                        DEFAULT_SNOW_COUNT, DEFAULT_TURB_STRENGTH, DEFAULT_VEL_RANGES,
                        DEFAULT_WIND_ANGLE, DEFAULT_WIND_STRENGTH, SEED, SEVERITY_TABLE, WEATHER_TYPE)
from .geometry import infer_depth
from .render import (merge_depth_and_assemble_videos, render_snow_edited_first_frame,
                     render_snow_video_folder)

PIPELINE_STAGES = ("preprocess", "anchoring", "simulation", "synthesis")


@dataclass
class SimParams:
    snow_count: int
    gravity: float
    wind_strength: float
    turb_strength: Optional[float]
    vel_ranges: dict
    wind_dir: np.ndarray


def resolve_sim_params(weather_type, severity=None, snow_count=None, gravity=None,
                       wind_strength=None, turb_strength=None, wind_angle=DEFAULT_WIND_ANGLE) -> SimParams:
    """Resolve simulation parameters with precedence:
        explicit value  >  severity table (sampled in range)  >  global default.

    Assumes the RNG has already been seeded by the caller; the severity sampling
    consumes draws in a fixed order so the particle field stays reproducible.
    """
    sev = SEVERITY_TABLE.get(severity)

    if snow_count is None and sev is not None:
        snow_count = random.randint(*sev["count"])
    elif snow_count is None:
        snow_count = DEFAULT_SNOW_COUNT

    if gravity is None:
        gravity = DEFAULT_GRAVITY

    if wind_strength is None and sev is not None:
        wind_strength = math.sqrt(random.uniform(*sev["wind"]))
    elif wind_strength is None:
        wind_strength = DEFAULT_WIND_STRENGTH

    if turb_strength is None and sev is not None and weather_type is not None:
        turb_strength = random.uniform(*sev["turb"][weather_type])
    elif turb_strength is None and weather_type is not None:
        turb_strength = DEFAULT_TURB_STRENGTH[weather_type]

    vel_ranges = sev["vel"] if sev is not None else DEFAULT_VEL_RANGES

    if severity is not None:
        print(f"[severity={severity}] snow_count={snow_count} gravity={gravity} "
              f"wind_strength={wind_strength:.2f} turb_strength={turb_strength} "
              f"vy={vel_ranges['vy']}")

    wind_rad = math.radians(wind_angle)
    wind_dir = np.array([math.cos(wind_rad), 0.0, math.sin(wind_rad)])

    return SimParams(snow_count, gravity, wind_strength, turb_strength, vel_ranges, wind_dir)


def _render(dataset_name, target_weather, severity, depth_scale, fps, first_frame, **overrides):
    """Seed, resolve params, and run the video or edited-first-frame renderer."""
    random.seed(SEED)
    np.random.seed(SEED)
    weather_type = WEATHER_TYPE.get(target_weather)
    p = resolve_sim_params(weather_type, severity, **overrides)
    data_root = f"./output/{dataset_name}"
    common = dict(target_weather=target_weather, weather_type=weather_type,
                  turb_strength=p.turb_strength, wind_strength=p.wind_strength, wind_dir=p.wind_dir,
                  gravity=p.gravity, snow_count=p.snow_count, vel_ranges=p.vel_ranges,
                  depth_scale=depth_scale, fps=fps)
    if first_frame:
        render_snow_edited_first_frame(data_root=data_root, **common)
    else:
        render_snow_video_folder(data_root=data_root, output_folder=data_root, **common)


# ==========================
#   STAGES (paper-aligned)
# ==========================
def stage_preprocess(args: argparse.Namespace) -> None:
    """Extract frames + estimate depth/camera geometry."""
    preprocess.run(args.dataset_name, fps=args.fps, sec=args.sec, target_size=args.target_size)
    infer_depth(args.dataset_name, fps=args.fps, duration=args.duration)


def stage_anchoring(args: argparse.Namespace) -> None:
    """Sec. 3.1 - semantic-aware appearance anchoring (edit the first frame)."""
    appearance.run(args.dataset_name, args.target_weather,
                   args.appearance_stage, args.particle_severity)


def stage_simulation(args: argparse.Namespace) -> None:
    """Sec. 3.2 - simulate particles, render the first-frame reference, and package
    the VACE-ready inputs into the run folder."""
    overrides = dict(snow_count=args.snow_count, gravity=args.gravity, wind_strength=args.wind_strength,
                     turb_strength=args.turb_strength, wind_angle=args.wind_angle)
    _render(args.dataset_name, args.target_weather, args.particle_severity,
            args.depth_scale, args.fps, first_frame=False, **overrides)
    merge_depth_and_assemble_videos(args.dataset_name, args.target_weather, fps=args.fps)
    _render(args.dataset_name, args.target_weather, args.particle_severity,
            args.depth_scale, args.fps, first_frame=True, **overrides)
    package.package_run(args.dataset_name, args.target_weather,
                        args.appearance_stage, args.particle_severity)


def stage_synthesis(args: argparse.Namespace) -> None:
    """Sec. 3.3 - geometry-grounded video synthesis (VACE)."""
    synthesis.run(args.dataset_name, args.target_weather, args.appearance_stage,
                  args.particle_severity, tea_cache_l1_thresh=args.tea_cache_l1_thresh,
                  models_dir=args.models_dir, offline=args.vace_offline)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the requested stages in order, sharing the dataset/weather settings."""
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in PIPELINE_STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. "
                         f"Choose from: {', '.join(PIPELINE_STAGES)}.")
    handlers = {"preprocess": stage_preprocess, "anchoring": stage_anchoring,
                "simulation": stage_simulation, "synthesis": stage_synthesis}
    for stage in stages:
        print(f"\n===== weathercrafter: {stage} =====")
        handlers[stage](args)
    print(f"\n[DONE] Results are in output/{args.dataset_name}/")


# ==========================
#   ARGUMENT WIRING
# ==========================
def _add_preprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame rate.")
    parser.add_argument("--sec", type=int, default=5, help="Seconds of video to extract.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Seconds of video to process during depth inference.")
    parser.add_argument("--target_size", type=int, default=504, help="Shortest-side size after resize.")


def _add_anchoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--target_weather", type=str, required=True, choices=["rainy", "snowy"])
    parser.add_argument("--appearance_stage", type=str, default="medium",
                        choices=["short", "medium", "long"],
                        help="Weather effect stage - how long the weather has been going, and "
                             "hence how far the background has changed (short | medium | long).")
    parser.add_argument("--particle_severity", type=str, default="moderate",
                        choices=["light", "moderate", "heavy"],
                        help="Particle severity the simulation stage will render; reaches the "
                             "video prompt (P_vid) so it matches the particle-augmented depth. "
                             "Keep it equal to the simulation stage's value.")


def _add_simulation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--target_weather", type=str, required=True, choices=["rainy", "snowy"])
    parser.add_argument("--particle_severity", type=str, default="moderate",
                        choices=["light", "moderate", "heavy"],
                        help="Severity preset for particle count / wind / turbulence.")
    parser.add_argument("--appearance_stage", type=str, default="medium",
                        choices=["short", "medium", "long"],
                        help="Used (with particle severity) to name the run folder.")
    parser.add_argument("--snow_count", type=int, default=None, help="Override particle count.")
    parser.add_argument("--depth_scale", type=float, default=DEFAULT_DEPTH_SCALE,
                        help="Particle weight when compositing into the depth map.")
    parser.add_argument("--turb_strength", type=float, default=None, help="Override turbulence strength.")
    parser.add_argument("--wind_strength", type=float, default=None, help="Override wind force coefficient.")
    parser.add_argument("--wind_angle", type=float, default=DEFAULT_WIND_ANGLE,
                        help="Wind direction in the horizontal (x-z) plane, degrees.")
    parser.add_argument("--gravity", type=float, default=None, help="Override gravity magnitude.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame rate for output videos.")


def _add_vace_weights_args(parser: argparse.ArgumentParser) -> None:
    """Where the VACE weights live. Shared by the `synthesis` and `pipeline` commands."""
    parser.add_argument("--models_dir", type=str,
                        default=os.environ.get("WEATHERCRAFTER_MODELS_DIR"),
                        help="Directory holding the VACE weights (~30 GB). Defaults to "
                             "./models relative to the working directory; point it at an "
                             "existing copy to skip the download. Env: WEATHERCRAFTER_MODELS_DIR.")
    parser.add_argument("--vace_offline", action="store_true",
                        default=bool(os.environ.get("WEATHERCRAFTER_VACE_OFFLINE")),
                        help="Resolve the weights by local path without contacting the "
                             "ModelScope hub. Env: WEATHERCRAFTER_VACE_OFFLINE.")


def _add_synthesis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--target_weather", type=str, required=True, choices=["rainy", "snowy"])
    parser.add_argument("--appearance_stage", type=str, default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--particle_severity", type=str, default="moderate",
                        choices=["light", "moderate", "heavy"])
    parser.add_argument("--tea_cache_l1_thresh", type=float, default=0.2,
                        help="VACE TeaCache threshold (larger = faster, lower quality).")
    _add_vace_weights_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weathercrafter",
                                     description="Depth-guided weather particle simulation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Extract frames + estimate depth/camera geometry.")
    _add_preprocess_args(p_pre)
    p_pre.set_defaults(func=stage_preprocess)

    p_anchor = sub.add_parser("anchoring", help="Sec. 3.1 - semantic-aware appearance anchoring.")
    _add_anchoring_args(p_anchor)
    p_anchor.set_defaults(func=stage_anchoring)

    p_sim = sub.add_parser("simulation", help="Sec. 3.2 - particle simulation + projection.")
    _add_simulation_args(p_sim)
    p_sim.set_defaults(func=stage_simulation)

    p_synth = sub.add_parser("synthesis", help="Sec. 3.3 - geometry-grounded video synthesis (VACE).")
    _add_synthesis_args(p_synth)
    p_synth.set_defaults(func=stage_synthesis)

    p_pipe = sub.add_parser("pipeline", help="Run several stages in order (the end-to-end pipeline).")
    p_pipe.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    p_pipe.add_argument("--target_weather", type=str, default="rainy", choices=["rainy", "snowy"])
    p_pipe.add_argument("--appearance_stage", type=str, default="medium",
                        choices=["short", "medium", "long"])
    p_pipe.add_argument("--particle_severity", type=str, default="moderate",
                        choices=["light", "moderate", "heavy"])
    p_pipe.add_argument("--tea_cache_l1_thresh", type=float, default=0.2,
                        help="VACE TeaCache threshold for the synthesis stage.")
    _add_vace_weights_args(p_pipe)
    p_pipe.add_argument("--stages", type=str, default=",".join(PIPELINE_STAGES),
                        help=f"Comma-separated stages to run, in order. Default: {','.join(PIPELINE_STAGES)}.")
    # Pipeline uses default sim params; expose the override knobs simulation needs.
    p_pipe.set_defaults(func=cmd_pipeline, fps=DEFAULT_FPS, depth_scale=DEFAULT_DEPTH_SCALE,
                        snow_count=None, gravity=None, wind_strength=None, turb_strength=None,
                        wind_angle=DEFAULT_WIND_ANGLE, sec=5, duration=DEFAULT_DURATION, target_size=504)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
