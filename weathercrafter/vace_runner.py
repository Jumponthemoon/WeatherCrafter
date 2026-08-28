"""VACE video-synthesis driver (paper Sec. 3.3).

Thin wrapper around the Wan2.1-VACE-14B pipeline, vendored into this repo so the
synthesis logic is version-controlled here. The heavy ``diffsynth`` library and
the model weights are NOT vendored - install them with ``pip install -e ".[synthesis]"``
(diffsynth only needs torch >= 2.0). ``weathercrafter/synthesis.py`` runs this script.

Derived from DiffSynth-Studio's run_vace.py.
"""
import argparse
import glob
import os

# Reduce fragmentation-driven OOMs on tight VRAM. Must be set before torch inits CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from diffsynth import VideoData, save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

NEGATIVE_PROMPT = ("transition,静态，转场,字幕，风格，作品，"
                   "画作，画面，静止，JPEG压缩残留，畸形的，"
                   "毁容的，形态畸形的肢体，手指融合，静止不动的画面")


VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-14B"


def _online_model_configs(models_dir):
    """Model configs that download (or reuse cached) weights via the ModelScope hub.

    ``local_model_path`` is passed explicitly: diffsynth otherwise defaults to
    "./models" relative to the *current working directory*, so running from a
    different folder would download the ~30 GB of weights all over again.
    """
    return [
        ModelConfig(model_id=VACE_MODEL_ID, local_model_path=models_dir,
                    origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id=VACE_MODEL_ID, local_model_path=models_dir,
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id=VACE_MODEL_ID, local_model_path=models_dir,
                    origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ]


def _offline_model_configs(models_dir):
    """Model configs that load cached weights by explicit local path - no hub access.

    Skips the per-run ModelScope connectivity check, which hangs on a flaky network even
    when every weight is already cached. Weights are located by searching ``models_dir``
    recursively (the T5 encoder + VAE live in the Wan2.1 base repo, not the VACE repo)."""
    def find(pattern, label, want_all=False):
        hits = sorted(glob.glob(os.path.join(models_dir, "**", pattern), recursive=True))
        if not hits:
            raise FileNotFoundError(
                f"Offline VACE: no cached {label} matching '{pattern}' under '{models_dir}'. "
                f"Run once online to download the weights, or unset WEATHERCRAFTER_VACE_OFFLINE."
            )
        return hits if want_all else hits[0]

    dit = find(os.path.join("Wan2.1-VACE-14B", "diffusion_pytorch_model*-of-*.safetensors"),
               "VACE DiT shards", want_all=True)
    t5 = find("models_t5_umt5-xxl-enc-bf16.pth", "T5 text encoder")
    vae = find("Wan2.1_VAE.pth", "VAE")
    return [
        ModelConfig(path=dit, offload_device="cpu"),
        ModelConfig(path=t5, offload_device="cpu"),
        ModelConfig(path=vae, offload_device="cpu"),
    ]


def build_pipeline(vram_limit=None, offline=False, models_dir="./models") -> "WanVideoPipeline":
    """Load the Wan2.1-VACE-14B pipeline with CPU offload + VRAM management.

    ``vram_limit`` (GB) caps GPU residency so weights + activations leave headroom for
    other GPU users (e.g. the desktop/compositor). Lower = more CPU offload (slower, but
    fits tighter cards). ``None`` lets diffsynth budget the whole card.
    ``offline`` loads cached weights by local path, skipping the ModelScope hub check."""
    model_configs = (_offline_model_configs(models_dir) if offline
                     else _online_model_configs(models_dir))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
    )
    if vram_limit is not None:
        pipe.enable_vram_management(vram_limit=vram_limit)
    else:
        pipe.enable_vram_management()
    return pipe


def _ensure_ext(filename: str, ext: str = ".mp4") -> str:
    return filename if filename.lower().endswith(ext) else filename + ext


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Wan2.1-VACE-14B video synthesis pipeline.")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument("-v", "--vace_depth_video_path", required=True, help="Depth / control video.")
    parser.add_argument("-r", "--vace_reference_image", nargs="+", help="Reference image(s).")
    parser.add_argument("-m", "--vace_video_mask_path", default="", help="(optional) Video mask.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("-q", "--quality", type=int, default=5, help="Output video quality (1-10).")
    parser.add_argument("--tea_cache_l1_thresh", type=float, default=0.15,
                        help="TeaCache rel-L1 threshold (larger = faster, lower quality; <=0 disables). "
                             "VACE-14B @50 steps: 0.15 ~2x (sweet spot), 0.2 ~2.5x.")
    parser.add_argument("--vram_limit", type=float,
                        default=(float(os.environ["WEATHERCRAFTER_VACE_VRAM_LIMIT_GB"])
                                 if os.environ.get("WEATHERCRAFTER_VACE_VRAM_LIMIT_GB") else None),
                        help="GPU VRAM budget in GB for model residency (lower = more CPU offload "
                             "to fit tighter GPUs or leave room for other GPU users). Defaults to the "
                             "WEATHERCRAFTER_VACE_VRAM_LIMIT_GB env var, else auto (whole card).")
    parser.add_argument("--no_trim", action="store_true",
                        help="Keep every generated frame. By default the output is trimmed to the "
                             "control video's length, dropping the unconditioned 4n+1 tail frames.")
    parser.add_argument("--offline", action="store_true",
                        default=bool(os.environ.get("WEATHERCRAFTER_VACE_OFFLINE")),
                        help="Load cached weights by local path, skipping the ModelScope hub check "
                             "(avoids hangs on flaky networks). Defaults to the "
                             "WEATHERCRAFTER_VACE_OFFLINE env var. Requires weights already cached.")
    parser.add_argument("--models_dir", default=os.environ.get("WEATHERCRAFTER_MODELS_DIR", "./models"),
                        help="Directory holding cached model weights (default ./models).")
    args = parser.parse_args()
    print(f"### VACE synthesis | prompt: {args.prompt}")

    pipe = build_pipeline(vram_limit=args.vram_limit, offline=args.offline, models_dir=args.models_dir)

    control_video = VideoData(args.vace_depth_video_path, height=args.height, width=args.width)
    mask = (VideoData(args.vace_video_mask_path, height=args.height, width=args.width)
            if args.vace_video_mask_path else None)

    call_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        vace_video=control_video,
        seed=args.seed,
        tiled=True,
        height=args.height, width=args.width,
    )
    if args.vace_reference_image:
        call_kwargs["vace_reference_image"] = [
            Image.open(p).convert("RGB").resize((args.width, args.height))
            for p in args.vace_reference_image
        ]
    if mask is not None:
        call_kwargs["vace_video_mask"] = mask

    # TeaCache acceleration (training-free). VACE-14B uses the Wan2.1-T2V-14B coefficients.
    if args.tea_cache_l1_thresh and args.tea_cache_l1_thresh > 0:
        call_kwargs["tea_cache_l1_thresh"] = args.tea_cache_l1_thresh
        call_kwargs["tea_cache_model_id"] = "Wan2.1-T2V-14B"
        print(f"[TeaCache] enabled: l1_thresh={args.tea_cache_l1_thresh}")
    else:
        print("[TeaCache] disabled")

    video = pipe(**call_kwargs)

    # Wan generates on a 4n+1 schedule (diffsynth defaults to 81 frames), which usually
    # overruns the control video. Frames past its end carry no conditioning and degrade,
    # so trim back to the control length.
    if not args.no_trim:
        n_control = len(control_video)
        if len(video) > n_control:
            print(f"[trim] {len(video)} -> {n_control} frames (control video length)")
            video = video[:n_control]

    out_name = _ensure_ext(os.path.basename(args.vace_depth_video_path))
    save_video(video, out_name, fps=args.fps, quality=args.quality)
    print(f"Saved output as {out_name}")


if __name__ == "__main__":
    main()
