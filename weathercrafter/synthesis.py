"""Geometry-grounded video synthesis (paper Sec. 3.3).

Drives the VACE video editor (our in-repo ``vace_runner.py``) on a run's
``preprocessed_results/`` and writes the final video into ``synthesis_results/``.

By default this runs in the **current** environment, so install the synthesis
extra into your main env:  ``pip install -e ".[synthesis]"`` (pulls a pinned
``diffsynth``; torch >= 2.0 already satisfies it). If you instead keep VACE in a
separate conda env, set ``WEATHERCRAFTER_VACE_ENV`` to that env's name.

VACE model weights (~30 GB) auto-download on first run to ``./models`` (relative
to the working directory) and are reused across runs.
"""
import os
import shutil
import subprocess
import sys

from .package import run_dir_for

# Vendored VACE driver (imports diffsynth, installed via the `synthesis` extra).
VACE_RUNNER = os.path.abspath(os.path.join(os.path.dirname(__file__), "vace_runner.py"))


def run(dataset_name: str, target_weather: str, appearance_stage: str, particle_severity: str,
        tea_cache_l1_thresh: float = 0.2, height: int = 480, width: int = 832) -> str:
    """Run VACE on the run's preprocessed inputs; return the saved result path."""
    run_dir = run_dir_for(dataset_name, target_weather, appearance_stage, particle_severity)
    prep = os.path.join(run_dir, "preprocessed_results")
    synth = os.path.join(run_dir, "synthesis_results")
    os.makedirs(synth, exist_ok=True)

    control = os.path.abspath(os.path.join(prep, "control_depth.mp4"))
    reference = os.path.abspath(os.path.join(prep, "reference.png"))
    mask = os.path.abspath(os.path.join(prep, "mask.mp4"))
    prompt_path = os.path.join(prep, "prompt.txt")
    for p in (control, reference, mask, prompt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing VACE input: {p} (run the simulation stage first)")
    with open(prompt_path) as f:
        prompt = f.read().strip()

    # Default: run in the current env. Optionally run in a separate conda env.
    vace_env = os.environ.get("WEATHERCRAFTER_VACE_ENV")
    launcher = ([sys.executable, VACE_RUNNER] if not vace_env
                else ["conda", "run", "-n", vace_env, "--no-capture-output", "python3", VACE_RUNNER])
    cmd = launcher + [
        "-p", prompt,
        "-v", control,
        "-r", reference,
        "-m", mask,
        "--height", str(height), "--width", str(width),
        "--tea_cache_l1_thresh", str(tea_cache_l1_thresh),
    ]
    print(f"[synthesis] running VACE ({vace_env or 'current env'}, thresh {tea_cache_l1_thresh})")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "VACE failed. Make sure the synthesis dependencies are installed:\n"
            "    pip install -e \".[synthesis]\"\n"
            "(or set WEATHERCRAFTER_VACE_ENV to a conda env that has diffsynth)."
        ) from e

    # vace_runner saves <control basename>.mp4 in the working dir; move it into synthesis_results/.
    produced = "control_depth.mp4"
    result_name = f"result_{target_weather}_a_{appearance_stage}_p_{particle_severity}.mp4"
    result_path = os.path.join(synth, result_name)
    if os.path.exists(produced):
        shutil.move(produced, result_path)
        print(f"[synthesis] saved {result_path}")
    else:
        print(f"[synthesis] WARNING: expected VACE output '{produced}' not found in {os.getcwd()}")
    return result_path
