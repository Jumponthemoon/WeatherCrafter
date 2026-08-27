"""Semantic-aware appearance anchoring (paper Sec. 3.1).

Uses GPT-4.1 to (1) parse the first frame into a structured scene description,
(2) author a weather-transformation prompt, and (3) summarize the resulting
edited scene. The results are saved to JSON and the transformation prompt is
handed to the FLUX Kontext editor (in-process) to repaint the first frame.

Requires the OpenAI API key in the environment:
    export OPENAI_API_KEY="sk-..."
"""
import argparse
import base64
import json
import os
import time

from openai import OpenAI

from . import flux_kontext

MODEL = "gpt-4.1"

SCENE_ANALYSIS_PROMPT = """
Analyze the provided image and output ONLY the following fields
exactly in this structure:

1. Scene Type
2. Subject
3. Static Elements (non-moving)
4. Dynamic Elements (capable of motion)
5. Lighting
6. Current Weather
7. Camera Angle (from: Low angle shot, High angle shot, Overhead shot, Aerial Shot, Point of View)
Do not add commentary. Do not explain. Only provide those fields.
"""

WEATHER_PROMPT_TEMPLATE = """
Using the following scene description:

{scene_description} without camera view information.

Transform the scene into **{target_weather}**.

Weather Effect Stage: {stage}

Duration Stage Interpretation:
- "short": weather has just started, minimal environmental
  changes with negligible and light coverage.
- "medium": weather is currently happening, moderate
  environmental changes and coverage.
- "long": weather has been ongoing for a long time, strong
  environmental changes with full coverage.

Transformation Rules:
- Explicitly mention preserving scene geometry and composition.
- Incorporate soft, diffuse overcast-style lighting with gentle atmospheric softness.
- Adjust atmosphere and surface conditions to reflect {target_weather} in its {stage} effect stage.
- Base the environmental transformation strictly on the effect stage.
- Ensure realism and avoid hallucinating new structures.
Return ONLY the final weather transformation prompt. Do not repeat the scene analysis.
"""

SUMMARY_PROMPT_TEMPLATE = """
Based on the following information:

Scene Analysis:
{scene_description}

Weather Transformation Instructions:
{final_weather_prompt}

Write a short, concise description of the resulting edited scene.

The description must:
- Describe the subject and event of the edited scene.
- Describe the environment to reflect weather in its {stage} effect stage.
- Describe the falling particle effects at {severity} intensity.
- No More than 1 sentence within 30 words.

Example:
"A high-angle view of a golfer walking on a thickly snow-covered course, golf cart and trees blanketed in white, with falling snowflakes creating a serene, misty winter scene." \

Return ONLY the scene description.
"""


def encode_image(path: str) -> str:
    """Return the base64-encoded contents of an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_gpt(client: OpenAI, text: str, image_b64: str | None = None) -> str:
    """Send a single text (and optional image) prompt to the model and return its text."""
    content = [{"type": "input_text", "text": text}]
    if image_b64 is not None:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"})
    response = client.responses.create(model=MODEL, input=[{"role": "user", "content": content}])
    return response.output_text


def analyze_scene(client: OpenAI, image_b64: str) -> str:
    """Parse the image into the structured scene description (paper P_scene)."""
    return ask_gpt(client, SCENE_ANALYSIS_PROMPT, image_b64=image_b64)


def author_weather_prompt(client: OpenAI, scene_description: str,
                          target_weather: str, stage: str) -> str:
    """Author the weather-transformation prompt (paper P_img) from the scene description.

    The appearance ``stage`` (short | medium | long) is the paper's ``duration_stage``:
    how long the weather has been going, and hence how much the environment has
    changed. It is the single control for this stage. The paper's separate
    ``severity`` input is not wired up here - only ``particle_severity`` (simulation
    stage) currently varies severity."""
    prompt = WEATHER_PROMPT_TEMPLATE.format(
        scene_description=scene_description,
        target_weather=target_weather,
        stage=stage,
    )
    return ask_gpt(client, prompt)


def summarize_edited_scene(client: OpenAI, scene_description: str, final_weather_prompt: str,
                           stage: str, severity: str) -> str:
    """Summarize the expected edited scene (paper P_vid).

    P_vid conditions the video model alongside the particle-augmented depth, so the two
    must agree on how dense the precipitation is: ``severity`` here is the *particle*
    severity the simulation stage will actually render, while ``stage`` governs the
    environment the first-frame edit established."""
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        scene_description=scene_description,
        final_weather_prompt=final_weather_prompt,
        stage=stage,
        severity=severity,
    )
    return ask_gpt(client, prompt)


def run(dataset_name: str, target_weather: str, stage: str, severity: str) -> None:
    """Run the full appearance-anchoring pipeline for one dataset's first frame.

    ``stage`` (short | medium | long) is the paper's ``duration_stage``: how long the
    weather has been going, and hence how far the environment has changed. It drives the
    FLUX edit prompt. ``severity`` (light | moderate | heavy) is the particle severity the
    simulation stage will render; it only reaches the summary (P_vid), so the text
    condition and the particle-augmented depth agree on precipitation density."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export your key first:\n"
            "    export OPENAI_API_KEY=\"sk-...\""
        )
    client = OpenAI(api_key=api_key)

    input_dir = f"output/{dataset_name}/geometry/images"
    output_dir = f"output/{dataset_name}/edited_first_frame"
    image_path = f"{input_dir}/0000.png"

    start = time.time()

    # Step 1 - scene analysis.
    scene_description = analyze_scene(client, encode_image(image_path))
    print("Scene analysis:\n", scene_description)

    # Step 2 - weather transformation prompt.
    final_weather_prompt = author_weather_prompt(client, scene_description, target_weather, stage)
    print("\nWeather transformation prompt:\n", final_weather_prompt)

    # Step 3 - edited scene summary.
    edited_scene_summary = summarize_edited_scene(client, scene_description, final_weather_prompt,
                                                  stage, severity)
    print(f"\nElapsed: {time.time() - start:.1f}s")

    # Step 4 - save results to JSON.
    results = {
        "target_weather": target_weather,
        "appearance_stage": stage,
        "particle_severity": severity,
        "scene_analysis": scene_description,
        "weather_transformation_prompt": final_weather_prompt,
        "edited_scene_summary": edited_scene_summary,
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = f"output/{dataset_name}/{target_weather}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {json_path}")

    # Step 5 - repaint the first-frame background with FLUX Kontext (in-process).
    flux_kontext.run_edit(
        [(image_path, final_weather_prompt)],
        output_dir=output_dir,
        target_weather=target_weather,
        error_handling=True,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--target_weather", type=str, required=True,
                        help="Target weather condition for transformation.")
    parser.add_argument("--appearance_stage", type=str, required=True,
                        choices=["short", "medium", "long"],
                        help="Weather effect stage - how long the weather has been going, "
                             "and hence how far the background has changed (short | medium | long).")
    parser.add_argument("--particle_severity", type=str, default="moderate",
                        choices=["light", "moderate", "heavy"],
                        help="Particle severity the simulation stage will render; reaches the "
                             "video prompt (P_vid) so it matches the particle-augmented depth.")


def main(args: argparse.Namespace) -> None:
    run(args.dataset_name, args.target_weather, args.appearance_stage, args.particle_severity)
