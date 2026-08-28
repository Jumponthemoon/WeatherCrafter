<div align="center">

## 🪄 Semantic-Aware, Physics-Informed, Geometry-Grounded<br>Weather Video Synthesis

<p align="center">
  <a href="https://arxiv.org/abs/2606.29020">
    <img src="https://img.shields.io/badge/arXiv-article-red" alt="arXiv">
  </a>
  <a href="https://jumponthemoon.github.io/w-crafter/">
    <img src="https://img.shields.io/badge/Project-link-blue" alt="Project">
  </a>
  <img src="https://img.shields.io/badge/ECCV-2026-green" alt="ECCV 2026">
</p>
<p>
<b>Chenghao Qian</b><sup>1</sup> &nbsp;·&nbsp; Nedko Savov<sup>2</sup> &nbsp;·&nbsp; Lingdong Kong<sup>3</sup> &nbsp;·&nbsp; Yeying Jin<sup>3</sup> &nbsp;·&nbsp; Rui Song<sup>5</sup><br>
Wenjing Li<sup>1,4</sup> &nbsp;·&nbsp; Zhun Zhong<sup>4</sup> &nbsp;·&nbsp; Jiaqi Ma<sup>5</sup> &nbsp;·&nbsp; Gustav Markkula<sup>1</sup> &nbsp;·&nbsp; Luc Van Gool<sup>2</sup>
</p>
<p>
<sup>1</sup>University of Leeds &nbsp;·&nbsp;
<sup>2</sup>INSAIT &nbsp;·&nbsp;
<sup>3</sup>National University of Singapore &nbsp;·&nbsp;
<sup>4</sup>Hefei University of Technology &nbsp;·&nbsp;
<sup>5</sup>UCLA
</p>
<br>
<img width="1100" height="158" alt="demo4_snow (1)" src="https://github.com/user-attachments/assets/2e8fec83-dcb8-4f0e-b0c1-b389e1b4c1b1" />
<img width="1100" height="158" alt="demo4_rain" src="https://github.com/user-attachments/assets/81639c2a-8e38-4de0-9de3-95f515fae4d4" />
<br>
</div>

We steer an **off-the-shelf** video diffusion editor with three structured priors —
 **semantics** (*what* the weather looks like), **dynamics** (*how* it evolves), and **geometry** (*where* it appears) to synthesize diverse, physically realistic weather on real videos, **without any finetuning**.



## ✨ Highlights
- 🧩 **Tri-prior interface** — a single, structured conditioning space that factorizes weather into **semantics · dynamics · geometry**, giving precise and interpretable control.
- 🌦️ **Diverse appearance** — a *semantic-aware* strategy binds the intended weather to scene semantics via a VLM + LLM, producing varied, realistic global appearances.
- ❄️ **Physical particle dynamics** — a *physics-informed* Gaussian particle field evolves under **gravity, wind, and turbulence**, activating latent weather priors in pretrained editors for dense, coherent particles.
- 📐 **Geometry grounding** — particles are gravity-aligned and projected with camera intrinsics/extrinsics into particle-augmented depth, ensuring spatially accurate, temporally consistent placement.

---

## 📰 News

- **`Jun 2026`** &nbsp;🎉 Paper accepted to **ECCV 2026**!
- **`Jun 2026`** &nbsp;🌐 [Project page](https://jumponthemoon.github.io/w-crafter/) is live, with the supplementary demo video.
- **`Aug 2026`** &nbsp;💻 Code Released!.
- **`Aug 2026`** &nbsp;✅ We have halved the inference time. Each clip now takes only ~20 mins.

---



## 🛠️ Installation

```bash
git clone https://github.com/Jumponthemoon/WeatherCrafter.git
cd WeatherCrafter

conda env create -f environment.yml
conda activate weathercrafter

# the synthesis (VACE) stage
pip install -e ".[synthesis]"
pip install --no-build-isolation \
    "diffsynth @ git+https://github.com/modelscope/DiffSynth-Studio.git@8332ece"
```

</details>
Model weights download on first use: Depth Anything 3 from HuggingFace, Wan2.1-VACE-14B from ModelScope.

## 🚀 Quick start

```bash
# API keys
export OPENAI_API_KEY="sk-..."
export BFL_API_KEY="..."

# Download dataset
bash scripts/download_examples.sh 

# Full pipeline run
python -m weathercrafter pipeline \
    --dataset_name drone \
    --target_weather snowy \
    --appearance_stage medium \
    --particle_severity moderate
```

For your own clip, put it at `data/<name>/<name>.mp4` and pass `--dataset_name <name>`.
The synthesis result lands in:

```
output/<name>/<weather>_a_<stage>_p_<severity>/synthesis_results/result_*.mp4
```

### 🎛️ Control axes

```
--target_weather      rainy | snowy
--appearance_stage    short | medium | long      how long the weather has been going
--particle_severity   light | moderate | heavy   how much is falling right now
```
`appearance_stage` drives the first-frame edit — how far the environment has changed,
from wet patches to deep accumulation. `particle_severity` drives the simulation —
particle count, wind, and turbulence.
### 🧱 Running stages one at a time

```bash
# Preprocessing
python -m weathercrafter preprocess --dataset_name drone

# Anchoring
python -m weathercrafter anchoring  --dataset_name drone --target_weather snowy \
    --appearance_stage medium --particle_severity moderate

# Simulation
python -m weathercrafter simulation --dataset_name drone --target_weather snowy \
    --appearance_stage medium --particle_severity moderate

# Synthesis
python -m weathercrafter synthesis  --dataset_name drone --target_weather snowy \
    --appearance_stage medium --particle_severity moderate
```


## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{qian2026weathervid,
  title     = {Semantic-Aware, Physics-Informed, Geometry-Grounded Weather Video Synthesis},
  author    = {Qian, Chenghao and Savov, Nedko and Kong, Lingdong and Jin, Yeying and
               Song, Rui and Li, Wenjing and Zhong, Zhun and Ma, Jiaqi and
               Markkula, Gustav and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---


