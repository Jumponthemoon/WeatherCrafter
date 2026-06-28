<h1 align="center">Semantic-Aware, Physics-Informed, Geometry-Grounded<br>Weather Video Synthesis</h1>

<p align="center">
  <a href="https://jumponthemoon.github.io/w-crafter/"><img src="https://img.shields.io/badge/🌐%20Project-Page-2f6df0"></a>
  <a href="https://jumponthemoon.github.io/w-crafter/static/paper.pdf"><img src="https://img.shields.io/badge/📄%20Paper-PDF-c8743a"></a>
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-lightgrey">
  <img src="https://img.shields.io/badge/ECCV-2026-72A350">
  <img src="https://img.shields.io/badge/Code-coming%20soon-e9b412">
</p>

<p align="center">
  <b>Official repository for our ECCV 2026 paper.</b>
</p>

<p align="center">
  <img src="https://jumponthemoon.github.io/w-crafter/static/images/teaser.png" width="100%">
</p>

> **TL;DR** — A controllable weather-synthesis framework that steers an off-the-shelf video editor with three
> structured priors — **semantics** (what it looks like), **dynamics** (how it evolves), and **geometry**
> (where it appears) — to produce diverse, physically realistic weather effects on real videos.

## 🔗 Links

- 🌐 **Project page (with demo video):** https://jumponthemoon.github.io/w-crafter/
- 📄 **Paper:** [PDF](https://jumponthemoon.github.io/w-crafter/static/paper.pdf)
- 📦 **arXiv:** coming soon

## 👥 Authors

Chenghao Qian<sup>1</sup>, Nedko Savov<sup>2</sup>, Lingdong Kong<sup>3</sup>, Yeying Jin<sup>3</sup>, Rui Song<sup>4</sup>,
Wenjing Li<sup>1,5</sup>\*, Zhun Zhong<sup>5</sup>\*, Jiaqi Ma<sup>4</sup>, Gustav Markkula<sup>1</sup>, Luc Van Gool<sup>2</sup>

<sup>1</sup>University of Leeds, UK &nbsp;·&nbsp; <sup>2</sup>INSAIT, Sofia University “St. Kliment Ohridski” &nbsp;·&nbsp;
<sup>3</sup>National University of Singapore &nbsp;·&nbsp; <sup>4</sup>University of California, Los Angeles &nbsp;·&nbsp;
<sup>5</sup>Hefei University of Technology, China
<br><sub>\* Corresponding authors</sub>

## 📝 Abstract

Weather synthesis aims to add weather effects to input videos while preserving scene identity, structure, and motion.
The key limitation of existing methods is the lack of diversity in weather appearance and effective control over weather
dynamics (e.g., temporal evolution and particle motion). To address this, we propose a **Semantic-Aware,
Physics-Informed, and Geometry-Grounded** framework that steers an off-the-shelf video editor to synthesize diverse
global appearances and detailed particle dynamics. We factorize synthesis into three conditional signals —
**semantics** specifies what the weather should look like, **dynamics** governs how it evolves over time, and
**geometry** determines where it should appear in the scene. Experiments demonstrate that our method produces diverse,
physically and visually realistic weather effects, and that our synthesized data significantly improves the robustness of
autonomous-driving semantic segmentation under adverse weather.

## 🚧 Code

The code is **coming soon** — we are cleaning it up for release. Please ⭐ **star** and 👁️ **watch** this repository to
be notified when it lands.

### Roadmap

- [ ] Inference / synthesis pipeline
- [ ] Pretrained checkpoints
- [ ] Physics-informed particle simulation module
- [ ] Geometry grounding & particle projection
- [ ] Evaluation scripts and benchmarks
- [ ] Example data and demo notebook

## 📚 Citation

BibTeX will be added once the proceedings are finalized.

```bibtex
% Coming soon.
```

## 🙏 Acknowledgements

This work was funded by the National Natural Science Foundation of China (Nos. 62572166 and 62402157) and the
Fundamental Research Funds for the Central Universities (No. JZ2025HGTB0219). It was also partially supported by the
Ministry of Education and Science of Bulgaria (support for INSAIT, part of the Bulgarian National Roadmap for Research
Infrastructure).
