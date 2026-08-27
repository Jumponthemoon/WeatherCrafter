"""WeatherCrafter: semantic-aware, physics-informed, geometry-grounded weather synthesis.

Package layout mirrors the paper's three stages:
  appearance.py  - Sec. 3.1 semantic-aware appearance anchoring
  simulation.py  - Sec. 3.2 physics-informed particle simulation
  geometry.py    - Sec. 3.3 geometry estimation, gravity alignment, projection
  render.py      -          particle projection, compositing, video assembly
  synthesis.py   -          drives the VACE video editor
  cli.py         -          `python -m weathercrafter <command>` entry point
"""

__version__ = "0.1.0"
