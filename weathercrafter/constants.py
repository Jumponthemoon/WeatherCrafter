"""Shared defaults and the per-severity parameter table."""

SEED = 42
DEFAULT_SNOW_COUNT = 6000
DEFAULT_DEPTH_SCALE = 0.6
DEFAULT_FPS = 15
DEFAULT_DURATION = 5      # seconds of video to process during depth inference
Z_MIN = 3                 # near plane for particle spawning / wrap-around
Z_MAX = 200               # far plane

DEFAULT_GRAVITY = 0.98      
DEFAULT_WIND_STRENGTH = 1.0 
DEFAULT_WIND_ANGLE = 0.0   


WEATHER_TYPE = {"snowy": "snow", "rainy": "rain"}
DEFAULT_TURB_STRENGTH = {"snow": 10.0, "rain": 3.0}


DEFAULT_VEL_RANGES = {"vx": (15.0, 25.0), "vy": (30.0, 50.0), "vz": (0.5, 1.0)}


SEVERITY_TABLE = {
    "light":    {"count": (1500, 2500), "wind": (1.0, 5.0),   "turb": {"rain": (0.0, 1.0), "snow": (0.0, 1.0)},
                 "vel": {"vx": (8.0, 15.0),  "vy": (20.0, 40.0), "vz": (0.5, 1.0)}},
    "moderate": {"count": (4000, 6000), "wind": (5.0, 10.0),  "turb": {"rain": (1.0, 2.0), "snow": (1.0, 2.0)},
                 "vel": {"vx": (8.0, 15.0), "vy": (40.0, 60.0), "vz": (0.5, 1.0)}},
    "heavy":    {"count": (6000, 9000), "wind": (10.0, 15.0), "turb": {"rain": (2.0, 3.0), "snow": (2.0, 3.0)},
                 "vel": {"vx": (8.0, 15.0), "vy": (60.0, 80.0), "vz": (0.5, 1.0)}},
}
