"""Fast difficulty-conditioned terrain generation with ONNX Runtime."""

from .generator import TerrainGenerator
from .io import (
    compose_terrain_grid,
    load_terrain,
    load_terrains,
    save_terrain,
    save_terrains,
    to_raw_heightfield,
)
from .render import render_terrain, render_terrains
from .simulators import SimulatorTerrain, apply as apply_terrain

__all__ = [
    "TerrainGenerator",
    "SimulatorTerrain",
    "apply_terrain",
    "compose_terrain_grid",
    "load_terrain",
    "load_terrains",
    "render_terrain",
    "render_terrains",
    "save_terrain",
    "save_terrains",
    "to_raw_heightfield",
]

__version__ = "0.4.2"
