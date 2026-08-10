"""IsaacLab adapter."""

from __future__ import annotations

from .base import SimulatorTerrain


def apply_isaaclab(
    terrain: SimulatorTerrain,
    importer: object | None,
    *,
    name: str = "adaptive_terrain",
    configure_env_origins: bool = False,
    origins: object | None = None,
):
    """Import a triangle mesh through an initialized TerrainImporter.

    IsaacLab and Isaac Sim must already be launched before this call. Returning
    the Trimesh object keeps it alive for callers that want to inspect/export it.
    """

    if importer is None or not callable(getattr(importer, "import_mesh", None)):
        raise TypeError(
            "IsaacLab requires target=an initialized isaaclab.terrains.TerrainImporter"
        )
    mesh = terrain.to_trimesh()
    importer.import_mesh(name, mesh)
    if configure_env_origins:
        configure = getattr(importer, "configure_env_origins", None)
        if not callable(configure):
            raise TypeError("The TerrainImporter has no configure_env_origins method")
        configure(origins)
    return mesh

