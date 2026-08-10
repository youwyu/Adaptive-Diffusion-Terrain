"""Gazebo Sim / Ignition Gazebo mesh adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from xml.sax.saxutils import escape

from .base import SimulatorTerrain


@dataclass(frozen=True)
class GazeboTerrain:
    directory: Path
    mesh_path: Path
    model_path: Path
    world_path: Path
    process: subprocess.Popen | None = None


def export_gazebo(
    terrain: SimulatorTerrain,
    output_dir: str | Path,
    *,
    name: str = "adaptive_terrain",
    world: str = "adaptive_terrain_world",
) -> GazeboTerrain:
    """Export an OBJ, SDF model, and directly launchable SDF world."""

    directory = Path(output_dir).expanduser().resolve()
    mesh_path = terrain.save_obj(directory / "meshes" / "terrain.obj")
    model_path = directory / "model.sdf"
    model_config_path = directory / "model.config"
    world_path = directory / "world.sdf"
    attribute_escapes = {'"': "&quot;", "'": "&apos;"}
    safe_name = escape(name, attribute_escapes)
    safe_world = escape(world, attribute_escapes)
    mesh_uri = escape(mesh_path.as_uri())
    model_uri = escape(model_path.as_uri())
    model_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{safe_name}">
    <static>true</static>
    <link name="terrain_link">
      <collision name="terrain_collision">
        <geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>
      </collision>
      <visual name="terrain_visual">
        <geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>
        <material>
          <ambient>0.38 0.52 0.28 1</ambient>
          <diffuse>0.38 0.52 0.28 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    )
    model_config_path.write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{safe_name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>adaptive-terrain-diffusion</name></author>
  <description>Generated consistency-diffusion terrain.</description>
</model>
"""
    )
    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{safe_world}">
    <physics name="physics" type="ignored">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <light type="directional" name="sun">
      <direction>-0.5 0.2 -0.9</direction>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    <include><uri>{model_uri}</uri></include>
  </world>
</sdf>
"""
    )
    return GazeboTerrain(
        directory=directory,
        mesh_path=mesh_path,
        model_path=model_path,
        world_path=world_path,
    )


def _gazebo_executable() -> tuple[str, str]:
    if executable := shutil.which("gz"):
        probe = subprocess.run(
            [executable, "sim", "--versions"], text=True, capture_output=True
        )
        if probe.returncode == 0:
            return executable, "gz"
    if executable := shutil.which("ign"):
        probe = subprocess.run(
            [executable, "gazebo", "--versions"], text=True, capture_output=True
        )
        if probe.returncode == 0:
            return executable, "ign"
    raise FileNotFoundError("Neither `gz` nor `ign` is installed or available on PATH")


def spawn_gazebo_model(
    asset: GazeboTerrain,
    *,
    world: str,
    name: str = "adaptive_terrain",
    timeout_ms: int = 5000,
) -> subprocess.CompletedProcess:
    """Spawn an exported model into an already running Gazebo world."""

    executable, family = _gazebo_executable()
    request = (
        f'sdf_filename: "{asset.model_path}", '
        f'name: "{name}", allow_renaming: true'
    )
    message_prefix = "gz.msgs" if family == "gz" else "ignition.msgs"
    command = [
        executable,
        "service",
        "-s",
        f"/world/{world}/create",
        "--reqtype",
        f"{message_prefix}.EntityFactory",
        "--reptype",
        f"{message_prefix}.Boolean",
        "--timeout",
        str(timeout_ms),
        "--req" if family == "gz" else "-r",
        request,
    ]
    return subprocess.run(command, check=True, text=True, capture_output=True)


def launch_gazebo_world(
    asset: GazeboTerrain,
    *,
    headless: bool = False,
    iterations: int | None = None,
) -> GazeboTerrain:
    """Launch a Gazebo process with the generated world already loaded."""

    executable, family = _gazebo_executable()
    command = [executable, "sim" if family == "gz" else "gazebo", "-r"]
    if headless:
        command.append("-s")
    if iterations is not None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        command.extend(("--iterations", str(iterations)))
    command.append(str(asset.world_path))
    process = subprocess.Popen(command)
    return GazeboTerrain(
        directory=asset.directory,
        mesh_path=asset.mesh_path,
        model_path=asset.model_path,
        world_path=asset.world_path,
        process=process,
    )


def apply_gazebo(
    terrain: SimulatorTerrain,
    target: object | None = None,
    *,
    output_dir: str | Path = "generated_gazebo_terrain",
    name: str = "adaptive_terrain",
    world: str = "empty",
    spawn: bool = True,
    launch: bool = False,
    headless: bool = False,
    iterations: int | None = None,
) -> GazeboTerrain:
    """Export and either launch Gazebo or spawn into a running world.

    ``target`` may be a world-name string. Set both ``spawn=False`` and
    ``launch=False`` for export-only operation.
    """

    if isinstance(target, str):
        world = target
    asset = export_gazebo(terrain, output_dir, name=name, world=world)
    if launch:
        return launch_gazebo_world(asset, headless=headless, iterations=iterations)
    if spawn:
        spawn_gazebo_model(asset, world=world, name=name)
    return asset


__all__ = [
    "GazeboTerrain",
    "apply_gazebo",
    "export_gazebo",
    "launch_gazebo_world",
    "spawn_gazebo_model",
]
