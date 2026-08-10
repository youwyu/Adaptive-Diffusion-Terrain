"""MuJoCo, MuJoCo Playground, and MuJoCo Warp adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from .base import SimulatorTerrain


def _mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise ImportError(
            "The MuJoCo adapters require MuJoCo in the simulator environment"
        ) from error
    return mujoco


def _normalized_heights(terrain: SimulatorTerrain) -> tuple[np.ndarray, float, float]:
    heights = terrain.physical_heights
    floor = float(heights.min())
    span = float(np.ptp(heights))
    if span <= np.finfo(np.float32).eps:
        normalized = np.zeros_like(heights, dtype=np.float32)
        size_z = 1.0e-6
    else:
        normalized = (heights - floor) / span
        size_z = span
    return np.ascontiguousarray(normalized, dtype=np.float32), floor, size_z


def build_mujoco_model(
    terrain: SimulatorTerrain,
    *,
    hfield_name: str = "adaptive_terrain",
    include_demo_body: bool = True,
):
    """Create a standalone MjModel containing the generated heightfield."""

    mujoco = _mujoco()
    normalized, floor, size_z = _normalized_heights(terrain)
    rows, columns = terrain.shape
    half_x = max(terrain.size[0] * 0.5, terrain.x_scale * 0.5)
    half_y = max(terrain.size[1] * 0.5, terrain.y_scale * 0.5)
    name = escape(hfield_name, {'"': "&quot;"})
    demo_body = ""
    if include_demo_body:
        z = float(terrain.physical_heights.max()) + 0.4
        demo_body = f"""
    <body name="demo_ball" pos="0 0 {z:.9g}">
      <freejoint/>
      <geom type="sphere" size="0.15" mass="1" rgba="0.9 0.25 0.15 1"/>
    </body>"""
    xml = f"""<mujoco model="adaptive_terrain">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/></visual>
  <asset>
    <hfield name="{name}" nrow="{rows}" ncol="{columns}"
      size="{half_x:.9g} {half_y:.9g} {size_z:.9g} 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1"/>
    <geom name="terrain" type="hfield" hfield="{name}" pos="0 0 {floor:.9g}"
      friction="1.0 0.005 0.0001" rgba="0.38 0.52 0.28 1"/>{demo_body}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = normalized.ravel(order="C")
    return model


def apply_mujoco(
    terrain: SimulatorTerrain,
    model: object | None = None,
    *,
    hfield_name: str = "adaptive_terrain",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_demo_body: bool = True,
):
    """Create a model or fill an existing same-shaped MuJoCo hfield.

    Existing compiled models cannot change topology. They must contain a
    placeholder ``hfield`` with the same ``nrow`` and ``ncol`` as the terrain.
    """

    mujoco = _mujoco()
    if model is None:
        return build_mujoco_model(
            terrain, hfield_name=hfield_name, include_demo_body=include_demo_body
        )

    hfield_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name
    )
    if hfield_id < 0:
        raise ValueError(
            f"MuJoCo model has no hfield named {hfield_name!r}; add a placeholder before compilation"
        )
    expected = terrain.shape
    actual = (int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id]))
    if actual != expected:
        raise ValueError(
            f"MuJoCo hfield {hfield_name!r} has shape {actual}, but terrain is {expected}; "
            "hfield dimensions are fixed when MJCF is compiled"
        )

    normalized, floor, size_z = _normalized_heights(terrain)
    address = int(model.hfield_adr[hfield_id])
    count = actual[0] * actual[1]
    model.hfield_data[address : address + count] = normalized.ravel(order="C")
    model.hfield_size[hfield_id] = (
        max(terrain.size[0] * 0.5, terrain.x_scale * 0.5),
        max(terrain.size[1] * 0.5, terrain.y_scale * 0.5),
        size_z,
        0.1,
    )
    matching_geoms = np.flatnonzero(
        (model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
        & (model.geom_dataid == hfield_id)
    )
    for geom_id in matching_geoms:
        model.geom_pos[geom_id] = (
            float(position[0]),
            float(position[1]),
            float(position[2]) + floor,
        )
    return model


def apply_mujoco_playground(
    terrain: SimulatorTerrain,
    env: object | None,
    *,
    hfield_name: str = "adaptive_terrain",
    rebuild: bool = True,
    impl: str | None = None,
):
    """Fill a Playground environment's placeholder and rebuild its MJX model.

    Call this immediately after constructing a custom ``MjxEnv`` and before
    invoking JIT-compiled reset or step functions.
    """

    if env is None or not hasattr(env, "mj_model"):
        raise TypeError("MuJoCo Playground requires target=an initialized MjxEnv")
    apply_mujoco(terrain, env.mj_model, hfield_name=hfield_name)
    if rebuild:
        try:
            from mujoco import mjx
        except ImportError as error:
            raise ImportError("MuJoCo Playground requires the mujoco-mjx package") from error
        if impl is None:
            impl = getattr(getattr(env, "_config", None), "impl", None)
        rebuilt = mjx.put_model(env.mj_model, impl=impl)
        if not hasattr(env, "_mjx_model"):
            raise TypeError(
                "This Playground environment does not expose _mjx_model; rebuild it in the environment constructor"
            )
        env._mjx_model = rebuilt
    return env


@dataclass(frozen=True)
class MujocoWarpTerrain:
    """Host MuJoCo model and the corresponding device-side Warp model."""

    mj_model: Any
    warp_model: Any


def apply_mujoco_warp(
    terrain: SimulatorTerrain,
    model: object | None = None,
    *,
    hfield_name: str = "adaptive_terrain",
    include_demo_body: bool = True,
    batch_sizes: dict[str, int] | None = None,
) -> MujocoWarpTerrain:
    """Apply the terrain and upload the resulting model through MuJoCo Warp."""

    mj_model = apply_mujoco(
        terrain,
        model,
        hfield_name=hfield_name,
        include_demo_body=include_demo_body,
    )
    try:
        import mujoco_warp
    except ImportError as error:
        raise ImportError(
            "The MuJoCo Warp adapter requires mujoco-warp in the simulator environment"
        ) from error
    warp_model = mujoco_warp.put_model(mj_model, batch_sizes=batch_sizes)
    return MujocoWarpTerrain(mj_model=mj_model, warp_model=warp_model)


__all__ = [
    "MujocoWarpTerrain",
    "apply_mujoco",
    "apply_mujoco_playground",
    "apply_mujoco_warp",
    "build_mujoco_model",
]

