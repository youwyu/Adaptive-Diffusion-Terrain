"""Optional adapters for common robotics simulators.

No simulator is imported until its adapter is actually used. This keeps all
simulator packages out of adaptive-terrain-diffusion's required dependencies.
"""

from __future__ import annotations

from .base import SimulatorTerrain


_ALIASES = {
    "isaac": "isaaclab",
    "isaac-lab": "isaaclab",
    "isaaclab": "isaaclab",
    "mujoco": "mujoco",
    "mj": "mujoco",
    "mujoco-playground": "mujoco_playground",
    "mujoco_playground": "mujoco_playground",
    "playground": "mujoco_playground",
    "mujoco-warp": "mujoco_warp",
    "mujoco_warp": "mujoco_warp",
    "warp": "mujoco_warp",
    "bullet": "pybullet",
    "pybullet": "pybullet",
    "gazebo": "gazebo",
    "gz": "gazebo",
    "ignition": "gazebo",
}


def apply(
    backend: str,
    terrain: SimulatorTerrain | object,
    target: object | None = None,
    **kwargs,
):
    """Apply a terrain to a simulator while importing only that simulator.

    Parameters are deliberately forwarded to the backend-specific function.
    See the tutorials for lifecycle-sensitive targets such as IsaacLab and
    MuJoCo Playground.
    """

    key = _ALIASES.get(str(backend).lower().replace(" ", "-"))
    if key is None:
        supported = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(f"Unsupported simulator {backend!r}. Supported adapters: {supported}")
    if not isinstance(terrain, SimulatorTerrain):
        terrain = SimulatorTerrain(terrain)

    if key == "isaaclab":
        from .isaaclab import apply_isaaclab

        return apply_isaaclab(terrain, target, **kwargs)
    if key == "mujoco":
        from .mujoco import apply_mujoco

        return apply_mujoco(terrain, target, **kwargs)
    if key == "mujoco_playground":
        from .mujoco import apply_mujoco_playground

        return apply_mujoco_playground(terrain, target, **kwargs)
    if key == "mujoco_warp":
        from .mujoco import apply_mujoco_warp

        return apply_mujoco_warp(terrain, target, **kwargs)
    if key == "pybullet":
        from .pybullet import apply_pybullet

        return apply_pybullet(terrain, target, **kwargs)
    from .gazebo import apply_gazebo

    return apply_gazebo(terrain, target, **kwargs)


__all__ = ["SimulatorTerrain", "apply"]

