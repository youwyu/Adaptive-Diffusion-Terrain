"""PyBullet heightfield adapter."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .base import SimulatorTerrain


def _pybullet():
    try:
        import pybullet
    except ImportError as error:
        raise ImportError(
            "The PyBullet adapter requires pybullet in the simulator environment"
        ) from error
    return pybullet


@dataclass(frozen=True)
class PyBulletTerrain:
    client_id: int
    body_id: int
    collision_shape_id: int
    owns_connection: bool = False


def apply_pybullet(
    terrain: SimulatorTerrain,
    client_id: int | None = None,
    *,
    gui: bool = False,
    name: str = "adaptive_terrain",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.38, 0.52, 0.28, 1.0),
) -> PyBulletTerrain:
    """Create a static Bullet heightfield, connecting when needed."""

    bullet = _pybullet()
    owns_connection = client_id is None
    if client_id is None:
        client_id = bullet.connect(bullet.GUI if gui else bullet.DIRECT)
    if client_id < 0 or not bullet.isConnected(client_id):
        raise RuntimeError("Could not connect to PyBullet")

    rows, columns = terrain.shape
    heights = terrain.physical_heights
    midpoint = 0.5 * float(heights.min() + heights.max())
    collision_shape_id = bullet.createCollisionShape(
        shapeType=bullet.GEOM_HEIGHTFIELD,
        meshScale=[terrain.x_scale, terrain.y_scale, 1.0],
        heightfieldData=heights.ravel(order="C").tolist(),
        # Bullet calls its x/sample-stride dimension "rows". NumPy's x
        # dimension is columns, so these names are intentionally exchanged.
        numHeightfieldRows=columns,
        numHeightfieldColumns=rows,
        physicsClientId=client_id,
    )
    body_id = bullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape_id,
        basePosition=[position[0], position[1], position[2] + midpoint],
        physicsClientId=client_id,
    )
    bullet.changeVisualShape(body_id, -1, rgbaColor=rgba, physicsClientId=client_id)
    try:
        bullet.addUserData(body_id, "terrain_name", name, physicsClientId=client_id)
    except TypeError:
        pass
    return PyBulletTerrain(
        client_id=client_id,
        body_id=body_id,
        collision_shape_id=collision_shape_id,
        owns_connection=owns_connection,
    )


def run_pybullet_demo(
    terrain: SimulatorTerrain,
    *,
    gui: bool = True,
    steps: int | None = None,
    real_time: bool = True,
) -> PyBulletTerrain:
    """Load the terrain and a test sphere, then run a small Bullet scene."""

    bullet = _pybullet()
    handle = apply_pybullet(terrain, gui=gui)
    client_id = handle.client_id
    bullet.setGravity(0, 0, -9.81, physicsClientId=client_id)
    sphere = bullet.createCollisionShape(
        bullet.GEOM_SPHERE, radius=0.15, physicsClientId=client_id
    )
    bullet.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=sphere,
        basePosition=[0, 0, float(terrain.physical_heights.max()) + 0.5],
        physicsClientId=client_id,
    )
    count = 240 if steps is None and not gui else steps
    index = 0
    try:
        while bullet.isConnected(client_id) and (count is None or index < count):
            bullet.stepSimulation(physicsClientId=client_id)
            if gui and real_time:
                time.sleep(1.0 / 240.0)
            index += 1
    except KeyboardInterrupt:
        pass
    return handle


__all__ = ["PyBulletTerrain", "apply_pybullet", "run_pybullet_demo"]
