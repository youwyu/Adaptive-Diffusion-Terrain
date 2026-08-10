"""Simulator-neutral terrain representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io import load_terrain


HorizontalScale = float | Sequence[float]


def _xy_scale(value: HorizontalScale) -> tuple[float, float]:
    if np.isscalar(value):
        x_scale = y_scale = float(value)
    else:
        if len(value) != 2:
            raise ValueError("horizontal_scale must be a scalar or an (x, y) pair")
        x_scale, y_scale = float(value[0]), float(value[1])
    if not np.isfinite([x_scale, y_scale]).all() or x_scale <= 0 or y_scale <= 0:
        raise ValueError("horizontal_scale values must be finite and positive")
    return x_scale, y_scale


@dataclass(frozen=True)
class SimulatorTerrain:
    """One heightfield plus the physical spacing needed by a simulator.

    ``heights`` are interpreted as meters. ``horizontal_scale`` is the sample
    spacing in meters, either a scalar or ``(x, y)``. ``vertical_scale`` is an
    optional multiplier and should normally remain 1.0 for generated terrain.
    """

    heights: object
    horizontal_scale: HorizontalScale = 0.1
    vertical_scale: float = 1.0

    def __post_init__(self) -> None:
        heights = np.ascontiguousarray(load_terrain(self.heights), dtype=np.float32)
        if min(heights.shape) < 2:
            raise ValueError("Simulator terrain needs at least two rows and two columns")
        if not np.isfinite(self.vertical_scale) or self.vertical_scale <= 0:
            raise ValueError("vertical_scale must be finite and positive")
        x_scale, y_scale = _xy_scale(self.horizontal_scale)
        object.__setattr__(self, "heights", heights)
        object.__setattr__(self, "horizontal_scale", (x_scale, y_scale))
        object.__setattr__(self, "vertical_scale", float(self.vertical_scale))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.heights.shape)

    @property
    def x_scale(self) -> float:
        return self.horizontal_scale[0]

    @property
    def y_scale(self) -> float:
        return self.horizontal_scale[1]

    @property
    def physical_heights(self) -> np.ndarray:
        return np.asarray(self.heights * self.vertical_scale, dtype=np.float32)

    @property
    def size(self) -> tuple[float, float, float]:
        """Physical ``(x, y, height-range)`` in meters."""

        rows, columns = self.shape
        return (
            (columns - 1) * self.x_scale,
            (rows - 1) * self.y_scale,
            float(np.ptp(self.physical_heights)),
        )

    def to_mesh(self, *, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return vertices and triangle faces without importing Trimesh."""

        rows, columns = self.shape
        x = np.arange(columns, dtype=np.float32) * self.x_scale
        y = np.arange(rows, dtype=np.float32) * self.y_scale
        if center:
            x -= x[-1] * 0.5
            y -= y[-1] * 0.5
        grid_x, grid_y = np.meshgrid(x, y)
        vertices = np.column_stack(
            (grid_x.ravel(), grid_y.ravel(), self.physical_heights.ravel())
        ).astype(np.float32, copy=False)

        upper_left = np.arange((rows - 1) * (columns - 1), dtype=np.int64)
        upper_left += np.repeat(np.arange(rows - 1, dtype=np.int64), columns - 1)
        faces = np.empty((upper_left.size * 2, 3), dtype=np.int64)
        faces[0::2] = np.column_stack(
            (upper_left, upper_left + 1, upper_left + columns)
        )
        faces[1::2] = np.column_stack(
            (upper_left + 1, upper_left + columns + 1, upper_left + columns)
        )
        return vertices, faces

    def to_trimesh(self):
        """Return a ``trimesh.Trimesh`` using a lazy optional import."""

        try:
            import trimesh
        except ImportError as error:
            raise ImportError(
                "IsaacLab mesh conversion requires Trimesh; install it in the simulator environment"
            ) from error
        vertices, faces = self.to_mesh()
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def save_obj(self, path: str | Path) -> Path:
        """Write a portable OBJ mesh using only NumPy."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        vertices, faces = self.to_mesh()
        triangle_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], triangle_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, np.finfo(np.float32).eps)
        lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
        lines.extend(f"vn {x:.9g} {y:.9g} {z:.9g}" for x, y, z in normals)
        lines.extend(
            f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
            for a, b, c in faces
        )
        destination.write_text("\n".join(lines) + "\n")
        return destination

    def apply(self, backend: str, target: object | None = None, **kwargs):
        """Apply this terrain using one of the lazy simulator adapters."""

        from . import apply

        return apply(backend, self, target=target, **kwargs)
