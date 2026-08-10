"""Optional Matplotlib-based 3D heightfield rendering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .io import load_terrain, load_terrains


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "3D rendering requires Matplotlib: pip install adaptive-terrain-diffusion[render]"
        ) from error
    return plt


def render_terrain(
    terrain: object,
    *,
    horizontal_scale: float = 0.1,
    vertical_scale: float = 1.0,
    stride: int = 1,
    cmap: str = "terrain",
    elevation: float = 35.0,
    azimuth: float = -135.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    ax=None,
):
    """Render one 2D heightfield as a 3D surface and return ``(figure, axes)``."""

    if horizontal_scale <= 0 or vertical_scale <= 0:
        raise ValueError("horizontal_scale and vertical_scale must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    plt = _matplotlib()
    value = load_terrain(terrain)
    sampled = value[::stride, ::stride] * vertical_scale
    y = np.arange(0, value.shape[0], stride, dtype=np.float32) * horizontal_scale
    x = np.arange(0, value.shape[1], stride, dtype=np.float32) * horizontal_scale
    xx, yy = np.meshgrid(x, y)
    if ax is None:
        figure = plt.figure(figsize=(8, 6))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    ax.plot_surface(xx, yy, sampled, cmap=cmap, linewidth=0, antialiased=True)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")
    if title:
        ax.set_title(title)
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return figure, ax


def render_terrains(
    terrains: object,
    *,
    columns: int | None = None,
    horizontal_scale: float = 0.1,
    vertical_scale: float = 1.0,
    stride: int = 2,
    cmap: str = "terrain",
    elevation: float = 35.0,
    azimuth: float = -135.0,
    titles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Render a batch of heightfields in a shared 3D figure."""

    plt = _matplotlib()
    values = load_terrains(terrains)
    count = len(values)
    columns = columns or int(math.ceil(math.sqrt(count)))
    if columns <= 0:
        raise ValueError("columns must be positive")
    rows = int(math.ceil(count / columns))
    figure = plt.figure(figsize=(4 * columns, 3.5 * rows))
    for index, terrain in enumerate(values):
        ax = figure.add_subplot(rows, columns, index + 1, projection="3d")
        render_terrain(
            terrain,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            stride=stride,
            cmap=cmap,
            elevation=elevation,
            azimuth=azimuth,
            title=titles[index] if titles and index < len(titles) else None,
            ax=ax,
        )
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return figure

