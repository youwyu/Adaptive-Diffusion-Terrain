"""Heightfield loading, saving, and simulator conversion utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[Sequence[float]]


def _clean_heightfield(value: object) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    terrain = np.asarray(value, dtype=np.float32)
    if terrain.ndim != 2:
        raise ValueError(f"A terrain must be a 2D heightfield, received shape {terrain.shape}")
    valid = np.isfinite(terrain) & (np.abs(terrain) < 1.0e20)
    if not valid.any():
        return np.zeros_like(terrain, dtype=np.float32)
    return np.where(valid, terrain, terrain[valid].min()).astype(np.float32, copy=False)


def load_terrain(source: str | Path | ArrayLike | object) -> np.ndarray:
    """Load one 2D heightfield from an array, tensor, TXT/CSV, NPY/NPZ, or image."""

    if not isinstance(source, (str, Path)):
        return _clean_heightfield(source)

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Terrain file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _clean_heightfield(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"Terrain archive is empty: {path}")
            key = "heightfield" if "heightfield" in archive.files else archive.files[0]
            return _clean_heightfield(archive[key])
    if suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        try:
            return _clean_heightfield(np.loadtxt(path, delimiter=delimiter, dtype=np.float32))
        except ValueError:
            return _clean_heightfield(np.loadtxt(path, delimiter=",", dtype=np.float32))
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError(
                "Image terrain loading requires Pillow: pip install adaptive-terrain-diffusion[images]"
            ) from error
        return _clean_heightfield(np.asarray(Image.open(path), dtype=np.float32))
    raise ValueError(f"Unsupported terrain format {suffix!r}: {path}")


def load_terrains(sources: object) -> np.ndarray:
    """Load one or more same-shaped terrains into an ``(N, H, W)`` float32 array."""

    if hasattr(sources, "detach") and hasattr(sources, "cpu"):
        sources = sources.detach().cpu().numpy()
    if isinstance(sources, np.ndarray):
        values = sources.astype(np.float32, copy=False)
        if values.ndim == 2:
            return _clean_heightfield(values)[None]
        if values.ndim == 3:
            return np.stack([_clean_heightfield(value) for value in values])
        raise ValueError(f"Terrains must have shape (H, W) or (N, H, W), got {values.shape}")
    if isinstance(sources, (str, Path)):
        return load_terrain(sources)[None]
    try:
        numeric = np.asarray(sources)
    except (TypeError, ValueError):
        numeric = np.asarray([], dtype=object)
    if numeric.dtype != object and numeric.ndim in {2, 3} and np.issubdtype(
        numeric.dtype, np.number
    ):
        return load_terrains(numeric)
    try:
        loaded = [load_terrain(source) for source in sources]
    except TypeError as error:
        raise TypeError("Expected a terrain or a sequence of terrains") from error
    if not loaded:
        raise ValueError("At least one source terrain is required")
    shapes = {terrain.shape for terrain in loaded}
    if len(shapes) != 1:
        raise ValueError(f"All source terrains must have the same shape, received {sorted(shapes)}")
    return np.stack(loaded).astype(np.float32, copy=False)


def save_terrain(path: str | Path, terrain: object) -> Path:
    """Save one heightfield. The format is selected from the filename suffix."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = _clean_heightfield(terrain)
    suffix = destination.suffix.lower()
    if suffix == ".npy":
        np.save(destination, value, allow_pickle=False)
    elif suffix == ".npz":
        np.savez_compressed(destination, heightfield=value)
    elif suffix in {".txt", ".csv"}:
        np.savetxt(destination, value, delimiter=",")
    elif suffix in {".tif", ".tiff", ".png"}:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError(
                "Image terrain saving requires Pillow: pip install adaptive-terrain-diffusion[images]"
            ) from error
        if suffix == ".png":
            span = float(np.ptp(value))
            image = np.zeros_like(value, dtype=np.uint16) if span == 0 else np.round(
                (value - value.min()) / span * 65535
            ).astype(np.uint16)
            Image.fromarray(image).save(destination)
        else:
            Image.fromarray(value, mode="F").save(destination)
    else:
        raise ValueError(
            f"Unsupported output format {suffix!r}; use .txt, .csv, .npy, .npz, .tif, or .png"
        )
    return destination


def save_terrains(
    directory: str | Path,
    terrains: object,
    *,
    suffix: str = ".txt",
    prefix: str = "terrain",
) -> list[Path]:
    """Save a batch as numbered files and return their paths."""

    values = load_terrains(terrains)
    output_dir = Path(directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    digits = max(3, len(str(len(values) - 1)))
    return [
        save_terrain(output_dir / f"{prefix}_{index:0{digits}d}{suffix}", terrain)
        for index, terrain in enumerate(values)
    ]


def compose_terrain_grid(
    terrains: object,
    *,
    rows: int | None = None,
    cols: int | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Compose same-sized terrains into the combined map used by ``TerrainContext``."""

    values = load_terrains(terrains)
    count, height, width = values.shape
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
    elif rows is None:
        if cols is None or cols <= 0:
            raise ValueError("cols must be positive")
        rows = int(math.ceil(count / cols))
    elif cols is None:
        if rows <= 0:
            raise ValueError("rows must be positive")
        cols = int(math.ceil(count / rows))
    if rows <= 0 or cols <= 0 or rows * cols < count:
        raise ValueError(f"A {rows}x{cols} grid cannot hold {count} terrains")
    combined = np.full((rows * height, cols * width), fill_value, dtype=np.float32)
    for index, terrain in enumerate(values):
        row, col = divmod(index, cols)
        combined[row * height : (row + 1) * height, col * width : (col + 1) * width] = terrain
    return combined


def to_raw_heightfield(
    terrain: object,
    *,
    vertical_scale: float = 0.1,
    dtype: np.dtype | type = np.int16,
) -> np.ndarray:
    """Convert model heights in meters to a simulator's integer height samples.

    The physical height represented by the result is ``raw * vertical_scale``.
    This is the representation expected by this repository's ``TerrainContext``
    and Isaac-style heightfield APIs.
    """

    if vertical_scale <= 0:
        raise ValueError("vertical_scale must be positive")
    if hasattr(terrain, "detach") and hasattr(terrain, "cpu"):
        terrain = terrain.detach().cpu().numpy()
    value = np.asarray(terrain, dtype=np.float32)
    was_2d = value.ndim == 2
    if value.ndim not in {2, 3}:
        raise ValueError(
            f"Heightfields must have shape (H, W) or (N, H, W), got {value.shape}"
        )
    batch = value[None] if was_2d else value
    batch = np.stack([_clean_heightfield(item) for item in batch])
    info = np.iinfo(dtype)
    raw = np.clip(np.rint(batch / vertical_scale), info.min, info.max).astype(dtype)
    return raw[0] if was_2d else raw
