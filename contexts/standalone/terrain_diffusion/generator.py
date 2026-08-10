"""High-level ONNX terrain generation and variation API."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Sequence

import numpy as np

from .io import load_terrains


Difficulty = int | Sequence[int] | np.ndarray | str | None


class TerrainGenerator:
    """Generate difficulty-conditioned heightfields with a one-step ONNX model."""

    native_size = 128
    num_classes = 100
    height_range = 22.16864013671875
    asinh_scale = 0.1
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_data = 0.5
    sampling_method = "onestep"
    sampling_steps = 1
    guidance_scale = 1.0

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        device: str | None = None,
        sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        """Load the bundled model on ``auto``, ``cpu``, or ``cuda[:index]``."""

        if checkpoint is not None:
            if model is not None:
                raise ValueError("Pass model or checkpoint, not both")
            model = checkpoint
        if model is None:
            resource = importlib.resources.files("terrain_diffusion").joinpath(
                "weights/terrain_consistency_unet_difficulty100.onnx"
            )
            if not resource.is_file():
                raise FileNotFoundError("The bundled ONNX model is missing; reinstall the package")
            model = resource

        self._validate_sampler(sampling_steps, guidance_scale)
        self.model_path = str(model)
        self.checkpoint_path = self.model_path  # compatibility with the previous API
        self.session, self.device = self._create_session(self.model_path, device)
        self.providers = tuple(self.session.get_providers())
        self.metadata = {
            "architecture": "consistency-unet-32",
            "native_size": self.native_size,
            "num_classes": self.num_classes,
            "sampling_method": self.sampling_method,
            "sampling_steps": self.sampling_steps,
            "runtime": "onnxruntime",
        }

    @staticmethod
    def _create_session(model_path: str, device: str | None):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "ONNX Runtime is required. Install onnxruntime, or onnxruntime-gpu for CUDA."
            ) from error

        requested = "auto" if device is None else str(device).lower()
        automatic = requested == "auto"
        available = set(ort.get_available_providers())
        if requested == "auto":
            requested = "cuda" if "CUDAExecutionProvider" in available else "cpu"

        if requested == "cpu":
            providers: list[object] = ["CPUExecutionProvider"]
        elif requested == "cuda" or requested.startswith("cuda:"):
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA was requested, but CUDAExecutionProvider is unavailable. "
                    "Install onnxruntime-gpu and its CUDA dependencies."
                )
            device_id = int(requested.split(":", 1)[1]) if ":" in requested else 0
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
            requested = f"cuda:{device_id}"
        else:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        if requested.startswith("cuda") and session.get_providers()[0] != "CUDAExecutionProvider":
            if not automatic:
                raise RuntimeError("ONNX Runtime could not initialize its CUDA provider")
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            requested = "cpu"
        return session, requested

    @staticmethod
    def _validate_sampler(
        sampling_steps: int | None, guidance_scale: float | None
    ) -> None:
        if sampling_steps not in (None, 1):
            raise ValueError("The distilled consistency model supports exactly one step")
        if guidance_scale not in (None, 1, 1.0):
            raise ValueError("The consistency model is directly conditioned and does not use CFG")

    @staticmethod
    def _validate_shape(shape: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(shape, (int, np.integer)):
            height = width = int(shape)
        else:
            if len(shape) != 2:
                raise ValueError("shape must be an integer or a (height, width) pair")
            height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Terrain dimensions must be positive, received {(height, width)}")
        return height, width

    @staticmethod
    def _rng(seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _difficulty_labels(
        self,
        difficulty: Difficulty,
        count: int,
        generator: np.random.Generator,
        *,
        source_count: int | None = None,
        variants: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        random_requested = difficulty is None or (
            isinstance(difficulty, str) and difficulty.lower() == "random"
        )
        if random_requested:
            displayed = generator.integers(1, self.num_classes + 1, size=count, dtype=np.int64)
        else:
            if isinstance(difficulty, str):
                try:
                    difficulty = int(difficulty)
                except ValueError as error:
                    raise ValueError(
                        "difficulty must be 1..100, a sequence, or 'random'"
                    ) from error
            if np.isscalar(difficulty):
                raw_values = np.full(count, difficulty)
            else:
                raw_values = np.asarray(list(difficulty))
                if raw_values.ndim != 1:
                    raise ValueError("difficulty must be a one-dimensional sequence")
                if raw_values.size == 1:
                    raw_values = np.full(count, raw_values[0])
                elif source_count is not None and raw_values.size == source_count and variants > 1:
                    raw_values = np.repeat(raw_values, variants)
                elif raw_values.size != count:
                    raise ValueError(
                        f"Expected 1 or {count} difficulty values, received {raw_values.size}"
                    )
            numeric = np.asarray(raw_values, dtype=np.float64)
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("Difficulty labels must be whole numbers from 1 through 100")
            displayed = numeric.astype(np.int64)
            if (displayed < 1).any() or (displayed > self.num_classes).any():
                raise ValueError(f"Difficulty must be between 1 and {self.num_classes}, inclusive")
        labels = ((displayed - 1) / (self.num_classes - 1)).astype(np.float32)
        return np.ascontiguousarray(labels), displayed

    def _strength_sigma(self, strength: float) -> float:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** strength

    def _denoise(
        self, noisy: np.ndarray, sigmas: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        feeds = {
            "noisy": np.ascontiguousarray(noisy, dtype=np.float32),
            "sigma": np.ascontiguousarray(sigmas, dtype=np.float32),
            "difficulty": np.ascontiguousarray(labels, dtype=np.float32),
        }
        return self.session.run(["denoised"], feeds)[0].astype(np.float32, copy=False)

    def _sample_native(
        self,
        labels: np.ndarray,
        *,
        generator: np.random.Generator,
        sources: np.ndarray | None = None,
        strength: float = 1.0,
        batch_size: int = 16,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs = []
        sigma_value = self.sigma_max if sources is None else self._strength_sigma(strength)
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            noise = generator.standard_normal(
                (stop - start, 1, self.native_size, self.native_size), dtype=np.float32
            )
            noisy = noise * np.float32(sigma_value)
            if sources is not None:
                noisy += sources[start:stop]
            sigmas = np.full(stop - start, sigma_value, dtype=np.float32)
            outputs.append(self._denoise(noisy, sigmas, labels[start:stop]))
        return np.concatenate(outputs, axis=0)

    def _to_model_space(self, terrains: np.ndarray) -> np.ndarray:
        values = np.asarray(terrains, dtype=np.float32)
        values = values - values.min(axis=(-2, -1), keepdims=True)
        values = np.clip(values, 0.0, self.height_range)
        normalizer = np.arcsinh(self.height_range / self.asinh_scale)
        unit = np.arcsinh(values / self.asinh_scale) / normalizer
        return np.ascontiguousarray((unit * 2.0 - 1.0)[:, None], dtype=np.float32)

    def _from_model_space(self, values: np.ndarray, *, zero_floor: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 4:
            values = values[:, 0]
        unit = np.clip((values + 1.0) * 0.5, 0.0, 1.0)
        normalizer = np.arcsinh(self.height_range / self.asinh_scale)
        heights = self.asinh_scale * np.sinh(unit * normalizer)
        heights = np.clip(heights, 0.0, self.height_range).astype(np.float32, copy=False)
        if zero_floor:
            heights -= heights.min(axis=(-2, -1), keepdims=True)
        return np.ascontiguousarray(heights)

    @staticmethod
    def _resize(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Bilinearly resize NCHW arrays using half-pixel coordinates."""

        values = np.asarray(values, dtype=np.float32)
        input_height, input_width = values.shape[-2:]
        output_height, output_width = shape
        if (input_height, input_width) == shape:
            return np.ascontiguousarray(values)

        x = (np.arange(output_width, dtype=np.float32) + 0.5) * input_width / output_width - 0.5
        x = np.clip(x, 0, input_width - 1)
        x0 = np.floor(x).astype(np.intp)
        x1 = np.minimum(x0 + 1, input_width - 1)
        wx = (x - x0).reshape((1,) * (values.ndim - 1) + (output_width,))
        horizontal = values[..., x0] * (1.0 - wx) + values[..., x1] * wx

        y = (np.arange(output_height, dtype=np.float32) + 0.5) * input_height / output_height - 0.5
        y = np.clip(y, 0, input_height - 1)
        y0 = np.floor(y).astype(np.intp)
        y1 = np.minimum(y0 + 1, input_height - 1)
        wy = (y - y0).reshape((1,) * (values.ndim - 2) + (output_height, 1))
        resized = horizontal[..., y0, :] * (1.0 - wy) + horizontal[..., y1, :] * wy
        return np.ascontiguousarray(resized, dtype=np.float32)

    def _resize_output(
        self,
        values: np.ndarray,
        shape: tuple[int, int],
        *,
        zero_floor: bool,
        smooth: bool,
    ) -> np.ndarray:
        heights = self._from_model_space(values, zero_floor=False)
        if smooth:
            heights = self._smooth_output(heights)
        result = np.clip(self._resize(heights[:, None], shape)[:, 0], 0.0, self.height_range)
        if zero_floor:
            result -= result.min(axis=(-2, -1), keepdims=True)
        return np.ascontiguousarray(result, dtype=np.float32)

    @staticmethod
    def _smooth_output(terrains: np.ndarray) -> np.ndarray:
        """Apply light separable smoothing while preserving each height range."""

        values = np.ascontiguousarray(terrains, dtype=np.float32)
        original_min = values.min(axis=(-2, -1), keepdims=True)
        original_span = values.max(axis=(-2, -1), keepdims=True) - original_min
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        mode = "reflect" if min(values.shape[-2:]) > 2 else "edge"
        padded_x = np.pad(values, ((0, 0), (0, 0), (2, 2)), mode=mode)
        horizontal = sum(kernel[i] * padded_x[..., i : i + values.shape[-1]] for i in range(5))
        padded_y = np.pad(horizontal, ((0, 0), (2, 2), (0, 0)), mode=mode)
        smoothed = sum(kernel[i] * padded_y[:, i : i + values.shape[-2], :] for i in range(5))
        smoothed_min = smoothed.min(axis=(-2, -1), keepdims=True)
        smoothed_span = smoothed.max(axis=(-2, -1), keepdims=True) - smoothed_min
        restored = (smoothed - smoothed_min) * (
            original_span / np.maximum(smoothed_span, np.finfo(np.float32).eps)
        ) + original_min
        return np.where(original_span > 0, restored, values).astype(np.float32, copy=False)

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        starts = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def _generate_tiled(
        self,
        labels: np.ndarray,
        shape: tuple[int, int],
        *,
        generator: np.random.Generator,
        source_model: np.ndarray | None,
        strength: float,
        overlap: int,
        batch_size: int,
        zero_floor: bool,
    ) -> np.ndarray:
        """Denoise overlapping views of one shared noisy canvas."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= overlap < self.native_size:
            raise ValueError(f"tile_overlap must be between 0 and {self.native_size - 1}")
        height, width = shape
        canvas_height = max(height, self.native_size)
        canvas_width = max(width, self.native_size)
        y_starts = self._tile_starts(canvas_height, self.native_size, overlap)
        x_starts = self._tile_starts(canvas_width, self.native_size, overlap)
        sigma_value = self.sigma_max if source_model is None else self._strength_sigma(strength)
        noisy_canvas = generator.standard_normal(
            (len(labels), 1, canvas_height, canvas_width), dtype=np.float32
        ) * np.float32(sigma_value)
        if source_model is not None:
            noisy_canvas += self._resize(source_model, (canvas_height, canvas_width))

        tasks = [
            (terrain_index, y, x)
            for terrain_index in range(len(labels))
            for y in y_starts
            for x in x_starts
        ]
        blend_1d = np.maximum(np.hanning(self.native_size).astype(np.float32), 1.0e-3)
        blend = np.outer(blend_1d, blend_1d)
        output = np.zeros((len(labels), canvas_height, canvas_width), dtype=np.float32)
        weights = np.zeros_like(output)

        for start in range(0, len(tasks), batch_size):
            chunk = tasks[start : start + batch_size]
            patches = np.concatenate(
                [
                    noisy_canvas[index : index + 1, :, y : y + self.native_size, x : x + self.native_size]
                    for index, y, x in chunk
                ],
                axis=0,
            )
            chunk_labels = np.asarray([labels[index] for index, _, _ in chunk], dtype=np.float32)
            sigmas = np.full(len(chunk), sigma_value, dtype=np.float32)
            denoised = self._denoise(patches, sigmas, chunk_labels)[:, 0]
            for patch, (index, y, x) in zip(denoised, chunk):
                output[index, y : y + self.native_size, x : x + self.native_size] += patch * blend
                weights[index, y : y + self.native_size, x : x + self.native_size] += blend

        model_values = output / np.maximum(weights, 1.0e-12)
        return self._from_model_space(model_values[:, :height, :width], zero_floor=zero_floor)

    def _resolve_size_mode(self, size_mode: str, shape: tuple[int, int]) -> str:
        mode = size_mode.lower()
        if mode == "auto":
            return "tile" if max(shape) > self.native_size else "resize"
        if mode not in {"resize", "tile"}:
            raise ValueError("size_mode must be 'auto', 'resize', or 'tile'")
        return mode

    def generate(
        self,
        num_terrains: int = 1,
        *,
        shape: int | Sequence[int] = (128, 128),
        difficulty: Difficulty = "random",
        seed: int | None = None,
        size_mode: str = "auto",
        tile_overlap: int = 32,
        batch_size: int = 16,
        sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        zero_floor: bool = True,
        smooth: bool = True,
        return_difficulties: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Generate one or more terrain heightfields."""

        self._validate_sampler(sampling_steps, guidance_scale)
        count = int(num_terrains)
        if count <= 0:
            raise ValueError("num_terrains must be positive")
        output_shape = self._validate_shape(shape)
        generator = self._rng(seed)
        labels, displayed = self._difficulty_labels(difficulty, count, generator)
        mode = self._resolve_size_mode(size_mode, output_shape)
        if mode == "resize":
            native = self._sample_native(labels, generator=generator, batch_size=batch_size)
            terrains = self._resize_output(
                native, output_shape, zero_floor=zero_floor, smooth=smooth
            )
        else:
            terrains = self._generate_tiled(
                labels,
                output_shape,
                generator=generator,
                source_model=None,
                strength=1.0,
                overlap=tile_overlap,
                batch_size=batch_size,
                zero_floor=zero_floor,
            )
            if smooth:
                terrains = self._smooth_output(terrains)
        return (terrains, displayed) if return_difficulties else terrains

    def generate_from(
        self,
        terrains: object,
        *,
        difficulty: Difficulty = "random",
        strength: float = 0.5,
        variants: int = 1,
        shape: int | Sequence[int] | None = None,
        seed: int | None = None,
        size_mode: str = "auto",
        tile_overlap: int = 32,
        batch_size: int = 16,
        sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        zero_floor: bool = True,
        smooth: bool = True,
        return_difficulties: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Create related terrains by consistency-denoising a noisy source."""

        self._validate_sampler(sampling_steps, guidance_scale)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        variants = int(variants)
        if variants <= 0:
            raise ValueError("variants must be positive")
        source_values = load_terrains(terrains)
        source_count = len(source_values)
        source_values = np.repeat(source_values, variants, axis=0)
        output_shape = self._validate_shape(source_values.shape[-2:] if shape is None else shape)
        generator = self._rng(seed)
        labels, displayed = self._difficulty_labels(
            difficulty,
            len(source_values),
            generator,
            source_count=source_count,
            variants=variants,
        )

        if strength == 0:
            generated = self._resize(source_values[:, None], output_shape)[:, 0]
            if zero_floor:
                generated -= generated.min(axis=(-2, -1), keepdims=True)
            return (generated, displayed) if return_difficulties else generated

        source_model = self._to_model_space(source_values)
        mode = self._resolve_size_mode(size_mode, output_shape)
        if mode == "resize":
            native_source = np.clip(
                self._resize(source_model, (self.native_size, self.native_size)), -1.0, 1.0
            )
            native = self._sample_native(
                labels,
                generator=generator,
                sources=native_source,
                strength=strength,
                batch_size=batch_size,
            )
            generated = self._resize_output(
                native, output_shape, zero_floor=zero_floor, smooth=smooth
            )
        else:
            generated = self._generate_tiled(
                labels,
                output_shape,
                generator=generator,
                source_model=self._resize(source_model, output_shape),
                strength=strength,
                overlap=tile_overlap,
                batch_size=batch_size,
                zero_floor=zero_floor,
            )
            if smooth:
                generated = self._smooth_output(generated)
        return (generated, displayed) if return_difficulties else generated

    vary = generate_from
