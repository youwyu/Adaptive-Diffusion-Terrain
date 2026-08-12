# Terrain Diffusion

Fast, difficulty-conditioned terrain generation with an endpoint-distilled one-step model and ONNX Runtime.

The teaser above shows all 100 difficulties across six simulator renderers.

## Install

```bash
pip install adaptive-terrain-diffusion
pip install "adaptive-terrain-diffusion[render,images]"  # optional rendering and TIFF support
```

## Python

```python
from terrain_diffusion import TerrainGenerator, render_terrains, save_terrains

generator = TerrainGenerator()  # automatically selects CUDA when available

terrains, difficulties = generator.generate(
    4,
    shape=(256, 384),
    difficulty="random",       # or any integer from 1 to 100
    seed=7,
    return_difficulties=True,
)

variations = generator.generate_from(
    terrains[0], difficulty=75, strength=0.5, variants=4
)

save_terrains("terrains", terrains, suffix=".npy")
render_terrains(terrains, save_path="terrains.png")
```

`generate_from` also accepts batches and `.txt`, `.csv`, `.npy`, `.npz`, or TIFF files. Larger-than-native maps use overlapping, blended generation automatically.

## CLI

```bash
terrain-diffusion generate --output terrains --num 4 --difficulty 60 --shape 256 384 --render
terrain-diffusion vary terrain.npy --output variants --difficulty 75 --variants 4
terrain-diffusion simulate mujoco --difficulty 100 --shape 1024 1024
```

## Simulators

`SimulatorTerrain.apply()` supports IsaacLab, MuJoCo, MuJoCo Playground, MuJoCo Warp, PyBullet, and Gazebo without adding simulator packages as dependencies.

```python
from terrain_diffusion import SimulatorTerrain

asset = SimulatorTerrain(terrains[0], horizontal_scale=0.1)
asset.apply("pybullet", client_id)
```

See the [simulator tutorial](https://github.com/youwyu/Adaptive-Diffusion-Terrain/blob/main/contexts/standalone/tutorials/simulators.md) for complete integrations.
