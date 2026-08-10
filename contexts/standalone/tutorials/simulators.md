# Using generated terrain in robotics simulators

This tutorial covers three workflows:

1. a one-command standalone demo for PyBullet, MuJoCo, and Gazebo;
2. a shared Python `SimulatorTerrain.apply(...)` API;
3. the exact integration point for IsaacLab, MuJoCo Playground, and MuJoCo
   Warp.

## Installation and isolation

The generator declares only NumPy and ONNX Runtime as required dependencies. Keep
simulator dependencies in the simulator environment; do not add them to the
generator's `pyproject.toml`.

For the `pings` environment used during development:

```shell
uv pip install --python /home/soicroot/miniconda3/envs/pings/bin/python \
  --no-cache -e /home/soicroot/Youwei/research/CoRL24/Adaptive-Diffusion-Terrain/contexts/standalone

uv pip install --python /home/soicroot/miniconda3/envs/pings/bin/python \
  --no-cache mujoco mujoco-warp playground pybullet trimesh lxml
```

The PyPI distribution named `playground` provides the
`mujoco_playground` module. IsaacLab must be installed together with its
matching Isaac Sim release; follow its installation instructions instead of
mixing a second Isaac Sim runtime into this environment. Gazebo Sim is normally
installed through the operating system.

Run scripts in `pings` with:

```shell
conda run -n pings python your_script.py
```

## Scale and coordinate conventions

`TerrainGenerator` outputs height in meters. Construct the simulator wrapper as
follows:

```python
from terrain_diffusion import SimulatorTerrain, TerrainGenerator

generator = TerrainGenerator(device="cuda")
heights = generator.generate(1, shape=(128, 128), difficulty=60, seed=7)[0]
terrain = SimulatorTerrain(
    heights,
    horizontal_scale=0.1,  # 10 cm between adjacent samples in x and y
    vertical_scale=1.0,    # heights are already meters
)
```

A 128×128 grid at 0.1 m spacing spans 12.7×12.7 m because there are 127
intervals. Do not call `to_raw_heightfield` for these adapters: that function is
only for APIs that subsequently multiply integer samples by a separate vertical
scale.

All adapters center the terrain around x=y=0. Heights retain their physical
value. A non-square map and different x/y resolution are supported with
`horizontal_scale=(x_spacing, y_spacing)`.

## One-command CLI

These commands generate a single terrain, store its source heightfield under
the output directory, and load it into the backend:

```shell
terrain-diffusion simulate pybullet --difficulty 60 --output runs/bullet_60
terrain-diffusion simulate mujoco --difficulty 60 --output runs/mujoco_60
terrain-diffusion simulate gazebo --difficulty 60 --output runs/gazebo_60
```

The PyBullet and MuJoCo scenes add a sphere so collision can be inspected.
Their GUI stays open until it is closed. Gazebo launches the generated world.
For a noninteractive smoke test:

```shell
terrain-diffusion simulate pybullet --difficulty 60 --headless --run-steps 240
terrain-diffusion simulate mujoco --difficulty 60 --headless --run-steps 240
terrain-diffusion simulate gazebo --difficulty 60 --headless --run-steps 240
```

Use `--terrain path/to/heightfield.npy` to avoid generation and `--export-only`
to write assets without starting the simulator. IsaacLab and Playground are not
CLI-launched because terrain must be inserted during their scene/environment
construction.

## Shared Python API

The same object dispatches to every backend using lazy imports:

```python
from terrain_diffusion import SimulatorTerrain, apply_terrain

terrain = SimulatorTerrain(heights, horizontal_scale=0.1)

# Equivalent forms:
result = terrain.apply("pybullet", client_id)
result = apply_terrain("pybullet", terrain, target=client_id)
```

Backend return values retain useful IDs or host/device models. The simulator
still owns its normal step, reset, viewer, and shutdown lifecycle.

## PyBullet

`apply` accepts an existing physics client or creates a DIRECT connection when
the target is omitted:

```python
import pybullet as p
from terrain_diffusion import SimulatorTerrain, TerrainGenerator

client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81, physicsClientId=client)

heights = TerrainGenerator().generate(1, difficulty=60)[0]
terrain = SimulatorTerrain(heights, horizontal_scale=0.1)
handle = terrain.apply("pybullet", client)

print(handle.body_id, handle.collision_shape_id)
while p.isConnected(client):
    p.stepSimulation(physicsClientId=client)
```

The adapter passes metric heights directly, uses
`meshScale=[x_spacing, y_spacing, 1]`, and offsets the Bullet body by the
heightfield midpoint so the returned z values remain physical heights.

## MuJoCo

Calling without a target creates an `MjModel` with a heightfield and demo
sphere:

```python
import mujoco
import mujoco.viewer

model = terrain.apply("mujoco")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
```

MuJoCo stores heightfield pixels in [0, 1]. The adapter performs only that
storage normalization and writes the real x/y half-extents, elevation range,
and minimum-height offset into the model, so world dimensions remain metric.

To update an existing compiled model, its MJCF must already contain a
same-shaped placeholder because `nrow` and `ncol` are compile-time topology:

```xml
<asset>
  <hfield name="adaptive_terrain" nrow="128" ncol="128"
          size="6.35 6.35 1 0.1"/>
</asset>
<worldbody>
  <geom name="floor" type="hfield" hfield="adaptive_terrain"/>
</worldbody>
```

Then apply it before creating long-lived render or accelerator state:

```python
model = mujoco.MjModel.from_xml_path("robot_with_placeholder.xml")
terrain.apply("mujoco", model, hfield_name="adaptive_terrain")
data = mujoco.MjData(model)
```

If a renderer already exists, upload the changed heightfield to its rendering
context with MuJoCo's `mjr_uploadHField`; creating the renderer after `apply`
avoids that extra step.

## MuJoCo Playground

Playground's rough locomotion environments already contain a 256×256 hfield
named `hfield`. Generate that shape, apply immediately after loading the env,
and let the adapter rebuild `_mjx_model` before any JIT-compiled reset/step:

```python
import jax
from mujoco_playground import registry
from terrain_diffusion import SimulatorTerrain, TerrainGenerator

heights = TerrainGenerator().generate(
    1, shape=(256, 256), difficulty=60, seed=7
)[0]

# The stock rough terrain spans 20 m: use 20 / (256 - 1) meters per sample.
terrain = SimulatorTerrain(heights, horizontal_scale=20.0 / 255.0)
env = registry.load("Go1JoystickRoughTerrain")
terrain.apply("mujoco-playground", env, hfield_name="hfield")

reset = jax.jit(env.reset)
step = jax.jit(env.step)
state = reset(jax.random.PRNGKey(0))
action = jax.numpy.zeros(env.action_size)
state = step(state, action)
```

For a new Playground environment, add the MuJoCo placeholder shown above to
its scene XML. Apply the terrain in the environment constructor after
`self._mj_model` is compiled and before assigning `self._mjx_model`. If you do
that directly inside the constructor, use `terrain.apply("mujoco",
self._mj_model, ...)`, followed by `mjx.put_model` exactly once.

## MuJoCo Warp

The Warp adapter first creates or updates the host `MjModel`, then uploads it
with `mujoco_warp.put_model`:

```python
import mujoco_warp

models = terrain.apply("mujoco-warp")
data = mujoco_warp.make_data(models.mj_model, nworld=4096)

for _ in range(1000):
    mujoco_warp.step(models.warp_model, data)
```

Pass an existing placeholder-containing `MjModel` as the second argument when
the robot is already defined:

```python
models = terrain.apply(
    "mujoco-warp", robot_model, hfield_name="adaptive_terrain"
)
```

Upload only after filling the host hfield. Changing `MjModel.hfield_data`
after `put_model` does not mutate the device-side Warp model.

## IsaacLab

IsaacLab must launch Isaac Sim before importing its simulation modules. In an
existing task, get the scene's `TerrainImporter`, remove or disable its old
ground, and call the adapter before `sim.reset()`:

```python
# AppLauncher must already be running at this point.
from isaaclab.sim.utils.prims import delete_prim
from terrain_diffusion import SimulatorTerrain, TerrainGenerator

importer = scene.terrain  # the initialized isaaclab.terrains.TerrainImporter

# A TerrainImporter automatically creates its configured terrain. Replace it.
for prim_path in list(importer.terrain_prim_paths):
    delete_prim(prim_path)
importer.terrain_prim_paths.clear()

heights = TerrainGenerator(device="cuda").generate(1, difficulty=60)[0]
terrain = SimulatorTerrain(heights, horizontal_scale=0.1)
mesh = terrain.apply(
    "isaaclab",
    importer,
    name="adaptive_terrain",
    configure_env_origins=True,
    origins=None,  # grid origins from importer.cfg.env_spacing
)
```

`target` must be a live `TerrainImporter`; the method calls its public
`import_mesh(name, trimesh.Trimesh)` entry point. For a manager-based task,
perform the replacement during scene construction, before cloning/resetting
environments. Set `env_spacing` in `TerrainImporterCfg` when requesting grid
origins. For curriculum origins, pass an array shaped `(levels, types, 3)` as
`origins` instead.

Run the complete example with the IsaacLab launcher, not plain Python:

```shell
cd /path/to/IsaacLab
./isaaclab.sh -p \
  /path/to/adaptive-terrain-diffusion/examples/simulators/isaaclab.py
```

## Gazebo Sim

The adapter exports a metric OBJ plus `model.sdf`, `model.config`, and a
directly launchable `world.sdf`:

```python
asset = terrain.apply(
    "gazebo",
    output_dir="generated_gazebo_terrain",
    spawn=False,
    launch=False,
)
print(asset.world_path)
```

Start the exported world yourself:

```shell
gz sim -r generated_gazebo_terrain/world.sdf
# Gazebo Fortress uses the older command name:
ign gazebo -r generated_gazebo_terrain/world.sdf
```

Or spawn into an already running world through its `/world/<name>/create`
service:

```python
terrain.apply(
    "gazebo",
    target="empty",
    output_dir="generated_gazebo_terrain",
    spawn=True,
)
```

For programmatic launch, set `launch=True`; the returned
`GazeboTerrain.process` is the child process to wait for or terminate. The
export uses a static triangle mesh for consistent geometry across Gazebo
versions and does not require Pillow or a grayscale heightmap encoding.
