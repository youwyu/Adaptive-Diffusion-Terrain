"""Build the README GIF from real terrain samples and simulator renderers.

Run from the repository root in the ``pings`` environment after sourcing ROS:

    source /opt/ros/humble/setup.bash
    MUJOCO_GL=egl python contexts/standalone/tools/build_teaser.py

The six panels use independent fixed latent noises. Within each panel, the
continuous difficulty condition changes from 1 through 100 while the robot
advances along a visible trajectory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from terrain_diffusion import SimulatorTerrain, TerrainGenerator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "assets" / "consistency-terrain-simulators.gif"
CAMERA_WORLD = Path(__file__).with_name("teaser") / "gazebo_camera.world"
TEASER_TOOLS = Path(__file__).with_name("teaser")
PANEL_SIZE = (400, 300)
VERTICAL_SCALE = 1.0
HORIZONTAL_SCALE = 0.1
BACKENDS = (
    ("isaaclab", "Isaac Lab · Go2"),
    ("mjlab", "MJLab · G1"),
    ("mujoco", "MuJoCo · Humanoid"),
    ("playground", "MuJoCo Playground · Go1"),
    ("pybullet", "PyBullet · Husky"),
    ("gazebo", "Gazebo · Rover"),
)
SEEDS = {
    "isaaclab": 5519,
    "mjlab": 6619,
    "mujoco": 1103,
    "playground": 2207,
    "pybullet": 3301,
    "gazebo": 4409,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/adtg-real-teaser"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path.home() / "miniconda3/envs/isaaclab/bin/python",
    )
    parser.add_argument(
        "--mjlab-python",
        type=Path,
        default=Path.home() / "miniconda3/envs/mjlab/bin/python",
    )
    return parser.parse_args()


def fixed_latent_terrains(generator: TerrainGenerator, seed: int) -> np.ndarray:
    """Evaluate all 100 conditions using one fixed latent noise."""

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((1, 1, 128, 128), dtype=np.float32)
    noisy = np.repeat(latent * np.float32(generator.sigma_max), 100, axis=0)
    labels = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    sigma = np.full(100, generator.sigma_max, dtype=np.float32)
    chunks = []
    for start in range(0, 100, 16):
        stop = min(start + 16, 100)
        chunks.append(generator._denoise(noisy[start:stop], sigma[start:stop], labels[start:stop]))
    heights = generator._from_model_space(np.concatenate(chunks), zero_floor=True)
    return generator._smooth_output(heights)


def prepare_terrains(work_dir: Path, device: str) -> dict[str, np.ndarray]:
    cache = work_dir / "terrains_6x100x128x128.npz"
    old_cache = work_dir / "terrains_4x100x128x128.npz"
    terrains = {}
    source = cache if cache.is_file() else old_cache
    if source.is_file():
        with np.load(source) as payload:
            terrains = {
                name: payload[name]
                for name, _ in BACKENDS
                if name in payload.files
            }
    missing = [name for name, _ in BACKENDS if name not in terrains]
    if missing:
        generator = TerrainGenerator(device=device)
        for name in missing:
            terrains[name] = fixed_latent_terrains(generator, SEEDS[name])
    np.savez_compressed(cache, **terrains, difficulties=np.arange(1, 101))
    return terrains


def scaled(terrain: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(terrain * np.float32(VERTICAL_SCALE))


def motion_pose(frame: int, x_radius: float, y_radius: float) -> tuple[float, float, float]:
    phase = 2.0 * np.pi * (frame - 1) / 100.0
    return float(phase), float(x_radius * np.sin(phase)), float(y_radius * np.cos(phase))


def terrain_height(terrain: np.ndarray, x: float, y: float, spacing: float) -> float:
    rows, columns = terrain.shape
    column = int(np.clip(round(x / spacing + (columns - 1) * 0.5), 0, columns - 1))
    row = int(np.clip(round(y / spacing + (rows - 1) * 0.5), 0, rows - 1))
    return float(terrain[row, column])


def save_frame(directory: Path, index: int, array: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
        directory / f"{index:03d}.png", optimize=True
    )


def capture_mujoco(terrains: np.ndarray, directory: Path, work_dir: Path) -> None:
    import mujoco

    if (directory / "100.png").is_file():
        return
    source = Path(mujoco.__file__).parent / "mjx/test_data/humanoid/humanoid.xml"
    dummy = work_dir / "mujoco_hfield.png"
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(dummy)
    spec = mujoco.MjSpec.from_file(str(source))
    spec.add_hfield(
        name="generated_terrain",
        file=str(dummy),
        size=[6.35, 6.35, 6.0, 0.1],
    )
    floor = spec.geom("floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "generated_terrain"
    floor.material = ""
    floor.rgba = [0.24, 0.43, 0.20, 1.0]
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.75
    base_qpos = model.qpos0.copy()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_SIZE[1], width=PANEL_SIZE[0])
    camera = mujoco.MjvCamera()
    camera.distance = 8.0
    camera.azimuth = 135
    camera.elevation = -27
    for difficulty, raw in enumerate(terrains, 1):
        terrain = scaled(raw)
        span = max(float(np.ptp(terrain)), 1.0e-5)
        model.hfield_data[:] = terrain.ravel() / span
        model.hfield_size[0] = [6.35, 6.35, span, 0.1]
        mujoco.mjr_uploadHField(model, renderer._mjr_context, 0)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = base_qpos
        phase, x, y = motion_pose(difficulty, 2.35, 1.55)
        data.qpos[0:3] = [
            x,
            y,
            terrain_height(terrain, x, y, HORIZONTAL_SCALE) + 1.35,
        ]
        yaw = phase + np.pi * 0.5
        data.qpos[3:7] = [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)]
        if model.nq > 7:
            gait = 0.18 * np.sin(phase * 4.0 + np.arange(model.nq - 7) * np.pi)
            data.qpos[7:] = base_qpos[7:] + gait
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        camera.lookat[:] = [0.0, 0.0, min(span * 0.3, 1.5)]
        renderer.update_scene(data, camera=camera)
        save_frame(directory, difficulty, renderer.render())
        if difficulty % 10 == 0:
            print(f"MuJoCo {difficulty}/100", flush=True)
    renderer.close()


def capture_playground(terrains: np.ndarray, directory: Path) -> None:
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    from mujoco_playground import registry

    if (directory / "100.png").is_file():
        return
    env = registry.load("Go1JoystickRoughTerrain")
    model = env.mj_model
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    for terrain_geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD):
        model.geom_matid[terrain_geom] = -1
        model.geom_rgba[terrain_geom] = [0.42, 0.32, 0.18, 1.0]

    fixed_height = max(float(np.ptp(scaled(terrain))) for terrain in terrains)
    model.hfield_size[0, 2] = fixed_height
    base_mjx_model = mjx.put_model(model)

    @jax.jit
    def one_mjx_step(hfield_data, qpos):
        mjx_model = base_mjx_model.replace(hfield_data=hfield_data)
        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos, ctrl=jnp.zeros(mjx_model.nu))
        return mjx.step(mjx_model, mjx.forward(mjx_model, data))

    renderer = mujoco.Renderer(model, height=PANEL_SIZE[1], width=PANEL_SIZE[0])
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.45]
    camera.distance = 7.8
    camera.azimuth = 132
    camera.elevation = -28
    for difficulty, raw in enumerate(terrains, 1):
        resized = np.asarray(
            Image.fromarray(scaled(raw)).resize((256, 256), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        # Bicubic interpolation can undershoot the model's zero floor by a
        # tiny amount; MuJoCo requires every hfield sample to be in [0, 1].
        resized = np.maximum(resized, np.float32(0.0))
        SimulatorTerrain(resized, horizontal_scale=0.05).apply(
            "mujoco-playground", env, hfield_name="hfield", rebuild=False
        )
        model.hfield_data[:] = resized.ravel() / fixed_height
        model.hfield_size[0, 2] = fixed_height
        mujoco.mjr_uploadHField(model, renderer._mjr_context, 0)
        phase, x, y = motion_pose(difficulty, 2.25, 1.45)
        qpos = model.qpos0.copy()
        qpos[0:3] = [x, y, terrain_height(resized, x, y, 0.05) + 0.48]
        yaw = phase + np.pi * 0.5
        qpos[3:7] = [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)]
        if model.nq >= 19:
            trot = np.sin(phase * 4.0 + np.array([0, np.pi, np.pi, 0]))
            for leg, offset in enumerate((7, 10, 13, 16)):
                qpos[offset : offset + 3] += [0.18 * trot[leg], 0.0, -0.32 * trot[leg]]
        state = one_mjx_step(jnp.asarray(model.hfield_data), jnp.asarray(qpos))
        qpos = np.asarray(jax.device_get(state.qpos))
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        save_frame(directory, difficulty, renderer.render())
        if difficulty % 10 == 0:
            print(f"MuJoCo Playground {difficulty}/100", flush=True)
    renderer.close()


def capture_pybullet(terrains: np.ndarray, directory: Path) -> None:
    import pybullet as bullet
    import pybullet_data

    if (directory / "100.png").is_file():
        return
    client = bullet.connect(bullet.DIRECT)
    bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    bullet.setGravity(0, 0, -9.81)
    robot = bullet.loadURDF("husky/husky.urdf", [0, 0, 1.0])
    view = bullet.computeViewMatrixFromYawPitchRoll([0, 0, 0.5], 8.2, 42, -29, 0, 2)
    projection = bullet.computeProjectionMatrixFOV(
        55, PANEL_SIZE[0] / PANEL_SIZE[1], 0.1, 50
    )
    terrain_body = None
    for difficulty, raw in enumerate(terrains, 1):
        terrain = scaled(raw)
        if terrain_body is not None:
            bullet.removeBody(terrain_body)
        shape = bullet.createCollisionShape(
            bullet.GEOM_HEIGHTFIELD,
            meshScale=[HORIZONTAL_SCALE, HORIZONTAL_SCALE, 1.0],
            heightfieldTextureScaling=32,
            heightfieldData=terrain.ravel().tolist(),
            numHeightfieldRows=128,
            numHeightfieldColumns=128,
        )
        terrain_body = bullet.createMultiBody(0, shape)
        bullet.changeVisualShape(terrain_body, -1, rgbaColor=[0.31, 0.45, 0.22, 1])
        terrain_midpoint = 0.5 * float(terrain.min() + terrain.max())
        phase, x, y = motion_pose(difficulty, 2.3, 1.4)
        yaw = phase + np.pi * 0.5
        bullet.resetBasePositionAndOrientation(
            robot,
            [x, y, terrain_height(terrain, x, y, HORIZONTAL_SCALE) - terrain_midpoint + 0.48],
            bullet.getQuaternionFromEuler([0, 0, yaw]),
        )
        left_speed = 5.0 + 1.2 * np.sin(phase * 2.0)
        right_speed = 5.0 - 1.2 * np.sin(phase * 2.0)
        for joint, speed in zip((2, 4, 3, 5), (left_speed, left_speed, right_speed, right_speed)):
            bullet.setJointMotorControl2(robot, joint, bullet.VELOCITY_CONTROL, targetVelocity=speed, force=80)
        for _ in range(10):
            bullet.stepSimulation()
        _, _, rgba, _, _ = bullet.getCameraImage(
            PANEL_SIZE[0],
            PANEL_SIZE[1],
            view,
            projection,
            renderer=bullet.ER_TINY_RENDERER,
            shadow=1,
            lightDirection=[-1, -1, -2],
        )
        image = np.asarray(rgba, dtype=np.uint8).reshape(
            PANEL_SIZE[1], PANEL_SIZE[0], 4
        )[:, :, :3]
        save_frame(directory, difficulty, image)
        if difficulty % 10 == 0:
            print(f"PyBullet {difficulty}/100", flush=True)
    bullet.disconnect(client)


def gazebo_model(terrain: np.ndarray, image_path: Path, frame: int) -> str:
    phase, robot_x, robot_y = motion_pose(frame, 2.25, 1.35)
    robot_z = terrain_height(terrain, robot_x, robot_y, 0.2) + 0.42
    yaw = phase + np.pi * 0.5
    wheels = "".join(
        f"""
      <link name="wheel_{index}"><pose>{robot_x + x} {robot_y + y} {robot_z - 0.13} 1.5708 0 {yaw}</pose>
        <visual name="visual"><geometry><cylinder><radius>0.14</radius><length>0.10</length></cylinder></geometry>
          <material><ambient>0.04 0.05 0.06 1</ambient><diffuse>0.08 0.09 0.10 1</diffuse></material></visual>
      </link>"""
        for index, (x, y) in enumerate(((0.28, 0.27), (0.28, -0.27), (-0.28, 0.27), (-0.28, -0.27)))
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.6"><model name="generated_terrain"><static>true</static>
  <link name="terrain"><visual name="visual"><geometry><mesh>
    <uri>file://{image_path}</uri>
  </mesh></geometry><material><ambient>0.34 0.24 0.13 1</ambient>
    <diffuse>0.52 0.38 0.20 1</diffuse><specular>0.05 0.05 0.05 1</specular>
  </material></visual></link>
  <link name="rover"><pose>{robot_x} {robot_y} {robot_z} 0 0 {yaw}</pose><visual name="body"><geometry><box><size>0.75 0.48 0.20</size></box></geometry>
    <material><ambient>0.65 0.18 0.04 1</ambient><diffuse>0.9 0.3 0.06 1</diffuse><specular>0.25 0.25 0.25 1</specular></material></visual>
    <visual name="sensor"><pose>0.18 0 0.18 0 0 0</pose><geometry><cylinder><radius>0.07</radius><length>0.16</length></cylinder></geometry>
      <material><ambient>0.05 0.08 0.1 1</ambient><diffuse>0.08 0.18 0.24 1</diffuse></material></visual>
  </link>{wheels}
</model></sdf>"""


def save_gazebo_mesh(terrain: np.ndarray, path: Path) -> None:
    resized = np.array(
        Image.fromarray(terrain).resize((65, 65), Image.Resampling.BICUBIC),
        dtype=np.float32,
        copy=True,
    )
    resized = np.maximum(resized, np.float32(0.0))
    SimulatorTerrain(resized, horizontal_scale=0.2).save_obj(path)


def capture_gazebo(terrains: np.ndarray, directory: Path, work_dir: Path) -> None:
    import rclpy
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ImageMessage

    if (directory / "100.png").is_file():
        return

    def decode_image(message: ImageMessage) -> np.ndarray:
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step
        )
        image = rows[:, : message.width * 3].reshape(
            message.height, message.width, 3
        )
        if message.encoding.lower() == "bgr8":
            image = image[:, :, ::-1]
        return image

    class GazeboCapture(Node):
        def __init__(self) -> None:
            super().__init__("terrain_teaser_capture")
            self.frame = None
            self.frame_count = 0
            self.create_subscription(
                ImageMessage, "/terrain_teaser/camera/image_raw", self.on_image, 1
            )
            self.spawn = self.create_client(SpawnEntity, "/spawn_entity")
            self.delete = self.create_client(DeleteEntity, "/delete_entity")

        def on_image(self, message: ImageMessage) -> None:
            self.frame = message
            self.frame_count += 1

    log_path = work_dir / "gazebo.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                "gzserver",
                "--verbose",
                str(CAMERA_WORLD),
                "-s",
                "libgazebo_ros_init.so",
                "-s",
                "libgazebo_ros_factory.so",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rclpy.init()
    node = GazeboCapture()
    try:
        if not node.spawn.wait_for_service(timeout_sec=25):
            raise TimeoutError("Gazebo spawn service did not start")
        previous = None
        heightmaps = work_dir / "gazebo_heightmaps"
        heightmaps.mkdir(exist_ok=True)
        for difficulty, raw in enumerate(terrains, 1):
            frame_path = directory / f"{difficulty:03d}.png"
            if frame_path.is_file():
                continue
            if previous is not None:
                request = DeleteEntity.Request()
                request.name = previous
                future = node.delete.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=10)
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f"Gazebo could not delete {previous}: {response}")
            terrain = scaled(raw)
            image_path = (heightmaps / f"{difficulty:03d}.obj").resolve()
            save_gazebo_mesh(terrain, image_path)
            name = f"generated_terrain_{difficulty:03d}"
            request = SpawnEntity.Request()
            request.name = name
            request.xml = gazebo_model(terrain, image_path, difficulty)
            request.initial_pose.orientation.w = 1.0
            request.reference_frame = "world"
            before = node.frame_count
            future = node.spawn.call_async(request)
            # The first heightmap insertion also initializes Gazebo's renderer
            # and can take noticeably longer than subsequent insertions.
            rclpy.spin_until_future_complete(node, future, timeout_sec=60)
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(f"Gazebo failed at {difficulty}: {response}")
            deadline = time.monotonic() + 15
            image = None
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.frame is None or node.frame_count < before + 3:
                    continue
                candidate = decode_image(node.frame)
                background = candidate[0, 0].astype(np.int16)
                visible = np.mean(
                    np.any(np.abs(candidate.astype(np.int16) - background) > 2, axis=2)
                )
                if visible > 0.1:
                    image = candidate
                    break
            if image is None:
                raise TimeoutError(
                    f"Gazebo did not render terrain at difficulty {difficulty}"
                )
            save_frame(directory, difficulty, image)
            previous = name
            if difficulty % 10 == 0:
                print(f"Gazebo {difficulty}/100", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        server.terminate()
        try:
            server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            server.kill()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def compose_frames(work_dir: Path) -> Path:
    output = work_dir / "composite"
    output.mkdir(exist_ok=True)
    heading = font(24, bold=True)
    label_font = font(16, bold=True)
    width = PANEL_SIZE[0] * len(BACKENDS)
    header = 42
    for difficulty in range(1, 101):
        canvas = Image.new("RGB", (width, PANEL_SIZE[1] + header), "#10161d")
        draw = ImageDraw.Draw(canvas)
        title = f"Consistency terrain difficulty  {difficulty:03d} / 100"
        box = draw.textbbox((0, 0), title, font=heading)
        draw.text(((width - box[2]) / 2, 6), title, fill="#f3f7fa", font=heading)
        for index, (name, label) in enumerate(BACKENDS):
            panel = Image.open(work_dir / name / f"{difficulty:03d}.png").convert("RGB")
            panel = ImageOps.fit(panel, PANEL_SIZE, method=Image.Resampling.LANCZOS)
            x = index * PANEL_SIZE[0]
            y = header
            canvas.paste(panel, (x, y))
            overlay = Image.new("RGBA", (PANEL_SIZE[0], 32), (8, 14, 20, 185))
            canvas.paste(overlay, (x, y), overlay)
            draw.text((x + 10, y + 6), label, fill="#ffffff", font=label_font)
        canvas.save(output / f"frame_{difficulty:03d}.png", optimize=True)
    return output


def encode_gif(frames: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ["-framerate", "12", "-start_number", "1", "-i", str(frames / "frame_%03d.png")]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *source,
            "-filter_complex",
            "[0:v]scale=1600:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    terrains = prepare_terrains(args.work_dir, args.device)
    subprocess.run(
        [str(args.isaac_python), str(TEASER_TOOLS / "capture_isaaclab.py"),
         "--terrains", str(args.work_dir / "terrains_6x100x128x128.npz"),
         "--output", str(args.work_dir / "isaaclab")],
        check=True,
        env={**os.environ, "MUJOCO_GL": "egl", "OMNI_KIT_ACCEPT_EULA": "YES"},
    )
    subprocess.run(
        [str(args.mjlab_python), str(TEASER_TOOLS / "capture_mjlab.py"),
         "--terrains", str(args.work_dir / "terrains_6x100x128x128.npz"),
         "--output", str(args.work_dir / "mjlab")],
        check=True,
        env={**os.environ, "MUJOCO_GL": "egl"},
    )
    capture_mujoco(terrains["mujoco"], args.work_dir / "mujoco", args.work_dir)
    capture_playground(terrains["playground"], args.work_dir / "playground")
    capture_pybullet(terrains["pybullet"], args.work_dir / "pybullet")
    capture_gazebo(terrains["gazebo"], args.work_dir / "gazebo", args.work_dir)
    frames = compose_frames(args.work_dir)
    encode_gif(frames, args.output.resolve())
    print(f"Saved README GIF to {args.output.resolve()}")


if __name__ == "__main__":
    main()
