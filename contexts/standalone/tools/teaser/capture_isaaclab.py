"""Capture the Isaac Lab panel for the project teaser."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--terrains", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if (args.output / "100.png").is_file():
    raise SystemExit(0)

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_assets import UNITREE_GO2_CFG


SIZE = (400, 300)
SPACING = 0.1


def terrain_height(terrain: np.ndarray, x: float, y: float) -> float:
    row = int(np.clip(round(y / SPACING + 63.5), 0, 127))
    column = int(np.clip(round(x / SPACING + 63.5), 0, 127))
    return float(terrain[row, column])


def create_mesh() -> UsdGeom.Mesh:
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain")
    xy = np.linspace(-6.35, 6.35, 128, dtype=np.float32)
    points = [(float(x), float(y), 0.0) for y in xy for x in xy]
    faces = []
    for row in range(127):
        for column in range(127):
            a = row * 128 + column
            faces.extend((a, a + 1, a + 129, a, a + 129, a + 128))
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
    mesh.GetFaceVertexIndicesAttr().Set(faces)
    mesh.GetSubdivisionSchemeAttr().Set("none")
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.28, 0.06), roughness=0.82)
    material.func("/World/Looks/Terrain", material)
    sim_utils.bind_visual_material("/World/Terrain", "/World/Looks/Terrain")
    return mesh


def set_heights(mesh: UsdGeom.Mesh, terrain: np.ndarray) -> None:
    xy = np.linspace(-6.35, 6.35, 128, dtype=np.float32)
    points = [Gf.Vec3f(float(x), float(y), float(terrain[row, column]))
              for row, y in enumerate(xy) for column, x in enumerate(xy)]
    mesh.GetPointsAttr().Set(points)


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "100.png").is_file():
        return
    terrains = np.load(args.terrains)["isaaclab"].astype(np.float32)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=1 / 60))
    sim_utils.DomeLightCfg(intensity=900.0, color=(0.85, 0.88, 0.92)).func("/World/Dome", sim_utils.DomeLightCfg())
    sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88)).func(
        "/World/Sun", sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88))
    )
    mesh = create_mesh()

    robot_cfg = UNITREE_GO2_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(robot_cfg)

    camera = Camera(CameraCfg(
        prim_path="/World/TeaserCamera",
        update_period=0.0,
        height=SIZE[1],
        width=SIZE[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=23.0,
            horizontal_aperture=26.0,
            clipping_range=(0.1, 100.0),
        ),
    ))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[3.8, 3.8, 2.8]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.45]], device=sim.device),
    )
    defaults = robot.data.default_joint_pos.clone()
    zeros = torch.zeros_like(defaults)

    for difficulty, terrain in enumerate(terrains, 1):
        set_heights(mesh, terrain)
        phase = 2.0 * np.pi * (difficulty - 1) / 100.0
        x, y = 1.55 * np.sin(phase), 0.95 * np.cos(phase)
        yaw = phase + np.pi * 0.5
        root = robot.data.default_root_state.clone()
        root[:, :3] = torch.tensor(
            [[x, y, terrain_height(terrain, x, y) + 0.48]], device=sim.device
        )
        root[:, 3:7] = torch.tensor(
            [[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]], device=sim.device
        )
        joints = defaults.clone()
        trot = torch.sin(torch.tensor(phase * 4.0, device=sim.device))
        joints[:, 1::3] += 0.16 * trot
        joints[:, 2::3] -= 0.28 * trot
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
        robot.write_joint_state_to_sim(joints, zeros)
        robot.set_joint_position_target(joints)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        Image.fromarray(rgb.astype(np.uint8)).save(args.output / f"{difficulty:03d}.png")
        if difficulty % 10 == 0:
            print(f"Isaac Lab {difficulty}/100", flush=True)


try:
    main()
finally:
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
