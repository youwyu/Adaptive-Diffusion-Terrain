"""Capture the MJLab (MuJoCo Warp) panel for the project teaser."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
import warp._src.context as wp_context
from PIL import Image

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.viewer import OffscreenRenderer, ViewerConfig


SIZE = (400, 300)
SPACING = 0.1

# MJLab 1.2 still reads this public alias, which Warp 1.14 moved internally.
wp.context = wp_context


def terrain_height(terrain: np.ndarray, x: float, y: float) -> float:
    row = int(np.clip(round(y / SPACING + 63.5), 0, 127))
    column = int(np.clip(round(x / SPACING + 63.5), 0, 127))
    return float(terrain[row, column])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrains", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "100.png").is_file():
        return
    terrains = np.load(args.terrains)["mjlab"].astype(np.float32)
    dummy = args.output.parent / "mjlab_hfield.png"
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(dummy)

    def add_terrain(spec: mujoco.MjSpec) -> None:
        spec.add_hfield(name="generated", file=str(dummy), size=[6.35, 6.35, 6.0, 0.1])
        geom = spec.worldbody.add_geom(
            name="generated_terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="generated"
        )
        geom.rgba = [0.25, 0.43, 0.18, 1.0]

    scene = Scene(
        SceneCfg(num_envs=1, entities={"robot": get_g1_robot_cfg()}, spec_fn=add_terrain),
        device="cuda:0",
    )
    model = scene.compile()
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device="cuda:0")
    renderer = OffscreenRenderer(
        model,
        ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.65), distance=8.0, azimuth=132.0, elevation=-27.0,
            width=SIZE[0], height=SIZE[1], max_extra_envs=0,
        ),
        scene,
    )
    renderer.initialize()
    default_qpos = torch.as_tensor(model.qpos0, device="cuda:0", dtype=torch.float32)

    for difficulty, terrain in enumerate(terrains, 1):
        span = max(float(np.ptp(terrain)), 1.0e-5)
        model.hfield_data[:] = terrain.ravel() / span
        model.hfield_size[0] = [6.35, 6.35, span, 0.1]
        mujoco.mjr_uploadHField(model, renderer.renderer._mjr_context, 0)
        phase = 2.0 * np.pi * (difficulty - 1) / 100.0
        x, y = 2.25 * np.sin(phase), 1.4 * np.cos(phase)
        yaw = phase + np.pi * 0.5
        qpos = default_qpos.clone()
        qpos[0:3] = torch.tensor([x, y, terrain_height(terrain, x, y) + 0.82], device="cuda:0")
        qpos[3:7] = torch.tensor([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], device="cuda:0")
        if qpos.numel() > 7:
            gait = 0.12 * torch.sin(
                torch.tensor(phase * 4.0, device="cuda:0")
                + torch.arange(qpos.numel() - 7, device="cuda:0") * np.pi
            )
            qpos[7:] += gait
        sim.data.qpos[0] = qpos
        sim.data.qvel[0].zero_()
        sim.forward()
        sim.step()
        renderer.update(sim.data)
        Image.fromarray(renderer.render().astype(np.uint8)).save(args.output / f"{difficulty:03d}.png")
        if difficulty % 10 == 0:
            print(f"MJLab {difficulty}/100", flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
