"""Command-line interface for terrain generation, variation, and rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generator import TerrainGenerator
from .io import load_terrains, save_terrains
from .render import render_terrains


def _difficulty(value: str):
    if value.lower() == "random":
        return "random"
    if "," in value:
        try:
            return [int(item.strip()) for item in value.split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "difficulty must be 'random', an integer, or comma-separated integers"
            ) from error
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "difficulty must be 'random', an integer, or comma-separated integers"
        ) from error


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        help="ONNX model; bundled model is the default",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:<index>"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Maximum inference batch size")


def _add_size_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(128, 128))
    parser.add_argument(
        "--size-mode",
        choices=("auto", "resize", "tile"),
        default="auto",
        help="auto tiles maps larger than the model's native 128x128 size",
    )
    parser.add_argument("--tile-overlap", type=int, default=32)


def _save_results(
    output: Path,
    terrains,
    difficulties,
    *,
    suffix: str,
    render: bool,
    horizontal_scale: float,
    vertical_scale: float,
) -> None:
    paths = save_terrains(output, terrains, suffix=suffix)
    metadata_path = output / "difficulties.txt"
    metadata_path.write_text(
        "\n".join(f"{path.name},{int(difficulty)}" for path, difficulty in zip(paths, difficulties))
        + "\n"
    )
    if render:
        render_terrains(
            terrains,
            titles=[f"difficulty {int(value)}" for value in difficulties],
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            save_path=output / "terrains_3d.png",
        )
    print(f"Saved {len(paths)} terrain(s) to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terrain-diffusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate new difficulty-conditioned terrains")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--num", type=int, default=1)
    generate.add_argument("--difficulty", type=_difficulty, default="random")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--format", choices=("txt", "npy", "npz", "tif"), default="txt")
    generate.add_argument("--render", action="store_true")
    generate.add_argument("--horizontal-scale", type=float, default=0.1)
    generate.add_argument("--vertical-scale", type=float, default=1.0)
    _add_model_options(generate)
    _add_size_options(generate)

    vary = subparsers.add_parser("vary", help="Generate new terrains from existing heightfields")
    vary.add_argument("inputs", nargs="+", type=Path)
    vary.add_argument("--output", type=Path, required=True)
    vary.add_argument("--difficulty", type=_difficulty, default="random")
    vary.add_argument("--strength", type=float, default=0.5)
    vary.add_argument("--variants", type=int, default=1)
    vary.add_argument("--seed", type=int)
    vary.add_argument("--format", choices=("txt", "npy", "npz", "tif"), default="txt")
    vary.add_argument("--render", action="store_true")
    vary.add_argument("--horizontal-scale", type=float, default=0.1)
    vary.add_argument("--vertical-scale", type=float, default=1.0)
    _add_model_options(vary)
    _add_size_options(vary)
    vary.set_defaults(shape=None)

    render = subparsers.add_parser("render", help="Render existing terrains as a 3D grid")
    render.add_argument("inputs", nargs="+", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--columns", type=int)
    render.add_argument("--stride", type=int, default=2)
    render.add_argument("--horizontal-scale", type=float, default=0.1)
    render.add_argument("--vertical-scale", type=float, default=1.0)

    simulate = subparsers.add_parser(
        "simulate",
        help="Generate or load one terrain and open it in a standalone simulator",
    )
    simulate.add_argument("backend", choices=("pybullet", "mujoco", "gazebo"))
    simulate.add_argument(
        "--terrain",
        type=Path,
        help="Use an existing heightfield instead of running the diffusion model",
    )
    simulate.add_argument("--output", type=Path, default=Path("generated_simulation"))
    simulate.add_argument("--difficulty", type=_difficulty, default="random")
    simulate.add_argument("--seed", type=int)
    simulate.add_argument("--horizontal-scale", type=float, default=0.1)
    simulate.add_argument("--vertical-scale", type=float, default=1.0)
    simulate.add_argument("--name", default="adaptive_terrain")
    simulate.add_argument("--world", default="empty", help="Gazebo world name")
    simulate.add_argument("--headless", action="store_true")
    simulate.add_argument(
        "--export-only", action="store_true", help="Prepare assets without opening a simulator"
    )
    simulate.add_argument(
        "--run-steps",
        type=int,
        help="Stop a headless demo after this many 0.002-second simulation steps",
    )
    _add_model_options(simulate)
    _add_size_options(simulate)
    return parser


def _simulation_asset(args):
    from .simulators import SimulatorTerrain

    args.output.mkdir(parents=True, exist_ok=True)
    if args.terrain is not None:
        terrain = load_terrains(args.terrain)[0]
        difficulty = None
    else:
        generator = TerrainGenerator(
            args.model,
            device=args.device,
        )
        batch, difficulties = generator.generate(
            1,
            shape=args.shape,
            difficulty=args.difficulty,
            seed=args.seed,
            size_mode=args.size_mode,
            tile_overlap=args.tile_overlap,
            batch_size=args.batch_size,
            return_difficulties=True,
        )
        terrain = batch[0]
        difficulty = int(difficulties[0])
    save_terrains(args.output, terrain, suffix=".npy", prefix="heightfield")
    if difficulty is not None:
        (args.output / "difficulty.txt").write_text(f"{difficulty}\n")
    return SimulatorTerrain(
        terrain,
        horizontal_scale=args.horizontal_scale,
        vertical_scale=args.vertical_scale,
    ), difficulty


def _run_simulation(args) -> None:
    asset, difficulty = _simulation_asset(args)
    label = "existing terrain" if difficulty is None else f"difficulty {difficulty}"
    steps = args.run_steps
    if steps is not None and steps <= 0:
        raise ValueError("--run-steps must be positive")

    if args.backend == "pybullet":
        mesh_path = asset.save_obj(args.output / f"{args.name}.obj")
        if args.export_only:
            print(f"Exported {label} for PyBullet to {mesh_path}")
            return
        from .simulators.pybullet import run_pybullet_demo

        run_pybullet_demo(
            asset,
            gui=not args.headless,
            steps=steps if steps is not None else (240 if args.headless else None),
        )
        print(f"Loaded {label} in PyBullet")
        return

    if args.backend == "mujoco":
        model = asset.apply("mujoco", hfield_name=args.name)
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required for this command") from error
        model_path = args.output / f"{args.name}.mjb"
        mujoco.mj_saveModel(model, str(model_path), None)
        if args.export_only:
            print(f"Exported {label} as {model_path}")
            return
        data = mujoco.MjData(model)
        if args.headless:
            for _ in range(steps if steps is not None else 240):
                mujoco.mj_step(model, data)
        else:
            import mujoco.viewer

            mujoco.viewer.launch(model, data)
        print(f"Loaded {label} in MuJoCo")
        return

    from .simulators.gazebo import apply_gazebo

    gazebo_asset = apply_gazebo(
        asset,
        output_dir=args.output / "gazebo_model",
        name=args.name,
        world=args.world,
        spawn=False,
        launch=not args.export_only,
        headless=args.headless,
        iterations=steps if steps is not None else (240 if args.headless else None),
    )
    if gazebo_asset.process is not None:
        try:
            gazebo_asset.process.wait()
        except KeyboardInterrupt:
            gazebo_asset.process.terminate()
    action = "Exported" if args.export_only else "Loaded"
    print(f"{action} {label} for Gazebo at {gazebo_asset.world_path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        terrains = load_terrains(args.inputs)
        render_terrains(
            terrains,
            columns=args.columns,
            stride=args.stride,
            horizontal_scale=args.horizontal_scale,
            vertical_scale=args.vertical_scale,
            save_path=args.output,
        )
        print(f"Saved 3D render to {args.output}")
        return
    if args.command == "simulate":
        _run_simulation(args)
        return

    generator = TerrainGenerator(
        args.model,
        device=args.device,
    )
    common = {
        "difficulty": args.difficulty,
        "seed": args.seed,
        "size_mode": args.size_mode,
        "tile_overlap": args.tile_overlap,
        "batch_size": args.batch_size,
        "return_difficulties": True,
    }
    if args.command == "generate":
        terrains, difficulties = generator.generate(
            args.num, shape=args.shape, **common
        )
    else:
        terrains, difficulties = generator.generate_from(
            args.inputs,
            strength=args.strength,
            variants=args.variants,
            shape=args.shape,
            **common,
        )
    _save_results(
        args.output,
        terrains,
        difficulties,
        suffix=f".{args.format}",
        render=args.render,
        horizontal_scale=args.horizontal_scale,
        vertical_scale=args.vertical_scale,
    )


if __name__ == "__main__":
    main()
