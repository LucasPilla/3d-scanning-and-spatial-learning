"""
Generate sanity-check motion for random ground-truth validation windows.

Each sample seeds real history (and, unless disabled, real text/goal/scene)
from a random window in the validation split, generates one window with the
trained model, and saves it in the same .npz format demo/offline.py reads,
so results can be inspected with:

    python -m demo.offline --input <one saved file>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.config import build_inference, load_config
from src.datasets.nymeriaplus.dataset import NymeriaDataset
from src.inference import RolloutState, inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-guidance", type=float, default=1.0)
    parser.add_argument("--goal-guidance", type=float, default=1.0)
    parser.add_argument("--scene-guidance", type=float, default=1.0)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-goal", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.path is None:
        parser.error("The configuration must set path to the processed dataset.")
    output = args.output or args.checkpoint.parent.parent / "validation"
    output.mkdir(parents=True, exist_ok=True)

    diffusion, statistics, device = build_inference(
        config, args.checkpoint, args.device
    )
    model = diffusion.model
    use_text = model.text_enabled and not args.no_text
    use_goal = model.goal_enabled and not args.no_goal
    use_scene = model.scene_enabled and not args.no_scene

    dataset = NymeriaDataset(
        config.path,
        split="val",
        window_size=config.window_size,
        history_size=config.history_size,
        scene_enabled=use_scene,
        text_enabled=False,
        goal_enabled=False,
    )

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.count, len(dataset)))

    for sample_index in indices:
        sample = dataset.samples[sample_index]
        sequence_name = sample["sequence"]
        arrays = dataset.sequence(sequence_name)
        features, joints, yaw = arrays[:3]
        bone_offsets = dataset.bone_offsets[sequence_name]

        start = sample["window_first"] + random.randrange(sample["window_count"])
        history_start = max(0, start - config.history_size)
        history = np.asarray(features[history_start:start], dtype=np.float32)
        anchor_frame = start - 1 if len(history) else start
        window_end = start + config.window_size - 1
        ground_truth = np.asarray(
            joints[start : start + config.window_size], dtype=np.float32
        )

        text = sample["text"] if use_text else ""
        goal_pelvis = None
        if use_goal:
            world_target = np.array(joints[window_end, 0], dtype=np.float32)
            anchor_pos = np.array(joints[anchor_frame, 0], dtype=np.float32)
            anchor_y = float(yaw[anchor_frame])
            rel = world_target - anchor_pos
            cos_y, sin_y = np.cos(-anchor_y), np.sin(-anchor_y)
            goal_pelvis = np.array([
                rel[0] * cos_y - rel[1] * sin_y,
                rel[0] * sin_y + rel[1] * cos_y,
                rel[2],
            ], dtype=np.float32)

        scene = scene_bounds = None
        if use_scene:
            scene = arrays[3]
            scene_bounds = dataset.scene_bounds_by_sequence[sequence_name]

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + sample_index)
        state = RolloutState(
            history,
            np.array(joints[anchor_frame, 0], dtype=np.float32),
            float(yaw[anchor_frame]),
        )
        _, generated, _ = inference(
            config,
            diffusion,
            statistics,
            bone_offsets,
            state,
            text=text,
            goal=goal_pelvis,
            text_guidance_scale=args.text_guidance,
            goal_guidance_scale=args.goal_guidance,
            scene_guidance_scale=args.scene_guidance,
            scene=scene,
            scene_bounds=scene_bounds,
            generator=generator,
        )

        destination = output / f"{sequence_name}_{start:06d}.npz"
        np.savez_compressed(
            destination,
            generated=generated,
            ground_truth=ground_truth,
            goal_frames=np.asarray([window_end], dtype=np.int64),
            goal_pelvis=(
                goal_pelvis[None] if goal_pelvis is not None
                else np.zeros((1, 2), dtype=np.float32)
            ),
            start_frame=np.asarray(start),
            fps=np.asarray(config.fps),
            prompt=np.asarray(text),
            sequence=np.asarray(sequence_name),
            checkpoint=np.asarray(str(args.checkpoint.resolve())),
            seed=np.asarray(args.seed + sample_index),
        )
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
