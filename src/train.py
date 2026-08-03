"""
Train the Nymeria motion-diffusion model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    build_dataset,
    build_diffusion,
    build_model,
    load_config,
    save_config,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def train(config, device=None) -> None:
    """
    Train all valid annotations once per epoch and save resumable checkpoints.
    """

    if config.name is None:
        raise ValueError("Set name in the configuration or pass --name")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = config.output / config.name
    run.mkdir(parents=True, exist_ok=True)
    save_config(config, run / "config.yaml")

    dataset = build_dataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )

    val_dataset = build_dataset(config, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )

    model = build_model(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = Adam(trainable, lr=config.learning_rate)
    start_epoch = 0
    if config.resume_checkpoint is not None:
        start_epoch = load_checkpoint(
            model, config.resume_checkpoint, optimizer, train=True, map_location="cpu"
        )

    diffusion = build_diffusion(config, model)
    writer = SummaryWriter(run / "tensorboard")
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            epoch_totals: dict[str, float] = {}
            for batch in loader:
                motion = batch["motion"].to(device)
                history = batch["history"].to(device)
                history_mask = batch["history_mask"].to(device)
                bone_offsets = batch["bone_offsets"].to(device)
                scene = batch.get("scene")
                text_features = batch.get("text_features")
                goal = batch.get("goal")
                if scene is not None:
                    scene = scene.to(device)
                if text_features is not None:
                    text_features = text_features.to(device)
                if goal is not None:
                    goal = goal.to(device)
                timesteps = torch.randint(
                    diffusion.timesteps, (len(motion),), device=device
                )
                optimizer.zero_grad(set_to_none=True)
                losses = diffusion.training_loss(
                    motion,
                    timesteps,
                    history,
                    history_mask,
                    bone_offsets=bone_offsets,
                    text=batch.get("text"),
                    text_features=text_features,
                    scene=scene,
                    goal=goal,
                )
                losses["total"].backward()
                optimizer.step()

                for name, value in losses.items():
                    value = value.detach()
                    epoch_totals[name] = epoch_totals.get(name, 0.0) + value

            for name, value in epoch_totals.items():
                writer.add_scalar(
                    f"train/{name}", value.item() / len(loader), epoch
                )

            # Validation loss on every epoch
            model.eval()
            val_totals: dict[str, float] = {}
            with torch.no_grad():
                for batch in val_loader:
                    motion = batch["motion"].to(device)
                    history = batch["history"].to(device)
                    history_mask = batch["history_mask"].to(device)
                    bone_offsets = batch["bone_offsets"].to(device)
                    scene = batch.get("scene")
                    text_features = batch.get("text_features")
                    goal = batch.get("goal")
                    if scene is not None:
                        scene = scene.to(device)
                    if text_features is not None:
                        text_features = text_features.to(device)
                    if goal is not None:
                        goal = goal.to(device)
                    timesteps = torch.randint(
                        diffusion.timesteps, (len(motion),), device=device
                    )
                    losses = diffusion.training_loss(
                        motion,
                        timesteps,
                        history,
                        history_mask,
                        bone_offsets=bone_offsets,
                        text=batch.get("text"),
                        text_features=text_features,
                        scene=scene,
                        goal=goal,
                    )
                    for name, value in losses.items():
                        value = value.detach()
                        val_totals[name] = val_totals.get(name, 0.0) + value

            for name, value in val_totals.items():
                writer.add_scalar(
                    f"val/{name}", value.item() / len(val_loader), epoch
                )

            if (
                (epoch + 1) % config.save_every == 0
                or epoch + 1 == config.epochs
            ):
                path = (
                    run / "checkpoints"
                    / f"{config.name}_epoch{epoch:03d}.pth"
                )
                save_checkpoint(model, optimizer, path, epoch)
                print(f"Saved {path}", flush=True)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--name")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("name", "workers"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.resume_checkpoint is not None:
        config.resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
    train(config, device=args.device)


if __name__ == "__main__":
    main()
