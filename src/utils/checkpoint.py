"""
Model and optimizer checkpoint saving and loading.
"""

from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(model, optimizer, path: str | Path, epoch: int) -> None:
    """
    Save trainable weights, optimizer, and epoch.
    """

    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model": trainable_state,
        "optimizer": optimizer.state_dict(),
    }, path)


def checkpoint_state_dict(path: str | Path, map_location="cpu") -> dict:
    """
    Read a checkpoint's saved trainable weights without touching a model --
    e.g. to detect which optional condition submodules it was trained with
    before constructing the model that will load it (see
    src.config.build_inference).
    """

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}


def load_checkpoint(
    model, path: str | Path, optimizer=None, *, train: bool = False, map_location="cpu"
) -> int | None:
    """
    Load trainable weights, and with `train=True`, also optimizer and epoch.
    """

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint["model"].items()
    }
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = expected - state_dict.keys()
    unexpected = state_dict.keys() - expected
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint trainable weights do not match the model; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(state_dict, strict=False)
    
    if not train:
        return None
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"]) + 1
