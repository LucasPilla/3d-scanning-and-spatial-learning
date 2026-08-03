"""
Configuration loading, validation, and object construction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from src.datasets.nymeriaplus.dataset import NymeriaDataset
from src.models.diffusion import MotionDiffusion
from src.utils.kinematics import FEATURE_DIM
from src.models.text_encoder import DEFAULT_MAX_TOKENS, build_text_encoder
from src.models.transformer import MotionTransformer
from src.utils.checkpoint import load_checkpoint
from src.utils.statistics import MotionStatistics


def load_config(path: str | Path):
    """
    Load YAML and apply the defaults documented in configs/default.yaml.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    # Dataset
    data.setdefault("path", None)
    data.setdefault("fps", 10.0)
    data.setdefault("window_size", 16)
    data.setdefault("history_size", 32)
    data.setdefault("max_open_sequences", 8)

    # Diffusion
    data.setdefault("timesteps", 50)

    # Transformer
    data.setdefault("input_dim", FEATURE_DIM)
    data.setdefault("output_dim", FEATURE_DIM)
    data.setdefault("model_dim", 512)
    data.setdefault("ff_size", 1024)
    data.setdefault("heads", 4)
    data.setdefault("layers", 8)
    data.setdefault("dropout", 0.1)

    # Text
    data.setdefault("text_enabled", True)
    data.setdefault("text_dropout", 0.1)
    data.setdefault("text_encoder", "clip")
    data.setdefault("text_cache", False)
    data.setdefault("max_text_tokens", DEFAULT_MAX_TOKENS)

    # Scene
    data.setdefault("scene_enabled", True)
    data.setdefault("scene_dropout", 0.1)
    data.setdefault("scene_voxels", 32)
    data.setdefault("scene_encoder", "cnn")
    data.setdefault("scene_crop_bounds", [-1.0, 1.0, -1.0, 1.0, -1.2, 0.8])

    # Goal
    data.setdefault("goal_enabled", True)
    data.setdefault("goal_dropout", 0.1)
    data.setdefault("max_goal_horizon", 32)
    data.setdefault("joint_dropout", 0.05)

    # Losses
    data.setdefault("feature_weight", 1.0)
    data.setdefault("position_weight", 0.1)

    # Training
    data.setdefault("name", None)
    data.setdefault("output", "runs")
    data.setdefault("resume_checkpoint", None)
    data.setdefault("learning_rate", 1e-4)
    data.setdefault("batch_size", 256)
    data.setdefault("epochs", 300)
    data.setdefault("workers", 4)
    data.setdefault("save_every", 10)

    if data["path"] is not None:
        data["path"] = Path(data["path"]).expanduser().resolve()
    if not isinstance(data["text_cache"], bool):
        raise ValueError("text_cache must be true or false")
    if int(data["max_text_tokens"]) < 1:
        raise ValueError("max_text_tokens must be a positive number of tokens")
    if data["output"] is not None:
        data["output"] = Path(data["output"]).expanduser().resolve()
    if data["resume_checkpoint"] is not None:
        data["resume_checkpoint"] = Path(data["resume_checkpoint"]).expanduser().resolve()
    return SimpleNamespace(**data)


def save_config(config, path: str | Path) -> None:
    """
    Write effective settings, including resolved paths.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plain = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config).items()
    }
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(plain, file, sort_keys=False)


def text_cache_path(dataset: str | Path, encoder: str, split: str = "train") -> Path:
    """
    Locate a processed dataset's cached text features for one encoder and split.
    """

    return Path(dataset) / "cache" / f"{encoder}_{split}.npy"


def load_statistics(config) -> MotionStatistics:
    """
    Load the processed dataset's normalization statistics.
    """

    if config.path is None:
        raise ValueError("The configuration must set the processed dataset path.")
    return MotionStatistics.load(config.path / "normalization.npz")


def build_dataset(config, split: str = "train"):
    """
    Construct the configured annotation dataset.
    """

    if config.path is None:
        raise ValueError("The configuration must set the processed dataset path.")
    candidate = text_cache_path(config.path, config.text_encoder, split=split)
    cache_path = (
        candidate
        if config.text_enabled and config.text_cache and candidate.exists()
        else None
    )
    return NymeriaDataset(
        config.path,
        split=split,
        window_size=config.window_size,
        history_size=config.history_size,
        max_open_sequences=config.max_open_sequences,
        scene_enabled=config.scene_enabled,
        scene_voxels=config.scene_voxels,
        scene_bounds=config.scene_crop_bounds,
        text_enabled=config.text_enabled,
        text_cache_path=cache_path,
        text_tokens=config.max_text_tokens,
        goal_enabled=config.goal_enabled,
        max_goal_horizon=config.max_goal_horizon,
    )


def build_model(config, *, cached_text_features: bool | None = None):
    """
    Construct the configured transformer without disabled condition modules.
    """

    if cached_text_features is None:
        cached_text_features = bool(config.text_enabled and config.text_cache)
    feature_dim = (
        build_text_encoder(config.text_encoder).feature_dim
        if cached_text_features else None
    )
    return MotionTransformer(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        model_dim=config.model_dim,
        ff_size=config.ff_size,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
        text_enabled=config.text_enabled,
        text_dropout=config.text_dropout,
        text_encoder_type=config.text_encoder,
        text_max_tokens=config.max_text_tokens,
        cached_text_features=cached_text_features,
        text_feature_dim=feature_dim,
        scene_enabled=config.scene_enabled,
        scene_dropout=config.scene_dropout,
        scene_voxels=config.scene_voxels,
        scene_encoder_type=config.scene_encoder,
        goal_enabled=config.goal_enabled,
        goal_dropout=config.goal_dropout,
        joint_dropout=config.joint_dropout,
    )


def build_inference(config, checkpoint, device=None):
    """
    Load a trained checkpoint ready to generate, with its statistics.

    Always builds the online text encoder: inference is given raw prompts, not
    rows of the training-order feature cache.
    """

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    statistics = load_statistics(config)
    model = build_model(config, cached_text_features=False).to(device)
    load_checkpoint(model, checkpoint)
    model.eval()
    diffusion = build_diffusion(config, model, statistics=statistics)
    return diffusion, statistics, device


def build_diffusion(config, model, *, statistics=None) -> MotionDiffusion:
    """
    Construct the diffusion wrapper with the configured loss weights.
    """

    device = next(model.parameters()).device
    return MotionDiffusion(
        model,
        timesteps=config.timesteps,
        window_size=config.window_size,
        statistics=load_statistics(config) if statistics is None else statistics,
        feature_weight=config.feature_weight,
        position_weight=config.position_weight,
    ).to(device)
