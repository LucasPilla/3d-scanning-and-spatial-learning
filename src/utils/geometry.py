"""
Geometry and kinematic transforms.
"""

from __future__ import annotations

import numpy as np
import torch


def yaw_rotation(yaw: float, *, dtype=np.float32) -> np.ndarray:
    """
    Return a world-to-local z-axis rotation for a facing yaw.
    """

    cosine = np.cos(-float(yaw))
    sine = np.sin(-float(yaw))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply batched local-to-world 4x4 transforms to flattened 3D points.
    """

    shape = points.shape
    reshaped = points.reshape(points.shape[0], -1, 3)
    rotation = transform[:, :3, :3]
    translation = transform[:, None, :3, 3]
    return (reshaped @ rotation.transpose(1, 2) + translation).reshape(shape)


def make_transform(position: torch.Tensor, yaw: torch.Tensor | float) -> torch.Tensor:
    """
    Create batched local-to-world transforms.
    """

    position = torch.as_tensor(position)
    if position.ndim == 1:
        position = position.unsqueeze(0)
    yaw = torch.as_tensor(yaw, device=position.device, dtype=position.dtype)
    if yaw.ndim == 0:
        yaw = yaw.repeat(len(position))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    transform = torch.eye(4, device=position.device, dtype=position.dtype).repeat(
        len(position), 1, 1
    )
    transform[:, 0, 0] = cosine
    transform[:, 0, 1] = -sine
    transform[:, 1, 0] = sine
    transform[:, 1, 1] = cosine
    transform[:, :3, 3] = position
    return transform


def facing_yaw(joints: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
    """
    Estimate facing yaw from the left/right hip direction.
    """

    lateral = joints[:, 1] - joints[:, 2]
    length = torch.linalg.vector_norm(lateral[:, [0, 1]], dim=1)
    yaw = torch.atan2(lateral[:, 1], lateral[:, 0])
    if fallback is None:
        fallback = torch.zeros_like(yaw)
    return torch.where(length > 1e-6, yaw, fallback)


def pad_history(normalized, history_size: int, feature_dim: int):
    """
    Right-align normalized history frames and mask the unused leading slots.

    Shared by training and rollout so the padding side and the mask polarity
    (True marks a slot the attention must ignore) cannot drift between the two.
    Returns unbatched `[history_size, feature_dim]` and `[history_size]`.
    """

    length = len(normalized)
    history = torch.zeros(history_size, feature_dim, dtype=torch.float32)
    mask = torch.ones(history_size, dtype=torch.bool)
    if length:
        history[-length:] = torch.as_tensor(normalized, dtype=torch.float32)
        mask[-length:] = False
    return history, mask
