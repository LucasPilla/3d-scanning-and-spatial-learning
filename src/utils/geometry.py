"""
Geometry and kinematic transforms.

The motion feature vector is anchored joint positions: `[T, JOINT_COUNT, 3]`
flattened to `[T, FEATURE_DIM]`, each frame's joints expressed relative to one
fixed anchor (the world position/yaw of the last history frame, or the
window's first frame when there is no history). Everything a window's model
sees shares that same anchor: history, the frames being generated, the goal,
and the scene crop. `anchor_local` below is the world-to-anchor transform.
"""

from __future__ import annotations

import numpy as np
import torch

# SMPL 24-joint kinematic tree.
SMPL_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
)

JOINT_COUNT = len(SMPL_PARENTS)
FEATURE_DIM = JOINT_COUNT * 3


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


def anchor_local(points_world, anchor_position, anchor_yaw):
    """
    Transform world-space points (..., 3) into the anchor's local frame.

    Works for a single point or a whole `[T, joints, 3]` array -- NumPy
    broadcasts `(..., 3) @ (3, 3)` over any leading shape, no reshape
    needed. The rotation matches `points_world`'s dtype, so float64
    accumulation (e.g. normalization statistics) isn't silently downcast.
    """

    points_world = np.asarray(points_world)
    rotation = yaw_rotation(float(anchor_yaw), dtype=points_world.dtype)
    return (points_world - anchor_position) @ rotation.T


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


def pad_frames(normalized, length: int, feature_dim: int, *, align: str = "right"):
    """
    Fixed-length buffer with real content at one end, padding at the other.

    `align="right"` (history convention): real content trails, padding leads.
    `align="left"` (target convention): real content leads, padding trails.
    Mask polarity is unchanged: True marks a slot attention must ignore.
    """

    actual = len(normalized)
    buffer = torch.zeros(length, feature_dim, dtype=torch.float32)
    mask = torch.ones(length, dtype=torch.bool)
    if actual:
        if align == "right":
            buffer[-actual:] = torch.as_tensor(normalized, dtype=torch.float32)
            mask[-actual:] = False
        else:
            buffer[:actual] = torch.as_tensor(normalized, dtype=torch.float32)
            mask[:actual] = False
    return buffer, mask
