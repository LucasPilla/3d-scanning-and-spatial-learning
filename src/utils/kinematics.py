"""
Rotation representations, root integration, and skeleton forward kinematics.

The motion feature vector is 148-D per frame:

    [0:3]      root translation delta in the previous frame's yaw-aligned frame
    [3:4]      root yaw delta
    [4:10]     root residual orientation, 6D
    [10:148]   23 body joint local rotations, 6D, in SMPL `body_pose` order

Root yaw is factored out of the root rotation on purpose. It carries the
navigation/facing direction the goal condition talks about, so keeping it as
its own channel means the model does not have to find heading buried inside a
generic 6D rotation, while the residual keeps leaning, bending, and torso
pitch/roll. The full root rotation is `R_z(yaw) @ R_residual`.

Because the root channels are frame-to-frame deltas and every other channel is
a local rotation, the feature vector is already invariant to the world frame:
only the goal and scene conditions need anchoring.
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
BODY_JOINT_COUNT = JOINT_COUNT - 1
FEATURE_DIM = 4 + 6 + BODY_JOINT_COUNT * 6

ROOT_DELTA = slice(0, 3)
ROOT_YAW_DELTA = slice(3, 4)
ROOT_ROTATION = slice(4, 10)
BODY_ROTATION = slice(10, FEATURE_DIM)


def wrap_angle(angle):
    """
    Wrap radians to (-pi, pi].
    """

    if torch.is_tensor(angle):
        return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def z_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """
    Build batched right-handed rotations about +z from yaw angles.
    """

    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    rows = torch.stack(
        (
            torch.stack((cosine, -sine, zero), dim=-1),
            torch.stack((sine, cosine, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    return rows


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle vectors [..., 3] to rotation matrices [..., 3, 3].
    """

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    # The small-angle branch keeps the axis finite; sin(x)/x -> 1 handles the
    # magnitude, so the resulting matrix is still exactly the identity at zero.
    safe_angle = angle.clamp_min(1e-8)
    axis = axis_angle / safe_angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    angle = angle.unsqueeze(-1)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    identity = identity.expand_as(skew)
    return (
        identity
        + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * (skew @ skew)
    )


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Take the first two columns of rotation matrices [..., 3, 3] as 6D.
    """

    columns = matrix[..., :, :2].transpose(-1, -2)
    return columns.reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """
    Recover rotation matrices from 6D via Gram-Schmidt (Zhou et al.).
    """

    first, second = rotation[..., :3], rotation[..., 3:]
    column_one = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = second - (column_one * second).sum(dim=-1, keepdim=True) * column_one
    column_two = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    column_three = torch.cross(column_one, column_two, dim=-1)
    return torch.stack((column_one, column_two, column_three), dim=-1)


def integrate_root(deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate root deltas [..., T, 4] from the identity pose at the anchor.

    Frame `t` is placed by rotating its translation delta into the frame that
    frame `t - 1` ended up in, so the returned positions and yaws are expressed
    in the anchor's canonical frame, which starts at the origin facing +x.
    """

    translation = deltas[..., :3]
    yaw_delta = deltas[..., 3]
    yaws = torch.cumsum(yaw_delta, dim=-1)
    previous_yaws = yaws - yaw_delta
    steps = (z_rotation(previous_yaws) @ translation.unsqueeze(-1)).squeeze(-1)
    return torch.cumsum(steps, dim=-2), yaws


def forward_kinematics(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    bone_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Pose a skeleton from local joint rotations and per-sample bone offsets.

    `local_rotations` is [..., T, 24, 3, 3] with entry 0 the full root
    rotation, `root_positions` is [..., T, 3], and `bone_offsets` is [..., 24, 3]
    holding each joint's rest position relative to its parent (row 0 is the
    rest root position and is unused here, since the root is placed directly).
    Returns joint positions [..., T, 24, 3].
    """

    offsets = bone_offsets.unsqueeze(-3)
    positions = [root_positions]
    rotations = [local_rotations[..., 0, :, :]]
    for joint in range(1, JOINT_COUNT):
        parent = SMPL_PARENTS[joint]
        parent_rotation = rotations[parent]
        offset = offsets[..., joint, :].unsqueeze(-1)
        positions.append(
            positions[parent] + (parent_rotation @ offset).squeeze(-1)
        )
        rotations.append(parent_rotation @ local_rotations[..., joint, :, :])
    return torch.stack(positions, dim=-2)


def features_to_rotations(features: torch.Tensor) -> torch.Tensor:
    """
    Rebuild full local rotations [..., T, 24, 3, 3] from denormalized features.

    Joint 0 is the complete root rotation `R_z(yaw) @ R_residual`, with the yaw
    integrated from the feature deltas.
    """

    _, yaws = integrate_root(features[..., ROOT_DELTA.start : ROOT_YAW_DELTA.stop])
    residual = rotation_6d_to_matrix(features[..., ROOT_ROTATION])
    root = z_rotation(yaws) @ residual
    body = rotation_6d_to_matrix(
        features[..., BODY_ROTATION].reshape(
            *features.shape[:-1], BODY_JOINT_COUNT, 6
        )
    )
    return torch.cat((root.unsqueeze(-3), body), dim=-3)


def features_to_joints(
    features: torch.Tensor, bone_offsets: torch.Tensor
) -> torch.Tensor:
    """
    Convert denormalized features [..., T, 148] to anchor-frame joints.

    The result is [..., T, 24, 3] with the anchor at the origin facing +x, the
    same frame the scene crop and the goal condition live in.
    """

    root_positions, _ = integrate_root(
        features[..., ROOT_DELTA.start : ROOT_YAW_DELTA.stop]
    )
    rotations = features_to_rotations(features)
    return forward_kinematics(rotations, root_positions, bone_offsets)


def build_features(
    root_positions: torch.Tensor,
    yaws: torch.Tensor,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Build the 148-D feature sequence from world-space SMPL motion.

    `root_positions` is [T, 3] world pelvis positions, `yaws` is [T] facing
    angles, `global_orient` is [T, 3] axis-angle, and `body_pose` is [T, 69]
    axis-angle. The first frame's root delta is zero, since it has no
    predecessor.
    """

    frames = root_positions.shape[0]
    steps = torch.zeros_like(root_positions)
    steps[1:] = root_positions[1:] - root_positions[:-1]
    translation = (
        z_rotation(-_shift_previous(yaws)) @ steps.unsqueeze(-1)
    ).squeeze(-1)
    yaw_delta = torch.zeros(frames, dtype=yaws.dtype, device=yaws.device)
    yaw_delta[1:] = wrap_angle(yaws[1:] - yaws[:-1])
    residual = z_rotation(-yaws) @ axis_angle_to_matrix(global_orient)
    body = axis_angle_to_matrix(body_pose.reshape(frames, BODY_JOINT_COUNT, 3))
    return torch.cat(
        (
            translation,
            yaw_delta.unsqueeze(-1),
            matrix_to_rotation_6d(residual),
            matrix_to_rotation_6d(body).reshape(frames, BODY_JOINT_COUNT * 6),
        ),
        dim=-1,
    )


def _shift_previous(values: torch.Tensor) -> torch.Tensor:
    """
    Shift a [T] sequence one step forward, repeating the first entry.
    """

    shifted = torch.empty_like(values)
    shifted[0] = values[0]
    shifted[1:] = values[:-1]
    return shifted
