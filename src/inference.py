"""
Generate one world-space motion window from inference conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.diffusion import MotionDiffusion
from src.utils.kinematics import (
    FEATURE_DIM,
    JOINT_COUNT,
    ROOT_DELTA,
    ROOT_YAW_DELTA,
    features_to_joints,
    integrate_root,
)
from src.utils.geometry import make_transform, pad_history, transform_points
from src.utils.scene import crop_scene
from src.utils.statistics import MotionStatistics


@dataclass(frozen=True)
class RolloutState:
    """
    Everything a rolling generation needs to continue from where it stopped.

    The model consumes canonical motion features, and joint rotations cannot
    be recovered from joint positions, so a rollout has to carry the features
    themselves rather than the rendered skeleton. `position` and `yaw` place
    the most recent frame in the world.
    """

    features: np.ndarray
    position: np.ndarray
    yaw: float

    @classmethod
    def cold_start(
        cls,
        start=(0.0, 0.0),
        *,
        floor_z: float = 0.0,
        spawn_height: float = 0.92,
        yaw: float = np.pi / 2,
    ) -> "RolloutState":
        """
        Begin a rollout with no history, standing at `start` on `floor_z`.
        """

        return cls(
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.asarray(
                [start[0], start[1], float(floor_z) + float(spawn_height)],
                dtype=np.float32,
            ),
            float(yaw),
        )


@torch.no_grad()
def inference(
    config,
    diffusion: MotionDiffusion,
    statistics: MotionStatistics,
    bone_offsets,
    state: RolloutState,
    *,
    text: str = "",
    goal=None,
    text_guidance_scale: float = 1.0,
    goal_guidance_scale: float = 1.0,
    scene_guidance_scale: float = 1.0,
    scene=None,
    scene_bounds=None,
    generator=None,
) -> tuple[np.ndarray, np.ndarray, RolloutState]:
    """
    Generate one window from a rollout state and optional conditions.

    `goal` is expected to be the anchor-local pelvis displacement [dx, dy, dz].

    Returns the window's canonical features, its world-space joints with shape
    [window_size, joints, 3], and the state to pass to the next call.
    """

    model = diffusion.model
    device = next(model.parameters()).device
    history_size = int(config.history_size)

    history_features = np.asarray(state.features, dtype=np.float32)
    if history_features.ndim != 2 or history_features.shape[1] != model.input_dim:
        raise ValueError(
            f"History must have shape [frames, {model.input_dim}], "
            f"got {history_features.shape}."
        )
    bone_offsets = torch.as_tensor(
        bone_offsets, device=device, dtype=torch.float32
    ).reshape(1, JOINT_COUNT, 3)

    anchor_position = torch.as_tensor(
        state.position, device=device, dtype=torch.float32
    ).reshape(1, 3)
    anchor_yaw = torch.full((1,), float(state.yaw), device=device)
    transform = make_transform(anchor_position, anchor_yaw)

    # Right-align available history and mask the unused leading frames.
    selected = history_features[-history_size:]
    if len(selected):
        selected = selected.copy()
        selected[0, ROOT_DELTA.start : ROOT_YAW_DELTA.stop] = 0.0
        selected = statistics.normalize(selected)
    normalized_history, history_mask = pad_history(
        selected, history_size, model.input_dim
    )
    normalized_history = normalized_history.unsqueeze(0).to(device)
    history_mask = history_mask.unsqueeze(0).to(device)

    # Prepare local goal condition
    goal_condition = None
    if goal is not None:
        goal_tensor = torch.as_tensor(goal, device=device, dtype=torch.float32).reshape(1, 3)
        goal_condition = statistics.normalize_goal(goal_tensor)

    # Crop the world occupancy grid around the same motion anchor.
    scene_condition = None
    if model.scene_enabled:
        if scene is None or scene_bounds is None:
            raise ValueError("Scene-enabled models require scene and scene bounds.")
        scene_condition = crop_scene(
            scene,
            scene_bounds,
            anchor_position[0].cpu().numpy(),
            float(anchor_yaw[0]),
            config.scene_crop_bounds,
            config.scene_voxels,
        )
        scene_condition = torch.from_numpy(scene_condition).unsqueeze(0).to(device)

    # Generate canonical features, pose the skeleton, then restore world space.
    normalized_motion = diffusion.sample_window(
        normalized_history,
        history_mask,
        text=text if model.text_enabled else None,
        scene=scene_condition,
        goal=goal_condition,
        text_guidance_scale=text_guidance_scale,
        goal_guidance_scale=goal_guidance_scale,
        scene_guidance_scale=scene_guidance_scale,
        generator=generator,
    )
    window_features = statistics.denormalize(normalized_motion)
    local_motion = features_to_joints(window_features, bone_offsets)
    world_motion = transform_points(
        local_motion.reshape(1, diffusion.window_size, -1), transform
    ).reshape(diffusion.window_size, JOINT_COUNT, 3)

    _, yaws = integrate_root(
        window_features[..., ROOT_DELTA.start : ROOT_YAW_DELTA.stop]
    )
    window_features = window_features.reshape(
        diffusion.window_size, model.input_dim
    ).cpu().numpy()
    next_state = RolloutState(
        np.concatenate((history_features, window_features))[-history_size:],
        world_motion[-1, 0].cpu().numpy(),
        float(anchor_yaw[0]) + float(yaws[0, -1]),
    )
    return window_features, world_motion.cpu().numpy(), next_state
