"""
Generate one world-space motion window from inference conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.diffusion import MotionDiffusion
from src.utils.floor import estimate_floor
from src.utils.geometry import (
    FEATURE_DIM,
    JOINT_COUNT,
    anchor_local,
    facing_yaw,
    make_transform,
    pad_frames,
    transform_points,
)
from src.utils.scene import crop_scene
from src.utils.statistics import MotionStatistics


@dataclass(frozen=True)
class RolloutState:
    """
    State carried between rollout steps.

    `features` are world-space joint positions for the trailing history
    frames ([frames, FEATURE_DIM]). `position`/`yaw` anchor the next window,
    matching NymeriaDataset's convention: x/y is the last frame's pelvis, z
    is the estimated floor height there (not pelvis height, which drifts
    with pose -- standing/sitting/lying put the pelvis at different heights
    above the same floor).
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
        yaw: float = np.pi / 2,
    ) -> "RolloutState":
        """
        Begin a rollout with no history, anchored at `start` on `floor_z`.

        No standing-height offset is added: `position` is the anchor (floor
        level), and the model generates an appropriately-standing pose above
        it on its own, same as every anchor during training.
        """

        return cls(
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.asarray([start[0], start[1], float(floor_z)], dtype=np.float32),
            float(yaw),
        )


@torch.no_grad()
def inference(
    config,
    diffusion: MotionDiffusion,
    statistics: MotionStatistics,
    state: RolloutState,
    *,
    text: str = "",
    goal=None,
    window_progress: float | None = None,
    text_guidance_scale: float = 1.0,
    scene_guidance_scale: float = 1.0,
    goal_guidance_scale: float = 1.0,
    scene=None,
    scene_bounds=None,
    generator=None,
) -> tuple[np.ndarray, np.ndarray, RolloutState, np.ndarray | None]:
    """
    Generate one window from a rollout state and optional conditions.

    `goal` is the anchor-local pelvis displacement [dx, dy, dz] -- a soft,
    CFG-guided condition like text and scene, not hard-inpainted.
    `goal_guidance_scale` controls how strongly it's followed.

    `window_progress` (0-1, or None) is where this window sits within the
    text instruction's span -- matches NymeriaDataset's training-time
    signal (see TextCondition). There's no annotation to place it within
    at inference time, so the caller decides; left None it's simply not
    conditioned on, the same as before this existed.

    Returns the window's anchor-local joint positions [window_size,
    FEATURE_DIM], the same joints in world space [window_size, joints, 3],
    the next rollout state, and world-space centers of occupied crop voxels
    (`None` if no scene was used).
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

    anchor_position = np.asarray(state.position, dtype=np.float32)
    anchor_yaw = float(state.yaw)
    transform = make_transform(
        torch.as_tensor(anchor_position, device=device, dtype=torch.float32),
        anchor_yaw,
    )

    # Right-align history, re-anchor to this window's anchor, mask unused
    # leading frames.
    selected = history_features[-history_size:]
    if len(selected):
        selected_joints = selected.reshape(-1, JOINT_COUNT, 3)
        local_joints = anchor_local(selected_joints, anchor_position, anchor_yaw)
        selected = statistics.normalize(
            local_joints.reshape(-1, model.input_dim)
        ).astype(np.float32, copy=False)
    normalized_history, history_mask = pad_frames(
        selected, history_size, model.input_dim, align="right"
    )
    normalized_history = normalized_history.unsqueeze(0).to(device)
    history_mask = history_mask.unsqueeze(0).to(device)

    goal_condition = None
    if goal is not None:
        goal_tensor = torch.as_tensor(goal, device=device, dtype=torch.float32).reshape(1, 3)
        goal_condition = statistics.normalize_pelvis(goal_tensor)

    # Crop around the same anchor; scene=None disables scene conditioning for
    # this window even on a scene-enabled model (e.g. a text/goal-only window).
    scene_condition = None
    scene_points = None
    if model.scene_enabled and scene is not None and scene_bounds is not None:
        scene_condition, crop_points, occupied, inside = crop_scene(
            scene,
            scene_bounds,
            anchor_position,
            anchor_yaw,
            config.scene_crop_bounds,
            config.scene_voxels,
            return_points=True,
        )
        # Occupied voxel centers in world space, for visualizing just this
        # crop rather than the full scene mesh.
        scene_points = crop_points[occupied & inside]
        scene_condition = torch.from_numpy(scene_condition).unsqueeze(0).to(device)

    # Generate anchor-local joint positions, then restore world space.
    progress_condition = (
        torch.tensor([window_progress], device=device, dtype=torch.float32)
        if window_progress is not None else None
    )
    normalized_motion = diffusion.sample_window(
        normalized_history,
        history_mask,
        text=text if model.text_enabled else None,
        scene=scene_condition,
        goal=goal_condition,
        window_progress=progress_condition,
        text_guidance_scale=text_guidance_scale,
        scene_guidance_scale=scene_guidance_scale,
        goal_guidance_scale=goal_guidance_scale,
        generator=generator,
    )
    local_motion = statistics.denormalize(normalized_motion)
    world_motion = transform_points(
        local_motion.reshape(1, diffusion.window_size, -1), transform
    ).reshape(diffusion.window_size, JOINT_COUNT, 3)

    local_features = local_motion.reshape(
        diffusion.window_size, model.input_dim
    ).cpu().numpy()
    world_motion_np = world_motion.cpu().numpy()
    world_features = world_motion_np.reshape(diffusion.window_size, model.input_dim)

    # Re-estimate the floor causally (src/utils/floor.py) from history + this
    # window's motion, so the next anchor stays floor-relative. Only the
    # trailing value is used; the run-up is needed for velocity/offset gating.
    history_joints = (
        history_features.reshape(-1, JOINT_COUNT, 3)
        if len(history_features)
        else np.empty((0, JOINT_COUNT, 3), dtype=np.float32)
    )
    floor_trace = estimate_floor(np.concatenate([history_joints, world_motion_np], axis=0))
    next_position = np.array(
        [world_motion_np[-1, 0, 0], world_motion_np[-1, 0, 1], floor_trace[-1]],
        dtype=np.float32,
    )
    next_yaw = float(facing_yaw(world_motion[-1:])[0])
    next_state = RolloutState(
        np.concatenate((history_features, world_features))[-history_size:],
        next_position,
        next_yaw,
    )
    return local_features, world_motion_np, next_state, scene_points
