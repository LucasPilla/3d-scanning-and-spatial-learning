"""
Scene occupancy cropping and voxel grid generation.
"""

from __future__ import annotations

import numpy as np

from src.utils.geometry import yaw_rotation


def local_meshgrid(bounds, size: int) -> np.ndarray:
    """
    Create a flattened xyz grid of cell centers within three bound pairs.
    """

    bounds = tuple(float(value) for value in bounds)
    size = int(size)
    axes = [
        bounds[index]
        + (np.arange(size, dtype=np.float32) + 0.5)
        * ((bounds[index + 1] - bounds[index]) / size)
        for index in range(0, 6, 2)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def scene_token_coordinates(bounds, resolution: int) -> np.ndarray:
    """
    Cell centers of a downsampled crop, ordered like the encoder's tokens.

    Must apply the same (z, x, y) transpose `crop_scene` does, or these
    coordinates would silently label the wrong tokens.
    """

    resolution = int(resolution)
    grid = local_meshgrid(bounds, resolution).reshape(
        resolution, resolution, resolution, 3
    )
    return grid.transpose(2, 0, 1, 3).reshape(-1, 3)


def crop_scene(
    occupancy: np.ndarray,
    scene_bounds: np.ndarray,
    anchor_position: np.ndarray,
    anchor_yaw: float,
    local_bounds,
    size: int,
    *,
    local_grid: np.ndarray | None = None,
    return_points: bool = False,
) -> np.ndarray:
    """
    Query one anchor-fixed local occupancy crop.

    `return_points=True` also returns the crop's world-space voxel centers
    and raw occupied/inside masks, e.g. for visualizing which points were
    actually occupied.
    """

    occupancy = np.asarray(occupancy)
    bounds = np.asarray(scene_bounds, dtype=np.float32)
    grid = (
        local_meshgrid(local_bounds, size)
        if local_grid is None
        else np.asarray(local_grid, dtype=np.float32)
    )
    if grid.shape != (int(size) ** 3, 3):
        raise ValueError(
            f"local_grid must have shape ({int(size) ** 3}, 3), got {grid.shape}"
        )
    world_to_local = yaw_rotation(anchor_yaw)
    points = grid @ world_to_local + np.asarray(anchor_position, dtype=np.float32)
    shape = np.asarray(occupancy.shape, dtype=np.float32)
    indices = np.floor((points - bounds[0]) / ((bounds[1] - bounds[0]) / shape)).astype(
        np.int64
    )
    integer_shape = np.asarray(occupancy.shape, dtype=np.int64)
    inside = np.all((indices >= 0) & (indices < integer_shape), axis=1)
    indices[~inside] = 0
    values = occupancy[indices[:, 0], indices[:, 1], indices[:, 2]].astype(
        np.bool_, copy=True
    )
    values[~inside] = True
    cropped = values.reshape(size, size, size).transpose(2, 0, 1)
    if return_points:
        return cropped, points, values, inside
    return cropped
