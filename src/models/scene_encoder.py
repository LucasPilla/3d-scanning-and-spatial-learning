"""
Encode voxelized scenes as spatial conditioning tokens.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN3DSceneEncoder(nn.Module):
    """
    Encode an occupancy grid with a small 3D CNN backbone.

    Strided 3D convs give real volumetric locality/equivariance, matching
    the occupancy grid's structure better than a repurposed 2D ViT would.
    Each surviving cell of the final feature volume becomes one token, so
    tokens keep the crop's coarse 3D layout. Stride-2 layer count is derived
    from `voxels`, keeping token count at `target_resolution ** 3` for any
    `voxels` that halves evenly down to it (see `__init__`'s assertion).
    """

    def __init__(self, voxels: int, target_resolution: int = 4):
        super().__init__()
        max_channels = 512
        self.output_resolution = int(target_resolution)
        self.token_count = self.output_resolution ** 3

        layers = []
        in_channels = 1
        out_channels = 32
        resolution = int(voxels)

        while -(-resolution // 2) >= self.output_resolution:
            layers += [
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True),
            ]
            resolution = -(-resolution // 2)
            in_channels = out_channels
            out_channels = min(max_channels, out_channels * 2)

        assert resolution == self.output_resolution, f"voxels={voxels} doesn't downsample evenly to {self.output_resolution}"

        self.model = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, occupancy: torch.Tensor) -> torch.Tensor:
        features = self.model(occupancy.unsqueeze(1))
        # [batch, channels, depth, height, width] -> one token per cell.
        return features.flatten(2).transpose(1, 2)
