"""
Encode voxelized scenes as spatial conditioning tokens.
"""

from __future__ import annotations

import torch
from torch import nn
from vit_pytorch import ViT


def build_scene_encoder(method: str, voxels: int) -> nn.Module:
    """
    Construct the configured scene encoder backbone.

    Every backbone returns [batch, token_count, output_dim] spatial tokens
    rather than one pooled vector, so different motion frames can attend to
    different regions of the crop.
    """

    method = str(method).lower()
    if method == "vit":
        return ViTSceneEncoder(voxels)
    if method == "cnn":
        return CNN3DSceneEncoder(voxels)
    raise ValueError(f"Unsupported scene encoder {method!r}; choose 'vit' or 'cnn'")


class ViTSceneEncoder(nn.Module):
    """
    Encode an occupancy grid into ViT patch tokens.

    This treats one grid axis as image channels and runs 2D patch attention
    over the other two, so it is a 2D patch encoder in disguise: nothing
    enforces 3D locality or translation-equivariance along the channel axis.
    Matches the original TRUMANS `scene_embedding` construction.
    """

    def __init__(self, voxels: int):
        super().__init__()
        voxels = int(voxels)
        patch = voxels // 4
        self.output_dim = 1024
        # num_classes=0 makes vit-pytorch skip its pooling head and return the
        # whole token sequence, which is what the motion blocks attend into.
        self.model = ViT(
            image_size=voxels,
            patch_size=patch,
            channels=voxels,
            num_classes=0,
            dim=self.output_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        # One CLS token ahead of the patch grid.
        self.token_count = (voxels // patch) ** 2 + 1

    def forward(self, occupancy: torch.Tensor) -> torch.Tensor:
        return self.model(occupancy)


class CNN3DSceneEncoder(nn.Module):
    """
    Encode an occupancy grid with a small 3D CNN backbone.

    Strided 3D convolutions give real volumetric locality and
    translation-equivariance along every axis, matching the occupancy
    grid's actual structure far more directly than a repurposed 2D ViT.
    Each surviving cell of the final feature volume becomes one token, so the
    tokens keep the coarse 3D layout of the crop; `voxels` only decides how
    many of them there are.
    """

    def __init__(self, voxels: int):
        super().__init__()
        self.output_dim = 256
        channels = (1, 32, 64, self.output_dim)
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True),
            ]
        self.model = nn.Sequential(*layers)
        resolution = int(voxels)
        for _ in range(len(channels) - 1):
            resolution = -(-resolution // 2)
        self.token_count = resolution ** 3

    def forward(self, occupancy: torch.Tensor) -> torch.Tensor:
        features = self.model(occupancy.unsqueeze(1))
        # [batch, channels, depth, height, width] -> one token per cell.
        return features.flatten(2).transpose(1, 2)
