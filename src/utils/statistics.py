"""
Standard-score statistics for anchored joint-position features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionStatistics:
    """
    Per-channel standard-score statistics for the 72-D anchored joint-position
    motion features (`JOINT_COUNT * 3`, see `src.utils.geometry`).

    The goal has no statistics of its own: it is the window's own last pelvis
    position, so it shares channels `[0:3]` with the motion feature.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float32)
            )

    @classmethod
    def load(cls, path: str | Path) -> "MotionStatistics":
        """
        Load feature standard-score statistics.
        """

        with np.load(path, allow_pickle=False) as values:
            return cls(values["mean"], values["std"])

    def normalize(self, values):
        """
        Standardize anchored joint-position features along the last dimension.
        """

        mean, std = self._match(self.mean, self.std, values)
        return (values - mean) / std

    def denormalize(self, values):
        """
        Restore standardized features to metric anchor-local joint positions.
        """

        mean, std = self._match(self.mean, self.std, values)
        return values * std + mean

    def normalize_pelvis(self, values):
        """
        Standardize an anchor-local pelvis position (dx, dy, dz) with the same
        per-channel statistics as the motion feature's own channels [0:3].
        """

        mean, std = self._match(self.mean[..., :3], self.std[..., :3], values)
        return (values - mean) / std

    @staticmethod
    def _match(mean, std, values):
        if torch.is_tensor(values):
            return (
                torch.as_tensor(mean, device=values.device, dtype=values.dtype),
                torch.as_tensor(std, device=values.device, dtype=values.dtype),
            )
        return mean, std
