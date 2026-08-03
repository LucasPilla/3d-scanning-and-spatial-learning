"""
Standard-score statistics for canonical motion features and pelvis goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionStatistics:
    """
    Per-channel standard-score statistics for the 148-D motion features.

    Goal statistics are kept separately because the goal is an anchor-local
    pelvis offset, which the delta-based motion features no longer describe.
    """

    mean: np.ndarray
    std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "std", "goal_mean", "goal_std"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float32)
            )

    @classmethod
    def load(cls, path: str | Path) -> "MotionStatistics":
        """
        Load feature and goal standard-score statistics.
        """

        with np.load(path, allow_pickle=False) as values:
            return cls(
                values["mean"],
                values["std"],
                values["goal_mean"],
                values["goal_std"],
            )

    def normalize(self, values):
        """
        Standardize canonical motion features along the last dimension.
        """

        mean, std = self._match(self.mean, self.std, values)
        return (values - mean) / std

    def denormalize(self, values):
        """
        Restore standardized features to canonical metric features.
        """

        mean, std = self._match(self.mean, self.std, values)
        return values * std + mean

    def normalize_goal(self, values):
        """
        Standardize an anchor-local pelvis goal displacement (dx, dy, dz).
        """

        mean, std = self._match(self.goal_mean, self.goal_std, values)
        return (values - mean) / std

    @staticmethod
    def _match(mean, std, values):
        if torch.is_tensor(values):
            return (
                torch.as_tensor(mean, device=values.device, dtype=values.dtype),
                torch.as_tensor(std, device=values.device, dtype=values.dtype),
            )
        return mean, std
