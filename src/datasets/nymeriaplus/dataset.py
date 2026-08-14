"""
Memory-mapped Nymeria annotations with one fixed-length window per segment.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.statistics import MotionStatistics
from src.utils.geometry import anchor_local, pad_frames
from src.utils.scene import crop_scene, local_meshgrid


class NymeriaDataset(Dataset):
    """
    Visit every valid annotation exactly once per epoch.
    """

    def __init__(
        self,
        folder: str | Path,
        *,
        split: str | None = None,
        window_size: int = 16,
        history_size: int = 16,
        max_open_sequences: int = 8,
        scene_enabled: bool = True,
        scene_voxels: int = 64,
        scene_bounds=(-1.5, 1.5, -1.5, 1.5, -0.3, 2.7),
        text_enabled: bool = True,
        text_cache_path: str | Path | None = None,
        text_tokens: int | None = None,
        goal_enabled: bool = True,
    ):
        self.folder = Path(folder)
        self.window_size = int(window_size)
        self.history_size = int(history_size)
        self.max_open_sequences = max(1, int(max_open_sequences))
        self.scene_enabled = bool(scene_enabled)
        self.scene_voxels = int(scene_voxels)
        self.scene_bounds = tuple(float(value) for value in scene_bounds)
        self._scene_grid = (
            local_meshgrid(self.scene_bounds, self.scene_voxels)
            if self.scene_enabled
            else None
        )
        self.text_enabled = bool(text_enabled)
        self.goal_enabled = bool(goal_enabled)

        self.statistics = MotionStatistics.load(self.folder / "normalization.npz")
        sequences_root = self.folder / "sequences"
        allowed = None
        if split is not None:
            splits_file = Path(__file__).parent / "splits.json"
            splits = json.loads(splits_file.read_text(encoding="utf-8"))
            allowed = set(splits[split])

        self.sequence_paths = {}
        self.samples = []
        # Small per-sequence arrays, read once so the sequence cache only has
        # to hold the memory-mapped ones.
        self.scene_bounds_by_sequence = {}
        for path in sorted(
            candidate for candidate in sequences_root.iterdir()
            if candidate.is_dir() and not candidate.name.startswith(".")
            and (allowed is None or candidate.name in allowed)
        ):
            joints = np.load(
                path / "joints.npy", mmap_mode="r", allow_pickle=False
            )
            with (path / "annotations.jsonl").open(encoding="utf-8") as file:
                annotations = [json.loads(line) for line in file if line.strip()]
            self.sequence_paths[path.name] = path
            if self.scene_enabled:
                self.scene_bounds_by_sequence[path.name] = np.load(
                    path / "scene_bounds.npy", allow_pickle=False
                )
            for annotation in annotations:
                # Each annotation becomes one sample; __getitem__ draws a
                # window_size-frame window from within its span, so every
                # annotation must be >= window_size frames (enforced by
                # preprocess.py's MIN_ANNOTATION_FRAMES; see assert below).
                # This list is exactly the annotations file, in order --
                # src/cache.py's text cache row order relies on that.
                start_frame = int(annotation["start_frame"])
                end_frame = min(int(annotation["end_frame"]), len(joints) - 1)
                assert end_frame - start_frame + 1 >= self.window_size, (
                    f"Annotation at {path.name}:{start_frame}-{end_frame} is shorter than "
                    f"window_size={self.window_size}; regenerate annotations.jsonl with "
                    f"MIN_ANNOTATION_FRAMES >= window_size (src/datasets/nymeriaplus/preprocess.py)."
                )
                self.samples.append({
                    "sequence": path.name,
                    "text": str(annotation["text"]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })

        # src/cache.py encodes exactly this list, in this order.
        self.texts = [sample["text"] for sample in self.samples]
        self.text_features = None
        if self.text_enabled and text_cache_path:
            self.text_features = np.load(
                text_cache_path, mmap_mode="r", allow_pickle=False
            )
            # Row order matches by construction; this only catches a cache
            # left over from a different corpus, split, or token budget.
            expected = (len(self.samples), text_tokens)
            if self.text_features.shape[:2] != expected and text_tokens:
                raise ValueError(
                    f"Text cache {Path(text_cache_path).name} has shape "
                    f"{tuple(self.text_features.shape)}, expected "
                    f"[{expected[0]}, {expected[1]}, features]; "
                    "regenerate it with src/cache.py"
                )
        self._open_sequences: OrderedDict[str, tuple[np.ndarray, ...]] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_sequences"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def sequence(self, name: str):
        """
        Return one sequence's memory-mapped arrays, caching recent ones.

        Only large arrays are cached; the two tiny ones were already read
        once in __init__, so a cache miss (common under shuffled sampling)
        never re-reads them.
        """

        if name in self._open_sequences:
            values = self._open_sequences.pop(name)
        else:
            path = self.sequence_paths[name]
            with np.load(path / "floor.npz", allow_pickle=False) as data:
                floor = data["floor"]
            values = [
                np.load(path / "joints.npy", mmap_mode="r", allow_pickle=False),
                np.load(path / "yaw.npy", mmap_mode="r", allow_pickle=False),
                floor,
            ]
            if self.scene_enabled:
                scene_file = path / "scene_occupancy.npz"
                if scene_file.exists():
                    with np.load(scene_file, allow_pickle=False) as data:
                        values.append(data["occupancy"])
                else:
                    values.append(
                        np.load(path / "scene.npy", mmap_mode="r", allow_pickle=False)
                    )
            values = tuple(values)
        self._open_sequences[name] = values
        if len(self._open_sequences) > self.max_open_sequences:
            self._open_sequences.popitem(last=False)
        return values

    def __getitem__(self, item: int) -> dict:
        item = int(item)
        sample = self.samples[item]
        # Random window_size-frame window within the annotation's span.
        # Every annotation is >= window_size frames (enforced in __init__),
        # so this range is never empty -- an annotation exactly window_size
        # long just always picks the same window.
        annotation_start = sample["start_frame"]
        latest_start = sample["end_frame"] - self.window_size + 1
        start = int(np.random.randint(annotation_start, latest_start + 1))
        # Where this window sits within the annotation's span (0 = earliest
        # possible start, 1 = latest) -- lets the model tell "just starting
        # to X" from "finishing X" even though every window here shares the
        # same text. 0.0 when the annotation is exactly window_size long, the
        # only start available.
        window_progress = (
            (start - annotation_start) / (latest_start - annotation_start)
            if latest_start > annotation_start else 0.0
        )
        real_end_frame = start + self.window_size - 1
        sequence = self.sequence(sample["sequence"])
        joints_all, yaw_all, floor_all = sequence[:3]

        available_history = start - max(0, start - self.history_size)
        full_history = available_history == self.history_size
        history_length = self.history_size if full_history and np.random.rand() < 0.5 else 0
        history_start = start - history_length
        anchor_index = history_length - 1 if history_length else 0
        anchor_global = history_start + anchor_index
        # x/y from the pelvis; z from estimated floor height, not pelvis z
        # (which drifts with pose -- see src/utils/floor.py), so the
        # anchor's vertical reference stays stable across poses.
        anchor_position = np.array(
            [joints_all[anchor_global, 0, 0], joints_all[anchor_global, 0, 1], floor_all[anchor_global]],
            dtype=np.float32,
        )
        anchor_yaw = float(yaw_all[anchor_global])

        window_joints = np.array(
            joints_all[history_start : real_end_frame + 1], dtype=np.float32
        )
        combined = anchor_local(window_joints, anchor_position, anchor_yaw).reshape(
            len(window_joints), -1
        )

        normalized = self.statistics.normalize(combined).astype(
            np.float32, copy=False
        )
        history, history_mask = pad_frames(
            normalized[:history_length], self.history_size, normalized.shape[-1],
            align="right",
        )
        motion, target_mask = pad_frames(
            normalized[history_length:], self.window_size, normalized.shape[-1],
            align="left",
        )

        result = {
            "motion": motion,
            "history": history,
            "history_mask": history_mask,
            "target_mask": target_mask,
            "annotation_index": item,
            "window_start": start,
            # World-space anchor pose, so a caller can transform generated
            # positions back to world space (e.g. to overlay scene geometry).
            "anchor_position": torch.from_numpy(anchor_position),
            "anchor_yaw": torch.tensor(anchor_yaw, dtype=torch.float32),
        }
        if self.goal_enabled:
            # Goal = the window's own last frame's pelvis position (3D
            # displacement from the anchor): "where you already end up" --
            # a clean steering target, since the window genuinely reaches
            # it. Soft/CFG-guided like text and scene, not hard-inpainted
            # (see MotionDiffusion). No arrival time is given since the
            # window length implies it.
            # real_end_frame <= end_frame <= len(joints_all) - 1 always
            # (annotations are >= window_size frames, see __init__), so no
            # bounds clamping is needed here.
            goal_world = np.asarray(joints_all[real_end_frame, 0], dtype=np.float32)
            goal_local = anchor_local(goal_world, anchor_position, anchor_yaw)
            result["goal"] = torch.from_numpy(
                np.asarray(
                    self.statistics.normalize_pelvis(goal_local), dtype=np.float32
                )
            )
        if self.scene_enabled:
            occupancy = sequence[3]
            bounds = self.scene_bounds_by_sequence[sample["sequence"]]
            result["scene"] = torch.from_numpy(
                crop_scene(
                    occupancy,
                    bounds,
                    anchor_position,
                    anchor_yaw,
                    self.scene_bounds,
                    self.scene_voxels,
                    local_grid=self._scene_grid,
                ).copy()
            )
        if self.text_enabled:
            result["window_progress"] = torch.tensor(window_progress, dtype=torch.float32)
            if self.text_features is None:
                result["text"] = sample["text"]
            else:
                # Writable copy off the memmap, kept at cache precision --
                # the model casts to its own dtype on device anyway.
                result["text_features"] = torch.from_numpy(
                    np.array(self.text_features[item])
                )
        return result
