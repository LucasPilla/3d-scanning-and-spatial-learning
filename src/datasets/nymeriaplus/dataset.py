"""
Memory-mapped Nymeria annotations with one random window per segment.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.kinematics import ROOT_DELTA, ROOT_YAW_DELTA
from src.utils.statistics import MotionStatistics
from src.utils.geometry import pad_history, yaw_rotation
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
        history_size: int = 32,
        max_open_sequences: int = 8,
        scene_enabled: bool = True,
        scene_voxels: int = 32,
        scene_bounds=(-1.0, 1.0, -1.0, 1.0, -1.2, 0.8),
        text_enabled: bool = True,
        text_cache_path: str | Path | None = None,
        text_tokens: int | None = None,
        goal_enabled: bool = True,
        max_goal_horizon: int = 32,
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
        self.max_goal_horizon = int(max_goal_horizon)

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
        self.bone_offsets = {}
        self.scene_bounds_by_sequence = {}
        for path in sorted(
            candidate for candidate in sequences_root.iterdir()
            if candidate.is_dir() and not candidate.name.startswith(".")
            and (allowed is None or candidate.name in allowed)
        ):
            features = np.load(
                path / "features.npy", mmap_mode="r", allow_pickle=False
            )
            with (path / "annotations.jsonl").open(encoding="utf-8") as file:
                annotations = [json.loads(line) for line in file if line.strip()]
            self.sequence_paths[path.name] = path
            self.bone_offsets[path.name] = np.load(
                path / "bone_offsets.npy", allow_pickle=False
            ).astype(np.float32, copy=False)
            if self.scene_enabled:
                self.scene_bounds_by_sequence[path.name] = np.load(
                    path / "scene_bounds.npy", allow_pickle=False
                )
            last_start = len(features) - self.window_size
            if last_start < 0:
                raise ValueError(
                    f"Sequence {path.name} has {len(features)} frames, fewer "
                    f"than window_size={self.window_size}"
                )
            for annotation in annotations:
                # Every annotation yields a sample: preprocess.py already drops
                # the ones too short to hold a window, so this list is exactly
                # the annotations file, which is what makes the text cache's
                # row order match by construction. Windows normally start
                # anywhere inside the annotation that leaves room; where the
                # training window is longer than the one preprocessing assumed,
                # clamping lets the window run a little past the annotation
                # rather than silently dropping it — the same latitude the goal
                # sampler already takes when it reaches beyond annotation ends.
                window_first = min(int(annotation["start_frame"]), last_start)
                window_last = min(
                    int(annotation["end_frame"]) - self.window_size + 1, last_start
                )
                self.samples.append({
                    "sequence": path.name,
                    "text": str(annotation["text"]),
                    "frame_count": len(features),
                    "window_first": window_first,
                    "window_count": max(1, window_last - window_first + 1),
                })

        # src/cache.py encodes exactly this list, in this order.
        self.texts = [sample["text"] for sample in self.samples]
        self.text_features = None
        if self.text_enabled and text_cache_path:
            self.text_features = np.load(
                text_cache_path, mmap_mode="r", allow_pickle=False
            )
            # The only cache check that survives: row order matches by
            # construction, but a file left over from a different corpus,
            # split, or token budget would otherwise be read as if it fit.
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
        Return one sequence's memory-mapped arrays, keeping recent ones open.

        Only the large arrays go through the cache; the two tiny ones are read
        once in `__init__`, so a miss under shuffled sampling — which is nearly
        every item, since the cache holds far fewer sequences than the split —
        never pays for them.
        """

        if name in self._open_sequences:
            values = self._open_sequences.pop(name)
        else:
            path = self.sequence_paths[name]
            values = [
                np.load(path / "features.npy", mmap_mode="r", allow_pickle=False),
                np.load(path / "joints.npy", mmap_mode="r", allow_pickle=False),
                np.load(path / "yaw.npy", mmap_mode="r", allow_pickle=False),
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
        start = np.random.randint(
            sample["window_first"],
            sample["window_first"] + sample["window_count"],
        )
        sequence = self.sequence(sample["sequence"])
        features_all, joints_all, yaw_all = sequence[:3]

        # Uniformly truncated history keeps every length from empty to full
        # in-distribution, matching cold-start and rolling inference windows.
        available_history = start - max(0, start - self.history_size)
        history_length = np.random.randint(available_history + 1)
        history_start = start - history_length
        combined = np.array(
            features_all[history_start : start + self.window_size], dtype=np.float32
        )
        # The first sliced frame has no predecessor inside the slice, so its
        # root delta is meaningless. Zeroing it is what puts the reconstruction
        # anchor at the origin, and it keeps training and rollout identical.
        combined[0, ROOT_DELTA.start : ROOT_YAW_DELTA.stop] = 0.0
        anchor_index = history_length - 1 if history_length else 0
        anchor_global = history_start + anchor_index
        world_to_local = yaw_rotation(float(yaw_all[anchor_global]))
        anchor_position = np.asarray(joints_all[anchor_global, 0], dtype=np.float32)

        normalized = self.statistics.normalize(combined).astype(
            np.float32, copy=False
        )
        motion = torch.from_numpy(normalized[history_length:])
        history, history_mask = pad_history(
            normalized[:history_length], self.history_size, normalized.shape[-1]
        )

        result = {
            "motion": motion,
            "history": history,
            "history_mask": history_mask,
            "bone_offsets": torch.from_numpy(
                self.bone_offsets[sample["sequence"]].copy()
            ),
            "annotation_index": item,
            "window_start": start,
        }
        if self.goal_enabled:
            # Sample the goal from a random future pelvis position within the
            # horizon, crossing annotation ends so late windows still see far
            # goals. The goal is the 3D displacement from the anchor pelvis;
            # vertical motion is what makes sitting, standing, and stairs
            # steerable. No arrival time is given — how soon the goal is due is
            # left to the text and scene semantics to imply.
            window_end = start + self.window_size - 1
            last_goal_frame = min(
                sample["frame_count"] - 1, window_end + self.max_goal_horizon
            )
            goal_frame = np.random.randint(window_end, last_goal_frame + 1)
            goal_world = np.asarray(joints_all[goal_frame, 0], dtype=np.float32)
            goal_local = (goal_world - anchor_position) @ world_to_local.T
            result["goal"] = torch.from_numpy(
                np.asarray(
                    self.statistics.normalize_goal(goal_local), dtype=np.float32
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
                    float(yaw_all[anchor_global]),
                    self.scene_bounds,
                    self.scene_voxels,
                    local_grid=self._scene_grid,
                ).copy()
            )
        if self.text_enabled:
            if self.text_features is None:
                result["text"] = sample["text"]
            else:
                # One writable copy off the memmap, at the cache's own
                # precision: the model casts to its parameter dtype on device
                # anyway, so upcasting here would only double the worker queue
                # and the host-to-device copy.
                result["text_features"] = torch.from_numpy(
                    np.array(self.text_features[item])
                )
        return result
