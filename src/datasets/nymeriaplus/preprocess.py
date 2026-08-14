"""
Preprocess raw NymeriaPlus.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.floor import estimate_floor
from src.utils.geometry import FEATURE_DIM, JOINT_COUNT, anchor_local, facing_yaw

# Annotations shorter than this are mostly noise; longer ones (up to 757
# frames observed) are composite/summary labels rather than atomic actions.
# Must also be >= configs/default.yaml's window_size: NymeriaDataset asserts
# every annotation is at least window_size frames long (see its __init__)
# rather than silently dropping/padding short ones. Changing either value
# needs re-running preprocessing (annotations.jsonl, floor.npz,
# normalization.npz, text caches).
MIN_ANNOTATION_FRAMES = 16  # 1.6s @ 10fps, matches window_size=16
MAX_ANNOTATION_FRAMES = 60  # 6.0s @ 10fps


def load_motion(
    path: Path, model, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMPL motion and extract joints and facing yaw.
    """

    with np.load(path, allow_pickle=False) as loaded:
        smpl = {key: np.asarray(loaded[key]) for key in loaded.files}

    timestamps = np.asarray(smpl["timestamps"], dtype=np.int64)
    count = len(timestamps)
    # Betas describe one participant despite being stored per frame;
    # averaging gives one consistent skeleton instead of frame-to-frame jitter.
    betas = torch.as_tensor(smpl["betas"], dtype=torch.float32).mean(0, keepdim=True)
    body_pose = torch.as_tensor(smpl["body_pose"], dtype=torch.float32)
    global_orient = torch.as_tensor(smpl["global_orient"], dtype=torch.float32)
    transl = torch.as_tensor(smpl["transl"], dtype=torch.float32)

    joints = torch.empty((count, JOINT_COUNT, 3), dtype=torch.float32)
    for first in range(0, count, batch_size):
        last = min(first + batch_size, count)
        frames = slice(first, last)
        with torch.no_grad():
            output = model(
                betas=betas.expand(last - first, -1),
                body_pose=body_pose[frames],
                global_orient=global_orient[frames],
                transl=transl[frames],
                return_verts=False,
            )
        joints[frames] = output.joints[:, :JOINT_COUNT]

    # Hip-based heading (shared with inference) -- global_orient's euler
    # decomposition is gimbal-locked for this y-up SMPL template in a z-up world.
    yaw = facing_yaw(joints)
    return timestamps, joints.numpy(), yaw.numpy().astype(np.float32)


def load_annotations(path: Path, timestamps: np.ndarray) -> list[dict]:
    """
    Read usable annotations and convert their times to SMPL frame bounds.
    """

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Start sequence from timestamp 0
    timestamps = timestamps - timestamps[0]
    time_origin = min(float(row["start_time"]) for row in rows)
    annotations = []
    for row in rows:
        start = float(row["start_time"]) - time_origin
        end = float(row["end_time"]) - time_origin

        # Nearest frame index for each timestamp
        start_frame = int(np.searchsorted(
            timestamps, start * 1_000_000.0, side="left"
        ))
        end_frame = int(np.searchsorted(
            timestamps, end * 1_000_000.0, side="right"
        ) - 1)

        length = end_frame - start_frame + 1
        if length < MIN_ANNOTATION_FRAMES or length > MAX_ANNOTATION_FRAMES:
            continue

        text = row["Describe my atomic actions"]
        text = re.sub(r"\bC\b", "the person", text.strip()).lower()
        annotations.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "text": text,
        })

    return annotations


def load_scene(
    recording: Path, joints: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load scene objects and voxelize them around the recorded motion.
    """

    point_clouds = []
    loaded_uids = set()

    # Load ShapeR meshes
    shaper = recording / "objects" / "shaper"
    metadata = shaper / "shaper_metadata.csv"
    with metadata.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Select best meshes by score
    best_meshes = {}
    for row in rows:
        uid = int(row["object_uid"])
        score = (int(row["score"]), int(row["variant_id"]))
        if uid not in best_meshes or score > best_meshes[uid][0]:
            best_meshes[uid] = (score, row)

    # Load meshes
    for uid, (_, row) in best_meshes.items():
        loaded = trimesh.load(
            shaper / row["file_name"], force="mesh", process=False
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(loaded.vertices, dtype=np.float32),
            faces=np.asarray(loaded.faces),
            process=False,
        )
        points = mesh.voxelized(pitch=voxel_size).fill().points.astype(np.float32)
        point_clouds.append(points)
        loaded_uids.add(uid)

    # Load Boxy as fallback
    boxy = recording / "objects" / "boxy"
    with (boxy / "3dbb.csv").open(newline="", encoding="utf-8") as file:
        boxes = {
            int(row["object_uid"]): row
            for row in csv.DictReader(file, skipinitialspace=True)
        }
    with (boxy / "scene_objects.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        objects = {
            int(row["object_uid"]): row
            for row in csv.DictReader(file, skipinitialspace=True)
        }

    for uid in (set(boxes) & set(objects)) - loaded_uids:
        box, obj = boxes[uid], objects[uid]
        lower = np.array(
            [float(box[f"p_local_obj_{axis}min[m]"]) for axis in "xyz"],
            dtype=np.float32,
        )
        upper = np.array(
            [float(box[f"p_local_obj_{axis}max[m]"]) for axis in "xyz"],
            dtype=np.float32,
        )
        rotation = Rotation.from_quat(
            [float(obj[f"q_wo_{axis}"]) for axis in "xyzw"]
        ).as_matrix()
        translation = np.array(
            [float(obj[f"t_wo_{axis}[m]"]) for axis in "xyz"], dtype=np.float32
        )
        axes = [
            np.arange(lower[i], upper[i] + voxel_size * 0.5, voxel_size)
            for i in range(3)
        ]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        point_clouds.append((points @ rotation.T + translation).astype(np.float32))

    # Build occupancy grid
    minimum = joints.reshape(-1, 3).min(axis=0) - 2.0
    requested_maximum = joints.reshape(-1, 3).max(axis=0) + 2.0
    shape = np.ceil(
        (requested_maximum - minimum) / voxel_size
    ).astype(np.int64)
    occupancy = np.zeros(tuple(shape), dtype=np.bool_)
    if point_clouds:
        points = np.concatenate(point_clouds)
        indices = np.floor((points - minimum) / voxel_size).astype(np.int64)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        occupancy[tuple(indices[inside].T)] = True
    maximum = minimum + shape * voxel_size
    return occupancy, np.asarray([minimum, maximum], dtype=np.float32)


def compute_normalization(
    processed: Path,
    *,
    history_size: int = 16,
) -> None:
    """
    Compute anchored-joint-position statistics, one pass per annotation.

    Only the train split contributes, so val/test never leak into training
    normalization. Uses each annotation's full span, not a window_size
    sample, for lower-noise per-channel stats. The goal reuses channels
    [0:3] (see MotionStatistics.normalize_pelvis), so it needs no stats of
    its own.
    """

    splits_file = Path(__file__).parent / "splits.json"
    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    train_sequences = set(splits["train"])
    total = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_squares = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_weight = 0.0

    sequences = [
        path for path in sorted((processed / "sequences").iterdir())
        if path.name in train_sequences
    ]
    for sequence in tqdm(
        sequences, desc="Computing normalization", unit="sequence"
    ):
        joints = np.load(sequence / "joints.npy", mmap_mode="r", allow_pickle=False)
        yaw = np.load(sequence / "yaw.npy", mmap_mode="r", allow_pickle=False)
        with np.load(sequence / "floor.npz", allow_pickle=False) as data:
            floor = data["floor"]
        frame_count = len(joints)
        annotations = (
            json.loads(line)
            for line in (sequence / "annotations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for annotation in annotations:
            start = int(annotation["start_frame"])
            real_end = min(int(annotation["end_frame"]), frame_count - 1)
            history_start = max(0, start - history_size)
            history_length = start - history_start
            anchor = history_start + (
                history_length - 1 if history_length else 0
            )
            anchor_position = np.array(
                [joints[anchor, 0, 0], joints[anchor, 0, 1], floor[anchor]],
                dtype=np.float64,
            )
            anchor_yaw = float(yaw[anchor])

            window_joints = np.array(
                joints[history_start : real_end + 1], dtype=np.float64
            )
            window = anchor_local(window_joints, anchor_position, anchor_yaw).reshape(
                len(window_joints), -1
            )
            total += window.sum(axis=0)
            total_squares += np.square(window).sum(axis=0)
            total_weight += len(window)

    mean, std = _mean_and_std(total, total_squares, total_weight)
    np.savez(processed / "normalization.npz", mean=mean, std=std)


def _mean_and_std(total, total_squares, weight) -> tuple[np.ndarray, np.ndarray]:
    """
    Finish a weighted mean/standard-deviation accumulation.
    """

    mean = total / weight
    variance = np.maximum(total_squares / weight - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    """
    Preprocess NymeriaPlus recordings from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-male-model", type=Path, required=True)
    parser.add_argument("--smpl-female-model", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.0625)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()

    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    (output_root / "sequences").mkdir(parents=True, exist_ok=True)

    models = {
        "male": smplx.SMPL(
            str(args.smpl_male_model),
            batch_size=args.batch_size,
        ).eval(),
        "female": smplx.SMPL(
            str(args.smpl_female_model),
            batch_size=args.batch_size,
        ).eval(),
    }

    count_annotations = 0
    recordings = [
        path for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]

    for recording in tqdm(recordings, desc="Preprocessing", unit="recording"):
        body_path = recording / "body" / "xdata_smpl_neutral.npz"
        action_path = recording / "narration" / "atomic_action.csv"
        metadata_path = recording / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gender = metadata["participant_gender"].lower()

        timestamps, joints, yaw = load_motion(
            body_path, models[gender], args.batch_size
        )
        annotations = load_annotations(action_path, timestamps)
        occupancy, bounds = load_scene(recording, joints, args.voxel_size)

        floor = estimate_floor(joints, causal=False)

        destination = output_root / "sequences" / recording.name
        destination.mkdir()
        np.save(destination / "joints.npy", joints)
        np.save(destination / "yaw.npy", yaw)
        np.savez_compressed(destination / "floor.npz", floor=floor)
        np.savez_compressed(destination / "scene_occupancy.npz", occupancy=occupancy)
        np.save(destination / "scene_bounds.npy", bounds)
        annotations_path = destination / "annotations.jsonl"
        with annotations_path.open("w", encoding="utf-8") as file:
            for annotation in annotations:
                file.write(json.dumps(annotation, sort_keys=True) + "\n")

        count_annotations += len(annotations)

    compute_normalization(output_root, history_size=args.history_size)

    print(
        f"Processed {len(recordings)} sequences and "
        f"{count_annotations} valid annotations.",
        flush=True,
    )


if __name__ == "__main__":
    main()
