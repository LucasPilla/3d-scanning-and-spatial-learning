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
from smplx.lbs import blend_shapes, vertices2joints
from tqdm import tqdm

from src.utils.kinematics import (
    FEATURE_DIM,
    JOINT_COUNT,
    ROOT_DELTA,
    ROOT_YAW_DELTA,
    SMPL_PARENTS,
    build_features,
)
from src.utils.geometry import facing_yaw, yaw_rotation


def bone_offsets(model, betas: torch.Tensor) -> torch.Tensor:
    """
    Rest joint positions relative to the parent, for one participant's shape.

    Row 0 keeps the absolute rest root position, which converts between a
    posed root position and SMPL's `transl`.
    """

    if tuple(model.parents[:JOINT_COUNT].tolist()) != SMPL_PARENTS:
        raise ValueError("The SMPL model's kinematic tree does not match SMPL_PARENTS")
    shaped = model.v_template + blend_shapes(betas, model.shapedirs)
    rest = vertices2joints(model.J_regressor, shaped)[0]
    offsets = rest.clone()
    offsets[1:] = rest[1:] - rest[list(SMPL_PARENTS[1:])]
    return offsets


def load_motion(
    path: Path, model, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMPL motion and extract joints, facing yaw, features, bone offsets.
    """

    with np.load(path, allow_pickle=False) as loaded:
        smpl = {key: np.asarray(loaded[key]) for key in loaded.files}

    timestamps = np.asarray(smpl["timestamps"], dtype=np.int64)
    count = len(timestamps)
    # Betas are stored per frame but describe a single participant. Averaging
    # them gives one skeleton, so forward kinematics over the extracted
    # rotations reproduces these joints exactly rather than approximately.
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

    # Hip-based heading, shared with inference; the euler decomposition of
    # global_orient is gimbal-locked for the y-up SMPL template in this
    # z-up world.
    yaw = facing_yaw(joints)
    features = build_features(joints[:, 0], yaw, global_orient, body_pose)
    offsets = bone_offsets(model, betas)
    return (
        timestamps,
        joints.numpy(),
        yaw.numpy().astype(np.float32),
        features.numpy().astype(np.float32),
        offsets.numpy().astype(np.float32),
    )


def load_annotations(
    path: Path, timestamps: np.ndarray, window_size: int
) -> list[dict]:
    """
    Read usable annotations and convert their times to SMPL frame bounds.
    """

    # Load text annotation file
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Start sequence from timestamp 0
    timestamps = timestamps - timestamps[0]
    time_origin = min(float(row["start_time"]) for row in rows)
    annotations = []
    for row in rows:

        # Get start and end timestamps
        start = float(row["start_time"]) - time_origin
        end = float(row["end_time"]) - time_origin

        # Get closest start and end frame
        start_frame = int(np.searchsorted(
            timestamps, start * 1_000_000.0, side="left"
        ))
        end_frame = int(np.searchsorted(
            timestamps, end * 1_000_000.0, side="right"
        ) - 1)

        if end_frame - start_frame + 1 < window_size:
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
    window_size: int = 16,
    history_size: int = 32,
    max_goal_horizon: int = 32,
    goal_samples: int = 2,
    seed: int = 0,
) -> None:
    """
    Compute weighted feature and goal statistics, and the default skeleton.

    Only the train split contributes, so validation and test locations never
    leak into the normalization the model trains against. Windows are weighted
    by `1 / window_count` so long annotations do not dominate, matching how the
    dataset draws one random window per annotation. Goal offsets are sampled a
    few times per window with the same uniform horizon draw the dataset uses.
    """

    splits_file = Path(__file__).parent / "splits.json"
    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    train_sequences = set(splits["train"])
    generator = np.random.default_rng(seed)
    total = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_squares = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_weight = 0.0
    goal_total = np.zeros(3, dtype=np.float64)
    goal_total_squares = np.zeros(3, dtype=np.float64)
    goal_weight = 0.0
    skeleton_total = np.zeros((JOINT_COUNT, 3), dtype=np.float64)
    skeleton_count = 0

    sequences = [
        path for path in sorted((processed / "sequences").iterdir())
        if path.name in train_sequences
    ]
    for sequence in tqdm(
        sequences, desc="Computing normalization", unit="sequence"
    ):
        features_all = np.load(
            sequence / "features.npy", mmap_mode="r", allow_pickle=False
        )
        joints = np.load(sequence / "joints.npy", mmap_mode="r", allow_pickle=False)
        yaw = np.load(sequence / "yaw.npy", mmap_mode="r", allow_pickle=False)
        skeleton_total += np.load(
            sequence / "bone_offsets.npy", allow_pickle=False
        )
        skeleton_count += 1
        frame_count = len(features_all)
        annotations = (
            json.loads(line)
            for line in (sequence / "annotations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for annotation in annotations:
            first = int(annotation["start_frame"])
            last = min(int(annotation["end_frame"]), frame_count - 1) - window_size + 1
            window_count = last - first + 1
            if window_count <= 0:
                continue
            window_weight = 1.0 / window_count
            for start in range(first, last + 1):
                history_start = max(0, start - history_size)
                history_length = start - history_start
                anchor = history_start + (
                    history_length - 1 if history_length else 0
                )
                window = np.array(
                    features_all[history_start : start + window_size],
                    dtype=np.float64,
                )
                # The first sliced frame has no predecessor inside the window,
                # so the dataset zeroes its root delta; mirror that here.
                window[0, ROOT_DELTA.start : ROOT_YAW_DELTA.stop] = 0.0
                total += window.sum(axis=0) * window_weight
                total_squares += np.square(window).sum(axis=0) * window_weight
                total_weight += len(window) * window_weight

                window_end = start + window_size - 1
                last_goal_frame = min(
                    frame_count - 1, window_end + max_goal_horizon
                )
                anchor_position = np.asarray(joints[anchor, 0], dtype=np.float64)
                world_to_local = yaw_rotation(float(yaw[anchor]), dtype=np.float64)
                goal_frames = generator.integers(
                    window_end, last_goal_frame + 1, size=goal_samples
                )
                offsets = (
                    np.asarray(joints[goal_frames, 0], dtype=np.float64)
                    - anchor_position
                ) @ world_to_local.T
                goal_total += offsets.sum(axis=0) * window_weight
                goal_total_squares += np.square(offsets).sum(axis=0) * window_weight
                goal_weight += goal_samples * window_weight

    mean, std = _mean_and_std(total, total_squares, total_weight)
    goal_mean, goal_std = _mean_and_std(
        goal_total, goal_total_squares, goal_weight
    )
    np.savez(
        processed / "normalization.npz",
        mean=mean,
        std=std,
        goal_mean=goal_mean,
        goal_std=goal_std,
    )
    np.save(
        processed / "skeleton.npy",
        (skeleton_total / max(skeleton_count, 1)).astype(np.float32),
    )


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

    # Parse CLI parameters
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-male-model", type=Path, required=True)
    parser.add_argument("--smpl-female-model", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.0625)
    parser.add_argument("--max-goal-horizon", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Input directory
    input_root = args.input.expanduser().resolve()

    # Output directory
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    (output_root / "sequences").mkdir(parents=True, exist_ok=True)

    # Initialize SMPL models
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

    # Process each recording sequence
    count_annotations = 0
    recordings = [
        path for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]

    for recording in tqdm(recordings, desc="Preprocessing", unit="recording"):

        # Path to motion file
        body_path = recording / "body" / "xdata_smpl_neutral.npz"

        # Path to text annotation file
        action_path = recording / "narration" / "atomic_action.csv"

        # Path to metadata
        metadata_path = recording / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Read participant gender
        gender = metadata["participant_gender"].lower()

        # Load motion
        timestamps, joints, yaw, features, bone_offsets = load_motion(
            body_path, models[gender], args.batch_size
        )

        # Load annotations
        annotations = load_annotations(
            action_path, timestamps, args.window_size
        )

        # Load scene
        occupancy, bounds = load_scene(recording, joints, args.voxel_size)

        # Save outputs
        destination = output_root / "sequences" / recording.name
        destination.mkdir()
        np.save(destination / "joints.npy", joints)
        np.save(destination / "yaw.npy", yaw)
        np.save(destination / "features.npy", features)
        np.save(destination / "bone_offsets.npy", bone_offsets)
        np.savez_compressed(destination / "scene_occupancy.npz", occupancy=occupancy)
        np.save(destination / "scene_bounds.npy", bounds)
        annotations_path = destination / "annotations.jsonl"
        with annotations_path.open("w", encoding="utf-8") as file:
            for annotation in annotations:
                file.write(json.dumps(annotation, sort_keys=True) + "\n")

        count_annotations += len(annotations)

    # Compute normalization values
    compute_normalization(
        output_root,
        window_size=args.window_size,
        history_size=args.history_size,
        max_goal_horizon=args.max_goal_horizon,
    )

    print(
        f"Processed {len(recordings)} sequences and "
        f"{count_annotations} valid annotations.",
        flush=True,
    )


if __name__ == "__main__":
    main()
