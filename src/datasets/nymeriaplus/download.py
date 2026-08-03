"""
Download and retain the NymeriaPlus data required by this project.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from nymeriaplus.downloader import DownloadManager
from tqdm import tqdm


def download_artifacts(
    url_data: dict,
    sequence_names,
    artifact_keys: set[str],
    output: Path,
    overwrite: bool,
    num_workers: int,
) -> None:
    """
    Download selected artifacts for the given sequences.
    """

    manifest = {
        "sequences": {
            name: {
                key: value
                for key, value in url_data["sequences"][name].items()
                if key in artifact_keys
            }
            for name in sequence_names
        },
        "sequence_config": url_data["sequence_config"],
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = Path(temporary_directory) / "urls.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        DownloadManager(manifest_path, output).download(
            ignore_existing=not overwrite,
            num_workers=num_workers,
        )


def main() -> None:
    """
    Download, filter, downsample, and prune NymeriaPlus sequences.
    """

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load the URL manifest and prepare the output directory
    url_data = json.loads(
        args.url_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequence_names = list(url_data["sequences"])

    # Download metadata before selecting the usable sequences
    download_artifacts(
        url_data,
        sequence_names,
        {"metadata_json"},
        output,
        args.overwrite,
        args.num_workers,
    )

    # Keep sequences containing every artifact required by preprocessing
    eligible = []
    for sequence_name in sequence_names:
        metadata = json.loads(
            (output / sequence_name / "metadata.json").read_text(encoding="utf-8")
        )
        if not (
            metadata["body_motion_smpl"]
            and metadata["objects_bounding_box"]
            and metadata["objects_shaper_mesh"]
            and metadata["atomic_action"]
        ):
            shutil.rmtree(output / sequence_name)
            shutil.rmtree(output / ".download_logs" / sequence_name)
            continue
        eligible.append(sequence_name)

    # Download the full artifacts only for eligible sequences
    DATA_ARTIFACTS = {
        "body_processed",
        "object_bounding_box",
        "object_mesh",
        "narration",
    }
    download_artifacts(
        url_data,
        eligible,
        DATA_ARTIFACTS,
        output,
        args.overwrite,
        args.num_workers,
    )

    # Retain only files consumed by preprocessing
    KEEP_PATHS = {
        "LICENSE",
        "metadata.json",
        "body/xdata_smpl_neutral.npz",
        "narration/atomic_action.csv",
        "objects/boxy/3dbb.csv",
        "objects/boxy/scene_objects.csv",
        "objects/boxy/instances.json",
        "objects/shaper",
    }
    for sequence_name in tqdm(eligible, desc="Preparing", unit="sequence"):
        sequence_directory = output / sequence_name
        smpl_path = sequence_directory / "body" / "xdata_smpl_neutral.npz"
        with np.load(smpl_path) as archive:
            smpl = {
                name: np.asarray(archive[name]) for name in archive.files
            }

        # Downsample SMPL motion to the requested frame rate
        timestamps = smpl["timestamps"]
        target_times = np.arange(
            timestamps[0],
            timestamps[-1] + 1,
            round(1_000_000 / args.fps),
        )
        keep = np.searchsorted(timestamps, target_times).clip(
            1, len(timestamps) - 1
        )
        previous = keep - 1
        keep[
            target_times - timestamps[previous]
            <= timestamps[keep] - target_times
        ] -= 1
        keep = np.unique(keep)
        np.savez_compressed(
            smpl_path,
            **{name: values[keep] for name, values in smpl.items()},
        )

        # Remove files not used by preprocessing
        for path in sequence_directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(sequence_directory).as_posix()
            if relative in KEEP_PATHS or any(
                relative.startswith(f"{kept}/") for kept in KEEP_PATHS
            ):
                continue
            path.unlink()

        # Remove empty directories left by pruning
        directories = sorted(
            (path for path in sequence_directory.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in directories:
            try:
                path.rmdir()
            except OSError:
                pass

    print(f"Finished {len(eligible)} sequences", flush=True)


if __name__ == "__main__":
    main()
