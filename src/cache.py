"""
Cache frozen text features for processed annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.nymeriaplus.dataset import NymeriaDataset
from src.models.text_encoder import (
    DEFAULT_MAX_TOKENS,
    TEXT_ENCODER_NAMES,
    build_text_encoder,
)


def build_text_cache(
    dataset: str | Path,
    encoder_type: str,
    split: str = "all",
    device="cpu",
    batch_size: int = 256,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    window_size: int = 16,
) -> None:
    """
    Encode annotations for dataset splits and save ordered feature arrays.

    `window_size` must match the training config's: NymeriaDataset asserts
    every annotation is at least that long, and a mismatch asserts wrong.
    """

    dataset = Path(dataset).expanduser().resolve()
    encoder_type = str(encoder_type).lower()
    splits = ["train", "val", "test"] if split == "all" else [split]

    encoder = build_text_encoder(encoder_type, max_tokens).to(device)
    encoder.eval()

    for s in splits:
        output = dataset / "cache" / f"{encoder_type}_{s}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)

        texts = NymeriaDataset(
            dataset,
            split=s,
            window_size=window_size,
            scene_enabled=False,
            text_enabled=False,
            goal_enabled=False,
        ).texts

        features = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float16,
            shape=(len(texts), encoder.max_tokens, encoder.output_dim),
        )
        batches = range(0, len(texts), batch_size)
        for first in tqdm(batches, desc=f"Caching text ({s})", unit="batch"):
            last = min(first + batch_size, len(texts))
            encoded = encoder(texts[first:last])
            features[first:last] = encoded.detach().cpu().numpy().astype(np.float16)
        features.flush()
        print(f"Saved {features.shape} text features to {output}", flush=True)


def main() -> None:
    """
    Cache annotation text using the selected encoder.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--text-encoder", choices=TEXT_ENCODER_NAMES, required=True
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="must match max_text_tokens in the training configuration",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="must match window_size in the training configuration",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    build_text_cache(
        args.dataset,
        args.text_encoder,
        split=args.split,
        device=device,
        batch_size=args.batch_size,
        max_tokens=args.max_text_tokens,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
