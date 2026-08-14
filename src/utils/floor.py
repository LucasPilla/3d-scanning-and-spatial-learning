"""
Per-frame floor height from foot-contact detection.

Used as the anchor's z (instead of raw pelvis z -- see `dataset.py` /
`inference.py`) because pelvis height drifts with pose (standing vs.
sitting vs. lying) while floor height doesn't.

Two causal gates decide contact per foot, each frame:
1. Velocity: a planted foot slows down; a swinging one doesn't.
2. Pelvis-to-foot offset: `pelvis_z - foot_z` must fall in a plausible
   standing/sitting band, which is what rejects kneeling.

The estimate is the min z among whichever foot(s) pass both gates; gaps
(neither foot passes) are filled per `estimate_floor`'s `causal` parameter.
A trailing, never-centered median then smooths jitter, so preprocessing and
online inference see the same signal shape.

A third torso-upright gate and a debounce/persistence requirement were
tried and dropped -- no improvement over velocity+offset alone.

Known limitations: isolated frames can be off by a few centimeters during
ambiguous transitions (e.g. rising off a bed edge) -- within the margins
the scene crop/anchor need, but not floor-exact. Bending forward with
straight legs (e.g. tying shoes) is an untested edge case.
"""

from __future__ import annotations

import numpy as np

# SMPL joint indices (see src/utils/geometry.py's SMPL_PARENTS ordering).
PELVIS = 0
LEFT_FOOT, RIGHT_FOOT = 10, 11


def estimate_floor(
    joints: np.ndarray,
    *,
    causal: bool = True,
    velocity_threshold: float = 0.015,
    median_window: int = 5,
    min_pelvis_offset: float = 0.35,
    max_pelvis_offset: float = 1.3,
) -> np.ndarray:
    """
    joints: [T, JOINT_COUNT, 3] world-space positions. Returns floor z,
    shape [T], one value per frame.

    `causal` controls how gaps between confirmed-contact frames are filled
    (the two contact gates themselves are always causal):

    - `causal=True` (default; used online by `src/inference.py`):
      forward-holds the last confirmed value, so frame t only ever depends
      on joints[:t+1] -- safe to call incrementally during rollout.
    - `causal=False` (offline preprocessing only): linearly interpolates
      between the confirmed frames bracketing each gap, using only those
      two values -- never raw joints inside the gap, so a momentary dip
      (e.g. a swinging leg while sitting) can't fool it. Needs a future
      confirmed contact, so it can't run online, but it avoids the
      forward-hold-then-snap artifact a long non-contact stretch (e.g.
      continuous stair descent) would otherwise inject into training
      targets.
    """

    pelvis_z = joints[:, PELVIS, 2]
    left = joints[:, LEFT_FOOT]
    right = joints[:, RIGHT_FOOT]

    left_speed = np.zeros(len(joints), dtype=np.float32)
    right_speed = np.zeros(len(joints), dtype=np.float32)
    left_speed[1:] = np.linalg.norm(np.diff(left, axis=0), axis=1)
    right_speed[1:] = np.linalg.norm(np.diff(right, axis=0), axis=1)
    left_speed[0], right_speed[0] = left_speed[1], right_speed[1]

    left_offset_ok = (pelvis_z - left[:, 2] >= min_pelvis_offset) & (pelvis_z - left[:, 2] <= max_pelvis_offset)
    right_offset_ok = (pelvis_z - right[:, 2] >= min_pelvis_offset) & (pelvis_z - right[:, 2] <= max_pelvis_offset)
    left_contact = (left_speed < velocity_threshold) & left_offset_ok
    right_contact = (right_speed < velocity_threshold) & right_offset_ok

    floor = np.full(len(joints), np.nan, dtype=np.float32)
    for t in range(len(joints)):
        candidates = []
        if left_contact[t]:
            candidates.append(left[t, 2])
        if right_contact[t]:
            candidates.append(right[t, 2])
        if candidates:
            floor[t] = min(candidates)

    if np.isnan(floor[0]):
        warmup = min(10, len(joints))
        floor[0] = min(left[:warmup, 2].min(), right[:warmup, 2].min())

    if causal:
        for t in range(1, len(floor)):
            if np.isnan(floor[t]):
                floor[t] = floor[t - 1]
    else:
        # Frame 0 is always non-NaN (fallback above), anchoring the left
        # edge; np.interp clamps a trailing gap to the last confirmed value,
        # matching what the causal forward-hold would do there.
        confirmed = ~np.isnan(floor)
        confirmed[0] = True
        indices = np.flatnonzero(confirmed)
        floor = np.interp(np.arange(len(floor)), indices, floor[indices]).astype(np.float32)

    if median_window > 1:
        padded = np.concatenate([np.full(median_window - 1, floor[0], dtype=np.float32), floor])
        floor = np.array([
            np.median(padded[t:t + median_window]) for t in range(len(floor))
        ], dtype=np.float32)

    return floor
