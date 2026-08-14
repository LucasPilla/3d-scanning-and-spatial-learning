"""
TRUMANS demo scene geometry, loaded directly from a local checkout of
TRUMANS' own Flask demo (github.com/jnnan/trumans_utils) -- its
`static/*.glb` assets, not the raw-scan `trumans_demo` data bundle.

TRUMANS' own demo (static/index.js) loads each of these GLBs independently
and adds it to the scene with no extra offset, so each mesh's baked-in glTF
node transform *is* the exact default world position their web demo shows
(read there as `model.children[0].position`, used to seed `initLoc` before
the user can drag objects around). trimesh reproduces that positioning for
free: `trimesh.load(path, force="mesh")` bakes the node transform into the
returned vertices. So loading `static/background.glb` plus every
`static/<name>.glb` in `_OBJECT_NAMES` and keeping their vertices as-is
reproduces TRUMANS' own default demo scene layout exactly, no guessed
placement needed (unlike the `trumans_demo` bundle's `objects_occ/*.obj`,
which carries no placement at all -- see git history of this file).

These web-demo GLBs are also already lightweight (low thousands of faces
each, background included) -- no decimation needed, unlike the raw-scan
`trumans_demo/objects_occ/background.obj` (~1.6M faces) used previously.

TRUMANS is y-up (their own occupancy grids span x:[-3,3], y:[0,2] height,
z:[-4,4] in that convention, and their demo's drag-plane sits at y=0 -- see
trumans_utils' static/index.js) while this project is z-up like NymeriaPlus,
so vertices are axis-swapped before being handed to demo.nymeriaplus_scene's
(otherwise scene-agnostic) occupancy/geometry helpers -- same downstream
contract nymeriaplus_scene.py uses for real Nymeria rooms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

_STATIC_DIR = "static"
_BACKGROUND_NAME = "background"
# Matches static/index.js's `allModelsUrl` (minus "background", which isn't
# draggable/interactive there either).
_OBJECT_NAMES = [
    "basin",
    "bed",
    "flower",
    "kitchen_chair_1",
    "kitchen_chair_2",
    "office_chair",
    "sofa",
    "table",
    "wc",
]

_CATEGORY_COLORS = {
    _BACKGROUND_NAME: [0.62, 0.62, 0.66],
    "bed": [0.75, 0.45, 0.45],
    "wc": [0.75, 0.75, 0.85],
    "basin": [0.7, 0.75, 0.85],
    "sofa": [0.5, 0.4, 0.35],
    "table": [0.55, 0.4, 0.25],
    "kitchen_chair_1": [0.45, 0.35, 0.25],
    "kitchen_chair_2": [0.45, 0.35, 0.25],
    "office_chair": [0.3, 0.3, 0.35],
    "flower": [0.35, 0.55, 0.3],
}


def _to_z_up(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRUMANS (x right, y up, z fwd) -> this project's (x right, y fwd, z up)."""

    converted = np.stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]], axis=1)
    # Swapping two axes mirrors the mesh (flips handedness), so triangles
    # must be rewound to keep outward-facing normals for the browser's
    # per-triangle shading (paintMeshDrawable in demo/assets/index.html).
    flipped_faces = faces[:, ::-1].copy()
    return converted.astype(np.float32), flipped_faces


def _load_mesh(path: Path, category: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No TRUMANS demo asset found at {path}")
    loaded = trimesh.load(path, force="mesh", process=False)
    vertices, faces = _to_z_up(
        np.asarray(loaded.vertices, dtype=np.float32),
        np.asarray(loaded.faces, dtype=np.int32),
    )
    return {
        "vertices": vertices,
        "faces": faces,
        "category": category,
        "color": _CATEGORY_COLORS.get(category, [0.6, 0.6, 0.6]),
    }


def load_scene(root: Path):
    """
    Returns (meshes, boxes, floor_z) in the same shape
    demo.nymeriaplus_scene.load_scene returns, for its
    build_occupancy/geometry_payload/room_center_xy to consume unchanged.
    `boxes` is always empty -- everything here comes in as mesh geometry,
    already positioned exactly like TRUMANS' own default demo scene.

    `root` is a checkout of github.com/jnnan/trumans_utils (i.e. it must
    contain a `static/` directory with the demo's *.glb assets), not the
    `trumans_demo` data bundle used previously.
    """

    static_dir = Path(root) / _STATIC_DIR
    meshes = [_load_mesh(static_dir / f"{_BACKGROUND_NAME}.glb", _BACKGROUND_NAME)]
    for name in _OBJECT_NAMES:
        meshes.append(_load_mesh(static_dir / f"{name}.glb", name))

    # TRUMANS' own demo drags objects along an invisible plane at y=0 (now z
    # after the axis swap) -- see static/index.js's drag `plane` -- so that's
    # this scene's floor, not derived from the mesh bounds.
    floor_z = 0.0
    return meshes, [], floor_z
