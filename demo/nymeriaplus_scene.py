"""
Real NymeriaPlus scene geometry for the demo, loaded from an actual Nymeria
test-split recording. Produces the same downstream shapes -- {meshes, boxes,
boxFaces} for /scene/geometry, (occupancy, bounds) for the model -- sourced
from genuine ShapeR meshes and Boxy boxes, the same source data and loading
convention tmp/render_gifs.py uses for the result GIFs, so the live demo
shows the model standing in and conditioned on a real room.
"""

from __future__ import annotations

import colorsys
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# 6 quad faces of a corner-numbered box, each wound CCW as seen from outside
# (right-hand rule) -- matches demo.py's own `_BOX_FACES` convention.
BOX_FACES = [
    (0, 2, 3, 1),  # z fixed low
    (4, 5, 7, 6),  # z fixed high
    (0, 1, 5, 4),  # y fixed low
    (2, 6, 7, 3),  # y fixed high
    (0, 4, 6, 2),  # x fixed low
    (1, 3, 7, 5),  # x fixed high
]

try:
    import fast_simplification
except ImportError:
    fast_simplification = None

# tmp/render_gifs.py's 3000-faces-per-object target assumes one fill() per
# OBJECT (flat/gradient shaded). The live demo now shades and fills per
# TRIANGLE (see paintMeshDrawable in demo/assets/index.html) for correct
# basic lighting instead of a per-object gradient hack, so canvas fill()
# count scales with total triangles, not object count -- a much lower
# per-mesh cap keeps a ~20-object room in the low thousands of fill() calls
# a frame instead of tens of thousands. ShapeR meshes come in around
# 30k-100k faces raw -- trimesh's own simplify_quadric_decimation needs an
# open3d backend we don't otherwise depend on, so this uses
# fast_simplification directly instead, same as tmp/render_gifs.py.
_TARGET_MESH_FACES = 400


def _category_color(category: str) -> list[float]:
    hue = (zlib.crc32(category.encode("utf-8")) % 360) / 360.0
    return [round(c, 3) for c in colorsys.hsv_to_rgb(hue, 0.55, 0.85)]


def _decimate_mesh(vertices: np.ndarray, faces: np.ndarray):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    clean_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    clean_faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(clean_faces) <= _TARGET_MESH_FACES or fast_simplification is None:
        return clean_vertices, clean_faces
    try:
        out_vertices, out_faces = fast_simplification.simplify(
            clean_vertices, clean_faces, target_count=_TARGET_MESH_FACES, agg=7
        )
    except Exception:
        return clean_vertices, clean_faces
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )


def _load_meshes(recording: Path):
    meshes = []
    loaded_uids = set()
    shaper_dir = recording / "objects" / "shaper"
    metadata_path = shaper_dir / "shaper_metadata.csv"
    if not metadata_path.is_file():
        return meshes, loaded_uids

    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    best = {}
    for row in rows:
        uid = int(row["object_uid"])
        score = (int(row["score"]), int(row["variant_id"]))
        if uid not in best or score > best[uid][0]:
            best[uid] = (score, row)

    for uid, (_, row) in best.items():
        mesh_path = shaper_dir / row["file_name"]
        if not mesh_path.is_file():
            continue
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices, faces = _decimate_mesh(
            np.asarray(loaded.vertices, dtype=np.float32),
            np.asarray(loaded.faces, dtype=np.int32),
        )
        category = row.get("category") or "object"
        meshes.append({
            "vertices": vertices,
            "faces": faces,
            "category": category,
            "color": _category_color(category),
        })
        loaded_uids.add(uid)
    return meshes, loaded_uids


def _load_boxes(recording: Path, loaded_uids: set):
    """
    Returns (boxes, floor_z). `floor_z` is the real recording's own floor
    height -- the top of Boxy's "Floor" box, which (unlike its XY footprint,
    see below) is trustworthy -- or `None` if this recording has no "Floor"
    annotation at all.
    """

    boxes = []
    floor_tops = []
    boxy_dir = recording / "objects" / "boxy"
    bbox_csv = boxy_dir / "3dbb.csv"
    objects_csv = boxy_dir / "scene_objects.csv"
    if not (bbox_csv.is_file() and objects_csv.is_file()):
        return boxes, None

    with bbox_csv.open(newline="", encoding="utf-8") as file:
        bbox_rows = {int(r["object_uid"]): r for r in csv.DictReader(file, skipinitialspace=True)}
    with objects_csv.open(newline="", encoding="utf-8") as file:
        object_rows = {int(r["object_uid"]): r for r in csv.DictReader(file, skipinitialspace=True)}
    categories = {}
    instances_path = boxy_dir / "instances.json"
    if instances_path.is_file():
        raw = json.loads(instances_path.read_text(encoding="utf-8"))
        categories = {int(uid): info.get("category") or "object" for uid, info in raw.items()}

    # Boxy is a fallback for objects ShapeR didn't reconstruct a mesh for,
    # same precedence _load_meshes' caller relies on.
    for uid in (set(bbox_rows) & set(object_rows)) - loaded_uids:
        box, obj = bbox_rows[uid], object_rows[uid]
        category = (categories.get(uid) or "object")
        lower = np.array([float(box[f"p_local_obj_{a}min[m]"]) for a in "xyz"], dtype=np.float32)
        upper = np.array([float(box[f"p_local_obj_{a}max[m]"]) for a in "xyz"], dtype=np.float32)
        corners_local = np.array(
            [
                [
                    upper[0] if bit & 1 else lower[0],
                    upper[1] if bit & 2 else lower[1],
                    upper[2] if bit & 4 else lower[2],
                ]
                for bit in range(8)
            ],
            dtype=np.float32,
        )
        rotation = Rotation.from_quat([float(obj[f"q_wo_{a}"]) for a in "xyzw"]).as_matrix()
        translation = np.array([float(obj[f"t_wo_{a}[m]"]) for a in "xyz"], dtype=np.float32)
        corners_world = (corners_local @ rotation.T + translation).astype(np.float32)

        if category.lower() == "floor":
            # Boxy's own "Floor" box regularly spans far more than the
            # actual furnished room (seen on this dataset: a single box
            # covering ~20x18m against a ~9x12m room the furniture actually
            # occupies) -- looks like it's picking up an entire building
            # slab rather than just this apartment. Using its huge XY
            # footprint would blow up the full-room occupancy bounds and
            # reintroduce the single-giant-object depth-sort bug already
            # fixed for the synthetic apartment's floor. Its HEIGHT is still
            # trustworthy though (this is the recording's own measured floor,
            # not a guess) -- keep just that; a tiled floor sized to the real
            # furniture bounds, at this real height, is added separately in
            # `_floor_tiles`.
            floor_tops.append(float(corners_world[:, 2].max()))
            continue
        if category.lower() == "wall":
            # Same problem as "Floor", confirmed on this dataset: Boxy's
            # "Wall" boxes are full room-spanning volumes (e.g. one measured
            # 4.75x9.65x3.15m -- nearly the whole room, floor to ceiling),
            # not thin wall slabs. Filled into the occupancy grid and
            # rendered as solid geometry, these read as huge blocks sitting
            # in what should be open floor space -- what showed up as "the
            # occupancy grid doesn't match the scene", since the real room
            # has nothing solid there. Unlike the floor, there's no cheap
            # substitute geometry worth synthesizing here, so walls are
            # just dropped: seeing slightly further into a neighboring room
            # is a much smaller problem than a room full of phantom blocks.
            continue
        boxes.append({
            "corners": corners_world,
            "category": category,
            "color": _category_color(category),
        })
    floor_z = max(floor_tops) if floor_tops else None
    return boxes, floor_z


# Keeps every floor object roughly furniture-scale so the browser's per-object depth
# sort (drawSceneGeometry in demo/assets/index.html) stays accurate; see the
# comment on the excluded raw "Floor" box above for why this is built rather
# than trusted from Boxy for its XY footprint.
_FLOOR_TILE = 1.5
_FLOOR_THICKNESS = 0.05


def _floor_tiles(meshes, boxes, floor_z: float | None) -> list[dict]:
    """
    `floor_z` is the recording's real floor height (top surface) from
    `_load_boxes`'s "Floor" box, if it had one -- falls back to just under
    the lowest furniture point only when a recording has no "Floor"
    annotation at all.
    """

    points = [mesh["vertices"] for mesh in meshes] + [box["corners"] for box in boxes]
    if not points:
        return []
    all_points = np.concatenate(points)
    xy_min = all_points[:, :2].min(axis=0)
    xy_max = all_points[:, :2].max(axis=0)
    if floor_z is None:
        floor_z = float(all_points[:, 2].min()) - _FLOOR_THICKNESS
    color = [0.72, 0.70, 0.67]

    tiles = []
    for tile_x in np.arange(xy_min[0], xy_max[0], _FLOOR_TILE):
        half_w = min(_FLOOR_TILE, xy_max[0] - tile_x) / 2
        for tile_y in np.arange(xy_min[1], xy_max[1], _FLOOR_TILE):
            half_d = min(_FLOOR_TILE, xy_max[1] - tile_y) / 2
            center = np.array(
                [tile_x + half_w, tile_y + half_d, floor_z - _FLOOR_THICKNESS / 2],
                dtype=np.float32,
            )
            half = np.array([half_w, half_d, _FLOOR_THICKNESS / 2], dtype=np.float32)
            lower, upper = center - half, center + half
            corners = np.array(
                [
                    [
                        upper[0] if bit & 1 else lower[0],
                        upper[1] if bit & 2 else lower[1],
                        upper[2] if bit & 4 else lower[2],
                    ]
                    for bit in range(8)
                ],
                dtype=np.float32,
            )
            tiles.append({"corners": corners, "category": "floor", "color": color})
    return tiles


def load_scene(recording: Path):
    """
    Load one real recording's objects once (meshes preferred, boxes as
    fallback per-object, plus a tiled floor at the recording's real floor
    height -- see `_floor_tiles`). Caller is expected to cache the result --
    this reads and decimates every mesh in the room, not cheap to repeat per
    request. Returns (meshes, boxes, floor_z): `floor_z` is the same real
    height the floor tiles were built at, for the caller to use as the
    click-to-floor fallback plane instead of assuming z=0.
    """

    meshes, loaded_uids = _load_meshes(recording)
    boxes, floor_z = _load_boxes(recording, loaded_uids)
    if not meshes and not boxes:
        raise FileNotFoundError(f"No scene geometry found under {recording}/objects")
    tiles = _floor_tiles(meshes, boxes, floor_z)
    boxes = boxes + tiles
    if floor_z is None and tiles:
        floor_z = float(tiles[0]["corners"][:, 2].max())
    return meshes, boxes, floor_z


def geometry_payload(meshes, boxes) -> dict:
    return {
        "meshes": [
            {
                "category": mesh["category"],
                "color": mesh["color"],
                "vertices": mesh["vertices"].round(4).tolist(),
                "faces": mesh["faces"].tolist(),
            }
            for mesh in meshes
        ],
        "boxes": [
            {
                "category": box["category"],
                "color": box["color"],
                "corners": box["corners"].round(4).tolist(),
            }
            for box in boxes
        ],
        "boxFaces": BOX_FACES,
    }


def build_occupancy(meshes, boxes, voxel_size: float = 0.05, margin: float = 0.5):
    """
    Voxelize every real object into one grid spanning the whole room (not
    cropped to any one trajectory), so the existing anchor-relative crop in
    src/inference.py works unchanged regardless of which scene is loaded.
    """

    point_clouds = []
    all_corners = []
    for mesh in meshes:
        all_corners.append(mesh["vertices"])
        solid = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
        try:
            points = solid.voxelized(pitch=voxel_size).fill().points.astype(np.float32)
        except Exception:
            points = mesh["vertices"]
        if len(points):
            point_clouds.append(points)
    for box in boxes:
        corners = box["corners"]
        all_corners.append(corners)
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        axes = [np.arange(lower[i], upper[i] + voxel_size * 0.5, voxel_size) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        point_clouds.append(grid.astype(np.float32))

    all_corners = np.concatenate(all_corners) if all_corners else np.zeros((1, 3), dtype=np.float32)
    minimum = all_corners.min(axis=0) - margin
    maximum = all_corners.max(axis=0) + margin
    shape = np.ceil((maximum - minimum) / voxel_size).astype(np.int64)
    occupancy = np.zeros(tuple(shape), dtype=np.bool_)
    if point_clouds:
        points = np.concatenate(point_clouds)
        indices = np.floor((points - minimum) / voxel_size).astype(np.int64)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        occupancy[tuple(indices[inside].T)] = True
    bounds = np.asarray([minimum, maximum], dtype=np.float32)
    return occupancy, bounds


def room_center_xy(boxes, meshes) -> tuple[float, float]:
    """A reasonable default --start: the real room's horizontal footprint center."""

    corners = [box["corners"] for box in boxes] + [mesh["vertices"] for mesh in meshes]
    if not corners:
        return (0.0, 0.0)
    all_points = np.concatenate(corners)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    return (float(center[0]), float(center[1]))
