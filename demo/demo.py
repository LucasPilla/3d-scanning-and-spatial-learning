"""
Run an explicit browser demo for text- and target-conditioned motion.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import torch

from demo import nymeriaplus_scene, trumans_scene
from src.config import build_inference, load_config
from src.inference import RolloutState, inference
from src.utils.geometry import FEATURE_DIM, SMPL_PARENTS, anchor_local


ASSET_PATH = Path(__file__).parent / "assets" / "index.html"

MAX_REQUEST_BYTES = 1_000_000
MAX_PROMPT_LENGTH = 2_000

# Curated real test-split recording (same one tmp/generate_gif_examples.py's
# three example clips use), tried under a couple of plausible raw-dataset
# locations. Override with --scene to point at any other recording directory.
DEFAULT_RECORDING = "20230724_s0_tyler_ayers_act0_bhdzn3"
DEFAULT_SCENE_ROOTS = [
    Path("/home/lucas/project/nymeria_dataset/download"),
    Path(__file__).resolve().parents[2] / "nymeria_dataset" / "download",
]
DEFAULT_TRUMANS_ROOT = Path("/home/lucas/project/trumans_utils")

DEFAULT_SPAWN_HEIGHT = 0.92
DEFAULT_YAW = float(np.pi / 2)

# SMPL 24-joint parent-index skeleton hierarchy, used by the browser renderer.
PARENTS = list(SMPL_PARENTS)
# Assumed floor height for a cold start with no scene loaded (and thus no
# real geometry to estimate a floor from yet) — the anchor's z is floor
# height (see RolloutState in src/inference.py), so this seeds it flat.
FLOOR_Z = 0.0
# Sane-input bound for target/start coordinates, not a UX constraint: real
# scene coordinates can sit far from the origin (a room's bounds might span
# x=0.7..10.2), unlike the no-scene flat-floor fallback below.
MAX_COORDINATE = 1_000.0
# Fallback clamp range for the no-scene flat floor, centered on --start.
WAYPOINT_LIMIT = 5.0


def compute_goal(state: RolloutState, target):
    """
    Convert a world-space waypoint into the anchor-local goal displacement.
    """

    if target is None:
        return None
    target_pos = np.asarray(target, dtype=np.float32).reshape(-1)
    if target_pos.size == 2:
        target_pos = np.append(target_pos, state.position[2])
    return anchor_local(target_pos, state.position, float(state.yaw)).astype(
        np.float32, copy=False
    )


def scene_preview_points(
    occupancy: np.ndarray,
    bounds: np.ndarray,
    max_points: int = 8_000,
) -> np.ndarray:
    """
    Downsample one scene's full occupancy grid to a point cloud for preview.

    A raw grid (order-10^6 voxels) is far too much to send per request or
    render on a 2D canvas, and most of it is redundant (a solid floor/wall
    doesn't need every interior voxel to read visually). Coarsen first, then
    cap with a fixed random subsample so the response size is bounded
    regardless of how much of the room happens to be occupied.
    """

    shape = np.asarray(occupancy.shape)
    stride = max(1, int(np.ceil((occupancy.size / 400_000) ** (1 / 3))))
    coarse = occupancy[::stride, ::stride, ::stride]
    indices = np.argwhere(coarse)
    if not len(indices):
        return np.empty((0, 3), dtype=np.float32)
    if len(indices) > max_points:
        choice = np.random.default_rng(0).choice(
            len(indices), max_points, replace=False
        )
        indices = indices[choice]
    cell_size = (bounds[1] - bounds[0]) / shape
    return (bounds[0] + (indices * stride + 0.5) * cell_size).astype(np.float32)


def surface_height(occupancy: np.ndarray, bounds: np.ndarray, x: float, y: float) -> float | None:
    """
    Height of the topmost occupied voxel in the (x, y) column, or `None` if
    that column is outside the grid or has nothing in it.

    This is what lets a click "land" on whatever is actually there — floor,
    stair tread, or the top of a couch cushion — instead of always assuming a
    flat floor.
    """

    shape = np.asarray(occupancy.shape)
    cell_size = (bounds[1] - bounds[0]) / shape
    column = np.floor(
        (np.array([x, y], dtype=np.float32) - bounds[0, :2]) / cell_size[:2]
    ).astype(np.int64)
    if not (0 <= column[0] < shape[0] and 0 <= column[1] < shape[1]):
        return None
    occupied = np.nonzero(occupancy[column[0], column[1], :])[0]
    if not len(occupied):
        return None
    return float(bounds[0, 2] + (occupied.max() + 1) * cell_size[2])


def _number(payload, key: str, default: float, minimum: float, maximum: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
    return value


def _parse_state(raw, history_size: int) -> RolloutState | None:
    """
    Validate the opaque rollout state the client echoes back to us.

    The client renders world-space joint positions, and `state.features` is
    itself the trailing history of world-space joints, so this is a shape
    check rather than a semantic decode — the client never needs to interpret
    the values.
    """

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("state must be null or an object.")

    raw_features = raw.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("state.features must be a list of frames.")
    if len(raw_features) > history_size:
        raise ValueError(
            f"state.features must contain at most {history_size} frames."
        )
    try:
        features = np.asarray(raw_features, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("state.features must contain numbers.") from error
    if not len(raw_features):
        features = np.empty((0, FEATURE_DIM), dtype=np.float32)
    if features.shape != (len(raw_features), FEATURE_DIM):
        raise ValueError(f"state.features must have shape [frames, {FEATURE_DIM}].")

    raw_position = raw.get("position")
    if not isinstance(raw_position, list) or len(raw_position) != 3:
        raise ValueError("state.position must be a three-number list.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_position
    ):
        raise ValueError("state.position must contain numbers.")
    position = np.asarray(raw_position, dtype=np.float32)

    yaw = raw.get("yaw")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
        raise ValueError("state.yaw must be a number.")
    if not (np.isfinite(features).all() and np.isfinite(position).all()):
        raise ValueError("state values must be finite.")
    if not math.isfinite(float(yaw)):
        raise ValueError("state.yaw must be finite.")
    return RolloutState(features, position, float(yaw))


def _parse_window_request(payload, history_size: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"prompt must contain at most {MAX_PROMPT_LENGTH} characters."
        )

    state = _parse_state(payload.get("state"), history_size)

    raw_target = payload.get("target")
    target = None
    if raw_target is not None:
        # Two coordinates mean "keep the current pelvis height"; a third lets a
        # caller aim at a specific height.
        if not isinstance(raw_target, list) or len(raw_target) not in (2, 3):
            raise ValueError("target must be null or a list of 2 or 3 numbers.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_target
        ):
            raise ValueError("target coordinates must be numbers.")
        target = np.asarray(raw_target, dtype=np.float32)
        if not np.isfinite(target).all():
            raise ValueError("target coordinates must be finite.")
        if (
            abs(float(target[0])) > MAX_COORDINATE
            or abs(float(target[1])) > MAX_COORDINATE
        ):
            raise ValueError(
                f"target must stay within +/-{MAX_COORDINATE:g} metres."
            )

    raw_start = payload.get("start")
    start = None
    if raw_start is not None:
        if not isinstance(raw_start, list) or len(raw_start) != 3:
            raise ValueError("start must be null or a list of 3 numbers.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_start
        ):
            raise ValueError("start coordinates must be numbers.")
        start = np.asarray(raw_start, dtype=np.float32)
        if not np.isfinite(start).all():
            raise ValueError("start coordinates must be finite.")
        if (
            abs(float(start[0])) > MAX_COORDINATE
            or abs(float(start[1])) > MAX_COORDINATE
        ):
            raise ValueError(f"start must stay within +/-{MAX_COORDINATE:g} metres.")

    raw_start_yaw = payload.get("startYaw")
    start_yaw = None
    if raw_start_yaw is not None:
        if isinstance(raw_start_yaw, bool) or not isinstance(raw_start_yaw, (int, float)):
            raise ValueError("startYaw must be a number.")
        start_yaw = float(raw_start_yaw)
        if not math.isfinite(start_yaw):
            raise ValueError("startYaw must be finite.")

    window_index = payload.get("windowIndex")
    if (
        isinstance(window_index, bool)
        or not isinstance(window_index, int)
        or window_index < 0
        or window_index > 2_147_483_647
    ):
        raise ValueError("windowIndex must be a non-negative integer.")

    return {
        "state": state,
        "prompt": prompt,
        "target": target,
        "start": start,
        "start_yaw": start_yaw,
        "text_guidance": _number(payload, "textGuidance", 1.0, 0.0, 10.0),
        "scene_guidance": _number(payload, "sceneGuidance", 1.0, 0.0, 10.0),
        "goal_guidance": _number(payload, "goalGuidance", 1.0, 0.0, 10.0),
        "window_index": window_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="World-space spawn point. Defaults to the loaded scene's footprint center.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help=(
            "Path to a real Nymeria recording directory (containing "
            "objects/shaper, objects/boxy) to condition on and render, or, "
            "with --trumans, a trumans_utils checkout directory (containing "
            f"static/*.glb). Defaults to the curated recording "
            f"{DEFAULT_RECORDING!r} if found under a known raw-dataset "
            "location."
        ),
    )
    parser.add_argument(
        "--trumans",
        action="store_true",
        help=(
            "Condition on and render TRUMANS' own default demo scene "
            "(background room + all interactive props, positioned exactly "
            "like github.com/jnnan/trumans_utils' Flask demo) instead of a "
            f"Nymeria recording. Looked up under {DEFAULT_TRUMANS_ROOT} by "
            "default, or --scene to point at a different trumans_utils "
            "checkout."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.path is None:
        parser.error("The configuration must set path to the processed dataset.")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.start is not None and not all(math.isfinite(value) for value in args.start):
        parser.error("--start coordinates must be finite.")

    diffusion, statistics, device = build_inference(
        config, args.checkpoint, args.device
    )
    model = diffusion.model

    # Real scene geometry only: genuine ShapeR meshes / Boxy boxes from a
    # held-out NymeriaPlus test-split recording (demo/nymeriaplus_scene.py),
    # or --trumans' fused TRUMANS demo room (demo/trumans_scene.py) -- the
    # same source data tmp/render_gifs.py uses for the result GIFs. Built
    # once eagerly since there's only ever one scene per demo run.
    scene_name = None
    scene = geometry = None
    start_xy = None
    # The floor's real world-space height for this scene -- the client uses
    # this as its click-to-floor fallback plane (see nymeriaplus_scene
    # .load_scene's docstring).
    floor_z = FLOOR_Z
    if model.scene_enabled and args.trumans:
        trumans_root = args.scene if args.scene is not None else DEFAULT_TRUMANS_ROOT
        trumans_meshes, trumans_boxes, trumans_floor_z = trumans_scene.load_scene(
            trumans_root
        )
        scene = nymeriaplus_scene.build_occupancy(trumans_meshes, trumans_boxes)
        geometry = nymeriaplus_scene.geometry_payload(trumans_meshes, trumans_boxes)
        scene_name = "trumans_demo"
        start_xy = nymeriaplus_scene.room_center_xy(trumans_boxes, trumans_meshes)
        floor_z = trumans_floor_z
    elif model.scene_enabled:
        recording_dir = args.scene
        if recording_dir is None:
            for root in DEFAULT_SCENE_ROOTS:
                candidate = root / DEFAULT_RECORDING
                if candidate.is_dir():
                    recording_dir = candidate
                    break
        if recording_dir is None:
            parser.error(
                "No real scene recording found under "
                f"{[str(root) for root in DEFAULT_SCENE_ROOTS]}; "
                "pass --scene <recording-dir> to use one."
            )
        real_meshes, real_boxes, real_floor_z = nymeriaplus_scene.load_scene(recording_dir)
        scene = nymeriaplus_scene.build_occupancy(real_meshes, real_boxes)
        geometry = nymeriaplus_scene.geometry_payload(real_meshes, real_boxes)
        scene_name = recording_dir.name
        start_xy = nymeriaplus_scene.room_center_xy(real_boxes, real_meshes)
        floor_z = real_floor_z

    if args.start is not None:
        start_xy = tuple(args.start)
    elif start_xy is None:
        start_xy = (0.0, 0.0)

    fps = float(args.fps if args.fps is not None else config.fps)
    if not math.isfinite(fps) or fps <= 0:
        parser.error("--fps must be a positive finite number.")

    settings = {
        "fps": fps,
        "parents": PARENTS,
        "start": [start_xy[0], start_xy[1]],
        "startYaw": DEFAULT_YAW,
        "textEnabled": bool(model.text_enabled),
        "sceneEnabled": bool(model.scene_enabled),
        "goalEnabled": bool(model.goal_enabled),
        "waypointLimit": WAYPOINT_LIMIT,
        "defaultHeight": DEFAULT_SPAWN_HEIGHT,
        "windowFrames": diffusion.window_size,
        "sceneName": scene_name if model.scene_enabled else None,
        "floorZ": floor_z,
    }
    encoded_settings = json.dumps(settings, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    page = ASSET_PATH.read_text(encoding="utf-8").replace(
        "__SETTINGS__", encoded_settings
    ).encode("utf-8")
    generation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content: bytes, content_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                # Resetting the demo can abort a browser request while its
                # already-running GPU work finishes under the generation lock.
                pass

        def _send_json(self, status: int, payload) -> None:
            content = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send(status, content, "application/json; charset=utf-8")

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/":
                self._send(200, page, "text/html; charset=utf-8")
                return
            if path == "/scene":
                if scene is None:
                    self._send_json(404, {"error": "Scene conditioning disabled."})
                    return
                occupancy, bounds = scene
                points = scene_preview_points(occupancy, bounds)
                self._send_json(
                    200,
                    {
                        "bounds": bounds.tolist(),
                        "points": points.round(3).tolist(),
                    },
                )
                return
            if path == "/scene/surface":
                if scene is None:
                    self._send_json(404, {"error": "Scene conditioning disabled."})
                    return
                params = parse_qs(query)
                try:
                    x = float(params["x"][0])
                    y = float(params["y"][0])
                except (KeyError, IndexError, ValueError):
                    self._send_json(400, {"error": "x and y must be numbers."})
                    return
                if not (math.isfinite(x) and math.isfinite(y)):
                    self._send_json(400, {"error": "x and y must be finite."})
                    return
                occupancy, bounds = scene
                height = surface_height(occupancy, bounds, x, y)
                self._send_json(200, {"height": height})
                return
            if path == "/scene/geometry":
                if geometry is None:
                    self._send_json(404, {"error": "Scene conditioning disabled."})
                    return
                self._send_json(200, geometry)
                return
            self._send_json(404, {"error": "Not found."})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/window":
                self._send_json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size.")
                request = _parse_window_request(
                    json.loads(self.rfile.read(length)), config.history_size
                )
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
                self._send_json(400, {"error": str(error)})
                return

            try:
                state = request["state"]
                if state is None:
                    # Only a cold start needs an explicit spawn point: the
                    # model is root-centric, so after the first window the
                    # anchor just comes from the motion itself. The client
                    # places `start` interactively (defaulting to the scene's
                    # center when one is loaded); fall back to --start only
                    # when it sends none at all.
                    start_yaw = (
                        request["start_yaw"]
                        if request["start_yaw"] is not None
                        else DEFAULT_YAW
                    )
                    if request["start"] is not None:
                        # The client sends pelvis height (surface + a standing
                        # offset, for a sensible-looking marker), but the
                        # anchor's z is floor height (see RolloutState's
                        # docstring in src/inference.py) — undo that offset
                        # rather than anchoring ~1m above the real floor.
                        start = np.asarray(request["start"], dtype=np.float32)
                        state = RolloutState(
                            np.empty((0, FEATURE_DIM), dtype=np.float32),
                            np.array(
                                [start[0], start[1], start[2] - DEFAULT_SPAWN_HEIGHT],
                                dtype=np.float32,
                            ),
                            start_yaw,
                        )
                    else:
                        state = RolloutState.cold_start(
                            start_xy,
                            floor_z=floor_z,
                            yaw=start_yaw,
                        )
                goal = compute_goal(state, request["target"])
                scene_occupancy, scene_bounds = (
                    (scene[0], scene[1]) if scene is not None else (None, None)
                )
                generator = torch.Generator(device=device)
                window_seed = (args.seed + request["window_index"]) % (2**63 - 1)
                generator.manual_seed(window_seed)
                with generation_lock:
                    started = time.perf_counter()
                    _, frames, next_state, scene_points = inference(
                        config,
                        diffusion,
                        statistics,
                        state,
                        text=request["prompt"],
                        goal=goal,
                        text_guidance_scale=request["text_guidance"],
                        scene_guidance_scale=request["scene_guidance"],
                        goal_guidance_scale=request["goal_guidance"],
                        scene=scene_occupancy,
                        scene_bounds=scene_bounds,
                        generator=generator,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                self._send_json(
                    200,
                    {
                        "windowIndex": request["window_index"],
                        "frames": frames.round(5).tolist(),
                        "state": {
                            "features": next_state.features.round(6).tolist(),
                            "position": next_state.position.round(6).tolist(),
                            "yaw": round(next_state.yaw, 6),
                        },
                        "scenePoints": (
                            scene_points.round(4).tolist()
                            if scene_points is not None else []
                        ),
                        "latencyMs": round(latency_ms, 2),
                    },
                )
            except Exception as error:
                traceback.print_exc()
                self._send_json(500, {"error": f"Generation failed: {error}"})

        def log_message(self, *_):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

    address = f"http://127.0.0.1:{args.port}"
    print(f"Loaded model on {device}. Open {address}", flush=True)
    try:
        Server(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
