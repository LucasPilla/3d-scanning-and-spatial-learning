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

import numpy as np
import torch

from src.config import build_inference, load_config
from src.inference import RolloutState, inference
from src.utils.kinematics import FEATURE_DIM, SMPL_PARENTS


ASSET_PATH = Path(__file__).parent / "assets" / "index.html"

MAX_REQUEST_BYTES = 1_000_000
MAX_PROMPT_LENGTH = 2_000
HISTORY_FRAMES = 32
WINDOW_FRAMES = 16

DEFAULT_SPAWN_HEIGHT = 0.92

# SMPL 24-joint parent-index skeleton hierarchy, used by the browser renderer.
PARENTS = list(SMPL_PARENTS)
# World height of the demo's flat floor. The model is root-centric and has no
# notion of a ground plane, so this only picks the height a cold-started
# character spawns at.
FLOOR_Z = 0.0
POSITION_LIMIT = 5.0


def compute_goal(state: RolloutState, target):
    """
    Convert a world-space waypoint into the anchor-local goal displacement.
    """

    if target is None:
        return None
    target_pos = np.asarray(target, dtype=np.float32).reshape(-1)
    if target_pos.size == 2:
        target_pos = np.append(target_pos, state.position[2])
    delta = target_pos - state.position

    yaw = float(state.yaw)
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    local_x = delta[0] * cos_y - delta[1] * sin_y
    local_y = delta[0] * sin_y + delta[1] * cos_y
    local_z = delta[2]
    return np.array([local_x, local_y, local_z], dtype=np.float32)


def _number(payload, key: str, default: float, minimum: float, maximum: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
    return value


def _parse_state(raw) -> RolloutState | None:
    """
    Validate the opaque rollout state the client echoes back to us.

    The client only renders joint positions, but the model consumes canonical
    features, so continuing a rollout means round-tripping the features and
    the anchor pose rather than the rendered skeleton.
    """

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("state must be null or an object.")

    raw_features = raw.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("state.features must be a list of frames.")
    if len(raw_features) > HISTORY_FRAMES:
        raise ValueError(
            f"state.features must contain at most {HISTORY_FRAMES} frames."
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


def _parse_window_request(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"prompt must contain at most {MAX_PROMPT_LENGTH} characters."
        )

    state = _parse_state(payload.get("state"))

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
            abs(float(target[0])) > POSITION_LIMIT
            or abs(float(target[1])) > POSITION_LIMIT
        ):
            raise ValueError(
                f"target must stay within +/-{POSITION_LIMIT:g} metres."
            )

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
        "text_guidance": _number(payload, "textGuidance", 1.0, 0.0, 10.0),
        "goal_guidance": _number(payload, "goalGuidance", 1.0, 0.0, 10.0),
        "scene_guidance": _number(payload, "sceneGuidance", 1.0, 0.0, 10.0),
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
        default=(0.0, 0.0),
        metavar=("X", "Y"),
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
    if not all(math.isfinite(value) for value in args.start):
        parser.error("--start coordinates must be finite.")

    diffusion, statistics, device = build_inference(
        config, args.checkpoint, args.device
    )
    model = diffusion.model
    bone_offsets = np.load(config.path / "skeleton.npy", allow_pickle=False)
    # Scene conditioning is disabled for now (no scene geometry to condition
    # on yet); Scene CFG stays in the UI but has no effect until this is
    # wired back up.
    if diffusion.window_size != WINDOW_FRAMES:
        parser.error(f"The demo requires window_size={WINDOW_FRAMES}.")
    if config.history_size != HISTORY_FRAMES:
        parser.error(f"The demo requires history_size={HISTORY_FRAMES}.")

    fps = float(args.fps if args.fps is not None else config.fps)
    if not math.isfinite(fps) or fps <= 0:
        parser.error("--fps must be a positive finite number.")

    settings = {
        "fps": fps,
        "parents": PARENTS,
        "start": [args.start[0], args.start[1]],
        "textEnabled": bool(model.text_enabled),
        "sceneEnabled": bool(model.scene_enabled),
        "waypointLimit": POSITION_LIMIT,
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
            if self.path.split("?", 1)[0] != "/":
                self._send_json(404, {"error": "Not found."})
                return
            self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/window":
                self._send_json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size.")
                request = _parse_window_request(json.loads(self.rfile.read(length)))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
                self._send_json(400, {"error": str(error)})
                return

            try:
                state = request["state"]
                if state is None:
                    # Only a cold start needs the floor: the model is
                    # root-centric, so after the first window the anchor's
                    # height just comes from the motion itself.
                    state = RolloutState.cold_start(
                        args.start,
                        floor_z=FLOOR_Z,
                        spawn_height=DEFAULT_SPAWN_HEIGHT,
                    )
                goal = compute_goal(state, request["target"])
                generator = torch.Generator(device=device)
                window_seed = (args.seed + request["window_index"]) % (2**63 - 1)
                generator.manual_seed(window_seed)
                with generation_lock:
                    started = time.perf_counter()
                    _, frames, next_state = inference(
                        config,
                        diffusion,
                        statistics,
                        bone_offsets,
                        state,
                        text=request["prompt"],
                        goal=goal,
                        text_guidance_scale=request["text_guidance"],
                        goal_guidance_scale=request["goal_guidance"],
                        scene_guidance_scale=request["scene_guidance"],
                        scene=None,
                        scene_bounds=None,
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
