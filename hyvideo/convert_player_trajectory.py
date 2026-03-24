"""
Convert CS:GO player replay data to HY-WorldPlay pose JSON format.

CS:GO replay JSON contains per-frame camera_position [x,y,z] and yaw/pitch
in Source engine coordinates. This script converts them to c2w 4x4 matrices
in HY-World's coordinate convention and outputs a pose JSON file compatible
with HY-WorldPlay inference.

Coordinate systems:
  Source engine: X=right, Y=forward, Z=up;  yaw around Z, pitch around local X
  HY-World:     X=right, Y=up,      Z=forward; yaw around Y, pitch around local X

Usage:
  python hyvideo/convert_player_trajectory.py \
      --input path/to/Alpha_instance_000.json \
      --output assets/pose/csgo_trajectory.json \
      --num_frames 125 --start_frame 0
"""

import argparse
import json
import math

import numpy as np


# ── rotation helpers (same convention as generate_custom_trajectory.py) ──

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


# ── coordinate conversion ────────────────────────────────────────────────

def source_to_hyworld_position(x_s, y_s, z_s):
    """Source (X=right, Y=forward, Z=up) -> HY-World (X=right, Y=up, Z=forward)."""
    return np.array([x_s, z_s, y_s])


def build_c2w(position_hw, yaw_rad, pitch_rad):
    """Build a 4x4 c2w matrix in HY-World convention from position and angles.

    Args:
        position_hw: [x, y, z] in HY-World coordinates
        yaw_rad: yaw in radians (HY-World: rotation around Y axis)
        pitch_rad: pitch in radians (HY-World: rotation around X axis)
    """
    R = rot_y(yaw_rad) @ rot_x(pitch_rad)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position_hw
    return T


# ── intrinsic matrix from FOV ───────────────────────────────────────────

def fov_to_intrinsic(fov_h_deg, width, height):
    """Compute 3x3 intrinsic matrix K from horizontal FOV and resolution.

    HY-World's pose_to_input normalises K so that:
        fx_norm = fx / (2*cx),  fy_norm = fy / (2*cy),  cx=0.5, cy=0.5
    We store the un-normalised K here; normalisation happens at inference.
    """
    fov_h_rad = math.radians(fov_h_deg)
    fx = width / (2.0 * math.tan(fov_h_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return [[fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]]


# ── main conversion ─────────────────────────────────────────────────────

def convert(args):
    # ── Step 1: load CS:GO data ──
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support two input formats:
    #   1) Sampled JSON from extract_round_start.py: {"metadata": {...}, "frames": [...]}
    #   2) Raw CS:GO player JSON: [{frame_count, tick, camera_position, ...}, ...]
    if isinstance(raw, dict) and "frames" in raw:
        # Sampled format — frames are already video-frame-aligned
        frames = raw["frames"]
        meta = raw.get("metadata", {})
        print(f"Loaded sampled JSON: {len(frames)} frames "
              f"(episode={meta.get('episode_id', '?')}, "
              f"freeze_end_tick={meta.get('round_freeze_end_tick', '?')})")
    elif isinstance(raw, list):
        frames = raw
    else:
        raise ValueError("Unrecognized input format")

    # Filter out null entries
    frames = [fr for fr in frames if fr is not None and fr.get("camera_position") is not None]

    total = len(frames)
    start = args.start_frame
    end = start + args.num_frames
    if end > total:
        print(f"Warning: requested frames [{start}, {end}) but only {total} frames available. "
              f"Clamping to {total}.")
        end = total

    video_frames = frames[start:end]
    actual_num_frames = len(video_frames)

    # ── Step 2: compute latent indices ──
    latent_num = (actual_num_frames - 1) // 4 + 1
    latent_indices = [i * 4 for i in range(latent_num)]
    # clamp last index
    latent_indices = [min(idx, actual_num_frames - 1) for idx in latent_indices]

    sampled = [video_frames[idx] for idx in latent_indices]
    print(f"Video frames: {actual_num_frames}, Latent poses: {latent_num}")

    # ── Step 3: extract positions & angles ──
    positions_hw = []
    yaws_rad = []
    pitches_rad = []

    for fr in sampled:
        x_s, y_s, z_s = fr["camera_position"]
        pos = source_to_hyworld_position(x_s, y_s, z_s)
        positions_hw.append(pos)

        # Source yaw: degrees, 0=east(+X), 90=north(+Y), CCW from above
        # HY-World yaw: rotation around Y-up axis
        # In Source, forward = [cos(yaw), sin(yaw), 0]
        # In HY-World forward is +Z, so: yaw_hw such that forward = [sin(yaw_hw), 0, cos(yaw_hw)]
        # Mapping: yaw_hw = -(yaw_source - 90°) converted to radians
        #   because Source yaw=90° means facing +Y (forward), which is HY-World +Z (yaw=0)
        yaw_src_deg = fr["yaw"]
        yaw_hw_rad = -math.radians(yaw_src_deg - 90.0)
        yaws_rad.append(yaw_hw_rad)

        # Source pitch: positive = looking down; HY-World pitch via rot_x: positive = looking down
        pitch_src_deg = fr["pitch"]
        pitch_hw_rad = math.radians(pitch_src_deg)
        pitches_rad.append(pitch_hw_rad)

    # ── Step 4: build absolute c2w and normalise to frame-0 = identity ──
    c2ws_abs = []
    for i in range(latent_num):
        c2w = build_c2w(positions_hw[i], yaws_rad[i], pitches_rad[i])
        c2ws_abs.append(c2w)

    T0_inv = np.linalg.inv(c2ws_abs[0])
    c2ws_rel = [T0_inv @ c2w for c2w in c2ws_abs]

    # ── Step 5: scale translations ──
    # Compute average per-step translation magnitude
    steps = []
    for i in range(1, len(c2ws_rel)):
        t = np.linalg.norm(c2ws_rel[i][:3, 3] - c2ws_rel[i - 1][:3, 3])
        steps.append(t)

    if args.scale_factor == "auto":
        avg_step = np.mean(steps) if steps and np.mean(steps) > 1e-8 else 1.0
        target_step = 0.08
        scale = target_step / avg_step
        print(f"Auto scale: avg_step={avg_step:.6f}, scale_factor={scale:.6f}")
    else:
        scale = float(args.scale_factor)
        print(f"Manual scale_factor={scale}")

    for c2w in c2ws_rel:
        c2w[:3, 3] *= scale

    # ── Step 6: build K ──
    w, h = args.resolution
    K = fov_to_intrinsic(args.fov, w, h)

    # ── Step 7: output JSON ──
    output = {}
    for i, c2w in enumerate(c2ws_rel):
        output[str(i)] = {
            "extrinsic": c2w.tolist(),
            "K": K,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(output)} poses to {args.output}")

    # ── verification summary ──
    print("\n── Verification ──")
    print(f"Frame 0 extrinsic is identity: {np.allclose(c2ws_rel[0], np.eye(4))}")
    if len(c2ws_rel) > 1:
        last_t = np.linalg.norm(c2ws_rel[-1][:3, 3])
        print(f"Last frame translation magnitude: {last_t:.4f}")
        scaled_steps = []
        for i in range(1, len(c2ws_rel)):
            t = np.linalg.norm(c2ws_rel[i][:3, 3] - c2ws_rel[i - 1][:3, 3])
            scaled_steps.append(t)
        print(f"Per-step translation: min={min(scaled_steps):.4f}, "
              f"max={max(scaled_steps):.4f}, avg={np.mean(scaled_steps):.4f}")


def parse_resolution(s):
    parts = s.split("x")
    return int(parts[0]), int(parts[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CS:GO player replay to HY-WorldPlay pose JSON"
    )
    parser.add_argument("--input", required=True,
                        help="Path to CS:GO player JSON (e.g. *_Alpha_instance_000.json)")
    parser.add_argument("--output", default="./assets/pose/csgo_trajectory.json",
                        help="Output pose JSON path")
    parser.add_argument("--num_frames", type=int, default=125,
                        help="Number of video frames to use (default: 125)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Starting frame index (default: 0)")
    parser.add_argument("--scale_factor", default="auto",
                        help="Translation scale factor. 'auto' scales avg step to ~0.08 (default: auto)")
    parser.add_argument("--fov", type=float, default=106.26,
                        help="Horizontal FOV in degrees (default: 106.26)")
    parser.add_argument("--resolution", type=str, default="1280x720",
                        help="Video resolution WxH (default: 1280x720)")

    args = parser.parse_args()
    args.resolution = parse_resolution(args.resolution)
    convert(args)
