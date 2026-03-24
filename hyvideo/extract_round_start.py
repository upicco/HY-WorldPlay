"""
Extract a video frame and sample player data starting from round_freeze_end.

Given a CS:GO replay dataset directory, this script:
  1. Reads world_events JSONL to find the round_freeze_end tick
  2. Computes the corresponding video frame index using start_tick and tf_ratio
  3. Extracts that frame as an image from the player's mp4
  4. Samples 125 video-frame-aligned entries from the player JSON (every tf_ratio ticks)
  5. Outputs a clean JSON file ready for convert_player_trajectory.py

Tick-to-frame mapping:
  video_frame_index = (tick - start_tick) / tf_ratio
  where tf_ratio = tickrate / video_fps (typically 128/64 = 2)

Usage:
  python hyvideo/extract_round_start.py \
      --data_dir path/to/4b48a4c7.../train \
      --player Alpha_instance_000 \
      --num_frames 125 \
      --output_dir ./output
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_files(data_dir, player):
    """Locate all relevant files for the given player in data_dir."""
    data_dir = Path(data_dir)
    files = list(data_dir.iterdir())

    # Find world_events file (there may be multiple, pick the one matching episode)
    world_events = [f for f in files if f.name.endswith("_world_events.jsonl")]

    # Find player-specific files by matching the player suffix
    player_json = None
    player_mp4 = None
    player_episode_info = None

    for f in files:
        if player in f.name:
            if f.name.endswith(".json") and "episode_info" not in f.name and "video_manifest" not in f.name:
                player_json = f
            elif f.name.endswith(".mp4") and "depth" not in f.name:
                player_mp4 = f
            elif f.name.endswith("_episode_info.json"):
                player_episode_info = f

    missing = []
    if not world_events:
        missing.append("world_events JSONL")
    if player_json is None:
        missing.append(f"player JSON for '{player}'")
    if player_mp4 is None:
        missing.append(f"player MP4 for '{player}'")
    if player_episode_info is None:
        missing.append(f"episode_info for '{player}'")

    if missing:
        print(f"Error: could not find: {', '.join(missing)}")
        print(f"  data_dir: {data_dir}")
        print(f"  player:   {player}")
        sys.exit(1)

    return {
        "world_events": world_events,
        "player_json": player_json,
        "player_mp4": player_mp4,
        "episode_info": player_episode_info,
    }


def load_episode_info(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_round_freeze_end_tick(world_events_files, episode_id=None):
    """Find the round_freeze_end tick from world_events JSONL files."""
    for wf in world_events_files:
        with open(wf, "r", encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line)
                if ev["event_type"] == "round_freeze_end":
                    if episode_id is None or ev.get("episode_id") == episode_id:
                        return ev["tick"]
    return None


def tick_to_video_frame(tick, start_tick, tf_ratio):
    """Convert a game tick to a video frame index."""
    return (tick - start_tick) // tf_ratio


def extract_video_frame(mp4_path, frame_index, output_path, fps):
    """Extract a single frame from mp4 using ffmpeg."""
    timestamp = frame_index / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", str(mp4_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    return True


def sample_frames(player_data, freeze_end_tick, tf_ratio, num_frames):
    """Sample num_frames video-frame-aligned entries starting from freeze_end_tick.

    Video frames correspond to ticks at intervals of tf_ratio.
    For each target tick, find the closest entry in player_data.
    """
    # Build tick -> entry lookup
    tick_to_entry = {}
    for entry in player_data:
        if entry is not None and entry.get("tick") is not None:
            tick_to_entry[entry["tick"]] = entry

    all_ticks = sorted(tick_to_entry.keys())

    sampled = []
    for i in range(num_frames):
        target_tick = freeze_end_tick + i * tf_ratio

        # Find closest tick in data
        # Binary search for efficiency
        lo, hi = 0, len(all_ticks) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if all_ticks[mid] < target_tick:
                lo = mid + 1
            else:
                hi = mid

        # Pick the closest between lo and lo-1
        best_idx = lo
        if lo > 0 and abs(all_ticks[lo - 1] - target_tick) < abs(all_ticks[lo] - target_tick):
            best_idx = lo - 1

        matched_tick = all_ticks[best_idx]
        entry = tick_to_entry[matched_tick]

        sampled.append({
            "sample_index": i,
            "target_tick": target_tick,
            "matched_tick": matched_tick,
            "tick_offset": matched_tick - target_tick,
            "camera_position": entry["camera_position"],
            "camera_rotation": entry["camera_rotation"],
            "yaw": entry["yaw"],
            "pitch": entry["pitch"],
            "x": entry["x"],
            "y": entry["y"],
            "z": entry["z"],
            "action": entry["action"],
        })

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frame and sample player data from round_freeze_end"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to the train directory containing player files")
    parser.add_argument("--player", default="Alpha_instance_000",
                        help="Player identifier suffix (default: Alpha_instance_000)")
    parser.add_argument("--num_frames", type=int, default=125,
                        help="Number of video frames to sample (default: 125)")
    parser.add_argument("--output_dir", default="./output",
                        help="Output directory (default: ./output)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: locate files ──
    files = find_files(args.data_dir, args.player)
    episode_info = load_episode_info(files["episode_info"])

    start_tick = episode_info["start_tick"]
    tf_ratio = episode_info["tf_ratio"]
    video_fps = episode_info["video_fps"]
    tickrate = episode_info["tickrate"]
    episode_id = f"{episode_info['episode_number']:06d}"

    print(f"Episode: {episode_id}")
    print(f"start_tick={start_tick}, tf_ratio={tf_ratio}, tickrate={tickrate}, video_fps={video_fps}")

    # ── Step 2: find round_freeze_end tick ──
    freeze_end_tick = find_round_freeze_end_tick(files["world_events"], episode_id)
    if freeze_end_tick is None:
        print("Error: round_freeze_end event not found in world_events")
        sys.exit(1)
    print(f"round_freeze_end tick: {freeze_end_tick}")

    # ── Step 3: compute video frame index ──
    if freeze_end_tick < start_tick:
        print(f"Warning: round_freeze_end tick ({freeze_end_tick}) is before video start_tick ({start_tick})")
        print("Using start_tick as the beginning frame instead.")
        freeze_end_tick = start_tick

    frame_index = tick_to_video_frame(freeze_end_tick, start_tick, tf_ratio)
    print(f"Video frame index: {frame_index}")

    # ── Step 4: extract video frame ──
    frame_output = os.path.join(args.output_dir, f"round_freeze_end_frame_{frame_index}.jpg")
    print(f"Extracting frame {frame_index} from {files['player_mp4'].name} ...")
    ok = extract_video_frame(files["player_mp4"], frame_index, frame_output, video_fps)
    if ok:
        print(f"Saved frame to: {frame_output}")
    else:
        print("Warning: failed to extract video frame (ffmpeg error)")

    # ── Step 5: load player data and sample ──
    print(f"Loading player data from {files['player_json'].name} ...")
    with open(files["player_json"], "r", encoding="utf-8") as f:
        player_data = json.load(f)

    sampled = sample_frames(player_data, freeze_end_tick, tf_ratio, args.num_frames)

    # ── Step 6: write output JSON ──
    output_meta = {
        "source_player": args.player,
        "episode_id": episode_id,
        "round_freeze_end_tick": freeze_end_tick,
        "start_tick": start_tick,
        "tf_ratio": tf_ratio,
        "tickrate": tickrate,
        "video_fps": video_fps,
        "video_frame_start": frame_index,
        "num_frames_sampled": len(sampled),
    }

    output = {
        "metadata": output_meta,
        "frames": sampled,
    }

    json_output = os.path.join(args.output_dir, f"sampled_{args.player}.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(sampled)} sampled frames to: {json_output}")

    # ── Summary ──
    print(f"\n── Summary ──")
    print(f"round_freeze_end tick: {freeze_end_tick} → video frame {frame_index}")
    print(f"Sampled ticks: [{sampled[0]['matched_tick']} ... {sampled[-1]['matched_tick']}]")
    print(f"Tick offsets (target vs matched): "
          f"max={max(abs(s['tick_offset']) for s in sampled)}")
    print(f"Frame 0: pos={sampled[0]['camera_position']}, yaw={sampled[0]['yaw']:.2f}, pitch={sampled[0]['pitch']:.2f}")
    if len(sampled) > 1:
        print(f"Frame {len(sampled)-1}: pos={sampled[-1]['camera_position']}, "
              f"yaw={sampled[-1]['yaw']:.2f}, pitch={sampled[-1]['pitch']:.2f}")


if __name__ == "__main__":
    main()
