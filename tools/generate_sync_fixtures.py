#!/usr/bin/env python3
"""Generate deterministic multi-source media for timeline regression tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


def write_master_audio(path, duration_seconds, sample_rate=16000):
    noise_rng = random.Random(20260930)
    envelope_rng = random.Random(613)
    step_times = [value for value in (2.0, 5.0, 8.5, 12.0, 15.5, 19.0, 23.0) if value < duration_seconds - 0.5]
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        noise_level = 0.06
        target_noise_level = noise_level
        for index in range(frame_count):
            t = index / sample_rate
            if index % max(1, int(sample_rate * 0.11)) == 0:
                target_noise_level = envelope_rng.uniform(0.015, 0.23)
            noise_level += (target_noise_level - noise_level) * 0.0025
            carrier = 0.018 * math.sin(2 * math.pi * (173 + 11 * math.sin(t * 0.37)) * t)
            speech_shape = 0.025 * math.sin(2 * math.pi * (310 + 35 * math.sin(t * 0.91)) * t)
            noise = noise_rng.uniform(-noise_level, noise_level)
            marker = 0.0
            for marker_index, marker_time in enumerate(step_times):
                local = t - marker_time
                if 0 <= local <= 0.22:
                    marker += (0.62 - local * 1.8) * math.sin(
                        2 * math.pi * (880 + marker_index * 137) * local
                    )
            value = max(-0.98, min(0.98, carrier + speech_shape + noise + marker))
            frames.extend(struct.pack("<h", int(value * 32767)))
        output.writeframes(frames)
    return step_times


def flash_filter(step_times, *, offset=0.0, scale=1.0):
    visible_times = [
        (step_time - offset) / scale
        for step_time in step_times
        if step_time >= offset and (step_time - offset) / scale >= 0
    ]
    if not visible_times:
        return "null"
    expression = "+".join(
        f"between(t\\,{value:.3f}\\,{value + 0.18:.3f})"
        for value in visible_times
    )
    return f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.92:t=fill:enable={expression}"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed")[-1600:])


def generate_fixtures(output_dir, *, duration=18.0, offset=1.75, scale=1.0):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate synchronization fixtures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_duration = offset + duration * scale + 2.0

    with tempfile.TemporaryDirectory(prefix="assembly-sync-") as temp_dir:
        master_audio = Path(temp_dir) / "master.wav"
        step_times = write_master_audio(master_audio, master_duration)
        reference_path = output_dir / "reference_landscape.mp4"
        alternate_path = output_dir / "alternate_portrait.mp4"
        voiceover_path = output_dir / "aligned_voiceover.wav"

        run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=1280x720:rate=24:duration={duration}",
                "-i",
                str(master_audio),
                "-t",
                str(duration),
                "-vf",
                flash_filter(step_times),
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                "-c:a",
                "aac",
                "-shortest",
                str(reference_path),
            ]
        )
        run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=720x1280:rate=24:duration={duration}",
                "-ss",
                str(offset),
                "-i",
                str(master_audio),
                "-t",
                str(duration),
                "-vf",
                flash_filter(step_times, offset=offset, scale=scale),
                "-af",
                f"atempo={scale}",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                "-c:a",
                "aac",
                "-shortest",
                str(alternate_path),
            ]
        )
        run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-ss",
                str(offset),
                "-i",
                str(master_audio),
                "-t",
                str(duration),
                "-af",
                f"atempo={scale}",
                "-c:a",
                "pcm_s16le",
                str(voiceover_path),
            ]
        )

    manifest = {
        "fixture_version": 1,
        "duration_seconds": duration,
        "reference": {
            "path": reference_path.name,
            "orientation": "landscape",
            "timeline_offset_seconds": 0.0,
            "timeline_scale": 1.0,
            "use_for_drafting": True,
        },
        "alternate": {
            "path": alternate_path.name,
            "orientation": "portrait",
            "timeline_offset_seconds": offset,
            "timeline_scale": scale,
            "use_for_drafting": False,
        },
        "voiceover": {
            "path": voiceover_path.name,
            "timeline_offset_seconds": offset,
            "timeline_scale": scale,
            "use_for_drafting": False,
        },
        "sync_markers": [round(value, 3) for value in step_times if value <= duration],
        "scripted_steps": [
            {"step": index + 1, "timeline_seconds": round(value, 3)}
            for index, value in enumerate(step_times)
            if value <= duration
        ],
    }
    manifest_path = output_dir / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--offset", type=float, default=1.75)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    manifest = generate_fixtures(
        args.output_dir,
        duration=max(8.0, args.duration),
        offset=max(0.0, args.offset),
        scale=min(2.0, max(0.5, args.scale)),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
