"""Audio-based timeline alignment for timed procedure sources."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - the caller reports the unavailable backend.
    np = None


WAVEFORM_SAMPLE_RATE = 4000
WAVEFORM_ENVELOPE_HZ = 40
WAVEFORM_MAX_OFFSET_SECONDS = 300
WAVEFORM_TIMEOUT_SECONDS = 300


def waveform_backend_available():
    return np is not None and bool(shutil.which("ffmpeg"))


def audio_envelope(
    audio_path,
    *,
    ffmpeg_path=None,
    sample_rate=WAVEFORM_SAMPLE_RATE,
    envelope_hz=WAVEFORM_ENVELOPE_HZ,
    timeout=WAVEFORM_TIMEOUT_SECONDS,
):
    """Decode audio and return a normalized transient-energy envelope."""
    if np is None:
        raise RuntimeError("numpy is required for waveform alignment")
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for waveform alignment")
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)

    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"audio decode timed out after {timeout}s") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(message or "ffmpeg audio decode failed")

    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32)
    window = max(1, int(round(sample_rate / envelope_hz)))
    usable = samples.size - (samples.size % window)
    if usable < window * envelope_hz * 3:
        raise RuntimeError("audio is too short for waveform alignment")
    frames = samples[:usable].reshape(-1, window)
    energy = np.sqrt(np.mean(np.square(frames), axis=1) + 1.0)
    envelope = np.log1p(energy)

    # Remove slow changes in microphone gain and emphasize shared impacts, speech,
    # and machine sounds that are useful as synchronization landmarks.
    baseline_width = max(3, int(envelope_hz * 2.0))
    kernel = np.ones(baseline_width, dtype=np.float32) / baseline_width
    left_pad = baseline_width // 2
    right_pad = baseline_width - 1 - left_pad
    padded = np.pad(envelope, (left_pad, right_pad), mode="edge")
    baseline = np.convolve(padded, kernel, mode="valid")
    transient = envelope - baseline
    transient += np.diff(envelope, prepend=envelope[0]) * 0.6
    median = float(np.median(transient))
    deviation = float(np.median(np.abs(transient - median)))
    if deviation < 1e-4:
        deviation = float(np.std(transient))
    if deviation < 1e-4:
        raise RuntimeError("audio does not contain enough waveform variation")
    normalized = np.clip((transient - median) / (deviation * 1.4826), -6.0, 6.0)
    normalized -= float(np.mean(normalized))
    standard_deviation = float(np.std(normalized))
    if standard_deviation < 1e-4:
        raise RuntimeError("audio does not contain usable waveform landmarks")
    normalized /= standard_deviation
    edge_width = min(len(normalized) // 5, max(1, int(envelope_hz * 0.75)))
    normalized[:edge_width] = 0
    normalized[-edge_width:] = 0
    return normalized.astype(np.float32), float(envelope_hz)


def _fft_cross_correlation(reference, candidate):
    full_length = len(reference) + len(candidate) - 1
    fft_length = 1 << max(1, full_length - 1).bit_length()
    reference_fft = np.fft.rfft(reference, fft_length)
    candidate_fft = np.fft.rfft(candidate[::-1], fft_length)
    correlation = np.fft.irfft(reference_fft * candidate_fft, fft_length)[:full_length]
    lags = np.arange(-(len(candidate) - 1), len(reference), dtype=np.int64)
    overlap = np.minimum(
        np.minimum(len(reference), len(candidate)),
        np.minimum(len(reference) - np.maximum(lags, 0), len(candidate) + np.minimum(lags, 0)),
    )
    return correlation, lags, overlap


def estimate_envelope_offset(
    reference,
    candidate,
    envelope_hz,
    *,
    max_offset_seconds=WAVEFORM_MAX_OFFSET_SECONDS,
    minimum_overlap_seconds=4.0,
):
    if np is None:
        return None
    correlation, lags, overlap = _fft_cross_correlation(reference, candidate)
    max_lag = int(max_offset_seconds * envelope_hz)
    minimum_overlap = max(2, int(minimum_overlap_seconds * envelope_hz))
    allowed = (np.abs(lags) <= max_lag) & (overlap >= minimum_overlap)
    if not np.any(allowed):
        return None

    scores = np.full(correlation.shape, -np.inf, dtype=np.float64)
    scores[allowed] = correlation[allowed] / overlap[allowed]
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    if not math.isfinite(best_score):
        return None

    exclusion = max(1, int(envelope_hz * 1.0))
    secondary = scores.copy()
    secondary[max(0, best_index - exclusion):best_index + exclusion + 1] = -np.inf
    second_score = float(np.max(secondary)) if np.any(np.isfinite(secondary)) else -1.0
    separation = best_score - second_score
    confidence = "low"
    if best_score >= 0.55 and separation >= 0.05:
        confidence = "high"
    elif best_score >= 0.30 and separation >= 0.015:
        confidence = "medium"
    return {
        "offset_seconds": round(float(lags[best_index]) / envelope_hz, 4),
        "correlation": round(best_score, 4),
        "peak_separation": round(separation, 4),
        "overlap_seconds": round(float(overlap[best_index]) / envelope_hz, 2),
        "confidence": confidence,
    }


def estimate_waveform_alignment(
    reference_audio_path,
    candidate_audio_path,
    *,
    ffmpeg_path=None,
    max_offset_seconds=WAVEFORM_MAX_OFFSET_SECONDS,
):
    """Estimate candidate-to-reference offset from shared audio landmarks."""
    reference, rate = audio_envelope(reference_audio_path, ffmpeg_path=ffmpeg_path)
    candidate, candidate_rate = audio_envelope(candidate_audio_path, ffmpeg_path=ffmpeg_path)
    if candidate_rate != rate:
        raise RuntimeError("waveform envelope rates do not match")
    estimate = estimate_envelope_offset(
        reference,
        candidate,
        rate,
        max_offset_seconds=max_offset_seconds,
    )
    if not estimate:
        return None
    estimate.update(
        {
            "method": "audio_waveform",
            "scale": 1.0,
            "reference_duration_seconds": round(len(reference) / rate, 2),
            "candidate_duration_seconds": round(len(candidate) / rate, 2),
            "envelope_hz": rate,
        }
    )
    return estimate
