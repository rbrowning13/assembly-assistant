#!/usr/bin/env python3
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, Response, jsonify, render_template, request, send_file
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from assembly_assistant.sync import estimate_waveform_alignment, waveform_backend_available

try:
    import cv2
except Exception:
    cv2 = None


STAGES = [
    ("upload_check", "Uploads"),
    ("transcription", "Transcript"),
    ("drawings", "Drawings"),
    ("images", "Images"),
    ("video", "Video"),
    ("questions", "Questions"),
    ("draft_sections", "Drafts"),
]

STATE_LOCK = threading.RLock()
JOBS = {}
SECTIONS = {}
PACKAGES = {}
SETTINGS = {
    "openai_api_key": "",
    "transcription_backend": "auto",
    "transcription_model": "whisper-1",
    "local_whisper_cli_path": "",
    "local_whisper_model_path": "",
    "local_whisper_no_gpu": True,
    "instruction_model": "gpt-4o-mini",
    "transcription_enabled": True,
    "default_processing": "Standard",
}
RUNNING_THREADS = {}
FRAME_MANIFEST_CACHE = {}
PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
STORAGE_DIR = APP_DIR / "storage"
DB_PATH = STORAGE_DIR / "assembly_assistant.sqlite3"
DB_READY = False
OPENAI_AUDIO_FILE_LIMIT_BYTES = 25 * 1024 * 1024
TIMESTAMPED_TRANSCRIPTION_MODEL = "whisper-1"
TRANSCRIPTION_CHUNK_TARGET_BYTES = 10 * 1024 * 1024
TRANSCRIPTION_CHUNK_SECONDS = 240
TRANSCRIPTION_AUDIO_BITRATE = "48k"
AUDIO_EXTRACTION_TIMEOUT_SECONDS = 300
AUDIO_CHUNK_TIMEOUT_SECONDS = 180
LOCAL_WHISPER_TIMEOUT_SECONDS = 1800
LOCAL_WHISPER_OUTPUT_WORD_LENGTH = 1
LOCAL_WHISPER_THREADS = min(8, max(4, (os.cpu_count() or 4) - 2))
LOCAL_WHISPER_BEST_OF = 1
LOCAL_WHISPER_BEAM_SIZE = 1
VIDEO_FRAME_SAMPLE_RATE_HZ = 3
VIDEO_FRAME_INTERVAL_SECONDS = 1 / VIDEO_FRAME_SAMPLE_RATE_HZ
VIDEO_EXTRACTION_TIMEOUT_SECONDS = 1200
DOCUMENT_TEXT_TIMEOUT_SECONDS = 180
DOCUMENT_RENDER_TIMEOUT_SECONDS = 600
DOCUMENT_OCR_PAGE_TIMEOUT_SECONDS = 120
FRAME_MANIFEST_NAME = "frame_manifest.json"
FRAME_MANIFEST_VERSION = 2
PROCESSING_MANIFEST_VERSION = 1
ENGINE_VERSION = "assembly-assistant-core-0.3"
ASSEMBLY_PROFILE_VERSION = "assembly-1"
INSTRUCTION_PROMPT_VERSION = "assembly-draft-1"
ADAPTIVE_BASE_SAMPLE_RATES = {
    "fast": 1.0,
    "standard": 2.0,
    "deep": 3.0,
}
ADAPTIVE_STEP_SAMPLE_RATE_HZ = 3.0
ADAPTIVE_PAUSE_SAMPLE_RATE_HZ = 4.0
ADAPTIVE_CUE_SAMPLE_RATE_HZ = 3.0
ADAPTIVE_IMAGE_CUE_SAMPLE_RATE_HZ = 6.0
ADAPTIVE_WINDOW_MERGE_GAP_SECONDS = 0.2
ADAPTIVE_TIMESTAMP_DEDUPE_SECONDS = 0.18
FRAME_ANALYSIS_BATCH_SIZE = 8
FRAME_ANALYSIS_MAX_FRAMES = 420
VIDEO_CANDIDATE_FRAME_LIMIT = 640
SUPPLEMENTAL_FRAME_ANALYSIS_LIMIT = 80
FRAME_ANALYSIS_PROFILE_BUDGETS = {
    "fast": {
        "frames_per_step": 4,
        "min_frames": 48,
        "max_frames": 180,
        "coverage_reserve": 8,
        "supplemental_per_weak_step": 2,
        "supplemental_max_frames": 32,
        "parallel_batches": 2,
        "candidate_multiplier": 2.0,
        "candidate_max": 260,
    },
    "standard": {
        "frames_per_step": 6,
        "min_frames": 72,
        "max_frames": 260,
        "coverage_reserve": 12,
        "supplemental_per_weak_step": 4,
        "supplemental_max_frames": 56,
        "parallel_batches": 2,
        "candidate_multiplier": 2.4,
        "candidate_max": 420,
    },
    "deep": {
        "frames_per_step": 10,
        "min_frames": 120,
        "max_frames": FRAME_ANALYSIS_MAX_FRAMES,
        "coverage_reserve": 24,
        "supplemental_per_weak_step": 8,
        "supplemental_max_frames": SUPPLEMENTAL_FRAME_ANALYSIS_LIMIT,
        "parallel_batches": 2,
        "candidate_multiplier": 2.8,
        "candidate_max": VIDEO_CANDIDATE_FRAME_LIMIT,
    },
}
HARD_FRAME_CUE_TAGS = {"good_image", "operator_image_cue"}
STEP_FRAME_CUE_TAGS = {"step", "step_window", "after_action_pause", "pause"}
SOFT_FRAME_CUE_TAGS = {"parts", "tool", "torque", "threadlocker", "quality", "safety", "fixture"}
TIMED_SOURCE_KINDS = {"video", "audio"}
SOURCE_ROLES = {
    "primary_video",
    "alternate_angle",
    "voiceover",
    "supporting_document",
    "reference_image",
    "reference",
}
SOURCE_AUTHORITY_LEVELS = {
    "authoritative",
    "primary_evidence",
    "supporting_context",
    "reference_only",
}
VIDEO_COVERAGE_INTERVAL_SECONDS = 6
VIDEO_CUE_BURST_SECONDS = 6
PROCESSING_WATCHDOG_SECONDS = 10 * 60
OPENCV_BLUR_MIN = 35
OPENCV_DUPLICATE_DELTA_MAX = 2.4
OPENCV_FOCUS_CHANGE_MIN = 6.0
OPENCV_BACKGROUND_CHANGE_MIN = 8.0
STEP_FRAME_TARGET_WINDOW_SECONDS = 5
STEP_EXACT_MIN_WINDOW_SECONDS = 1.5
STEP_FALLBACK_NEAR_SECONDS = 8
STEP_FALLBACK_GLOBAL_SECONDS = 14
FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.jpg$")
OPERATOR_PHRASE_GUIDE = [
    {
        "label": "Step boundary",
        "phrases": ["new step", "next step", "start step", "step complete", "moving on"],
        "purpose": "Creates a likely new instruction step at that timestamp.",
    },
    {
        "label": "Capture frame",
        "phrases": ["take picture here", "good image here", "capture this", "show this", "good shot"],
        "purpose": "Boosts nearby frames for step media selection.",
    },
    {
        "label": "Tools",
        "phrases": ["tool is", "using tool", "driver is", "fixture is", "jig is"],
        "purpose": "Adds or confirms the tool callout for the step.",
    },
    {
        "label": "Torque",
        "phrases": ["torque is", "torque value is", "torque to", "no torque required"],
        "purpose": "Adds torque callouts or bypasses a missing-torque flag.",
    },
    {
        "label": "Parts",
        "phrases": ["part is", "parts are", "component is", "consumable is", "apply threadlocker"],
        "purpose": "Adds parts and consumables to the step.",
    },
    {
        "label": "Quality",
        "phrases": ["quality check", "verify", "inspect", "confirm", "make sure"],
        "purpose": "Adds a quality check callout.",
    },
    {
        "label": "Safety",
        "phrases": ["safety note", "caution", "keep clear", "hot surface", "pinch point"],
        "purpose": "Adds a safety callout.",
    },
    {
        "label": "Orientation",
        "phrases": ["orientation is", "front faces", "arrow points", "flush with", "seated fully"],
        "purpose": "Improves wording for alignment, orientation, and final condition.",
    },
]


def short_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def timestamp():
    return time.strftime("%I:%M:%S %p").lstrip("0")


def datestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def epoch_now():
    return time.time()


def safe_filename(name):
    raw = Path(name or "upload.bin").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or "upload.bin"


def file_kind(file):
    name = (file.get("name") or "").lower()
    content_type = (file.get("type") or "").lower()
    suffix = Path(name).suffix.lower()
    if content_type.startswith("video/") or suffix in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        return "video"
    if content_type.startswith("audio/") or suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        return "audio"
    if content_type == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if content_type.startswith("image/") or suffix in {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".dng"}:
        return "image"
    if content_type.startswith("text/") or suffix in {".txt", ".md", ".csv"}:
        return "text"
    return "unknown"


def optional_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_source_role(kind, first_timed=False):
    if kind == "video":
        return "primary_video" if first_timed else "alternate_angle"
    if kind == "audio":
        return "voiceover"
    if kind in {"pdf", "text"}:
        return "supporting_document"
    if kind == "image":
        return "reference_image"
    return "reference"


def default_source_authority(kind):
    if kind in TIMED_SOURCE_KINDS:
        return "primary_evidence"
    return "supporting_context"


def normalize_timeline_sync(file):
    raw = file.get("timeline_sync") if isinstance(file.get("timeline_sync"), dict) else {}
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in {"auto", "manual", "reference"}:
        mode = "auto"
    sync = {
        "mode": mode,
        "start_source_seconds": optional_float(raw.get("start_source_seconds")),
        "start_timeline_seconds": optional_float(raw.get("start_timeline_seconds")),
        "end_source_seconds": optional_float(raw.get("end_source_seconds")),
        "end_timeline_seconds": optional_float(raw.get("end_timeline_seconds")),
    }
    file["timeline_sync"] = sync
    return sync


def manual_timeline_transform(file):
    sync = normalize_timeline_sync(file)
    if sync.get("mode") != "manual":
        return None
    source_start = sync.get("start_source_seconds")
    timeline_start = sync.get("start_timeline_seconds")
    if source_start is None or timeline_start is None:
        return None

    scale = 1.0
    source_end = sync.get("end_source_seconds")
    timeline_end = sync.get("end_timeline_seconds")
    if source_end is not None and timeline_end is not None and source_end > source_start:
        scale = (timeline_end - timeline_start) / (source_end - source_start)
        if not 0.5 <= scale <= 1.5:
            return None
    return {
        "mode": "manual",
        "source_origin_seconds": source_start,
        "timeline_origin_seconds": timeline_start,
        "scale": scale,
        "offset_seconds": timeline_start - source_start,
        "has_drift_anchor": source_end is not None and timeline_end is not None,
    }


def source_timeline_transform(source):
    if isinstance(source, (int, float)):
        return {
            "mode": "offset",
            "source_origin_seconds": 0.0,
            "timeline_origin_seconds": float(source),
            "scale": 1.0,
            "offset_seconds": float(source),
        }
    source = source or {}
    if source.get("timeline_reference") or source.get("_timeline_reference"):
        return {
            "mode": "reference",
            "source_origin_seconds": 0.0,
            "timeline_origin_seconds": 0.0,
            "scale": 1.0,
            "offset_seconds": 0.0,
        }
    timeline_sync = source.get("timeline_sync") or source.get("_timeline_sync")
    manual_source = source
    if timeline_sync is not source.get("timeline_sync"):
        manual_source = dict(source)
        manual_source["timeline_sync"] = timeline_sync or {}
    manual = manual_timeline_transform(manual_source)
    if manual:
        return manual
    offset = optional_float(source.get("timeline_offset_seconds"))
    if offset is None:
        offset = optional_float(source.get("_timeline_offset_seconds")) or 0.0
    scale = optional_float(source.get("timeline_scale"))
    if scale is None:
        scale = optional_float(source.get("_timeline_scale")) or 1.0
    return {
        "mode": "auto",
        "source_origin_seconds": 0.0,
        "timeline_origin_seconds": offset,
        "scale": scale,
        "offset_seconds": offset,
    }


def apply_source_timeline(source_seconds, source=None):
    transform = source_timeline_transform(source)
    source_value = float(source_seconds or 0)
    timeline_value = transform["timeline_origin_seconds"] + (
        source_value - transform["source_origin_seconds"]
    ) * transform["scale"]
    return max(0.0, timeline_value)


def normalize_job_sources(job):
    files = active_job_files(job)
    timed_files = [file for file in files if file_kind(file) in TIMED_SOURCE_KINDS]
    first_timed_id = file_identity(timed_files[0]) if timed_files else None

    for file in files:
        kind = file_kind(file)
        file["kind"] = kind
        file.setdefault("source_id", f"src-{uuid.uuid4().hex[:10]}")
        role = str(file.get("source_role") or "").strip().lower()
        if role not in SOURCE_ROLES:
            role = default_source_role(kind, file_identity(file) == first_timed_id)
        file["source_role"] = role
        file.setdefault("source_label", "")
        authority = str(file.get("authority_level") or "").strip().lower()
        file["authority_level"] = authority if authority in SOURCE_AUTHORITY_LEVELS else default_source_authority(kind)
        file.setdefault("revision", "")
        file.setdefault("applicability", "")
        normalize_timeline_sync(file)

    if not timed_files:
        return

    references = [file for file in timed_files if file.get("timeline_reference")]
    reference = references[0] if references else timed_files[0]
    for file in timed_files:
        file["timeline_reference"] = file is reference
        if file is reference:
            file["timeline_offset_seconds"] = 0.0
            file["timeline_scale"] = 1.0
            file["timeline_sync"]["mode"] = "reference"
            file["timeline_alignment"] = {
                "method": "reference_source",
                "confidence": "reference",
                "reference_file": file.get("name"),
            }
        elif file["timeline_sync"].get("mode") == "reference":
            file["timeline_sync"]["mode"] = "auto"

    narration_sources = [file for file in timed_files if file.get("use_for_drafting")]
    narration = narration_sources[0] if narration_sources else reference
    for file in timed_files:
        file["use_for_drafting"] = file is narration


def drafting_source_files(job):
    normalize_job_sources(job)
    return [
        file
        for file in active_job_files(job)
        if file.get("kind") in TIMED_SOURCE_KINDS and file.get("use_for_drafting")
    ]


def human_size(byte_count):
    size = float(byte_count or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024


def parse_multipart_files(content_type, body):
    match = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not match:
        return []
    boundary = ("--" + match.group(1)).encode("utf-8")
    files = []
    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        header_blob, data = part.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = header_blob.decode("latin-1", errors="replace").split("\r\n")
        disposition = ""
        content_type_header = "application/octet-stream"
        for header in headers:
            lower = header.lower()
            if lower.startswith("content-disposition:"):
                disposition = header.split(":", 1)[1].strip()
            if lower.startswith("content-type:"):
                content_type_header = header.split(":", 1)[1].strip()
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if not filename_match or not filename_match.group(1):
            continue
        files.append(
            {
                "name": filename_match.group(1),
                "type": content_type_header,
                "data": data,
            }
        )
    return files


def store_uploaded_files(job, uploads, source="job upload", question_id=None, clarification_for=None):
    originals_dir = STORAGE_DIR / "jobs" / job["id"] / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {
        file.get("sha256"): file
        for file in job.get("files", [])
        if file.get("sha256") and not is_removed_file(file)
    }
    added = []
    skipped = []
    for upload in uploads:
        data = upload.get("data") or b""
        digest = hashlib.sha256(data).hexdigest()
        original_name = upload.get("name") or "upload.bin"
        if digest in existing_files:
            skipped.append(existing_files[digest])
            continue
        clean_name = safe_filename(original_name)
        stored_name = f"{digest[:12]}_{clean_name}"
        stored_path = originals_dir / stored_name
        stored_path.write_bytes(data)
        record = {
            "name": original_name,
            "type": upload.get("type") or "application/octet-stream",
            "kind": file_kind({"name": original_name, "type": upload.get("type")}),
            "size": human_size(len(data)),
            "bytes": len(data),
            "sha256": digest,
            "stored_path": str(stored_path.relative_to(APP_DIR)),
            "source": source,
            "processing_status": "stored",
        }
        if question_id:
            record["question_id"] = question_id
        if clarification_for:
            record["clarification_for"] = clarification_for
        job["files"].append(record)
        normalize_job_sources(job)
        existing_files[digest] = record
        added.append(record)
    if added or skipped:
        if added and source == "job upload":
            job["needs_processing"] = True
            if job.get("status") not in {"Processing", "Drafting Sections", "Questions Pending"}:
                job["status"] = "Needs Processing"
        add_log(job, f"Stored {len(added)} files. Skipped {len(skipped)} duplicates.")
    else:
        save_state()
    return added, skipped


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_chunked_job_upload(job, params, data):
    upload_id = safe_filename((params.get("upload_id") or [""])[0])
    original_name = (params.get("name") or ["upload.bin"])[0] or "upload.bin"
    content_type = (params.get("type") or ["application/octet-stream"])[0] or "application/octet-stream"
    try:
        offset = int((params.get("offset") or ["0"])[0] or 0)
        total_size = int((params.get("size") or ["0"])[0] or 0)
    except Exception:
        return {"error": "Invalid chunk metadata."}, 400
    final = (params.get("final") or ["0"])[0] == "1"
    if not upload_id or offset < 0 or total_size <= 0:
        return {"error": "Missing chunk metadata."}, 400

    dirs = ensure_job_dirs(job)
    chunk_dir = dirs["root"] / "upload_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    part_path = chunk_dir / f"{upload_id}.part"
    mode = "r+b" if part_path.exists() else "wb"
    with part_path.open(mode) as handle:
        handle.seek(offset)
        handle.write(data or b"")

    received = part_path.stat().st_size
    if not final:
        return {"ok": True, "complete": False, "received": received, "added": [], "skipped": []}, 200
    if received != total_size:
        return {"error": f"Incomplete upload: received {received} of {total_size} bytes."}, 400

    digest = file_sha256(part_path)
    existing_files = {
        file.get("sha256"): file
        for file in job.get("files", [])
        if file.get("sha256") and not is_removed_file(file)
    }
    if digest in existing_files:
        part_path.unlink(missing_ok=True)
        add_log(job, "Chunked upload skipped: duplicate file.")
        return {"ok": True, "complete": True, "added": [], "skipped": [existing_files[digest]]}, 200

    clean_name = safe_filename(original_name)
    stored_name = f"{digest[:12]}_{clean_name}"
    stored_path = dirs["originals"] / stored_name
    shutil.move(str(part_path), stored_path)
    record = {
        "name": original_name,
        "type": content_type,
        "kind": file_kind({"name": original_name, "type": content_type}),
        "size": human_size(total_size),
        "bytes": total_size,
        "sha256": digest,
        "stored_path": str(stored_path.relative_to(APP_DIR)),
        "source": "job upload",
        "processing_status": "stored",
    }
    job["files"].append(record)
    normalize_job_sources(job)
    job["needs_processing"] = True
    if job.get("status") not in {"Processing", "Drafting Sections", "Questions Pending"}:
        job["status"] = "Needs Processing"
    add_log(job, f"Stored 1 large file by chunked upload: {original_name}.")
    return {"ok": True, "complete": True, "added": [record], "skipped": []}, 200


def store_section_media(section, uploads):
    media_dir = STORAGE_DIR / "sections" / section["id"] / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    section.setdefault("media_uploads", [])
    existing_files = {
        file.get("sha256"): file
        for file in section.get("media_uploads", [])
        if file.get("sha256")
    }
    added = []
    skipped = []
    for upload in uploads:
        data = upload.get("data") or b""
        original_name = upload.get("name") or "upload.bin"
        kind = file_kind({"name": original_name, "type": upload.get("type")})
        if kind != "image":
            skipped.append(
                {
                    "name": original_name,
                    "type": upload.get("type") or "application/octet-stream",
                    "kind": kind,
                    "reason": "Only image uploads can be attached directly to section steps.",
                }
            )
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in existing_files:
            skipped.append(existing_files[digest])
            continue
        clean_name = safe_filename(original_name)
        stored_name = f"{digest[:12]}_{clean_name}"
        stored_path = media_dir / stored_name
        stored_path.write_bytes(data)
        record = {
            "name": original_name,
            "type": upload.get("type") or "application/octet-stream",
            "kind": "image",
            "size": human_size(len(data)),
            "bytes": len(data),
            "sha256": digest,
            "stored_path": str(stored_path.relative_to(APP_DIR)),
            "source": "section step upload",
            "uploaded_at": datestamp(),
        }
        section["media_uploads"].append(record)
        existing_files[digest] = record
        added.append(record)
    section["updated_at"] = datestamp()
    save_state()
    return added, skipped


def init_db():
    global DB_READY
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    DB_READY = True


def save_state():
    if not DB_READY:
        return
    with STATE_LOCK:
        payload = json.dumps(
            {
                "jobs": JOBS,
                "sections": SECTIONS,
                "packages": PACKAGES,
                "settings": SETTINGS,
            },
            sort_keys=True,
        )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES ('main', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (payload,),
        )


def mark_interrupted_work(job):
    interrupted = job.get("status") in {"Processing", "Drafting Sections"}
    for stage in job.get("stages", {}).values():
        if stage.get("status") == "running":
            interrupted = True
            stage["status"] = "failed"
            stage["note"] = "Interrupted by app restart"
    if interrupted:
        job["status"] = "Needs Attention"
        flags = job.setdefault("flags", [])
        if "Processing interrupted" not in flags:
            flags.append("Processing interrupted")
        job["last_activity"] = timestamp()
        recompute_progress(job)
        if active_processing_manifest(job):
            finish_job_processing_timing(
                job,
                status="interrupted",
                outcome="Application restarted before processing completed.",
            )


def load_state():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = 'main'"
        ).fetchone()
    if not row:
        return False

    data = json.loads(row[0])
    with STATE_LOCK:
        JOBS.clear()
        JOBS.update(data.get("jobs") or {})
        SECTIONS.clear()
        SECTIONS.update(data.get("sections") or {})
        PACKAGES.clear()
        PACKAGES.update(data.get("packages") or {})
        SETTINGS.update(data.get("settings") or {})
        for job in JOBS.values():
            migrate_job_defaults(job)
            mark_interrupted_work(job)
    save_state()
    return True


def job_dir(job):
    return STORAGE_DIR / "jobs" / job["id"]


def ensure_job_dirs(job):
    paths = {
        "root": job_dir(job),
        "originals": job_dir(job) / "originals",
        "audio": job_dir(job) / "audio",
        "audio_chunks": job_dir(job) / "audio_chunks",
        "frames": job_dir(job) / "frames",
        "thumbnails": job_dir(job) / "thumbnails",
        "transcripts": job_dir(job) / "transcripts",
        "ocr": job_dir(job) / "ocr",
        "analysis": job_dir(job) / "analysis",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def stored_path(file):
    value = file.get("stored_path")
    if not value:
        return None
    path = APP_DIR / value
    return path if path.exists() else None


def stored_relative_path(value):
    if not value:
        return None
    path = APP_DIR / value
    return path if path.exists() else None


def file_identity(file):
    return file.get("sha256") or file.get("stored_path") or f"{file.get('name')}:{file.get('bytes') or file.get('size')}"


def is_removed_file(file):
    return bool(file.get("removed"))


def active_job_files(job):
    return [file for file in job.get("files", []) if not is_removed_file(file)]


def mark_job_needs_processing(job, message=None):
    job["needs_processing"] = True
    if job.get("status") not in {"Processing", "Drafting Sections", "Questions Pending"}:
        job["status"] = "Needs Processing"
    job["draft_sections"] = []
    job["questions"] = []
    job["question_review"] = {}
    job["media_selection"] = {}
    if message:
        add_log(job, message)
    else:
        save_state()


def remove_job_upload(job, identity):
    removed = None
    for file in job.get("files", []):
        if is_removed_file(file):
            continue
        if identity in {file.get("sha256"), file.get("stored_path"), file_identity(file)}:
            file["removed"] = True
            file["processing_status"] = "removed"
            file["removed_at"] = timestamp()
            removed = file
            break
    if not removed:
        return None

    removed_name = removed.get("name")
    removed_sha = removed.get("sha256")
    removed_audio = removed.get("audio_path")
    removed_transcript = removed.get("transcript_path")
    removed_frames = set(removed.get("frame_paths") or [])
    job["artifacts"] = [
        artifact
        for artifact in job.get("artifacts", [])
        if not (
            artifact.get("source_file") == removed_name
            or artifact.get("stored_path") in removed_frames
            or artifact.get("stored_path") in {removed_audio, removed_transcript}
        )
    ]
    job["frame_analysis"] = [
        item
        for item in job.get("frame_analysis", [])
        if item.get("stored_path") not in removed_frames
    ]
    if removed_sha:
        removed["sha256"] = removed_sha
    mark_job_needs_processing(job, f"Removed upload from processing set: {removed_name}. Stored file retained.")
    normalize_job_sources(job)
    return removed


def update_job_source(job, identity, data):
    normalize_job_sources(job)
    target = next(
        (
            file
            for file in active_job_files(job)
            if identity in {file.get("source_id"), file_identity(file), file.get("sha256"), file.get("stored_path")}
        ),
        None,
    )
    if not target:
        return None, "Source file not found."

    kind = target.get("kind") or file_kind(target)
    role = str(data.get("source_role") or target.get("source_role") or "").strip().lower()
    if role not in SOURCE_ROLES:
        return None, "Invalid source role."
    authority = str(data.get("authority_level") or target.get("authority_level") or "").strip().lower()
    if authority not in SOURCE_AUTHORITY_LEVELS:
        return None, "Invalid authority level."
    sync = None
    transform = None
    if kind in TIMED_SOURCE_KINDS:
        mode = str(data.get("sync_mode") or target.get("timeline_sync", {}).get("mode") or "auto").lower()
        if mode not in {"auto", "manual", "reference"}:
            return None, "Invalid synchronization mode."
        if target.get("timeline_reference") or bool(data.get("timeline_reference")):
            mode = "reference"
        sync = {
            "mode": mode,
            "start_source_seconds": optional_float(data.get("start_source_seconds")),
            "start_timeline_seconds": optional_float(data.get("start_timeline_seconds")),
            "end_source_seconds": optional_float(data.get("end_source_seconds")),
            "end_timeline_seconds": optional_float(data.get("end_timeline_seconds")),
        }
        if mode == "manual":
            candidate = dict(target)
            candidate["timeline_sync"] = sync
            transform = manual_timeline_transform(candidate)
            if not transform:
                return None, "Manual synchronization needs a valid start anchor and optional matching end anchors."

    target["source_role"] = role
    target["source_label"] = str(data.get("source_label") or "").strip()[:120]
    target["authority_level"] = authority
    target["revision"] = str(data.get("revision") or "").strip()[:120]
    target["applicability"] = str(data.get("applicability") or "").strip()[:500]

    if kind in TIMED_SOURCE_KINDS:
        if bool(data.get("timeline_reference")):
            for file in active_job_files(job):
                if file.get("kind") in TIMED_SOURCE_KINDS:
                    file["timeline_reference"] = file is target
        if bool(data.get("use_for_drafting")):
            for file in active_job_files(job):
                if file.get("kind") in TIMED_SOURCE_KINDS:
                    file["use_for_drafting"] = file is target
        if target.get("timeline_reference"):
            mode = "reference"
            sync["mode"] = "reference"
        target["timeline_sync"] = sync
        if mode == "manual":
            target["timeline_offset_seconds"] = round(transform["offset_seconds"], 6)
            target["timeline_scale"] = round(transform["scale"], 9)
            target["timeline_alignment"] = {
                "method": "manual_anchors",
                "confidence": "manual",
                "reference_file": next(
                    (
                        file.get("name")
                        for file in active_job_files(job)
                        if file.get("kind") in TIMED_SOURCE_KINDS and file.get("timeline_reference")
                    ),
                    "",
                ),
                "offset_seconds": round(transform["offset_seconds"], 6),
                "scale": round(transform["scale"], 9),
                "has_drift_anchor": transform["has_drift_anchor"],
            }
        elif mode == "auto" and not target.get("timeline_reference"):
            target["timeline_alignment"] = {
                "method": "pending_auto_alignment",
                "confidence": "pending",
                "reason": "Run transcription to estimate alignment from matching timed words.",
            }

    normalize_job_sources(job)
    mark_job_needs_processing(job, f"Updated source setup: {target.get('name')}.")
    return target, None


def upsert_artifact(job, record):
    artifacts = job.setdefault("artifacts", [])
    stored = record.get("stored_path")
    for artifact in artifacts:
        if stored and artifact.get("stored_path") == stored:
            artifact.update(record)
            return artifact
    artifacts.append(record)
    return record


def public_settings():
    local_cli = configured_local_whisper_cli_path()
    local_model = configured_local_whisper_model_path()
    backend = resolve_transcription_backend()
    return {
        "has_openai_api_key": bool(SETTINGS.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")),
        "transcription_backend": SETTINGS.get("transcription_backend") or "auto",
        "effective_transcription_backend": backend.get("backend"),
        "transcription_model": SETTINGS.get("transcription_model") or TIMESTAMPED_TRANSCRIPTION_MODEL,
        "local_whisper_cli_path": SETTINGS.get("local_whisper_cli_path") or local_cli or "",
        "local_whisper_model_path": SETTINGS.get("local_whisper_model_path") or local_model or "",
        "local_whisper_no_gpu": bool(SETTINGS.get("local_whisper_no_gpu", True)),
        "has_local_whisper_cli": bool(local_cli),
        "has_local_whisper_model": bool(local_model),
        "instruction_model": SETTINGS.get("instruction_model") or "gpt-4o-mini",
        "transcription_enabled": bool(SETTINGS.get("transcription_enabled", True)),
        "default_processing": SETTINGS.get("default_processing") or "Standard",
        "storage_root": str(STORAGE_DIR.relative_to(APP_DIR)),
        "operator_phrase_guide": OPERATOR_PHRASE_GUIDE,
    }


def get_openai_api_key():
    return SETTINGS.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""


def configured_local_whisper_cli_path():
    configured = (SETTINGS.get("local_whisper_cli_path") or "").strip()
    if configured and Path(configured).exists():
        return configured
    return shutil.which("whisper-cli") or shutil.which("whisper-command") or ""


def discover_local_whisper_model():
    search_roots = [
        STORAGE_DIR / "models",
        APP_DIR / "models",
        Path("/opt/homebrew/share/whisper-cpp"),
        Path("/usr/local/share/whisper-cpp"),
        Path.home() / "models",
    ]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.glob("ggml-*.bin"):
            name = path.name.lower()
            if "for-tests" in name:
                continue
            candidates.append(path)
    if not candidates:
        return ""
    preference = ["large-v3", "large-v2", "large", "medium.en", "medium", "small.en", "small", "base.en", "base", "tiny.en", "tiny"]

    def rank(path):
        name = path.name.lower()
        for index, token in enumerate(preference):
            if token in name:
                return index
        return len(preference)

    return str(sorted(candidates, key=rank)[0])


def configured_local_whisper_model_path():
    configured = (SETTINGS.get("local_whisper_model_path") or "").strip()
    if configured and Path(configured).exists():
        return configured
    return discover_local_whisper_model()


def resolve_transcription_backend():
    requested = SETTINGS.get("transcription_backend") or "auto"
    cli_path = configured_local_whisper_cli_path()
    model_path = configured_local_whisper_model_path()
    local_ready = bool(cli_path and model_path)
    if requested == "local_whisper":
        return {
            "requested": requested,
            "backend": "local_whisper",
            "cli_path": cli_path,
            "model_path": model_path,
            "ready": local_ready,
        }
    if requested == "openai":
        return {"requested": requested, "backend": "openai", "ready": bool(get_openai_api_key())}
    if local_ready:
        return {
            "requested": requested,
            "backend": "local_whisper",
            "cli_path": cli_path,
            "model_path": model_path,
            "ready": True,
        }
    return {"requested": requested, "backend": "openai", "ready": bool(get_openai_api_key())}


def effective_timestamped_transcription_model(model):
    requested = model or TIMESTAMPED_TRANSCRIPTION_MODEL
    if requested == TIMESTAMPED_TRANSCRIPTION_MODEL:
        return requested
    return TIMESTAMPED_TRANSCRIPTION_MODEL


def iter_multipart_fields(fields):
    if isinstance(fields, dict):
        iterator = fields.items()
    else:
        iterator = fields
    for name, value in iterator:
        if isinstance(value, (list, tuple)):
            for item in value:
                yield name, item
        else:
            yield name, value


def multipart_body(fields, file_field, file_path, content_type="audio/wav"):
    boundary = f"----AssemblyAssistant{uuid.uuid4().hex}"
    chunks = []
    for name, value in iter_multipart_fields(fields):
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_filename(file_path.name)}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


def transcribe_with_openai(audio_path, model):
    api_key = get_openai_api_key()
    if not api_key:
        return None, "missing_api_key"
    if audio_path.stat().st_size > OPENAI_AUDIO_FILE_LIMIT_BYTES:
        return None, "audio_over_25mb_needs_chunking"

    effective_model = effective_timestamped_transcription_model(model)
    last_error = None
    for attempt in range(1, 3):
        boundary, body = multipart_body(
            [
                ("model", effective_model),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ],
            "file",
            audio_path,
            mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream",
        )
        req = urllib_request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
                data["_requested_model"] = model
                data["_effective_model"] = effective_model
                data["_timestamp_granularities"] = ["segment", "word"]
                return data, None
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"openai_http_{exc.code}: {detail[:500]}"
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt == 1:
                time.sleep(2.5)
                continue
            return None, last_error
        except Exception as exc:
            last_error = f"openai_error: {exc}"
            if attempt == 1:
                time.sleep(1.5)
    return None, last_error


def local_whisper_audio_path(audio_path, ffmpeg):
    supported = {".wav", ".mp3", ".flac", ".ogg"}
    if audio_path.suffix.lower() in supported:
        return audio_path, None
    if not ffmpeg:
        return None, "ffmpeg_missing_for_local_whisper_audio_conversion"
    wav_path = audio_path.with_name(f"{audio_path.stem}_local_whisper.wav")
    if wav_path.exists() and wav_path.stat().st_mtime >= audio_path.stat().st_mtime:
        return wav_path, None
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=AUDIO_EXTRACTION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None, f"local_whisper_audio_conversion_timed_out after {AUDIO_EXTRACTION_TIMEOUT_SECONDS}s"
    if result.returncode != 0 or not wav_path.exists():
        return None, f"local_whisper_audio_conversion_failed: {(result.stderr or result.stdout or 'ffmpeg failed')[-700:]}"
    return wav_path, None


def parse_whisper_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 1000 if numeric > 1000 else numeric
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(text)
    except Exception:
        return None


def parse_whisper_offset(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return parse_whisper_timestamp(value)
    return numeric / 1000


def local_whisper_segment_times(item):
    timestamps = item.get("timestamps") or {}
    offsets = item.get("offsets") or {}
    start = parse_whisper_timestamp(timestamps.get("from"))
    end = parse_whisper_timestamp(timestamps.get("to"))
    if start is None:
        start = parse_whisper_timestamp(timestamps.get("from_ms"))
    if end is None:
        end = parse_whisper_timestamp(timestamps.get("to_ms"))
    if start is None:
        start = parse_whisper_offset(offsets.get("from"))
    if end is None:
        end = parse_whisper_offset(offsets.get("to"))
    if start is None:
        start = 0.0
    if end is None or end <= start:
        end = start + 0.5
    return start, end


def words_from_local_segment(text, start, end):
    tokens = re.findall(r"\S+", text or "")
    if not tokens:
        return []
    span = max(0.25 * len(tokens), float(end) - float(start))
    word_duration = span / max(1, len(tokens))
    words = []
    for index, token in enumerate(tokens):
        word_start = float(start) + word_duration * index
        word_end = min(float(end), word_start + word_duration)
        if word_end <= word_start:
            word_end = word_start + 0.25
        words.append({"word": token, "start": round(word_start, 2), "end": round(word_end, 2)})
    return words


def normalize_local_whisper_json(data):
    transcription = data.get("transcription") or data.get("segments") or []
    segments = []
    words = []
    text_parts = []
    for index, item in enumerate(transcription):
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", item.get("text") or item.get("transcript") or "").strip()
        if not text:
            continue
        start, end = local_whisper_segment_times(item)
        segment = {
            "id": index,
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
            "source": "local_whisper",
        }
        segments.append(segment)
        text_parts.append(text)
        item_words = item.get("words") if isinstance(item.get("words"), list) else []
        if item_words:
            for word in item_words:
                if not isinstance(word, dict):
                    continue
                word_text = word.get("word") or word.get("text") or ""
                word_start = parse_whisper_timestamp(word.get("start") or word.get("from"))
                word_end = parse_whisper_timestamp(word.get("end") or word.get("to"))
                if word_text and word_start is not None:
                    words.append({"word": word_text, "start": round(word_start, 2), "end": round(word_end or word_start + 0.25, 2)})
        else:
            words.extend(words_from_local_segment(text, start, end))
    return {
        "text": " ".join(text_parts).strip(),
        "segments": segments,
        "words": words,
        "duration": max([segment["end"] for segment in segments] or [0]),
        "raw_response": data,
    }


def local_whisper_json_candidates(output_prefix, result):
    candidates = [Path(f"{output_prefix}.json"), output_prefix.with_suffix(".json")]
    output_text = f"{result.stderr or ''}\n{result.stdout or ''}" if result else ""
    for match in re.findall(r"output_json:\s+saving output to '([^']+\.json)'", output_text):
        candidates.append(Path(match))
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def wait_for_local_whisper_json(output_prefix, result, timeout_seconds=3):
    candidates = local_whisper_json_candidates(output_prefix, result)
    deadline = time.time() + timeout_seconds
    while True:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if time.time() >= deadline:
            return None
        time.sleep(0.05)


def local_whisper_json_from_error(error_text):
    for match in re.findall(r"output_json:\s+saving output to '([^']+\.json)'", error_text or ""):
        candidate = Path(match)
        if not candidate.is_absolute():
            candidate = APP_DIR / candidate
        if candidate.exists():
            return candidate
    return None


def transcribe_with_local_whisper(audio_path, ffmpeg):
    backend = resolve_transcription_backend()
    cli_path = backend.get("cli_path") or configured_local_whisper_cli_path()
    model_path = backend.get("model_path") or configured_local_whisper_model_path()
    if not cli_path:
        return None, "local_whisper_cli_missing"
    if not model_path:
        return None, "local_whisper_model_missing"
    input_path, conversion_error = local_whisper_audio_path(audio_path, ffmpeg)
    if conversion_error:
        return None, conversion_error
    output_dir = audio_path.parent if os.access(audio_path.parent, os.W_OK) else Path(tempfile.gettempdir())
    output_prefix = output_dir / f"{audio_path.stem}_local_{uuid.uuid4().hex[:8]}"
    cmd = [
        cli_path,
        "-t",
        str(LOCAL_WHISPER_THREADS),
        "-m",
        model_path,
        "-f",
        str(input_path),
        "-l",
        "en",
        "-bo",
        str(LOCAL_WHISPER_BEST_OF),
        "-bs",
        str(LOCAL_WHISPER_BEAM_SIZE),
        "-oj",
        "-ojf",
        "-ml",
        str(LOCAL_WHISPER_OUTPUT_WORD_LENGTH),
        "-sow",
        "-of",
        str(output_prefix),
        "-np",
    ]
    if SETTINGS.get("local_whisper_no_gpu", True):
        cmd.insert(1, "-ng")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=LOCAL_WHISPER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None, f"local_whisper_timed_out after {LOCAL_WHISPER_TIMEOUT_SECONDS}s"
    json_path = wait_for_local_whisper_json(output_prefix, result)
    if not json_path:
        return None, f"local_whisper_failed: {(result.stderr or result.stdout or 'whisper-cli failed')[-900:]}"
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"local_whisper_json_read_failed: {exc}"
    normalized = normalize_local_whisper_json(raw)
    if result.returncode != 0:
        normalized["_warning"] = f"whisper-cli returned {result.returncode} after writing JSON"
        normalized["_stderr_tail"] = (result.stderr or result.stdout or "")[-900:]
    normalized["_requested_model"] = model_path
    normalized["_effective_model"] = Path(model_path).name
    normalized["_backend"] = "local_whisper"
    normalized["_cli_path"] = cli_path
    normalized["_timestamp_granularities"] = ["segment", "word_approx"]
    return normalized, None


def media_duration_seconds(media_path, ffprobe=None):
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe or not media_path or not Path(media_path).exists():
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float((result.stdout or "").strip())
    except Exception:
        return None
    return duration if duration > 0 else None


def timed_seconds(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def offset_timed_records(records, offset_seconds):
    adjusted = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        item = dict(record)
        start = timed_seconds(item.get("start"))
        end = timed_seconds(item.get("end"))
        if start is not None:
            item["start"] = round(max(0, start + offset_seconds), 2)
        if end is not None:
            item["end"] = round(max(0, end + offset_seconds), 2)
        if start is not None or end is not None:
            item["chunk_offset_seconds"] = round(offset_seconds, 2)
        adjusted.append(item)
    return adjusted


def split_audio_for_transcription(job_id, audio_path, ffmpeg):
    if audio_path.stat().st_size <= TRANSCRIPTION_CHUNK_TARGET_BYTES:
        return [audio_path], None

    chunk_root = ensure_job_dirs(JOBS[job_id])["audio_chunks"]
    chunk_dir = chunk_root / f"{audio_path.stem}_{uuid.uuid4().hex[:8]}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_suffix = audio_path.suffix or ".m4a"
    pattern = chunk_dir / f"{audio_path.stem}_%03d{chunk_suffix}"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(TRANSCRIPTION_CHUNK_SECONDS),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(pattern),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIO_CHUNK_TIMEOUT_SECONDS,
        )
        chunk_error = None
    except subprocess.TimeoutExpired:
        result = None
        chunk_error = f"audio_chunking_timed_out after {AUDIO_CHUNK_TIMEOUT_SECONDS}s"
    chunks = sorted(chunk_dir.glob(f"{audio_path.stem}_*{chunk_suffix}"))
    if chunk_error or result.returncode != 0 or not chunks:
        return [], f"audio_chunking_failed: {chunk_error or (result.stderr or result.stdout or 'ffmpeg failed')[-700:]}"

    oversized = [chunk.name for chunk in chunks if chunk.stat().st_size > OPENAI_AUDIO_FILE_LIMIT_BYTES]
    if oversized:
        return [], f"audio_chunking_failed: chunk over OpenAI 25 MB limit ({', '.join(oversized[:3])})"
    return chunks, None


def transcribe_audio_file(job_id, media_file, audio_path, model, ffmpeg):
    chunks, split_error = split_audio_for_transcription(job_id, audio_path, ffmpeg)
    if split_error:
        return None, split_error

    combined_text = []
    combined_segments = []
    combined_words = []
    chunk_records = []
    ffprobe = shutil.which("ffprobe")
    running_offset = 0.0
    requested_model = model or TIMESTAMPED_TRANSCRIPTION_MODEL
    backend = resolve_transcription_backend()
    effective_model = effective_timestamped_transcription_model(requested_model) if backend.get("backend") == "openai" else Path(backend.get("model_path") or "").name
    if len(chunks) > 1:
        with STATE_LOCK:
            job = JOBS[job_id]
            add_log(job, f"Audio split into {len(chunks)} transcription chunks for {media_file.get('name')}.")
    if backend.get("backend") == "local_whisper":
        if not backend.get("ready"):
            return None, "local_whisper_not_configured: set whisper-cli path and ggml model path in Settings"
        with STATE_LOCK:
            job = JOBS[job_id]
            add_log(job, f"Using local whisper.cpp model {effective_model} for timestamped transcription.")
    elif effective_model != requested_model:
        with STATE_LOCK:
            job = JOBS[job_id]
            add_log(job, f"Using {effective_model} for word timestamps; requested model was {requested_model}.")
    elif backend.get("requested") == "auto":
        with STATE_LOCK:
            job = JOBS[job_id]
            add_log(job, "Local Whisper model not configured; Auto transcription is using OpenAI.")

    for index, chunk_path in enumerate(chunks, start=1):
        chunk_offset = running_offset
        chunk_duration = media_duration_seconds(chunk_path, ffprobe)
        with STATE_LOCK:
            job = JOBS[job_id]
            percent = min(95, 18 + round((index - 1) / max(1, len(chunks)) * 72))
            note = f"Transcribing chunk {index}/{len(chunks)}"
            set_stage(job, "transcription", "running", percent, note)
            add_log(job, f"{note} for {media_file.get('name')} ({human_size(chunk_path.stat().st_size)}, offset {chunk_offset:.2f}s).")
        if backend.get("backend") == "local_whisper":
            transcription, error = transcribe_with_local_whisper(chunk_path, ffmpeg)
        else:
            transcription, error = transcribe_with_openai(chunk_path, model)
        if error:
            return None, f"{error} on chunk {index}/{len(chunks)}"

        text = (transcription or {}).get("text", "").strip()
        if text:
            combined_text.append(text)
        segments = offset_timed_records((transcription or {}).get("segments", []), chunk_offset)
        words = offset_timed_records((transcription or {}).get("words", []), chunk_offset)
        combined_segments.extend(segments)
        combined_words.extend(words)
        provider_duration = timed_seconds((transcription or {}).get("duration"))
        duration_for_offset = chunk_duration or provider_duration or TRANSCRIPTION_CHUNK_SECONDS
        chunk_records.append(
            {
                "name": chunk_path.name,
                "size": human_size(chunk_path.stat().st_size),
                "offset_seconds": round(chunk_offset, 2),
                "duration_seconds": round(duration_for_offset, 2),
                "segment_count": len(segments),
                "word_count": len(words),
                "backend": backend.get("backend"),
                "text": text,
                "raw_response": transcription,
            }
        )
        with STATE_LOCK:
            job = JOBS[job_id]
            add_log(job, f"Captured {len(words)} timed words and {len(segments)} timed segments for chunk {index}/{len(chunks)}.")
            if (transcription or {}).get("_warning"):
                add_log(job, f"Local Whisper warning: {(transcription or {}).get('_warning')}.")
        running_offset += duration_for_offset

    return (
        {
            "text": "\n\n".join(combined_text),
            "segments": combined_segments,
            "words": combined_words,
            "chunks": chunk_records,
            "timing": {
                "requested_backend": backend.get("requested"),
                "backend": backend.get("backend"),
                "requested_model": requested_model,
                "effective_model": effective_model,
                "granularity": "word+segment",
                "chunk_count": len(chunks),
                "duration_seconds": round(running_offset, 2),
            },
            "raw_response": {"chunks": chunk_records},
        },
        None,
    )


def run_upload_check(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, "upload_check", "running", 5, "Classifying uploaded sources")
        job["status"] = "Processing"
        add_log(job, "Upload check started.")
        paths = ensure_job_dirs(job)
        job["storage_root"] = str(paths["root"].relative_to(APP_DIR))

    files_seen = 0
    readable = 0
    kinds = {}
    with STATE_LOCK:
        job = JOBS[job_id]
        normalize_job_sources(job)
        for file in active_job_files(job):
            files_seen += 1
            kind = file_kind(file)
            file["kind"] = kind
            kinds[kind] = kinds.get(kind, 0) + 1
            path = stored_path(file)
            file["readable"] = bool(path) or file.get("source") in {None, "seeded"}
            file["processing_status"] = "ready" if file["readable"] else "missing source file"
            if file["readable"]:
                readable += 1
        note = f"{readable}/{files_seen} readable · " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        set_stage(job, "upload_check", "done", 100, note if files_seen else "No files attached")
        add_log(job, f"Upload check complete: {note if files_seen else 'no files attached'}.")
    return True


def write_transcript_file(job, media_file, audio_record=None, status="pending_provider", transcription=None, error=None):
    paths = ensure_job_dirs(job)
    base = media_file.get("sha256") or safe_filename(media_file.get("name")).rsplit(".", 1)[0]
    transcript_path = paths["transcripts"] / f"{base[:24]}_transcript.json"
    raw_response = None
    if transcription:
        raw_response = transcription.get("raw_response", transcription)
    payload = {
        "source_file": media_file.get("name"),
        "source_sha256": media_file.get("sha256"),
        "audio_file": audio_record.get("stored_path") if audio_record else None,
        "status": status,
        "text": (transcription or {}).get("text", ""),
        "segments": (transcription or {}).get("segments", []),
        "words": (transcription or {}).get("words", []),
        "chunks": (transcription or {}).get("chunks", []),
        "timing": (transcription or {}).get("timing", {}),
        "timeline_offset_seconds": float(media_file.get("timeline_offset_seconds") or 0),
        "timeline_scale": float(media_file.get("timeline_scale") or 1),
        "timeline_sync": media_file.get("timeline_sync") or {},
        "timeline_alignment": media_file.get("timeline_alignment") or {},
        "raw_response": raw_response,
        "error": error,
        "note": "Audio extraction is complete. Transcription provider will fill this file later." if status == "pending_provider" else "",
    }
    transcript_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    media_file["transcript_path"] = str(transcript_path.relative_to(APP_DIR))
    media_file["transcript_status"] = status


def transcript_file_has_word_timing(path):
    if not path or not Path(path).exists():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    words = payload.get("words") or []
    if not words:
        return False
    for word in words[:50]:
        if isinstance(word, dict) and timed_seconds(word.get("start")) is not None and timed_seconds(word.get("end")) is not None:
            return True
    return False


def completed_transcript_path(media_file):
    if media_file.get("transcript_status") != "complete":
        return None
    path = stored_relative_path(media_file.get("transcript_path"))
    if not transcript_file_has_word_timing(path):
        media_file["transcript_status"] = "needs_timestamp_refresh"
        media_file["processing_status"] = "transcript needs timestamp refresh"
        return None
    return path


def recover_failed_local_whisper_transcript(job, media_file):
    failed_transcript_path = stored_relative_path(media_file.get("transcript_path"))
    if not failed_transcript_path:
        return False
    try:
        failed_payload = json.loads(failed_transcript_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if failed_payload.get("status") != "transcription_failed":
        return False
    raw_json_path = local_whisper_json_from_error(failed_payload.get("error"))
    if not raw_json_path:
        return False
    try:
        raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    transcription = normalize_local_whisper_json(raw)
    if not transcription.get("text") or not transcription.get("words"):
        return False
    _, audio_record = cached_audio_record(media_file)
    transcription["chunks"] = [
        {
            "name": raw_json_path.name,
            "size": human_size(raw_json_path.stat().st_size),
            "offset_seconds": 0,
            "duration_seconds": round(timed_seconds(transcription.get("duration"), 0) or 0, 2),
            "segment_count": len(transcription.get("segments") or []),
            "word_count": len(transcription.get("words") or []),
            "backend": "local_whisper",
            "text": transcription.get("text", ""),
            "raw_response": raw,
            "recovered_from_failed_run": True,
        }
    ]
    transcription["timing"] = {
        "requested_backend": SETTINGS.get("transcription_backend") or "auto",
        "backend": "local_whisper",
        "requested_model": SETTINGS.get("local_whisper_model_path") or "",
        "effective_model": Path(SETTINGS.get("local_whisper_model_path") or "").name,
        "granularity": "word+segment",
        "chunk_count": 1,
        "duration_seconds": round(timed_seconds(transcription.get("duration"), 0) or 0, 2),
        "recovered_from_failed_run": True,
    }
    write_transcript_file(job, media_file, audio_record, status="complete", transcription=transcription)
    media_file["processing_status"] = "transcribed"
    if audio_record:
        audio_record["processing_status"] = "transcribed"
        upsert_artifact(job, audio_record)
    return True


def cached_audio_record(media_file):
    audio_path = stored_relative_path(media_file.get("audio_path"))
    if not audio_path:
        return None, None
    record = {
        "name": audio_path.name,
        "type": mimetypes.guess_type(audio_path.name)[0] or "audio/mp4",
        "kind": "audio",
        "size": human_size(audio_path.stat().st_size),
        "bytes": audio_path.stat().st_size,
        "stored_path": str(audio_path.relative_to(APP_DIR)),
        "source": "extracted audio",
        "source_file": media_file.get("name"),
        "processing_status": "ready for transcription",
    }
    if media_file.get("transcript_status") == "complete":
        record["processing_status"] = "transcribed"
    return audio_path, record


def transcription_provider_label():
    backend = resolve_transcription_backend()
    if backend.get("backend") == "local_whisper":
        model = Path(backend.get("model_path") or "").name or "model not configured"
        return f"local whisper.cpp ({model})"
    return f"OpenAI {effective_timestamped_transcription_model(SETTINGS.get('transcription_model') or TIMESTAMPED_TRANSCRIPTION_MODEL)}"


def payload_word_tokens(payload, max_words=2500):
    words = []
    for word in payload.get("words") or []:
        if not isinstance(word, dict):
            continue
        token = timed_word_token(word.get("word") or word.get("text"))
        start = timed_seconds(word.get("start"))
        if token and start is not None:
            words.append({"token": token, "start": start})
        if len(words) >= max_words:
            break
    return words


def estimate_transcript_offset(reference_payload, candidate_payload):
    reference_words = payload_word_tokens(reference_payload)
    candidate_words = payload_word_tokens(candidate_payload)
    if len(reference_words) < 20 or len(candidate_words) < 20:
        return None

    shingle_size = 5
    reference_index = {}
    for index in range(0, len(reference_words) - shingle_size + 1):
        key = tuple(word["token"] for word in reference_words[index:index + shingle_size])
        if len(set(key)) < 3:
            continue
        starts = reference_index.setdefault(key, [])
        if len(starts) < 4:
            starts.append(reference_words[index]["start"])

    offsets = []
    for index in range(0, len(candidate_words) - shingle_size + 1):
        key = tuple(word["token"] for word in candidate_words[index:index + shingle_size])
        if len(set(key)) < 3:
            continue
        for reference_start in reference_index.get(key, []):
            offsets.append(reference_start - candidate_words[index]["start"])
        if len(offsets) >= 240:
            break

    if len(offsets) < 4:
        return None

    offset = statistics.median(offsets)
    deviations = [abs(item - offset) for item in offsets]
    median_deviation = statistics.median(deviations) if deviations else 0
    confidence = "high" if len(offsets) >= 12 and median_deviation <= 0.75 else "medium"
    if median_deviation > 2.0:
        confidence = "low"
    return {
        "offset_seconds": round(offset, 2),
        "match_count": len(offsets),
        "median_deviation_seconds": round(median_deviation, 2),
        "confidence": confidence,
    }


def alignment_audio_path(file):
    audio_rel = file.get("audio_path")
    if audio_rel:
        audio_path = APP_DIR / audio_rel
        if audio_path.exists():
            return audio_path
    source_path = stored_path(file)
    if source_path and source_path.exists() and file_kind(file) in TIMED_SOURCE_KINDS:
        return source_path
    return None


def align_transcript_timelines(job):
    """Align timed sources, preferring audio waveform evidence over text."""
    normalize_job_sources(job)
    sources = [
        (file, transcript_payload_for_file(file))
        for file in active_job_files(job)
        if file.get("kind") in TIMED_SOURCE_KINDS
    ]

    if not sources:
        return []

    reference_file, reference_payload = next(
        ((file, payload) for file, payload in sources if file.get("timeline_reference")),
        sources[0],
    )
    reference_file["timeline_offset_seconds"] = 0
    reference_file["timeline_scale"] = 1
    reference_file["timeline_sync"]["mode"] = "reference"
    reference_file["timeline_alignment"] = {
        "method": "reference_source",
        "confidence": "reference",
        "reference_file": reference_file.get("name"),
    }

    alignments = []
    for file, payload in sources:
        if file is reference_file:
            continue
        manual = manual_timeline_transform(file)
        if manual:
            file["timeline_offset_seconds"] = round(manual["offset_seconds"], 6)
            file["timeline_scale"] = round(manual["scale"], 9)
            file["timeline_alignment"] = {
                "method": "manual_anchors",
                "reference_file": reference_file.get("name"),
                "confidence": "manual",
                "offset_seconds": round(manual["offset_seconds"], 6),
                "scale": round(manual["scale"], 9),
                "has_drift_anchor": manual["has_drift_anchor"],
            }
            alignments.append({
                "source_file": file.get("name"),
                **file["timeline_alignment"],
                "match_count": 0,
            })
            continue

        waveform_estimate = None
        waveform_error = ""
        reference_audio = alignment_audio_path(reference_file)
        candidate_audio = alignment_audio_path(file)
        if waveform_backend_available() and reference_audio and candidate_audio:
            try:
                waveform_estimate = estimate_waveform_alignment(reference_audio, candidate_audio)
            except Exception as exc:
                waveform_error = str(exc)[:500]
        if waveform_estimate and waveform_estimate.get("confidence") in {"high", "medium"}:
            file["timeline_offset_seconds"] = waveform_estimate["offset_seconds"]
            file["timeline_scale"] = waveform_estimate.get("scale") or 1
            file["timeline_alignment"] = {
                "reference_file": reference_file.get("name"),
                **waveform_estimate,
            }
            alignments.append({"source_file": file.get("name"), **file["timeline_alignment"]})
            continue

        transcript_estimate = None
        if reference_payload and payload:
            transcript_estimate = estimate_transcript_offset(reference_payload, payload)
        if transcript_estimate:
            file["timeline_offset_seconds"] = transcript_estimate["offset_seconds"]
            file["timeline_scale"] = 1
            file["timeline_alignment"] = {
                "method": "transcript_word_match",
                "reference_file": reference_file.get("name"),
                **transcript_estimate,
            }
            alignments.append({"source_file": file.get("name"), **file["timeline_alignment"]})
            continue

        reasons = []
        if waveform_estimate:
            reasons.append(
                f"waveform confidence was low ({waveform_estimate.get('correlation', 0)} correlation)"
            )
        elif waveform_error:
            reasons.append(f"waveform analysis failed: {waveform_error}")
        elif not waveform_backend_available():
            reasons.append("waveform backend is unavailable")
        else:
            reasons.append("shared source audio is unavailable")
        reasons.append("not enough shared timed words")
        file["timeline_offset_seconds"] = float(file.get("timeline_offset_seconds") or 0)
        file["timeline_scale"] = float(file.get("timeline_scale") or 1)
        file["timeline_alignment"] = {
            "method": "not_aligned",
            "reference_file": reference_file.get("name"),
            "confidence": "low",
            "reason": "; ".join(reasons) + ".",
            "waveform_candidate": waveform_estimate or {},
        }
    return alignments


def run_transcription_stage(job_id):
    ffmpeg = shutil.which("ffmpeg")
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, "transcription", "running", 5, "Preparing audio extraction")
        add_log(job, "Transcript stage started.")

    if not ffmpeg:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "transcription", "failed", 0, "ffmpeg not installed")
            job["flags"].append("ffmpeg missing")
            add_log(job, "Transcript stage failed: ffmpeg not installed.")
        return False

    with STATE_LOCK:
        job = JOBS[job_id]
        normalize_job_sources(job)
        media_files = [
            file
            for file in active_job_files(job)
            if file.get("kind") in {"video", "audio"} and stored_path(file)
        ]

    if not media_files:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "transcription", "done", 100, "No readable audio/video sources")
        add_log(job, "Transcript stage complete: no readable audio/video sources.")
        return True

    extracted = 0
    transcribed = 0
    reused_transcripts = 0
    reused_audio = 0
    extraction_failed = 0
    transcription_failed = 0
    pending_provider = 0
    required_extraction_failed = 0
    required_transcription_failed = 0
    required_pending_provider = 0
    aligned_sources = 0
    for index, file in enumerate(media_files, start=1):
        input_path = stored_path(file)
        digest = file.get("sha256") or hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()
        output_path = ensure_job_dirs(JOBS[job_id])["audio"] / f"{digest[:12]}_{safe_filename(file.get('name')).rsplit('.', 1)[0]}.m4a"
        transcript_path = completed_transcript_path(file)
        if transcript_path:
            with STATE_LOCK:
                job = JOBS[job_id]
                percent = 10 + round((index - 1) / max(1, len(media_files)) * 80)
                set_stage(job, "transcription", "running", percent, f"Reusing transcript {index}/{len(media_files)}")
                _, audio_record = cached_audio_record(file)
                if audio_record:
                    upsert_artifact(job, audio_record)
                file["processing_status"] = "transcribed"
                reused_transcripts += 1
                add_log(job, f"Reused completed transcript for {file.get('name')}.")
            continue

        with STATE_LOCK:
            job = JOBS[job_id]
            recovered_transcript = recover_failed_local_whisper_transcript(job, file)
            if recovered_transcript:
                percent = 10 + round((index - 1) / max(1, len(media_files)) * 80)
                set_stage(job, "transcription", "running", percent, f"Recovered transcript {index}/{len(media_files)}")
                reused_transcripts += 1
                add_log(job, f"Recovered completed local Whisper transcript for {file.get('name')} from saved JSON output.")
        if recovered_transcript:
            continue

        audio_path, audio_record = cached_audio_record(file)
        should_transcribe = False
        transcription_model = TIMESTAMPED_TRANSCRIPTION_MODEL
        if audio_path and audio_record:
            with STATE_LOCK:
                job = JOBS[job_id]
                percent = 10 + round((index - 1) / max(1, len(media_files)) * 80)
                set_stage(job, "transcription", "running", percent, f"Reusing audio {index}/{len(media_files)}")
                file["audio_path"] = audio_record["stored_path"]
                file["processing_status"] = "audio extracted"
                audio_record = upsert_artifact(job, audio_record)
                reused_audio += 1
                add_log(job, f"Reused extracted audio for {file.get('name')}: {audio_path.name}.")
                if SETTINGS.get("transcription_enabled", True):
                    should_transcribe = True
                    transcription_model = SETTINGS.get("transcription_model") or TIMESTAMPED_TRANSCRIPTION_MODEL
                    set_stage(job, "transcription", "running", min(95, percent + 8), f"Transcribing {index}/{len(media_files)}")
                    add_log(job, f"Sending audio to {transcription_provider_label()} for timestamped transcription.")
                else:
                    write_transcript_file(job, file, audio_record, status="disabled")
                    file["processing_status"] = "audio extracted, transcription disabled"
                    add_log(job, "Transcript provider skipped: transcription is disabled in Settings.")
        else:
            audio_path = output_path
            audio_record = None
            with STATE_LOCK:
                job = JOBS[job_id]
                percent = 10 + round((index - 1) / max(1, len(media_files)) * 80)
                set_stage(job, "transcription", "running", percent, f"Extracting audio {index}/{len(media_files)}")
                add_log(job, f"Extracting audio from {file.get('name')}.")
            cmd = [
                ffmpeg,
                "-nostdin",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "aac",
                "-b:a",
                TRANSCRIPTION_AUDIO_BITRATE,
                str(output_path),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=AUDIO_EXTRACTION_TIMEOUT_SECONDS,
                )
                extraction_error = None
            except subprocess.TimeoutExpired:
                result = None
                extraction_error = f"audio extraction timed out after {AUDIO_EXTRACTION_TIMEOUT_SECONDS}s"
            with STATE_LOCK:
                job = JOBS[job_id]
                if not extraction_error and result.returncode == 0 and output_path.exists():
                    audio_record = {
                        "name": output_path.name,
                        "type": mimetypes.guess_type(output_path.name)[0] or "audio/mp4",
                        "kind": "audio",
                        "size": human_size(output_path.stat().st_size),
                        "bytes": output_path.stat().st_size,
                        "stored_path": str(output_path.relative_to(APP_DIR)),
                        "source": "extracted audio",
                        "source_file": file.get("name"),
                        "processing_status": "ready for transcription",
                    }
                    file["audio_path"] = audio_record["stored_path"]
                    file["processing_status"] = "audio extracted"
                    audio_record = upsert_artifact(job, audio_record)
                    extracted += 1
                    add_log(job, f"Audio extracted: {output_path.name}.")
                    if SETTINGS.get("transcription_enabled", True):
                        should_transcribe = True
                        transcription_model = SETTINGS.get("transcription_model") or TIMESTAMPED_TRANSCRIPTION_MODEL
                        set_stage(job, "transcription", "running", min(95, percent + 8), f"Transcribing {index}/{len(media_files)}")
                        add_log(job, f"Sending audio to {transcription_provider_label()} for timestamped transcription.")
                    else:
                        write_transcript_file(job, file, audio_record, status="disabled")
                        file["processing_status"] = "audio extracted, transcription disabled"
                        add_log(job, "Transcript provider skipped: transcription is disabled in Settings.")
                else:
                    extraction_failed += 1
                    if file.get("use_for_drafting"):
                        required_extraction_failed += 1
                    file["processing_status"] = "audio extraction failed"
                    file["audio_error"] = extraction_error or (result.stderr or result.stdout or "ffmpeg failed")[-1000:]
                    add_log(job, f"Audio extraction failed for {file.get('name')}: {file['audio_error']}")
        if should_transcribe and audio_record:
            transcription, error = transcribe_audio_file(job_id, file, audio_path, transcription_model, ffmpeg)
            with STATE_LOCK:
                job = JOBS[job_id]
                if transcription:
                    write_transcript_file(job, file, audio_record, status="complete", transcription=transcription)
                    file["processing_status"] = "transcribed"
                    audio_record["processing_status"] = "transcribed"
                    transcribed += 1
                    add_log(job, f"Transcript complete for {file.get('name')}.")
                else:
                    status = "pending_provider" if error == "missing_api_key" else "transcription_failed"
                    write_transcript_file(job, file, audio_record, status=status, error=error)
                    file["processing_status"] = "audio extracted, transcript pending" if error == "missing_api_key" else "transcription failed"
                    audio_record["processing_status"] = file["processing_status"]
                    if error == "missing_api_key":
                        pending_provider += 1
                        if file.get("use_for_drafting"):
                            required_pending_provider += 1
                        if "OpenAI key missing" not in job["flags"]:
                            job["flags"].append("OpenAI key missing")
                        add_log(job, "Transcript provider skipped: OpenAI API key is not set.")
                    else:
                        transcription_failed += 1
                        if file.get("use_for_drafting"):
                            required_transcription_failed += 1
                        if "Transcription issue" not in job["flags"]:
                            job["flags"].append("Transcription issue")
                        add_log(job, f"Transcript provider failed for {file.get('name')}: {error}")
    with STATE_LOCK:
        job = JOBS[job_id]
        alignments = align_transcript_timelines(job)
        aligned_sources = len(alignments)
        if alignments:
            for alignment in alignments:
                if alignment.get("method") == "manual_anchors":
                    drift_note = " with drift correction" if alignment.get("has_drift_anchor") else ""
                    add_log(
                        job,
                        f"Timeline aligned {alignment.get('source_file')} from manual anchors at {alignment.get('offset_seconds')}s offset{drift_note}.",
                    )
                elif alignment.get("method") == "audio_waveform":
                    add_log(
                        job,
                        (
                            f"Timeline aligned {alignment.get('source_file')} from shared audio at "
                            f"{alignment.get('offset_seconds')}s offset ({alignment.get('confidence')} confidence, "
                            f"{alignment.get('correlation')} correlation)."
                        ),
                    )
                else:
                    add_log(
                        job,
                        f"Timeline aligned {alignment.get('source_file')} at {alignment.get('offset_seconds')}s offset ({alignment.get('confidence')} confidence, {alignment.get('match_count')} word matches).",
                    )
    with STATE_LOCK:
        job = JOBS[job_id]
        note = f"{extracted} audio files extracted"
        if reused_audio:
            note += f" · {reused_audio} audio reused"
        if reused_transcripts:
            note += f" · {reused_transcripts} transcript reused"
        if transcribed:
            note += f" · {transcribed} transcribed"
        if aligned_sources:
            note += f" · {aligned_sources} source timelines aligned"
        if pending_provider:
            note += f" · {pending_provider} pending provider"
        if transcription_failed:
            note += f" · {transcription_failed} transcription failed"
        if extraction_failed:
            note += f" · {extraction_failed} extraction failed"
            if "Audio extraction issue" not in job["flags"]:
                job["flags"].append("Audio extraction issue")
        cache = job.setdefault("processing_cache", {})
        cache["transcription"] = {
            "audio_reused": reused_audio,
            "transcripts_reused": reused_transcripts,
            "audio_extracted": extracted,
            "transcribed": transcribed,
            "updated_at": datestamp(),
        }
        status = "done"
        if required_extraction_failed or required_transcription_failed:
            status = "failed"
        elif required_pending_provider:
            status = "blocked"
        set_stage(job, "transcription", status, 100 if not required_extraction_failed else 75, note)
        add_log(job, f"Transcript stage complete: {note}.")
    return not required_extraction_failed and not required_transcription_failed and not required_pending_provider


def base_stages():
    return {
        key: {
            "key": key,
            "label": label,
            "status": "idle",
            "percent": 0,
            "note": "",
            "started_at": "",
            "started_epoch": None,
            "ended_at": "",
            "ended_epoch": None,
            "duration_seconds": None,
        }
        for key, label in STAGES
    }


def add_log(job, message):
    now = epoch_now()
    job["logs"].append({"time": timestamp(), "at": datestamp(), "epoch": now, "message": message})
    job["last_activity"] = timestamp()
    job["last_activity_epoch"] = now
    if len(job["logs"]) > 120:
        job["logs"] = job["logs"][-120:]
    save_state()


def recompute_progress(job):
    total = sum(stage["percent"] for stage in job["stages"].values())
    job["progress"] = round(total / len(STAGES))


def set_stage(job, key, status, percent=None, note=None):
    stage = job["stages"][key]
    now = epoch_now()
    previous_status = stage.get("status")
    stage["status"] = status
    stage["updated_at"] = datestamp()
    stage["updated_epoch"] = now
    if status == "running" and (previous_status != "running" or not stage.get("started_epoch")):
        stage["started_at"] = datestamp()
        stage["started_epoch"] = now
        stage["ended_at"] = ""
        stage["ended_epoch"] = None
        stage["duration_seconds"] = None
    if status in {"done", "failed", "blocked"}:
        stage.setdefault("started_at", stage.get("updated_at") or datestamp())
        stage.setdefault("started_epoch", stage.get("updated_epoch") or now)
        stage["ended_at"] = datestamp()
        stage["ended_epoch"] = now
        started = float(stage.get("started_epoch") or now)
        stage["duration_seconds"] = round(max(0, now - started), 2)
    if percent is not None:
        stage["percent"] = percent
    if note is not None:
        stage["note"] = note
    recompute_progress(job)
    job["last_activity"] = timestamp()
    save_state()


def start_job_processing_timing(job):
    now = epoch_now()
    job["processing_timing"] = {
        "started_at": datestamp(),
        "started_epoch": now,
        "ended_at": "",
        "ended_epoch": None,
        "duration_seconds": None,
        "profile": processing_profile_key(job),
    }
    normalize_job_sources(job)
    run = {
        "manifest_version": PROCESSING_MANIFEST_VERSION,
        "run_id": short_id("run"),
        "status": "running",
        "started_at": datestamp(),
        "started_epoch": now,
        "ended_at": "",
        "ended_epoch": None,
        "duration_seconds": None,
        "engine": {
            "version": ENGINE_VERSION,
            "profile_id": "assembly",
            "profile_version": ASSEMBLY_PROFILE_VERSION,
            "prompt_version": INSTRUCTION_PROMPT_VERSION,
        },
        "configuration": {
            "processing_profile": processing_profile_key(job),
            "expected_step_count": job.get("expected_step_count"),
            "focus_area": job.get("focus_area") or "center",
            "transcription_backend": SETTINGS.get("transcription_backend") or "auto",
            "transcription_model": SETTINGS.get("transcription_model") or TIMESTAMPED_TRANSCRIPTION_MODEL,
            "instruction_model": SETTINGS.get("instruction_model") or "gpt-4o-mini",
            "transcription_enabled": bool(SETTINGS.get("transcription_enabled", True)),
        },
        "sources": [
            {
                "source_id": file.get("source_id") or "",
                "name": file.get("name") or "",
                "kind": file.get("kind") or file_kind(file),
                "sha256": file.get("sha256") or "",
                "bytes": int(file.get("bytes") or 0),
                "source_role": file.get("source_role") or "",
                "authority_level": file.get("authority_level") or "",
                "revision": file.get("revision") or "",
                "applicability": file.get("applicability") or "",
                "use_for_drafting": bool(file.get("use_for_drafting")),
                "timeline_reference": bool(file.get("timeline_reference")),
            }
            for file in active_job_files(job)
        ],
        "stages": {},
        "metrics": {},
        "failures": [],
    }
    runs = job.setdefault("processing_runs", [])
    runs.append(run)
    if len(runs) > 20:
        del runs[:-20]
    job["active_processing_run_id"] = run["run_id"]


def active_processing_manifest(job):
    run_id = job.get("active_processing_run_id")
    return next(
        (run for run in reversed(job.get("processing_runs") or []) if run.get("run_id") == run_id),
        None,
    )


def refresh_processing_manifest(job, status=None, outcome=""):
    run = active_processing_manifest(job)
    if not run:
        return None
    run["status"] = status or run.get("status") or "running"
    if outcome:
        run["outcome"] = outcome
    run["stages"] = {
        key: {
            "status": stage.get("status") or "idle",
            "percent": stage.get("percent") or 0,
            "note": stage.get("note") or "",
            "duration_seconds": stage.get("duration_seconds"),
        }
        for key, stage in (job.get("stages") or {}).items()
    }
    run["source_alignments"] = [
        {
            "source_id": file.get("source_id") or "",
            "name": file.get("name") or "",
            "offset_seconds": float(file.get("timeline_offset_seconds") or 0),
            "scale": float(file.get("timeline_scale") or 1),
            "alignment": file.get("timeline_alignment") or {},
        }
        for file in active_job_files(job)
        if file.get("kind") in TIMED_SOURCE_KINDS
    ]
    run["metrics"] = {
        "files": len(active_job_files(job)),
        "document_evidence": len(job.get("document_evidence") or []),
        "transcript_sources": len(transcript_results(job)),
        "timed_words": sum(len(item.get("words") or []) for item in transcript_results(job)),
        "raw_frames": sum(len(file.get("frame_paths") or []) for file in active_job_files(job)),
        "scored_frames": len(job.get("frame_analysis") or []),
        "sections": len(job.get("draft_sections") or []),
        "steps": sum(len(section.get("steps") or []) for section in job.get("draft_sections") or []),
        "questions": len(job.get("questions") or []),
        "qa_issues": int((job.get("draft_validation") or {}).get("issue_count") or 0),
        "cache": json.loads(json.dumps(job.get("processing_cache") or {})),
    }
    run["failures"] = [
        {
            "stage": key,
            "status": stage.get("status"),
            "note": stage.get("note") or "",
        }
        for key, stage in (job.get("stages") or {}).items()
        if stage.get("status") in {"failed", "blocked"}
    ]
    return run


def finish_job_processing_timing(job, status="complete", outcome=""):
    timing = job.setdefault("processing_timing", {})
    now = epoch_now()
    started = float(timing.get("started_epoch") or now)
    timing["ended_at"] = datestamp()
    timing["ended_epoch"] = now
    timing["duration_seconds"] = round(max(0, now - started), 2)
    timing["profile"] = processing_profile_key(job)
    run = refresh_processing_manifest(job, status=status, outcome=outcome)
    if run:
        run["ended_at"] = datestamp()
        run["ended_epoch"] = now
        run_started = float(run.get("started_epoch") or now)
        run["duration_seconds"] = round(max(0, now - run_started), 2)
        job["active_processing_run_id"] = ""
    save_state()


def unresolved_questions(job):
    return [q for q in job["questions"] if q["status"] == "pending"]


def set_question_flag(job, pending_count):
    flags = [
        flag
        for flag in job.get("flags", [])
        if "questions pending" not in flag
    ]
    if pending_count:
        flags.append(f"{pending_count} questions pending")
    job["flags"] = flags


def find_question(question_id):
    for job in JOBS.values():
        for question in job.get("questions", []):
            if question["id"] == question_id:
                return job, question
    return None, None


def transcript_payload_for_file(file):
    transcript_rel = file.get("transcript_path")
    if not transcript_rel:
        return None
    path = APP_DIR / transcript_rel
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    data["_path"] = transcript_rel
    data["_source_file"] = file.get("name") or data.get("source_file")
    data["_timeline_offset_seconds"] = float(file.get("timeline_offset_seconds") or data.get("timeline_offset_seconds") or 0)
    data["_timeline_scale"] = float(file.get("timeline_scale") or data.get("timeline_scale") or 1)
    data["_timeline_sync"] = file.get("timeline_sync") or data.get("timeline_sync") or {}
    data["_timeline_alignment"] = file.get("timeline_alignment") or data.get("timeline_alignment") or {}
    data["_timeline_reference"] = bool(file.get("timeline_reference"))
    data["_use_for_drafting"] = bool(file.get("use_for_drafting"))
    data["_source_id"] = file.get("source_id") or ""
    return data


def transcript_payloads(job):
    normalize_job_sources(job)
    payloads = []
    for file in active_job_files(job):
        payload = transcript_payload_for_file(file)
        if payload:
            payloads.append(payload)
    return payloads


def drafting_transcript_payloads(job):
    payloads = [payload for payload in transcript_payloads(job) if payload.get("_use_for_drafting")]
    return payloads or transcript_payloads(job)[:1]


def transcript_text(job):
    texts = []
    for payload in drafting_transcript_payloads(job):
        text = (payload.get("text") or "").strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def transcript_sentence_spans(text):
    spans = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n+|$)", text or ""):
        value = re.sub(r"\s+", " ", match.group(0)).strip(" .!?\n\t")
        if value:
            spans.append(
                {
                    "text": value,
                    "char_start": match.start(),
                    "char_end": match.end(),
                }
            )
    if spans:
        return spans
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    for start in range(0, len(cleaned), 180):
        chunk = cleaned[start:start + 220].strip()
        if chunk:
            spans.append({"text": chunk, "char_start": start, "char_end": min(len(cleaned), start + len(chunk))})
    return spans


def timed_word_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def transcript_timed_words(job):
    words = []
    for payload in drafting_transcript_payloads(job):
        source_file = payload.get("_source_file") or payload.get("source_file")
        for word in payload.get("words") or []:
            if not isinstance(word, dict):
                continue
            token = timed_word_token(word.get("word") or word.get("text"))
            if not token:
                continue
            start = timed_seconds(word.get("start"))
            end = timed_seconds(word.get("end"), start)
            if start is None:
                continue
            start = apply_source_timeline(start, payload)
            end = max(start, apply_source_timeline(end if end is not None else start, payload))
            words.append(
                {
                    "word": word.get("word") or word.get("text") or token,
                    "token": token,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "source_file": source_file,
                }
            )
    return sorted(words, key=lambda item: item["start"])


def timed_words_text(words):
    text_parts = []
    spans = []
    cursor = 0
    for index, word in enumerate(words):
        value = re.sub(r"\s+", " ", str(word.get("word") or "")).strip()
        if not value:
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(value)
        cursor += len(value)
        spans.append({"index": index, "char_start": start, "char_end": cursor})
    return "".join(text_parts), spans


def word_index_for_char(spans, char_position):
    for span in spans:
        if span["char_start"] <= char_position < span["char_end"]:
            return span["index"]
    if not spans:
        return None
    closest = min(spans, key=lambda span: abs(span["char_start"] - char_position))
    return closest["index"]


STEP_BOUNDARY_RE = re.compile(
    r"\b(new step|next step|start step|step complete|moving on|step\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))\b",
    flags=re.IGNORECASE,
)
SOFT_STEP_BOUNDARY_RE = re.compile(r"\b(after that|from there|next we|then we|now we|next up)\b", flags=re.IGNORECASE)
OPERATOR_IMAGE_CUE_RE = re.compile(
    r"\b("
    r"take (?:a )?(?:picture|photo|shot)(?: here)?|"
    r"good (?:image|picture|photo|shot)(?: here)?|"
    r"use this (?:image|picture|photo|shot|frame)|"
    r"show this|"
    r"capture this|"
    r"close[- ]?up|"
    r"this is the (?:image|picture|photo|shot|frame)"
    r")\b",
    flags=re.IGNORECASE,
)


def expected_step_count_for(job):
    return parse_expected_step_count(job.get("expected_step_count"))


def dedupe_step_boundaries(candidates, words):
    candidates = sorted(candidates, key=lambda item: (-item["score"], item["index"]))
    selected = []
    for candidate in candidates:
        index = int(candidate.get("index") or 0)
        if index <= 0 or index >= len(words):
            continue
        too_close = False
        for existing in selected:
            time_delta = abs(float(candidate.get("time") or 0) - float(existing.get("time") or 0))
            if abs(index - existing["index"]) < 5 or time_delta < 1.2:
                too_close = True
                break
        if not too_close:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["index"])


def proportional_step_boundary_indexes(word_count, target_step_count):
    if not target_step_count or target_step_count <= 1 or word_count <= 8:
        return []
    return [
        max(1, min(word_count - 1, round((index / target_step_count) * word_count)))
        for index in range(1, target_step_count)
    ]


def select_step_boundary_indexes(candidates, words, target_step_count=None):
    target_internal = max(0, int(target_step_count or 0) - 1)
    if target_internal:
        selected = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            index = candidate["index"]
            if all(abs(index - existing) >= 8 for existing in selected):
                selected.append(index)
            if len(selected) >= target_internal:
                break
        for index in proportional_step_boundary_indexes(len(words), target_step_count):
            if len(selected) >= target_internal:
                break
            if all(abs(index - existing) >= 8 for existing in selected):
                selected.append(index)
        return sorted(set(selected))

    selected = [
        candidate["index"]
        for candidate in candidates
        if candidate.get("explicit") or candidate.get("score", 0) >= 58
    ]
    if len(selected) > 60:
        selected = [
            candidate["index"]
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True)[:60]
        ]
    return sorted(set(selected))


def clean_step_slice_text(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = STEP_BOUNDARY_RE.sub("", cleaned, count=1).strip(" .,:;-")
    cleaned = re.sub(r"^(okay|ok|all right|so|now|then|and then)\b[\s,.-]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned or text


def transcript_step_slices(job):
    words = transcript_timed_words(job)
    target_count = expected_step_count_for(job)
    if words:
        transcript, spans = timed_words_text(words)
        candidates = []
        for match in STEP_BOUNDARY_RE.finditer(transcript):
            index = word_index_for_char(spans, match.start())
            if index is not None:
                candidates.append(
                    {
                        "index": index,
                        "time": words[index]["start"],
                        "score": 96,
                        "reason": f"operator cue: {match.group(0)}",
                        "explicit": True,
                    }
                )
        for match in SOFT_STEP_BOUNDARY_RE.finditer(transcript):
            index = word_index_for_char(spans, match.start())
            if index is not None:
                candidates.append(
                    {
                        "index": index,
                        "time": words[index]["start"],
                        "score": 54,
                        "reason": f"transition cue: {match.group(0)}",
                        "explicit": False,
                    }
                )
        for index in range(1, len(words)):
            gap = float(words[index]["start"]) - float(words[index - 1].get("end", words[index - 1]["start"]))
            if gap >= 1.25:
                candidates.append(
                    {
                        "index": index,
                        "time": words[index]["start"],
                        "score": min(86, 44 + gap * 9),
                        "reason": f"narration pause: {gap:.1f}s",
                        "explicit": False,
                    }
                )
        candidates = dedupe_step_boundaries(candidates, words)
        boundary_indexes = select_step_boundary_indexes(candidates, words, target_count)
        boundaries = [0] + boundary_indexes + [len(words)]
        candidate_by_index = {candidate["index"]: candidate for candidate in candidates}
        slices = []
        for index in range(0, len(boundaries) - 1):
            start_idx = boundaries[index]
            end_idx = boundaries[index + 1]
            if end_idx <= start_idx:
                continue
            slice_words = words[start_idx:end_idx]
            raw_text = clean_step_slice_text(" ".join(word["word"] for word in slice_words))
            if not raw_text or len(raw_text.split()) < 3:
                continue
            boundary = candidate_by_index.get(start_idx) or {}
            confidence = "high" if boundary.get("explicit") else "medium" if boundary else "low"
            slices.append(
                {
                    "id": f"rough-{len(slices) + 1:03d}",
                    "index": len(slices) + 1,
                    "raw_text": raw_text,
                    "start_time": round(float(slice_words[0]["start"]), 2),
                    "end_time": round(float(slice_words[-1].get("end", slice_words[-1]["start"])), 2),
                    "boundary_reason": boundary.get("reason") or ("target step count" if target_count else "timed transcript"),
                    "confidence": confidence,
                    "source": "word_timestamps",
                }
            )
        return slices

    segments = transcript_timed_segments(job, 0)
    if not segments:
        return []
    return [
        {
            "id": f"rough-{index:03d}",
            "index": index,
            "raw_text": clean_step_slice_text(segment.get("text") or ""),
            "start_time": round(float(segment.get("start") or 0), 2),
            "end_time": round(float(segment.get("end") or 0), 2),
            "boundary_reason": "provider segment",
            "confidence": "medium",
            "source": segment.get("source") or "segment",
        }
        for index, segment in enumerate(segments, start=1)
        if segment.get("text")
    ]


def transcript_timed_segments(job, duration_seconds):
    explicit = []
    for payload in drafting_transcript_payloads(job):
        source_file = payload.get("_source_file") or payload.get("source_file")
        for segment in payload.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            text = re.sub(r"\s+", " ", segment.get("text") or "").strip()
            try:
                start = apply_source_timeline(float(segment.get("start")), payload)
                end = apply_source_timeline(float(segment.get("end")), payload)
            except Exception:
                continue
            if text and end > start:
                explicit.append(
                    {
                        "text": text,
                        "start": max(0, start),
                        "end": max(start, end),
                        "source": "provider",
                        "source_file": source_file,
                    }
                )
    if explicit:
        non_monotonic = any(explicit[index]["start"] + 1 < explicit[index - 1]["start"] for index in range(1, len(explicit)))
        explicit.sort(key=lambda item: item["start"])
        max_end = max(item["end"] for item in explicit)
        if not non_monotonic and max_end > 0:
            return explicit

    words = transcript_timed_words(job)
    if words:
        grouped = []
        current = []
        current_start = None
        for word in words:
            if current_start is None:
                current_start = word["start"]
            current.append(word)
            sentence_end = bool(re.search(r"[.!?]$", str(word.get("word") or "")))
            long_enough = len(current) >= 24 or word["end"] - current_start >= 10
            if sentence_end or long_enough:
                grouped.append(current)
                current = []
                current_start = None
        if current:
            grouped.append(current)
        return [
            {
                "text": " ".join(item["word"] for item in group),
                "start": group[0]["start"],
                "end": max(group[-1]["end"], group[0]["start"] + 0.5),
                "source": "word_timestamps",
                "source_file": group[0].get("source_file"),
            }
            for group in grouped
        ]

    text = transcript_text(job)
    spans = transcript_sentence_spans(text)
    if not spans:
        return []
    duration = max(float(duration_seconds or 0), float(len(spans) * 6))
    text_len = max(1, len(text))
    timed = []
    for span in spans:
        start = duration * (span["char_start"] / text_len)
        end = duration * (span["char_end"] / text_len)
        if end <= start:
            end = start + 2
        timed.append(
            {
                "text": span["text"],
                "start": round(max(0, start), 2),
                "end": round(min(duration, max(end, start + 2)), 2),
                "source": "transcript_position",
            }
        )
    return timed


def frame_artifacts(job):
    return [
        artifact
        for artifact in job.get("artifacts", [])
        if artifact.get("kind") == "frame" and artifact.get("stored_path")
    ]


def extracted_frame_files(frame_dir):
    frames = []
    manifest = load_frame_manifest(frame_dir)
    manifest_records = {
        item.get("name"): item
        for item in manifest.get("frames", [])
        if item.get("name")
    }
    for frame_path in frame_dir.glob("frame_*.jpg"):
        match = FRAME_FILE_RE.match(frame_path.name)
        if match:
            frames.append((int(match.group(1)), frame_path))
    if manifest_records:
        return [
            frame_path
            for _, frame_path in sorted(
                frames,
                key=lambda item: (
                    float((manifest_records.get(item[1].name) or {}).get("timestamp_seconds", item[0])),
                    int((manifest_records.get(item[1].name) or {}).get("frame_index", item[0])),
                    item[0],
                ),
            )
        ]
    return [frame_path for _, frame_path in sorted(frames)]


def frame_number(frame_path):
    match = FRAME_FILE_RE.match(frame_path.name)
    if not match:
        return 0
    return int(match.group(1))


def frame_manifest_path(frame_dir):
    return frame_dir / FRAME_MANIFEST_NAME


def load_frame_manifest(frame_dir):
    path = frame_manifest_path(Path(frame_dir))
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return {}
    cache_key = str(path)
    cached = FRAME_MANIFEST_CACHE.get(cache_key)
    if cached and cached.get("mtime") == mtime:
        return cached.get("data") or {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict) or data.get("version") != FRAME_MANIFEST_VERSION:
        return {}
    data["_by_name"] = {
        item.get("name"): item
        for item in data.get("frames", [])
        if item.get("name")
    }
    FRAME_MANIFEST_CACHE[cache_key] = {"mtime": mtime, "data": data}
    return data


def save_frame_manifest(frame_dir, manifest):
    path = frame_manifest_path(Path(frame_dir))
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(path)
    FRAME_MANIFEST_CACHE.pop(str(path), None)


def frame_manifest_record(frame_path):
    manifest = load_frame_manifest(Path(frame_path).parent)
    return (manifest.get("_by_name") or {}).get(Path(frame_path).name)


def frame_source_timestamp_seconds(frame_path):
    record = frame_manifest_record(frame_path)
    if record and record.get("source_timestamp_seconds") is not None:
        try:
            return max(0, float(record.get("source_timestamp_seconds")))
        except Exception:
            pass
    number = frame_number(frame_path)
    return max(0, (number - 1) * VIDEO_FRAME_INTERVAL_SECONDS)


def frame_timestamp_seconds(frame_path, timeline_source=None):
    record = frame_manifest_record(frame_path)
    if record:
        try:
            source_seconds = float(record.get("source_timestamp_seconds") or 0)
            if timeline_source is not None:
                return apply_source_timeline(source_seconds, timeline_source)
            return apply_source_timeline(
                source_seconds,
                {
                    "timeline_offset_seconds": record.get("timeline_offset_seconds") or 0,
                    "timeline_scale": record.get("timeline_scale") or 1,
                    "timeline_sync": record.get("timeline_sync") or {},
                },
            )
        except Exception:
            pass
    return apply_source_timeline(frame_source_timestamp_seconds(frame_path), timeline_source or 0)


def frame_sampling_metadata(frame_path):
    record = frame_manifest_record(frame_path) or {}
    return {
        "sampling_reason": record.get("sampling_reason") or "",
        "sampling_tier": record.get("sampling_tier") or "",
        "sample_rate_hz": record.get("sample_rate_hz"),
    }


def cached_video_frames(file, frame_dir, schedule_hash=None):
    raw_frames = extracted_frame_files(frame_dir) if frame_dir.exists() else []
    if raw_frames:
        manifest = load_frame_manifest(frame_dir)
        if schedule_hash:
            if (
                manifest
                and manifest.get("schedule_hash") == schedule_hash
                and all(path.exists() for path in raw_frames)
            ):
                return raw_frames
            return []
        interval = file.get("frame_interval_seconds")
        if interval not in {None, VIDEO_FRAME_INTERVAL_SECONDS}:
            return []
        return raw_frames
    expected_paths = [
        APP_DIR / stored
        for stored in file.get("frame_paths", [])
        if stored
    ]
    if not expected_paths:
        return []
    if not all(path.exists() for path in expected_paths):
        return []
    if schedule_hash:
        manifest = load_frame_manifest(expected_paths[0].parent)
        if manifest and manifest.get("schedule_hash") == schedule_hash:
            return sorted(expected_paths, key=lambda path: frame_timestamp_seconds(path, file))
        return []
    interval = file.get("frame_interval_seconds")
    if interval not in {None, VIDEO_FRAME_INTERVAL_SECONDS}:
        return []
    return sorted(expected_paths, key=frame_number)


def cue_pattern_matches(pattern, text):
    if hasattr(pattern, "finditer"):
        return pattern.finditer(text or "")
    return re.finditer(pattern, text or "", flags=re.IGNORECASE)


def transcript_cue_times(job, duration_seconds):
    text = transcript_text(job)
    cue_patterns = [
        ("good_image", OPERATOR_IMAGE_CUE_RE, 2.4),
        ("step", r"\b(new step|next step|start step|step complete|moving on)\b", 1.4),
        ("parts", r"\b(part is|parts are|component is|consumable is|parts?|component|kit|bag)\b", 1.0),
        ("tool", r"\b(tool is|using tool|driver is|fixture is|jig is|tool|driver|screwdriver|wrench|torque tool)\b", 1.0),
        ("torque", r"\b(torque is|torque value is|torque to|no torque required|torque|in-?lb|ft-?lb|n-?m|nm)\b", 1.1),
        ("threadlocker", r"\b(apply threadlocker|loctite|locktite|threadlocker|thread locker)\b", 1.1),
        ("quality", r"\b(quality check|verify|inspect|confirm|make sure|flush|seated)\b", 1.3),
        ("safety", r"\b(safety note|safety|caution|keep clear|pinch point|hot surface|carefully)\b", 1.0),
        ("fixture", r"\b(fixture is|jig is|fixture|jig|clamp)\b", 1.0),
    ]
    cues = []
    timed_words = transcript_timed_words(job)
    if timed_words:
        cleaned, spans = timed_words_text(timed_words)
        for tag, pattern, weight in cue_patterns:
            for match in cue_pattern_matches(pattern, cleaned):
                word_index = word_index_for_char(spans, match.start())
                if word_index is None:
                    continue
                cues.append(
                    {
                        "tag": tag,
                        "time": timed_words[word_index]["start"],
                        "weight": weight,
                        "phrase": match.group(0),
                        "source": "word_timestamp",
                    }
                )
        return cues

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned or duration_seconds <= 0:
        return []
    for tag, pattern, weight in cue_patterns:
        for match in cue_pattern_matches(pattern, cleaned):
            position = match.start() / max(1, len(cleaned))
            cues.append(
                {
                    "tag": tag,
                    "time": position * duration_seconds,
                    "weight": weight,
                    "phrase": match.group(0),
                    "source": "text_position",
                }
            )
    return cues


def probe_video_duration(input_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float((result.stdout or "").strip())
    except Exception:
        return None
    return duration if duration > 0 else None


def adaptive_base_sample_rate(job):
    profile = str(job.get("profile") or SETTINGS.get("default_processing") or "Standard").strip().lower()
    return ADAPTIVE_BASE_SAMPLE_RATES.get(profile, ADAPTIVE_BASE_SAMPLE_RATES["standard"])


def processing_profile_key(job):
    profile = str((job or {}).get("profile") or SETTINGS.get("default_processing") or "Standard").strip().lower()
    return profile if profile in FRAME_ANALYSIS_PROFILE_BUDGETS else "standard"


def frame_budget_config(job):
    profile = processing_profile_key(job)
    config = dict(FRAME_ANALYSIS_PROFILE_BUDGETS["standard"])
    config.update(FRAME_ANALYSIS_PROFILE_BUDGETS.get(profile, {}))
    config["profile"] = profile
    return config


def add_sampling_window(windows, start, end, rate, reason, tier, duration_seconds):
    try:
        start = float(start)
        end = float(end)
        rate = float(rate)
    except Exception:
        return
    duration = float(duration_seconds or 0)
    if duration <= 0 or rate <= 0:
        return
    start = max(0.0, min(duration, start))
    end = max(0.0, min(duration, end))
    if end <= start:
        return
    windows.append(
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "rate": round(rate, 3),
            "reason": reason,
            "tier": tier,
        }
    )


def action_words_near_pause(words, index):
    context = " ".join(item.get("word") or "" for item in words[max(0, index - 12):index + 4]).lower()
    return bool(re.search(r"\b(attach|install|insert|place|press|push|set|seat|align|flip|turn|add|connect|mount|weld|inspect|verify|check)\b", context))


def merge_sampling_windows(windows):
    if not windows:
        return []
    tier_rank = {"base": 0, "step": 1, "pause": 2, "cue": 3, "good_image": 4}
    grouped = {}
    for window in windows:
        key = (window.get("tier") or "", float(window.get("rate") or 0))
        grouped.setdefault(key, []).append(window)
    merged = []
    for (tier, rate), items in grouped.items():
        current = None
        for item in sorted(items, key=lambda value: (float(value.get("start") or 0), float(value.get("end") or 0))):
            if current and float(item["start"]) <= float(current["end"]) + ADAPTIVE_WINDOW_MERGE_GAP_SECONDS:
                current["end"] = round(max(float(current["end"]), float(item["end"])), 3)
                if item.get("reason") and item.get("reason") not in current.get("reason", ""):
                    current["reason"] = f"{current.get('reason')}; {item.get('reason')}"
            else:
                if current:
                    merged.append(current)
                current = dict(item)
        if current:
            merged.append(current)
    return sorted(
        merged,
        key=lambda item: (
            float(item.get("start") or 0),
            -tier_rank.get(item.get("tier") or "", 0),
            -float(item.get("rate") or 0),
        ),
    )


def build_adaptive_sampling_schedule(job, duration_seconds):
    duration = float(duration_seconds or 0)
    base_rate = adaptive_base_sample_rate(job)
    if duration <= 0:
        return {
            "version": FRAME_MANIFEST_VERSION,
            "mode": "legacy_fixed",
            "profile": job.get("profile") or "Standard",
            "duration_seconds": 0,
            "base_rate_hz": base_rate,
            "windows": [],
            "hash": "",
        }
    windows = []
    add_sampling_window(windows, 0, duration, base_rate, "base coverage", "base", duration)

    for step in transcript_step_slices(job):
        start = timed_seconds(step.get("start_time"))
        end = timed_seconds(step.get("end_time"))
        if start is None or end is None:
            continue
        add_sampling_window(
            windows,
            start - 0.5,
            end + 1.25,
            max(base_rate, ADAPTIVE_STEP_SAMPLE_RATE_HZ),
            f"step window: {step.get('id') or step.get('index')}",
            "step",
            duration,
        )

    for cue in transcript_cue_times(job, duration):
        cue_time = timed_seconds(cue.get("time"))
        if cue_time is None:
            continue
        tag = cue.get("tag") or "cue"
        if tag == "good_image":
            add_sampling_window(
                windows,
                cue_time - 2.0,
                cue_time + 4.0,
                max(base_rate, ADAPTIVE_IMAGE_CUE_SAMPLE_RATE_HZ),
                f"operator image cue: {cue.get('phrase') or tag}",
                "good_image",
                duration,
            )
        elif tag in {"quality", "safety", "tool", "torque", "parts", "threadlocker", "fixture", "step"}:
            add_sampling_window(
                windows,
                cue_time - 1.25,
                cue_time + 2.25,
                max(base_rate, ADAPTIVE_CUE_SAMPLE_RATE_HZ),
                f"{tag} cue: {cue.get('phrase') or tag}",
                "cue",
                duration,
            )

    words = transcript_timed_words(job)
    for index in range(1, len(words)):
        gap = float(words[index].get("start") or 0) - float(words[index - 1].get("end") or words[index - 1].get("start") or 0)
        if gap >= 1.1 and action_words_near_pause(words, index):
            add_sampling_window(
                windows,
                float(words[index - 1].get("end") or words[index - 1].get("start") or 0) - 0.25,
                float(words[index].get("start") or 0) + 1.25,
                max(base_rate, ADAPTIVE_PAUSE_SAMPLE_RATE_HZ),
                f"after-action pause: {gap:.1f}s",
                "pause",
                duration,
            )

    merged = merge_sampling_windows(windows)
    schedule = {
        "version": FRAME_MANIFEST_VERSION,
        "mode": "adaptive",
        "profile": job.get("profile") or "Standard",
        "duration_seconds": round(duration, 3),
        "base_rate_hz": base_rate,
        "windows": merged,
    }
    schedule["hash"] = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return schedule


def sampling_tier_priority(tier):
    return {"base": 0, "step": 1, "pause": 2, "cue": 3, "good_image": 4}.get(tier or "", 0)


def remove_frame_files(frame_dir):
    for old_frame in frame_dir.glob("frame_*.jpg"):
        old_frame.unlink(missing_ok=True)
    frame_manifest_path(frame_dir).unlink(missing_ok=True)
    FRAME_MANIFEST_CACHE.pop(str(frame_manifest_path(frame_dir)), None)


def extract_window_frames(ffmpeg, input_path, frame_dir, window, start_number):
    pattern = frame_dir / "frame_%04d.jpg"
    start = float(window.get("start") or 0)
    end = float(window.get("end") or start)
    rate = float(window.get("rate") or 0)
    duration = max(0.0, end - start)
    if duration <= 0 or rate <= 0:
        return [], None
    before = {path.name for path in frame_dir.glob("frame_*.jpg")}
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-vf",
        f"fps={rate},scale=1280:-2",
        "-q:v",
        "3",
        "-start_number",
        str(start_number),
        str(pattern),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(VIDEO_EXTRACTION_TIMEOUT_SECONDS, max(60, int(duration * 3) + 45)),
        )
    except subprocess.TimeoutExpired:
        return [], f"ffmpeg timed out while sampling {window.get('tier') or 'window'} frames"
    if result.returncode != 0:
        return [], (result.stderr or result.stdout or "ffmpeg failed")[-1000:]
    new_frames = [
        path
        for path in extracted_frame_files(frame_dir)
        if path.name not in before
    ]
    new_frames = sorted(new_frames, key=frame_number)
    records = []
    for offset, path in enumerate(new_frames):
        source_timestamp = min(end, start + (offset / rate))
        records.append(
            {
                "path": path,
                "name": path.name,
                "source_timestamp_seconds": round(source_timestamp, 3),
                "sampling_reason": window.get("reason") or "",
                "sampling_tier": window.get("tier") or "",
                "sample_rate_hz": rate,
            }
        )
    return records, None


def dedupe_frame_records(records):
    if not records:
        return [], 0
    clusters = []
    for record in sorted(records, key=lambda item: float(item.get("source_timestamp_seconds") or 0)):
        timestamp_seconds = float(record.get("source_timestamp_seconds") or 0)
        if clusters and timestamp_seconds - float(clusters[-1][0].get("source_timestamp_seconds") or 0) <= ADAPTIVE_TIMESTAMP_DEDUPE_SECONDS:
            clusters[-1].append(record)
        else:
            clusters.append([record])
    kept = []
    removed = 0
    for cluster in clusters:
        winner = sorted(
            cluster,
            key=lambda item: (
                sampling_tier_priority(item.get("sampling_tier")),
                float(item.get("sample_rate_hz") or 0),
                -frame_number(item.get("path")),
            ),
            reverse=True,
        )[0]
        kept.append(winner)
        for item in cluster:
            if item is winner:
                continue
            removed += 1
            try:
                item["path"].unlink(missing_ok=True)
            except Exception:
                pass
    return sorted(kept, key=lambda item: float(item.get("source_timestamp_seconds") or 0)), removed


def extract_adaptive_video_frames(ffmpeg, input_path, frame_dir, file, schedule):
    remove_frame_files(frame_dir)
    timeline_offset_seconds = float(file.get("timeline_offset_seconds") or 0)
    timeline_transform = source_timeline_transform(file)
    records = []
    window_errors = []
    next_number = 1
    for window in schedule.get("windows") or []:
        window_records, error = extract_window_frames(ffmpeg, input_path, frame_dir, window, next_number)
        if error and not records:
            return [], {}, error
        if error:
            window_errors.append(
                {
                    "tier": window.get("tier") or "",
                    "reason": window.get("reason") or "",
                    "error": error,
                }
            )
            continue
        records.extend(window_records)
        if window_records:
            next_number = max(frame_number(item["path"]) for item in window_records) + 1
    records, duplicate_count = dedupe_frame_records(records)
    manifest_frames = []
    for index, record in enumerate(records, start=1):
        source_timestamp = float(record.get("source_timestamp_seconds") or 0)
        rel_path = str(record["path"].relative_to(APP_DIR))
        manifest_frames.append(
            {
                "frame_index": index,
                "name": record["path"].name,
                "stored_path": rel_path,
                "timestamp_seconds": round(apply_source_timeline(source_timestamp, file), 3),
                "source_timestamp_seconds": round(max(0, source_timestamp), 3),
                "timeline_offset_seconds": round(timeline_offset_seconds, 3),
                "timeline_scale": round(float(timeline_transform.get("scale") or 1), 9),
                "timeline_sync": file.get("timeline_sync") or {},
                "sampling_reason": record.get("sampling_reason") or "",
                "sampling_tier": record.get("sampling_tier") or "",
                "sample_rate_hz": record.get("sample_rate_hz"),
            }
        )
    manifest = {
        "version": FRAME_MANIFEST_VERSION,
        "created_at": datestamp(),
        "source_file": file.get("name") or "",
        "source_sha256": file.get("sha256") or "",
        "mode": schedule.get("mode") or "adaptive",
        "profile": schedule.get("profile") or "",
        "duration_seconds": schedule.get("duration_seconds") or 0,
        "base_rate_hz": schedule.get("base_rate_hz"),
        "schedule_hash": schedule.get("hash") or "",
        "windows": schedule.get("windows") or [],
        "window_errors": window_errors,
        "duplicate_timestamps_removed": duplicate_count,
        "frames": manifest_frames,
    }
    save_frame_manifest(frame_dir, manifest)
    stats = {
        "schedule_windows": len(schedule.get("windows") or []),
        "base_rate_hz": schedule.get("base_rate_hz"),
        "duplicate_timestamps_removed": duplicate_count,
        "window_error_count": len(window_errors),
        "manifest_frames": len(manifest_frames),
    }
    return extracted_frame_files(frame_dir), stats, None


def score_candidate_frame(frame_path, cue_times, duration_seconds, timeline_source=None):
    timestamp_seconds = frame_timestamp_seconds(frame_path, timeline_source)
    sampling = frame_sampling_metadata(frame_path)
    score = 10
    tags = []
    reasons = []

    sampling_tier = sampling.get("sampling_tier") or ""
    sampling_reason = sampling.get("sampling_reason") or ""
    if sampling_tier:
        tags.append(sampling_tier)
        if sampling_reason:
            reasons.append(sampling_reason)
    if sampling_tier == "good_image":
        score += 42
        tags.extend(["good_image", "operator_image_cue"])
    elif sampling_tier == "cue":
        score += 22
        tags.append("cue_window")
    elif sampling_tier == "pause":
        score += 18
        tags.append("after_action_pause")
    elif sampling_tier == "step":
        score += 16
        tags.append("step_window")
    elif sampling_tier == "base":
        tags.append("coverage")

    if timestamp_seconds <= 2:
        score += 12
        tags.append("start")
    if duration_seconds and abs(duration_seconds - timestamp_seconds) <= 5:
        score += 14
        tags.append("end")
    if not sampling_tier and timestamp_seconds % VIDEO_COVERAGE_INTERVAL_SECONDS == 0:
        score += 10
        tags.append("coverage")

    best_delta = None
    for cue in cue_times:
        delta = abs(timestamp_seconds - cue["time"])
        if delta <= VIDEO_CUE_BURST_SECONDS:
            bonus = max(0, (VIDEO_CUE_BURST_SECONDS - delta + 1) * 10 * cue["weight"])
            score += bonus
            tags.append(cue["tag"])
            if best_delta is None or delta < best_delta:
                best_delta = delta
                reasons = [f"{cue['tag']} cue: {cue['phrase']}"]

    if best_delta is not None and best_delta <= 1:
        score += 18

    return {
        "score": round(score, 2),
        "tags": sorted(set(tags)),
        "reason": ", ".join(reasons) if reasons else "coverage sample",
        "sampling_reason": sampling_reason,
        "sampling_tier": sampling_tier,
        "sample_rate_hz": sampling.get("sample_rate_hz"),
    }


def focus_crop(gray, focus_area="center"):
    if focus_area != "center":
        return gray
    height, width = gray.shape[:2]
    crop_width = max(1, int(width * 0.72))
    crop_height = max(1, int(height * 0.72))
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    return gray[y0:y0 + crop_height, x0:x0 + crop_width]


def opencv_frame_metrics(frames, focus_area="center"):
    if cv2 is None:
        return {}, {"enabled": False, "reason": "opencv not installed"}

    metrics = {}
    previous_small = None
    previous_focus_small = None
    read_failed = 0
    for frame in frames:
        image = cv2.imread(str(frame))
        if image is None:
            read_failed += 1
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus = focus_crop(gray, focus_area)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        focus_blur = float(cv2.Laplacian(focus, cv2.CV_64F).var())
        brightness = float(gray.mean())
        focus_brightness = float(focus.mean())
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
        focus_small = cv2.resize(focus, (96, 54), interpolation=cv2.INTER_AREA)
        delta = None
        focus_delta = None
        if previous_small is not None:
            delta = float(cv2.absdiff(small, previous_small).mean())
        if previous_focus_small is not None:
            focus_delta = float(cv2.absdiff(focus_small, previous_focus_small).mean())
        previous_small = small
        previous_focus_small = focus_small
        metrics[frame.name] = {
            "blur": round(blur, 2),
            "focus_blur": round(focus_blur, 2),
            "brightness": round(brightness, 2),
            "focus_brightness": round(focus_brightness, 2),
            "delta": round(delta, 2) if delta is not None else None,
            "focus_delta": round(focus_delta, 2) if focus_delta is not None else None,
            "focus_area": focus_area,
        }
    return metrics, {"enabled": True, "read_failed": read_failed}


def apply_opencv_scoring(meta, cv_meta):
    if not cv_meta:
        return meta
    score = float(meta.get("score", 0))
    tags = set(meta.get("tags") or [])
    reasons = [meta.get("reason") or ""]

    blur = cv_meta.get("blur")
    focus_blur = cv_meta.get("focus_blur")
    brightness = cv_meta.get("brightness")
    focus_brightness = cv_meta.get("focus_brightness")
    delta = cv_meta.get("delta")
    focus_delta = cv_meta.get("focus_delta")
    strong_cue = bool(tags & (HARD_FRAME_CUE_TAGS | STEP_FRAME_CUE_TAGS))

    tags.add("opencv")
    if blur is not None:
        if blur < OPENCV_BLUR_MIN:
            tags.add("blurry")
            score -= 38 if not strong_cue else 18
            reasons.append(f"low sharpness {blur}")
        else:
            tags.add("sharp")
            score += min(18, blur / 35)
    if focus_blur is not None:
        if focus_blur < OPENCV_BLUR_MIN and not strong_cue:
            tags.add("focus_blurry")
            score -= 24
        elif focus_blur >= OPENCV_BLUR_MIN:
            tags.add("focus_sharp")
            score += min(16, focus_blur / 42)
    if brightness is not None:
        if brightness < 28:
            tags.add("dark")
            score -= 16
        elif brightness > 235:
            tags.add("bright")
            score -= 12
        else:
            score += 3
    if focus_brightness is not None:
        if focus_brightness < 28:
            tags.add("focus_dark")
            score -= 10
        elif focus_brightness > 235:
            tags.add("focus_bright")
            score -= 8
        else:
            score += 2
    if delta is not None:
        if delta <= OPENCV_DUPLICATE_DELTA_MAX:
            tags.add("near_duplicate")
            score -= 22 if not strong_cue else 8
        elif delta >= 9:
            tags.add("visual_change")
            score += min(22, delta * 1.5)
    if focus_delta is not None:
        if focus_delta >= OPENCV_FOCUS_CHANGE_MIN:
            tags.add("focus_change")
            score += min(34, focus_delta * 3.0)
        elif delta is not None and delta >= OPENCV_BACKGROUND_CHANGE_MIN:
            tags.add("background_change")
            score -= 18

    meta["score"] = round(max(0, score), 2)
    meta["tags"] = sorted(tags)
    meta["reason"] = "; ".join(reason for reason in reasons if reason)
    meta["opencv"] = cv_meta
    return meta


def select_candidate_frames(job, frames, file=None):
    if not frames:
        return [], {}, {"raw_frames": 0, "selected_frames": 0, "opencv": {"enabled": cv2 is not None}}
    timeline_source = file or 0
    focus_area = normalize_focus_area(job.get("focus_area"))
    duration_seconds = max(frame_timestamp_seconds(frame, timeline_source) for frame in frames)
    cue_times = transcript_cue_times(job, duration_seconds)
    cv_metrics, cv_stats = opencv_frame_metrics(frames, focus_area)
    scored = [
        (
            apply_opencv_scoring(
                score_candidate_frame(frame, cue_times, duration_seconds, timeline_source),
                cv_metrics.get(frame.name),
            ),
            frame,
        )
        for frame in frames
    ]
    selected = {}
    selection_sources = {}
    rejected_blurry = 0
    rejected_duplicate = 0

    def keep(frame, meta, source):
        selected[frame] = meta
        selection_sources.setdefault(frame, set()).add(source)

    for meta, frame in scored:
        timestamp_seconds = frame_timestamp_seconds(frame, timeline_source)
        tags = set(meta.get("tags") or [])
        hard_cue = bool(tags & HARD_FRAME_CUE_TAGS)
        step_cue = bool(tags & STEP_FRAME_CUE_TAGS)
        soft_cue = bool(tags & SOFT_FRAME_CUE_TAGS)
        strong_cue = hard_cue or step_cue
        endpoint_coverage = timestamp_seconds <= 2 or (duration_seconds and abs(duration_seconds - timestamp_seconds) <= 5)
        if endpoint_coverage:
            keep(frame, meta, "coverage")
            continue
        if "blurry" in tags and not strong_cue:
            rejected_blurry += 1
            continue
        if "near_duplicate" in tags and not (hard_cue or "visual_change" in tags):
            rejected_duplicate += 1
            continue
        if hard_cue and meta["score"] >= 20:
            keep(frame, meta, "hard_cue")
        elif step_cue and meta["score"] >= 34:
            keep(frame, meta, "step_window")
        elif soft_cue and ("sharp" in tags or "focus_sharp" in tags or meta["score"] >= 70):
            keep(frame, meta, "soft_cue")
        elif "visual_change" in tags and meta["score"] >= 30:
            keep(frame, meta, "visual_change")
        elif meta["score"] >= 52:
            keep(frame, meta, "high_score")
        elif "sharp" in tags and meta["score"] >= 44:
            keep(frame, meta, "sharp")

    ranked = sorted(scored, key=lambda item: item[0]["score"], reverse=True)
    duration_baseline = min(len(frames), max(3, min(20, int(duration_seconds // 60) + 2)))
    if len(selected) < duration_baseline:
        for meta, frame in ranked:
            tags = set(meta.get("tags") or [])
            if "blurry" in tags and not (tags & (HARD_FRAME_CUE_TAGS | STEP_FRAME_CUE_TAGS)):
                continue
            if "near_duplicate" in tags and "visual_change" not in tags:
                continue
            keep(frame, meta, "duration_baseline")
            if len(selected) >= duration_baseline:
                break

    candidate_limit = dynamic_candidate_limit(job, len(frames), duration_seconds)
    hard_ceiling_applied = False
    if len(selected) > candidate_limit:
        protected_sources = {"coverage", "hard_cue", "step_window"}
        protected_items = [
            item
            for item in selected.items()
            if selection_sources.get(item[0], set()) & protected_sources
        ]
        unprotected_items = [
            item
            for item in selected.items()
            if not (selection_sources.get(item[0], set()) & protected_sources)
        ]
        ranked_protected = sorted(protected_items, key=lambda item: item[1]["score"], reverse=True)
        ranked_unprotected = sorted(unprotected_items, key=lambda item: item[1]["score"], reverse=True)
        selected = dict((ranked_protected + ranked_unprotected)[:candidate_limit])
        selection_sources = {
            frame: selection_sources.get(frame, set())
            for frame in selected
        }
        hard_ceiling_applied = True

    for frame, meta in selected.items():
        sources = sorted(selection_sources.get(frame, []))
        tags = set(meta.get("tags") or [])
        tags.update(f"selected_{source}" for source in sources)
        meta["tags"] = sorted(tags)
        meta["selection_sources"] = sources

    ordered_frames = sorted(selected, key=lambda frame: frame_timestamp_seconds(frame, timeline_source))
    meta_by_name = {frame.name: selected[frame] for frame in ordered_frames}
    source_counts = {}
    for frame in ordered_frames:
        for source in selection_sources.get(frame, []):
            source_counts[source] = source_counts.get(source, 0) + 1
    stats = {
        "raw_frames": len(frames),
        "selected_frames": len(ordered_frames),
        "cue_count": len(cue_times),
        "opencv": cv_stats,
        "blurry_rejected": rejected_blurry,
        "duplicate_rejected": rejected_duplicate,
        "cue_selected": source_counts.get("hard_cue", 0) + source_counts.get("step_window", 0) + source_counts.get("soft_cue", 0),
        "hard_cue_selected": source_counts.get("hard_cue", 0),
        "step_window_selected": source_counts.get("step_window", 0),
        "soft_cue_selected": source_counts.get("soft_cue", 0),
        "visual_change_selected": source_counts.get("visual_change", 0),
        "coverage_selected": source_counts.get("coverage", 0),
        "sharp_selected": source_counts.get("sharp", 0),
        "duration_baseline_selected": source_counts.get("duration_baseline", 0),
        "hard_ceiling_applied": hard_ceiling_applied,
        "candidate_limit": candidate_limit,
        "configured_hard_limit": VIDEO_CANDIDATE_FRAME_LIMIT,
        "focus_area": focus_area,
    }
    return ordered_frames, meta_by_name, stats


def rough_step_count_for_limit(job):
    expected = parse_expected_step_count(job.get("expected_step_count"))
    if expected:
        return expected
    step_parse = job.get("step_parse") or {}
    rough_count = parse_expected_step_count(step_parse.get("rough_step_count"))
    if rough_count:
        return rough_count
    rough_steps = job.get("transcript_step_slices") or []
    if rough_steps:
        return len(rough_steps)
    return None


def dynamic_candidate_limit(job, raw_count, duration_seconds=0):
    step_count = rough_step_count_for_limit(job)
    if not step_count:
        step_count = max(8, int(max(duration_seconds or 0, raw_count or 0) // 30) + 1)
    config = frame_budget_config(job)
    duration_bonus = min(48, int(max(0, float(duration_seconds or 0)) // 120) * 4)
    desired = max(
        config["min_frames"],
        int(step_count * config["frames_per_step"] * config["candidate_multiplier"]) + config["coverage_reserve"] + duration_bonus,
    )
    candidate_max = int(config.get("candidate_max") or VIDEO_CANDIDATE_FRAME_LIMIT)
    return max(1, min(int(raw_count or desired), desired, candidate_max, VIDEO_CANDIDATE_FRAME_LIMIT))


def dynamic_frame_analysis_limit(job, candidate_count):
    step_count = rough_step_count_for_limit(job)
    if not step_count:
        step_count = max(8, int(candidate_count // 24) + 1)
    config = frame_budget_config(job)
    desired = max(config["min_frames"], step_count * config["frames_per_step"] + config["coverage_reserve"])
    return max(1, min(int(candidate_count or desired), desired, config["max_frames"], FRAME_ANALYSIS_MAX_FRAMES))


def frame_budget_score(frame):
    score = float(frame.get("local_priority") or 0) + float(frame.get("instruction_value") or 0) * 0.35
    tags = set(frame.get("candidate_tags") or [])
    if tags & HARD_FRAME_CUE_TAGS:
        score += 60
    if tags & STEP_FRAME_CUE_TAGS:
        score += 22
    if "visual_change" in tags:
        score += 18
    if "focus_change" in tags:
        score += 16
    if "sharp" in tags or "focus_sharp" in tags:
        score += 10
    if "near_duplicate" in tags:
        score -= 22
    if "blurry" in tags or "focus_blurry" in tags:
        score -= 30
    return score


def step_windows_for_frame_budget(job, duration_seconds):
    rough_steps = transcript_step_slices(job) or job.get("transcript_step_slices") or []
    windows = []
    for index, step in enumerate(rough_steps):
        start = timed_seconds(step.get("start_time"))
        end = timed_seconds(step.get("end_time"))
        if start is None or end is None:
            continue
        end = max(start + STEP_EXACT_MIN_WINDOW_SECONDS, end)
        windows.append(
            {
                "index": index,
                "start": max(0.0, start - 0.75),
                "end": min(float(duration_seconds), end + 1.25) if duration_seconds else end + 1.25,
                "target": min(float(duration_seconds), max(start, end - 0.5)) if duration_seconds else max(start, end - 0.5),
                "source": step.get("id") or f"rough-{index + 1:03d}",
            }
        )
    if windows:
        return windows

    step_count = rough_step_count_for_limit(job) or 0
    if not step_count or not duration_seconds:
        return []
    span = max(float(duration_seconds or 0) / step_count, STEP_EXACT_MIN_WINDOW_SECONDS)
    for index in range(step_count):
        start = index * span
        end = duration_seconds if index == step_count - 1 else (index + 1) * span
        windows.append(
            {
                "index": index,
                "start": max(0.0, start - 0.5),
                "end": min(float(duration_seconds), end + 0.75),
                "target": min(float(duration_seconds), max(start, end - 0.5)),
                "source": f"estimated-{index + 1:03d}",
            }
        )
    return windows


def evenly_spaced_candidates(candidates, count, duration_seconds, excluded_paths=None):
    if count <= 0 or not candidates:
        return []
    excluded_paths = excluded_paths or set()
    ordered = sorted(candidates, key=lambda frame: float(frame.get("timestamp_seconds") or 0))
    duration = float(duration_seconds or ordered[-1].get("timestamp_seconds") or 0)
    if duration <= 0:
        return []
    selected = []
    for index in range(count):
        target = (index + 0.5) * duration / count
        nearby = [
            frame
            for frame in ordered
            if frame.get("stored_path") not in excluded_paths
            and frame.get("stored_path") not in {item.get("stored_path") for item in selected}
        ]
        if not nearby:
            break
        best = min(
            nearby,
            key=lambda frame: (
                abs(float(frame.get("timestamp_seconds") or 0) - target),
                -frame_budget_score(frame),
            ),
        )
        selected.append(best)
    return selected


def prioritize_frame_analysis_candidates(job, candidates):
    if not candidates:
        return [], {"profile": processing_profile_key(job), "analysis_limit": 0}

    config = frame_budget_config(job)
    analysis_limit = dynamic_frame_analysis_limit(job, len(candidates))
    duration_seconds = max(float(frame.get("timestamp_seconds") or 0) for frame in candidates)
    windows = step_windows_for_frame_budget(job, duration_seconds)
    frames_per_step = int(config["frames_per_step"])
    selected = {}
    selection_sources = {}

    def keep(frame, source):
        stored = frame.get("stored_path")
        if not stored:
            return
        selected[stored] = frame
        selection_sources.setdefault(stored, set()).add(source)

    hard_cue_frames = [
        frame
        for frame in candidates
        if set(frame.get("candidate_tags") or []) & HARD_FRAME_CUE_TAGS
    ]
    for frame in sorted(hard_cue_frames, key=frame_budget_score, reverse=True):
        keep(frame, "hard_cue")
        if len(selected) >= analysis_limit:
            break

    step_budgeted_count = 0
    for window in windows:
        if len(selected) >= analysis_limit:
            break
        start = float(window["start"])
        end = float(window["end"])
        target = float(window["target"])
        in_window = [
            frame
            for frame in candidates
            if start <= float(frame.get("timestamp_seconds") or 0) <= end
        ]
        if not in_window:
            in_window = [
                frame
                for frame in candidates
                if abs(float(frame.get("timestamp_seconds") or 0) - target) <= STEP_FALLBACK_NEAR_SECONDS
            ]
        ranked = sorted(
            in_window,
            key=lambda frame: (
                frame_budget_score(frame),
                -abs(float(frame.get("timestamp_seconds") or 0) - target),
            ),
            reverse=True,
        )
        picked_for_step = 0
        for frame in ranked:
            stored = frame.get("stored_path")
            if not stored or stored in selected:
                continue
            tags = set(frame.get("candidate_tags") or [])
            if "blurry" in tags and not (tags & HARD_FRAME_CUE_TAGS):
                continue
            keep(frame, "step_budget")
            step_budgeted_count += 1
            picked_for_step += 1
            if picked_for_step >= frames_per_step or len(selected) >= analysis_limit:
                break

    coverage_frames = evenly_spaced_candidates(
        candidates,
        int(config["coverage_reserve"]),
        duration_seconds,
        excluded_paths=set(selected),
    )
    for frame in coverage_frames:
        if len(selected) >= analysis_limit:
            break
        keep(frame, "coverage_reserve")

    fill_count = 0
    for frame in sorted(candidates, key=frame_budget_score, reverse=True):
        if len(selected) >= analysis_limit:
            break
        stored = frame.get("stored_path")
        if not stored or stored in selected:
            continue
        tags = set(frame.get("candidate_tags") or [])
        if "blurry" in tags and not (tags & HARD_FRAME_CUE_TAGS):
            continue
        if "near_duplicate" in tags and "visual_change" not in tags and not (tags & HARD_FRAME_CUE_TAGS):
            continue
        keep(frame, "priority_fill")
        fill_count += 1

    if not selected:
        for frame in sorted(candidates, key=frame_budget_score, reverse=True)[:analysis_limit]:
            keep(frame, "best_available_fallback")

    ordered = sorted(selected.values(), key=lambda frame: float(frame.get("timestamp_seconds") or 0))
    stats = {
        "profile": config["profile"],
        "analysis_limit": analysis_limit,
        "candidate_frames": len(candidates),
        "selected_frames": len(ordered),
        "rough_step_windows": len(windows),
        "frames_per_step": frames_per_step,
        "hard_cue_frames": len([path for path, sources in selection_sources.items() if "hard_cue" in sources]),
        "step_budgeted_frames": step_budgeted_count,
        "coverage_reserve_frames": len([path for path, sources in selection_sources.items() if "coverage_reserve" in sources]),
        "priority_fill_frames": fill_count,
        "best_available_fallback_frames": len([path for path, sources in selection_sources.items() if "best_available_fallback" in sources]),
        "ceiling_applied": len(candidates) > len(ordered),
    }
    return ordered, stats



def clear_frame_artifacts_for_file(job, file):
    job["artifacts"] = [
        artifact
        for artifact in job.get("artifacts", [])
        if not (
            artifact.get("kind") == "frame"
            and artifact.get("source_file") == file.get("name")
        )
    ]


def video_frame_record(job, file, frame_path, meta=None, source="sampled video frame"):
    meta = meta or {}
    analysis_by_path = {
        item.get("stored_path"): item
        for item in job.get("frame_analysis", [])
        if item.get("stored_path")
    }
    match = FRAME_FILE_RE.match(frame_path.name)
    frame_index = int(match.group(1)) if match else 1
    timeline_offset_seconds = float(file.get("timeline_offset_seconds") or 0)
    timeline_transform = source_timeline_transform(file)
    source_timestamp_seconds = frame_source_timestamp_seconds(frame_path)
    timestamp_seconds = frame_timestamp_seconds(frame_path, file)
    sampling = frame_sampling_metadata(frame_path)
    stored = str(frame_path.relative_to(APP_DIR))
    record = {
        "name": frame_path.name,
        "type": "image/jpeg",
        "kind": "frame",
        "size": human_size(frame_path.stat().st_size),
        "bytes": frame_path.stat().st_size,
        "stored_path": stored,
        "source": source,
        "source_file": file.get("name"),
        "processing_status": "candidate keyframe",
        "timestamp": f"~{round(timestamp_seconds, 2)}s",
        "timestamp_seconds": round(timestamp_seconds, 2),
        "source_timestamp_seconds": round(source_timestamp_seconds, 2),
        "timeline_offset_seconds": round(timeline_offset_seconds, 2),
        "timeline_scale": round(float(timeline_transform.get("scale") or 1), 9),
        "source_id": file.get("source_id") or "",
        "source_label": file.get("source_label") or "",
        "local_priority": meta.get("score", 0),
        "candidate_tags": meta.get("tags", []),
        "candidate_reason": meta.get("reason", ""),
        "opencv_metrics": meta.get("opencv") or {},
        "sampling_reason": sampling.get("sampling_reason") or "",
        "sampling_tier": sampling.get("sampling_tier") or "",
        "sample_rate_hz": sampling.get("sample_rate_hz"),
    }
    analysis = analysis_by_path.get(stored)
    if analysis:
        record["analysis"] = analysis
        record["processing_status"] = "scored keyframe"
        record["instruction_value"] = analysis.get("instruction_value", 0)
        record["pointing"] = analysis.get("pointing", False)
        record["discard"] = analysis.get("discard", False)
    return record


def record_video_frames(job, file, frames, candidate_meta=None):
    candidate_meta = candidate_meta or {}
    frame_records = []
    for fallback_index, frame_path in enumerate(frames, start=1):
        meta = candidate_meta.get(frame_path.name) or {}
        frame_records.append(upsert_artifact(job, video_frame_record(job, file, frame_path, meta)))
    manifest = load_frame_manifest(frames[0].parent) if frames else {}
    file["frame_paths"] = [record["stored_path"] for record in frame_records]
    if manifest:
        file["frame_extraction_mode"] = manifest.get("mode") or "adaptive"
        file["frame_schedule_hash"] = manifest.get("schedule_hash") or ""
        file["frame_manifest_path"] = str(frame_manifest_path(frames[0].parent).relative_to(APP_DIR))
        file["frame_interval_seconds"] = None
    else:
        file["frame_extraction_mode"] = "fixed"
        file["frame_interval_seconds"] = VIDEO_FRAME_INTERVAL_SECONDS
    file["video_status"] = f"{len(frame_records)} frames extracted"
    return frame_records


def split_sentences(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [
        sentence.strip(" -")
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip(" -")
    ]


def normalize_instruction(sentence):
    text = sentence.strip()
    replacements = [
        (r"^(all right,\s*)?", ""),
        (r"^(new step|next step|start step|step complete)\s*[:\-]?\s*", ""),
        (r"^(so\s+)?(i am|i'm)\s+first\s+going\s+to\s+", ""),
        (r"^(so\s+)?(i am|i'm)\s+going\s+to\s+", ""),
        (r"^(so\s+)?we('re| are)\s+going\s+to\s+", ""),
        (r"^(so\s+)?we\s+need\s+to\s+", ""),
        (r"^(now|then|next|and then|so)\s+", ""),
        (r"^i\s+will\s+", ""),
        (r"^we\s+will\s+", ""),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = text.strip(" ,.")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def action_sentences(text):
    actions = []
    keywords = {
        "open", "remove", "set", "lay", "turn", "tear", "dump", "discard",
        "assemble", "press", "find", "put", "place", "grab", "stick", "flip",
        "add", "install", "push", "kit", "new step", "next step", "tool",
        "torque", "loctite", "locktite", "threadlocker", "fixture", "jig",
        "quality check", "safety note",
    }
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(re.search(rf"\b{word}\b", lowered) for word in keywords):
            instruction = normalize_instruction(sentence)
            if instruction:
                actions.append(instruction)
    return actions


def step_parts(text):
    words = []
    patterns = [
        r"\b(?:part|parts|component|components)\s+(?:are|is|used|needed|callout|number)?\s*[:\-]?\s*[^.]+",
        r"\b(?:loctite|locktite|threadlocker|thread locker)\b",
        r"\b(?:kit|bag)\s+\d+\b",
        r"\b(?:yellow|red|black|white|gray|blue)\s+[^.,;]+?(?:piece|panel|axle|hub|rectangle|square|round|pins?)\b",
        r"\b(?:driver|legs|torso|head|helmet|wrench|hair piece|chassis|wheel hubs?|side panels?|front axles?|rear wheel axles?|cylinder heads?)\b",
        r"\b\d+[- ]?by[- ]?\d+(?:[- ]?by[- ]?\d+)?\s+[^.,;]+?(?:piece|panel|round|square|rectangle)?\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            value = re.sub(r"\s+", " ", match).strip(" .,")
            if value and value.lower() not in {item.lower() for item in words}:
                words.append(value)
    return words[:8]


def step_quality(text):
    lowered = text.lower()
    checks = []
    quality_match = re.search(r"\b(?:quality check|verify|inspect|confirm)\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if quality_match:
        checks.append(normalize_instruction(quality_match.group(1)))
    if "flush" in lowered:
        checks.append("Verify the parts are seated flush.")
    if "seated" in lowered:
        checks.append("Verify the part is fully seated.")
    if "symmetr" in lowered or "opposite" in lowered or "both" in lowered:
        checks.append("Verify the left and right sides match.")
    if "orientation" in lowered or "flip" in lowered:
        checks.append("Verify orientation matches the reference.")
    return " ".join(dict.fromkeys(checks))


def step_safety(text):
    lowered = text.lower()
    safety_match = re.search(r"\b(?:safety note|safety|caution)\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if safety_match:
        return normalize_instruction(safety_match.group(1))
    if any(word in lowered for word in ["tear", "open", "carefully"]):
        return "Open packaging carefully and keep loose parts contained."
    if any(word in lowered for word in ["press", "push"]):
        return "Keep fingers clear while pressing parts together."
    return ""


def step_tools(text):
    lowered = text.lower()
    cue_match = re.search(r"\b(?:tool|tooling|fixture|jig)\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if cue_match:
        return cue_match.group(1).strip(" .")
    tool_patterns = [
        ("Fixture", r"\b(fixture|jig|clamp)\b"),
        ("Torque tool", r"\btorque\s+(tool|wrench|driver)\b"),
        ("Driver", r"\b(use|using)\s+(a\s+)?(driver|screwdriver)\b"),
        ("Wrench", r"\b(use|using)\s+(a\s+)?wrench\b"),
    ]
    for label, pattern in tool_patterns:
        if re.search(pattern, lowered):
            return label
    return "Hands"


def step_torque(text):
    torque_cue = re.search(r"\b(?:torque|torque value|torque callout)\s*(?:is|to|:)?\s*([^.]+)", text, flags=re.IGNORECASE)
    if torque_cue:
        value = torque_cue.group(1).strip(" .")
        if value:
            return value
    match = re.search(r"\b\d+(?:\.\d+)?\s?(?:in-?lb|ft-?lb|n-?m|nm)\b", text, flags=re.IGNORECASE)
    return match.group(0) if match else "Not required"


def media_for_step(frames, index, total):
    if not frames:
        return []
    frame_index = min(len(frames) - 1, round(index / max(1, total - 1) * (len(frames) - 1)))
    return [frames[frame_index].get("stored_path")]


def step_alignment_words(step):
    return keywords_for_text(
        " ".join(
            [
                step.get("text") or "",
                " ".join(step.get("parts") or []),
                step.get("quality") or "",
                step.get("safety") or "",
            ]
        )
    )


def fallback_step_timing(step_index, total_steps, duration_seconds):
    duration = max(float(duration_seconds or 0), float(max(total_steps, 1) * 8))
    if total_steps <= 1:
        target = duration * 0.5
        span = min(duration, 18)
    else:
        target = (step_index / (total_steps - 1)) * duration
        span = max(10, duration / max(total_steps, 1) * 0.8)
    start = max(0, target - span * 0.45)
    end = min(duration, max(target + span * 0.55, start + 4))
    return start, end, target


def refine_step_timing_with_words(step_words, timed_words, start, end):
    if not step_words or not timed_words:
        return None
    window_start = max(0, float(start or 0) - 1.5)
    window_end = float(end or 0) + 2.5
    matches = [
        word
        for word in timed_words
        if window_start <= word.get("start", 0) <= window_end
        and word.get("token") in step_words
    ]
    if len(matches) < 2:
        return None
    refined_start = max(0, float(matches[0]["start"]) - 0.5)
    refined_end = max(float(matches[-1]["end"]) + 1.0, refined_start + 2.0)
    refined_end = min(max(end, refined_end), refined_end + 2.0)
    target = min(refined_end, max(refined_start, float(matches[-1]["end"]) + 0.75))
    return refined_start, refined_end, target, len(matches)


def assign_step_timings(job, sections):
    frames = frame_analysis_records(job)
    duration = max([float(frame.get("timestamp_seconds") or 0) for frame in frames] or [0])
    segments = transcript_timed_segments(job, duration)
    timed_words = transcript_timed_words(job)
    flat_steps = [
        step
        for section in sections
        for step in section.get("steps") or []
    ]
    total_steps = len(flat_steps)
    matched = 0
    cursor = 0
    last_end = 0
    for step_index, step in enumerate(flat_steps):
        step_words = step_alignment_words(step)
        source_start = timed_seconds(step.get("source_step_start"))
        source_end = timed_seconds(step.get("source_step_end"))
        if source_start is not None and source_end is not None and source_end >= source_start:
            start = max(last_end, source_start)
            end = max(source_end, start + 0.5)
            target = min(end, max(start, end - 0.5))
            step["start_time"] = round(start, 2)
            step["end_time"] = round(end, 2)
            step["target_time"] = round(target, 2)
            step["timing_source"] = "rough_step_boundary"
            step["timing_confidence"] = step.get("transcript_confidence") or step.get("timing_confidence") or "medium"
            last_end = max(last_end, end)
            matched += 1
            continue
        best = None
        expected_segment = round(step_index / max(1, total_steps - 1) * max(0, len(segments) - 1)) if segments else 0
        for segment_index in range(cursor, len(segments)):
            segment = segments[segment_index]
            segment_words = keywords_for_text(segment.get("text") or "")
            overlap = len(step_words & segment_words)
            if not overlap:
                continue
            proximity_penalty = abs(segment_index - max(cursor, expected_segment)) * 0.2
            score = overlap * 10 - proximity_penalty
            if segment_index < cursor:
                score -= 4
            if best is None or score > best[0]:
                best = (score, overlap, segment_index, segment)

        if best and best[0] >= 7:
            _, overlap, segment_index, segment = best
            start = max(last_end, max(0, float(segment.get("start") or 0) - 2))
            end = min(max(duration, float(segment.get("end") or 0) + 5), max(float(segment.get("end") or 0) + 5, start + 5))
            target = min(end, start + (end - start) * 0.72)
            refined = refine_step_timing_with_words(step_words, timed_words, start, end)
            word_match_count = 0
            if refined:
                start, end, target, word_match_count = refined
                start = max(last_end, start)
                end = max(end, start + 2.0)
                target = min(end, max(start, target))
            step["start_time"] = round(start, 2)
            step["end_time"] = round(end, 2)
            step["target_time"] = round(target, 2)
            step["timing_source"] = "word_timestamps" if refined else (segment.get("source") or "transcript")
            step["timing_confidence"] = "high" if refined or overlap >= 3 else "medium"
            if refined:
                step["timing_word_matches"] = word_match_count
            cursor = max(cursor, segment_index + 1)
            last_end = max(last_end, end)
            matched += 1
        else:
            start, end, target = fallback_step_timing(step_index, total_steps, duration)
            start = max(start, last_end)
            end = max(end, start + 5)
            if duration:
                end = min(duration, end)
                if end <= start:
                    start = max(0, min(start, duration - 5))
                    end = min(duration, start + 5)
            target = min(end, max(start, target))
            step["start_time"] = round(start, 2)
            step["end_time"] = round(end, 2)
            step["target_time"] = round(target, 2)
            step["timing_source"] = "proportional"
            step["timing_confidence"] = "low"
            last_end = max(last_end, end)
    return {
        "steps": total_steps,
        "matched_steps": matched,
        "segments": len(segments),
        "duration_seconds": round(duration, 2),
    }


def title_for_group(group, fallback_index):
    joined = " ".join(group).lower()
    if any(word in joined for word in ["box", "kit", "manual", "bag", "contents"]):
        return "Prepare kits and source material"
    if any(word in joined for word in ["driver", "torso", "helmet", "wrench", "hair"]):
        return "Assemble driver subassembly"
    if any(word in joined for word in ["chassis", "axle", "wheel", "side panel", "hub"]):
        return "Build chassis subassembly"
    return f"Assembly section {fallback_index}"


def group_actions(actions):
    groups = []
    current = []
    current_key = None
    for action in actions:
        lowered = action.lower()
        explicit_boundary = bool(re.search(r"\b(new step|next step|start step|step complete)\b", lowered))
        if any(word in lowered for word in ["box", "kit", "manual", "bag", "contents"]):
            key = "prep"
        elif any(word in lowered for word in ["driver", "torso", "helmet", "wrench", "hair"]):
            key = "driver"
        elif any(word in lowered for word in ["chassis", "axle", "wheel", "panel", "hub", "piece"]):
            key = "chassis"
        else:
            key = current_key or "assembly"
        if current and (explicit_boundary or key != current_key):
            groups.append(current)
            current = []
        current_key = key
        current.append(action)
        if len(current) >= 7:
            groups.append(current)
            current = []
            current_key = None
    if current:
        groups.append(current)
    return groups


def transcript_questions_for(job):
    text = transcript_text(job)
    if not text:
        return []
    questions = []
    lowered = text.lower()
    fastener_context = re.search(r"\b(install|insert|tighten|secure|fasten|attach|torque)\b[^.]{0,80}\b(bolt|screw|fastener|nut)\b|\b(bolt|screw|fastener|nut)\b[^.]{0,80}\b(install|insert|tighten|secure|fasten|attach|torque)\b", lowered)
    torque_value = re.search(r"\b\d+(?:\.\d+)?\s?(?:in-?lb|ft-?lb|n-?m|nm)\b", lowered)
    if fastener_context and not torque_value:
        questions.append(
            {
                "id": short_id("q"),
                "job_id": job["id"],
                "title": "Confirm torque",
                "prompt": "The transcript mentions a fastener but does not include a torque value. Should this step require torque?",
                "source": "transcript",
                "why": "A fastener appears without a torque value.",
                "choices": ["No torque required", "Add torque value", "Use standard work"],
                "status": "pending",
                "answer": "",
                "attachment_note": "",
                "attachment_files": [],
            }
        )
    tool_frames = [
        frame
        for frame in job.get("frame_analysis", [])
        if frame.get("tool_visible") and not frame.get("discard")
    ]
    named_tool = re.search(r"\b(tool|driver|screwdriver|wrench|torque tool|drill|rivet gun|welder|fixture|jig)\b", lowered)
    if tool_frames and not named_tool:
        questions.append(
            {
                "id": short_id("q"),
                "job_id": job["id"],
                "title": "Confirm tool",
                "prompt": "A tool appears in video frames, but the transcript does not name it. Should a tool be listed for these steps?",
                "source": "video",
                "why": "Vision detected a possible tool without a matching transcript cue.",
                "choices": ["No tool required", "Add tool name", "Operator hand only"],
                "status": "pending",
                "answer": "",
                "attachment_note": "",
                "attachment_files": [],
            }
        )
    ambiguous = [
        sentence
        for sentence in split_sentences(text)
        if re.search(r"\b(maybe|not sure|unclear|never mind)\b", sentence, flags=re.IGNORECASE)
    ]
    if ambiguous:
        questions.append(
            {
                "id": short_id("q"),
                "job_id": job["id"],
                "title": "Clarify transcript",
                "prompt": f"Clarify this transcript wording before drafting: {ambiguous[0]}",
                "source": "transcript",
                "why": "The transcript contains uncertain wording.",
                "choices": ["Use transcript wording", "Correct part name", "Not applicable"],
                "status": "pending",
                "answer": "",
                "attachment_note": "",
                "attachment_files": [],
            }
        )
    return questions[:3]


def question_review_for(job, questions):
    text = transcript_text(job)
    lowered = text.lower()
    checks = []
    bypass_reasons = []
    question_titles = {question.get("title") for question in questions}

    def add_check(label, result, detail):
        checks.append({"label": label, "result": result, "detail": detail})
        if result == "cleared":
            bypass_reasons.append(detail)

    torque_present = bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:in-?lb|ft-?lb|n-?m|nm)\b", lowered))
    fastener_context = bool(re.search(r"\b(install|insert|tighten|secure|fasten|attach|torque)\b[^.]{0,80}\b(bolt|screw|fastener|nut)\b|\b(bolt|screw|fastener|nut)\b[^.]{0,80}\b(install|insert|tighten|secure|fasten|attach|torque)\b", lowered))
    if "Confirm torque" in question_titles:
        add_check("Torque", "question", "Fastener context appears without a torque value.")
    elif torque_present:
        add_check("Torque", "cleared", "Torque value appears in transcript.")
    elif fastener_context:
        add_check("Torque", "cleared", "Fastener context found, but wording did not require a torque decision.")
    else:
        add_check("Torque", "cleared", "No fastener torque ambiguity detected.")

    tool_frames = [frame for frame in job.get("frame_analysis", []) if frame.get("tool_visible") and not frame.get("discard")]
    named_tool = bool(re.search(r"\b(tool|driver|screwdriver|wrench|torque tool|drill|rivet gun|welder|fixture|jig)\b", lowered))
    if "Confirm tool" in question_titles:
        add_check("Tools", "question", "Vision detected a possible tool that was not named.")
    elif tool_frames and named_tool:
        add_check("Tools", "cleared", "Tool cue appears in transcript and video.")
    elif tool_frames:
        add_check("Tools", "cleared", "Tool-like frames were detected but confidence was not high enough to ask.")
    else:
        add_check("Tools", "cleared", "No unclear tool use detected.")

    if "Clarify transcript" in question_titles:
        add_check("Transcript clarity", "question", "Transcript contains uncertain wording.")
    else:
        add_check("Transcript clarity", "cleared", "No uncertain transcript wording detected.")

    qc_safety_cues = bool(re.search(r"\b(quality check|verify|inspect|confirm|flush|seated|safety|caution|carefully)\b", lowered))
    if qc_safety_cues:
        add_check("Quality / Safety", "cleared", "Quality or safety cues were retained for drafting.")
    else:
        add_check("Quality / Safety", "cleared", "No unresolved quality or safety ambiguity detected.")

    return {
        "generated_questions": len(questions),
        "checks": checks,
        "bypass_reasons": bypass_reasons,
        "confidence_note": "User decisions required before drafting." if questions else "No actionable user questions were generated; checks were auto-cleared.",
    }


def question_context(job):
    answered = []
    for question in job.get("questions", []):
        if question.get("status") == "answered" and question.get("answer"):
            answered.append(f"{question.get('title')}: {question.get('answer')}")
        elif question.get("status") == "bypassed":
            answered.append(f"{question.get('title')}: bypassed")
    return "\n".join(answered)


def cue_summary(job):
    text = transcript_text(job).lower()
    cues = []
    cue_patterns = [
        ("Step cues", r"\b(new step|next step|start step|step complete)\b"),
        ("Tools", r"\b(tool|tooling|driver|screwdriver|torque tool)\b"),
        ("Parts", r"\b(part|parts|component|kit|bag)\b"),
        ("Torque", r"\b(torque|in-?lb|ft-?lb|n-?m|nm)\b"),
        ("Threadlocker", r"\b(loctite|locktite|threadlocker|thread locker)\b"),
        ("Fixtures", r"\b(fixture|jig|clamp)\b"),
        ("Quality", r"\b(quality check|verify|inspect|confirm|flush|seated)\b"),
        ("Safety", r"\b(safety|caution|carefully)\b"),
    ]
    for label, pattern in cue_patterns:
        count = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if count:
            cues.append({"label": label, "count": count})
    return cues


def parse_expected_step_count(value):
    if value in {None, ""}:
        return None
    try:
        count = int(float(value))
    except Exception:
        return None
    if count <= 0:
        return None
    return min(200, count)


def normalize_focus_area(value):
    value = (value or "center").strip().lower()
    return value if value in {"center", "full"} else "center"


def make_job(name, product, work_area, station, profile="Standard", files=None, expected_step_count=None, focus_area="center"):
    job_id = short_id("job")
    job = {
        "id": job_id,
        "job_no": f"J-{len(JOBS) + 101}",
        "name": name,
        "product": product,
        "work_area": work_area,
        "station": station,
        "profile": profile,
        "expected_step_count": parse_expected_step_count(expected_step_count),
        "focus_area": normalize_focus_area(focus_area),
        "status": "Ready",
        "active": True,
        "deleted": False,
        "processing_started": False,
        "needs_processing": False,
        "progress": 0,
        "flags": [],
        "files": files or [],
        "stages": base_stages(),
        "questions": [],
        "question_review": {},
        "document_evidence": [],
        "transcript_step_slices": [],
        "step_parse": {},
        "draft_sections": [],
        "processing_runs": [],
        "active_processing_run_id": "",
        "logs": [],
        "created_at": timestamp(),
        "last_activity": timestamp(),
    }
    JOBS[job_id] = job
    normalize_job_sources(job)
    add_log(job, "Job created.")
    return job


def initialize_empty_state():
    save_state()


def migrate_job_defaults(job):
    job.setdefault("deleted", False)
    job.setdefault("needs_processing", False)
    job.setdefault(
        "processing_started",
        any(log.get("message") == "Processing queued." for log in job.get("logs", [])),
    )
    job.setdefault("draft_source", "")
    job.setdefault("frame_analysis", [])
    job.setdefault("question_review", {})
    job.setdefault("document_evidence", [])
    job.setdefault("draft_validation", {})
    job.setdefault("draft_audit", {})
    job.setdefault("processing_cache", {})
    job.setdefault("processing_runs", [])
    job.setdefault("active_processing_run_id", "")
    job.setdefault("watchdog", {})
    job.setdefault("last_activity_epoch", epoch_now())
    job.setdefault("expected_step_count", None)
    job.setdefault("focus_area", "center")
    job.setdefault("transcript_step_slices", [])
    job.setdefault("step_parse", {})
    normalize_job_sources(job)
    if (
        job.get("processing_started")
        and not active_job_files(job)
        and job.get("status") not in {"Ready", "Deleted"}
    ):
        job["status"] = "Needs Attention"
        job["needs_processing"] = False
        flags = job.setdefault("flags", [])
        if "No uploaded files" not in flags:
            flags.append("No uploaded files")
        upload_stage = job.get("stages", {}).get("upload_check")
        if upload_stage:
            upload_stage["status"] = "failed"
            upload_stage["percent"] = 0
            upload_stage["note"] = "No uploaded files attached"
        recompute_progress(job)
    for stage in job.get("stages", {}).values():
        stage.setdefault("updated_at", datestamp())
        stage.setdefault("updated_epoch", job.get("last_activity_epoch") or epoch_now())
        stage.setdefault("started_at", "")
        stage.setdefault("started_epoch", None)
        stage.setdefault("ended_at", "")
        stage.setdefault("ended_epoch", None)
        stage.setdefault("duration_seconds", None)
    job.setdefault("processing_timing", {})
    for artifact in job.get("artifacts", []):
        if artifact.get("kind") == "frame" and "timestamp_seconds" not in artifact:
            match = re.search(r"~?(\d+(?:\.\d+)?)s", str(artifact.get("timestamp") or ""))
            if match:
                artifact["timestamp_seconds"] = float(match.group(1))


def extract_json_text_from_response(data):
    if isinstance(data, dict) and data.get("output_text"):
        return data["output_text"]
    for output in data.get("output", []) if isinstance(data, dict) else []:
        for content in output.get("content", []) if isinstance(output, dict) else []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return content["text"]
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def call_openai_json(prompt, model=None, timeout=180):
    api_key = get_openai_api_key()
    if not api_key:
        return None, "missing_api_key"
    body = json.dumps(
        {
            "model": model or SETTINGS.get("instruction_model") or "gpt-4o-mini",
            "input": prompt,
            "text": {"format": {"type": "json_object"}},
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib_request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = extract_json_text_from_response(data)
        if not text:
            return None, "openai_no_json_text"
        return json.loads(text), None
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"openai_http_{exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"openai_error: {exc}"


def frame_data_url(rel_path):
    path = APP_DIR / rel_path
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def call_openai_frame_json(prompt, frames, model=None, timeout=240):
    api_key = get_openai_api_key()
    if not api_key:
        return None, "missing_api_key"
    content = [{"type": "input_text", "text": prompt}]
    for frame in frames:
        rel_path = frame.get("stored_path")
        if not rel_path:
            continue
        try:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Frame {frame.get('frame_label')}: {rel_path} at {frame.get('timestamp_seconds', 0)} seconds. "
                        f"Source time: {frame.get('source_timestamp_seconds', frame.get('timestamp_seconds', 0))} seconds. "
                        f"Local cue tags: {', '.join(frame.get('candidate_tags') or []) or 'none'}. "
                        f"Local cue reason: {frame.get('candidate_reason') or 'none'}. "
                        f"Sampling: {frame.get('sampling_tier') or 'unknown'}"
                        f"{' - ' + frame.get('sampling_reason') if frame.get('sampling_reason') else ''}. "
                        f"Local priority: {frame.get('local_priority', 0)}."
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": frame_data_url(rel_path),
                    "detail": "low",
                }
            )
        except Exception:
            continue
    if len(content) <= 1:
        return None, "no_readable_frames"
    body = json.dumps(
        {
            "model": model or SETTINGS.get("instruction_model") or "gpt-4o-mini",
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_object"}},
            "temperature": 0.1,
        }
    ).encode("utf-8")
    req = urllib_request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = extract_json_text_from_response(data)
        if not text:
            return None, "openai_no_json_text"
        return json.loads(text), None
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"openai_http_{exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"openai_error: {exc}"


def clamp_score(value, default=0):
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return default


def normalize_frame_analysis(frame, incoming):
    incoming = incoming or {}
    pointing = bool(incoming.get("pointing") or incoming.get("operator_pointing"))
    instruction_value = clamp_score(incoming.get("instruction_value"), 45)
    clarity = clamp_score(incoming.get("clarity"), 50)
    relevance = clamp_score(incoming.get("assembly_relevance"), 50)
    tags = set(incoming.get("tags") or [])
    tags.update(frame.get("candidate_tags") or [])
    visible_parts = incoming.get("visible_parts") or []
    operator_image_cue = "good_image" in tags or bool(OPERATOR_IMAGE_CUE_RE.search(frame.get("candidate_reason") or ""))
    blocked = bool(incoming.get("blocked_or_obstructed"))
    if pointing:
        instruction_value = min(100, instruction_value + 12)
        tags.add("pointing")
    if incoming.get("newly_added_part_visible"):
        instruction_value = min(100, instruction_value + 8)
        tags.add("newly_added_part_visible")
    after_state = bool(incoming.get("after_state"))
    usable_operator_image_cue = operator_image_cue and not blocked
    if usable_operator_image_cue:
        tags.add("operator_image_cue")
        clarity = max(clarity, 65)
        relevance = max(relevance, 65)
        instruction_value = max(instruction_value, 72)
        if visible_parts and not incoming.get("before_state"):
            after_state = True
            tags.add("after_state")
    for key in ["tool_visible", "fastener_visible", "quality_check", "before_state", "after_state", "blocked_or_obstructed"]:
        if incoming.get(key):
            tags.add(key)
    discard = (bool(incoming.get("discard")) or blocked) and not pointing and not usable_operator_image_cue
    return {
        "stored_path": frame.get("stored_path"),
        "frame_label": frame.get("frame_label"),
        "timestamp_seconds": frame.get("timestamp_seconds", 0),
        "source_timestamp_seconds": frame.get("source_timestamp_seconds"),
        "timeline_offset_seconds": frame.get("timeline_offset_seconds", 0),
        "visual_summary": incoming.get("visual_summary") or incoming.get("action_summary") or "",
        "visible_parts": visible_parts,
        "pointing": pointing,
        "pointing_target": incoming.get("pointing_target") or "",
        "newly_added_part_visible": bool(incoming.get("newly_added_part_visible")),
        "tool_visible": bool(incoming.get("tool_visible")),
        "fastener_visible": bool(incoming.get("fastener_visible")),
        "quality_check": bool(incoming.get("quality_check")),
        "before_state": bool(incoming.get("before_state")),
        "after_state": after_state,
        "blocked_or_obstructed": bool(incoming.get("blocked_or_obstructed")),
        "tags": sorted(tags),
        "clarity": clarity,
        "assembly_relevance": relevance,
        "instruction_value": instruction_value,
        "discard": discard,
        "reason": incoming.get("reason") or "",
        "local_priority": frame.get("local_priority", 0),
        "candidate_reason": frame.get("candidate_reason", ""),
        "opencv_metrics": frame.get("opencv_metrics") or {},
        "sampling_reason": frame.get("sampling_reason") or "",
        "sampling_tier": frame.get("sampling_tier") or "",
        "sample_rate_hz": frame.get("sample_rate_hz"),
    }


def normalize_ai_sections(job, data):
    frame_paths = {frame.get("stored_path") for frame in frame_artifacts(job)}
    sections = []
    for section in (data or {}).get("sections", []):
        steps = []
        for step in section.get("steps", []):
            media = [
                path
                for path in step.get("media", [])
                if path in frame_paths
            ]
            steps.append(
                {
                    "text": step.get("text") or "",
                    "tools": step.get("tools") or "Hands",
                    "torque": step.get("torque") or "Not required",
                    "quality": step.get("quality") or "",
                    "safety": step.get("safety") or "",
                    "parts": step.get("parts") or [],
                    "media": media,
                    "source_step_id": step.get("source_step_id") or "",
                }
            )
        steps = [step for step in steps if step["text"]]
        if not steps:
            continue
        sections.append(
            {
                "id": short_id("draft"),
                "title": section.get("title") or f"Assembly section {len(sections) + 1}",
                "product": job["product"],
                "work_area": job["work_area"],
                "station": job["station"],
                "tags": section.get("tags") or [],
                "steps": steps,
                "draft_source": "openai",
            }
        )
    return sections


def rough_step_prompt_payload(rough_steps):
    payload = []
    for step in rough_steps:
        raw_text = re.sub(r"\s+", " ", step.get("raw_text") or "").strip()
        payload.append(
            {
                "id": step.get("id"),
                "index": step.get("index"),
                "start_time": step.get("start_time"),
                "end_time": step.get("end_time"),
                "raw_text": raw_text[:900],
                "boundary_reason": step.get("boundary_reason"),
                "confidence": step.get("confidence"),
            }
        )
    return payload


def apply_rough_step_metadata(sections, rough_steps):
    rough_by_id = {step.get("id"): step for step in rough_steps if step.get("id")}
    flat_steps = [
        step
        for section in sections
        for step in section.get("steps") or []
    ]
    for index, step in enumerate(flat_steps):
        rough = rough_by_id.get(step.get("source_step_id"))
        if not rough and index < len(rough_steps):
            rough = rough_steps[index]
            step["source_step_id"] = rough.get("id")
        if not rough:
            continue
        start = timed_seconds(rough.get("start_time"))
        end = timed_seconds(rough.get("end_time"))
        if start is not None and end is not None and end >= start:
            step["source_step_start"] = round(start, 2)
            step["source_step_end"] = round(end, 2)
            step["start_time"] = round(start, 2)
            step["end_time"] = round(max(end, start + 0.5), 2)
            step["target_time"] = round(max(start, end - 0.5), 2)
            step["timing_source"] = "rough_step_boundary"
            step["timing_confidence"] = rough.get("confidence") or "medium"
        step["raw_transcript"] = rough.get("raw_text") or ""
        step["boundary_reason"] = rough.get("boundary_reason") or ""
        step["transcript_confidence"] = rough.get("confidence") or "medium"
    return sections


def openai_draft_sections_for(job):
    text = transcript_text(job)
    if not text:
        return [], "missing_transcript"
    frames = [frame.get("stored_path") for frame in frame_artifacts(job)]
    rough_steps = transcript_step_slices(job)
    expected_count = expected_step_count_for(job)
    supporting_context = document_context(job)
    transcript_excerpt = text[:12000]
    if len(text) > len(transcript_excerpt):
        transcript_excerpt += "\n\n[Transcript truncated for prompt; rough timestamped steps above are primary.]"
    prompt = f"""
You are creating manufacturing assembly instruction sections from a real processing transcript.
Return JSON only, with this exact shape:
{{
  "sections": [
    {{
      "title": "short section title",
      "tags": ["tag"],
      "steps": [
        {{
          "source_step_id": "rough-001",
          "text": "imperative instruction",
          "tools": "Hands or observed tool",
          "torque": "Not required or exact value",
          "quality": "observable quality check or empty",
          "safety": "specific safety callout or empty",
          "parts": ["parts or consumables named in transcript"],
          "media": ["one or more provided frame paths"]
        }}
      ]
    }}
  ]
}}

Rules:
- Do not invent drawings, frame names, torque values, tools, parts, or safety risks.
- Use supporting-document facts only when the quoted evidence applies to this product/configuration.
- Prefer authoritative sources over supporting context. Surface contradictions instead of silently choosing one.
- Do not invent technical values when a document is absent, unreadable, revision-unknown, or ambiguous.
- Use only these frame paths in media: {json.dumps(frames[:160])}
- Prefer clear, concrete, shop-floor wording.
- Rewrite rough operator narration into professional, direct operator language. Do not copy filler words.
- Split into reusable sections based on natural subassemblies.
- Keep each step specific and actionable.
- Use the rough transcript steps as the primary step boundaries. Preserve source_step_id on each returned step.
- Do not collapse multiple physical placements into one step. If the rough step contains multiple clear placements, split it and reuse the same source_step_id.
- For a detailed assembly video, prefer a complete step list over a short summary. A normal result may be 10-30 steps.
- If expected step count is provided, stay close unless the transcript clearly proves otherwise.
- If a fastener has no torque value, write "Not required" only if the transcript does not imply torque control.
- The word "wrench" is a part if the transcript says the driver holds it; do not call it an assembly tool unless it is used as a tool.

Job:
Product: {job.get("product")}
Work area: {job.get("work_area")}
Station: {job.get("station")}
Expected step count: {expected_count or "not provided"}

Question answers:
{question_context(job) or "None"}

Rough transcript steps with timestamps:
{json.dumps(rough_step_prompt_payload(rough_steps), indent=2)}

Transcript excerpt:
{transcript_excerpt}

Supporting document evidence:
{supporting_context or "None processed"}
"""
    data, error = call_openai_json(prompt)
    if error:
        return [], error
    sections = normalize_ai_sections(job, data)
    sections = apply_rough_step_metadata(sections, rough_steps)
    return sections, None if sections else "openai_returned_no_sections"


def heuristic_draft_sections_for(job):
    text = transcript_text(job)
    frames = frame_artifacts(job)
    rough_steps = transcript_step_slices(job)
    if rough_steps:
        actions = [normalize_instruction(step.get("raw_text") or "") for step in rough_steps]
        actions = [action for action in actions if action]
    else:
        actions = action_sentences(text)
    if not actions:
        actions = ["Review the transcript and create assembly steps before releasing this section."]
    groups = group_actions(actions)
    total_steps = sum(len(group) for group in groups)
    step_counter = 0
    rough_counter = 0
    sections = []
    for group_index, group in enumerate(groups, start=1):
        steps = []
        for action in group:
            rough = rough_steps[rough_counter] if rough_counter < len(rough_steps) else {}
            media = media_for_step(frames, step_counter, total_steps)
            step = {
                "text": action,
                "tools": step_tools(action),
                "torque": step_torque(action),
                "quality": step_quality(action),
                "safety": step_safety(action),
                "parts": step_parts(action),
                "media": media,
                "source_step_id": rough.get("id") or "",
            }
            if rough:
                apply_rough_step_metadata([{"steps": [step]}], [rough])
            steps.append(step)
            step_counter += 1
            rough_counter += 1
        sections.append(
            {
                "id": short_id("draft"),
                "title": title_for_group(group, group_index),
                "product": job["product"],
                "work_area": job["work_area"],
                "station": job["station"],
                "tags": [tag for tag in [job.get("product"), "transcript draft"] if tag],
                "steps": steps,
                "draft_source": "heuristic",
            }
        )
    return sections


def draft_sections_for(job):
    rough_steps = transcript_step_slices(job)
    job["transcript_step_slices"] = rough_steps
    job["step_parse"] = {
        "expected_step_count": expected_step_count_for(job),
        "rough_step_count": len(rough_steps),
        "source": "word_timestamps" if rough_steps and rough_steps[0].get("source") == "word_timestamps" else "segments" if rough_steps else "none",
        "high_confidence_steps": sum(1 for step in rough_steps if step.get("confidence") == "high"),
        "medium_confidence_steps": sum(1 for step in rough_steps if step.get("confidence") == "medium"),
        "low_confidence_steps": sum(1 for step in rough_steps if step.get("confidence") == "low"),
    }
    if rough_steps:
        add_log(job, f"Transcript parser created {len(rough_steps)} rough timed steps" + (f" against expected {job['step_parse']['expected_step_count']}." if job["step_parse"].get("expected_step_count") else "."))
    sections, error = openai_draft_sections_for(job)
    if sections:
        job["draft_source"] = "openai"
        return sections
    if error and error != "missing_api_key":
        add_log(job, f"AI draft generation failed; using transcript fallback: {error}")
    job["draft_source"] = "heuristic"
    return heuristic_draft_sections_for(job)


STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "assembly", "attach", "before", "being",
    "both", "build", "down", "each", "find", "from", "going", "have", "here", "into",
    "just", "need", "next", "onto", "part", "parts", "piece", "place", "press",
    "section", "seen", "step", "subassembly", "that", "the", "then", "there", "these", "this", "those", "until", "we",
    "will", "with", "we're", "weve", "you",
}


def keywords_for_text(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
    return {
        word.strip("-_")
        for word in words
        if len(word.strip("-_")) > 2 and word.strip("-_") not in STOPWORDS
    }


def frame_analysis_records(job):
    by_path = {
        item.get("stored_path"): item
        for item in job.get("frame_analysis", [])
        if item.get("stored_path")
    }
    records = []
    for artifact in frame_artifacts(job):
        rel_path = artifact.get("stored_path")
        analysis = by_path.get(rel_path) or artifact.get("analysis") or {}
        records.append(
            {
                "stored_path": rel_path,
                "source_file": artifact.get("source_file") or analysis.get("source_file") or "",
                "timestamp_seconds": artifact.get("timestamp_seconds", analysis.get("timestamp_seconds", 0)),
                "source_timestamp_seconds": artifact.get("source_timestamp_seconds"),
                "timeline_offset_seconds": artifact.get("timeline_offset_seconds", 0),
                "visual_summary": analysis.get("visual_summary") or "",
                "visible_parts": analysis.get("visible_parts") or [],
                "pointing": bool(analysis.get("pointing") or artifact.get("pointing")),
                "pointing_target": analysis.get("pointing_target") or "",
                "newly_added_part_visible": bool(analysis.get("newly_added_part_visible")),
                "tool_visible": bool(analysis.get("tool_visible")),
                "fastener_visible": bool(analysis.get("fastener_visible")),
                "quality_check": bool(analysis.get("quality_check")),
                "before_state": bool(analysis.get("before_state")),
                "after_state": bool(analysis.get("after_state")),
                "blocked_or_obstructed": bool(analysis.get("blocked_or_obstructed")),
                "tags": analysis.get("tags") or artifact.get("candidate_tags") or [],
                "opencv_metrics": analysis.get("opencv_metrics") or artifact.get("opencv_metrics") or {},
                "sampling_reason": analysis.get("sampling_reason") or artifact.get("sampling_reason") or "",
                "sampling_tier": analysis.get("sampling_tier") or artifact.get("sampling_tier") or "",
                "sample_rate_hz": analysis.get("sample_rate_hz") or artifact.get("sample_rate_hz"),
                "clarity": clamp_score(analysis.get("clarity"), 50),
                "assembly_relevance": clamp_score(analysis.get("assembly_relevance"), 50),
                "instruction_value": clamp_score(analysis.get("instruction_value") or artifact.get("instruction_value"), 45),
                "discard": bool(analysis.get("discard") or artifact.get("discard")),
                "reason": analysis.get("reason") or "",
            }
        )
    return sorted(records, key=lambda item: item.get("timestamp_seconds", 0))


def frame_words(frame):
    return keywords_for_text(
        " ".join(
            [
                frame.get("visual_summary") or "",
                " ".join(frame.get("visible_parts") or []),
                frame.get("pointing_target") or "",
                " ".join(frame.get("tags") or []),
            ]
        )
    )


def section_context_words(section):
    texts = [
        section.get("title") or "",
        " ".join(section.get("tags") or []),
    ]
    for step in section.get("steps") or []:
        texts.extend(
            [
                step.get("text") or "",
                " ".join(step.get("parts") or []),
                step.get("quality") or "",
                step.get("safety") or "",
            ]
        )
    return keywords_for_text(" ".join(texts))


def section_contexts(sections):
    return [section_context_words(section) for section in sections]


def competing_context_words(contexts, current_index):
    current = contexts[current_index] if current_index < len(contexts) else set()
    words = set()
    for index, context in enumerate(contexts):
        if index != current_index:
            words.update(context - current)
    return words


def frame_context_match(frame, current_context=None, competing_context=None):
    words = frame_words(frame)
    current_hits = sorted(words & (current_context or set()))
    competing_hits = sorted(words & (competing_context or set()))
    return current_hits, competing_hits


def frame_matches_competing_context(frame, current_context=None, competing_context=None):
    current_hits, competing_hits = frame_context_match(frame, current_context, competing_context)
    return bool(competing_hits) and len(competing_hits) >= max(1, len(current_hits) + 1)


def step_exact_media_window(step, target_time, duration_seconds=None):
    start = timed_seconds(step.get("start_time"))
    end = timed_seconds(step.get("end_time"))
    target = float(target_time or 0)
    if start is None or end is None:
        start = max(0, target - STEP_EXACT_MIN_WINDOW_SECONDS / 2)
        end = target + STEP_EXACT_MIN_WINDOW_SECONDS / 2
    if end <= start:
        start = max(0, min(start, target) - 0.4)
        end = max(start + STEP_EXACT_MIN_WINDOW_SECONDS, target + 0.9)
    if end - start < STEP_EXACT_MIN_WINDOW_SECONDS:
        center = min(max(target, start), end)
        start = min(start, max(0, center - 0.45))
        end = max(end, center + STEP_EXACT_MIN_WINDOW_SECONDS - (center - start))
    if duration_seconds:
        end = min(float(duration_seconds), end)
        start = max(0, min(start, end))
    return start, end


def step_timing_alignment_adjustment(frame, target_time, exact_start=None, exact_end=None):
    if exact_start is None or exact_end is None:
        return 0
    timestamp_seconds = float(frame.get("timestamp_seconds") or 0)
    tags = set(frame.get("tags") or [])
    strong_after_state = bool(
        frame.get("pointing")
        or frame.get("after_state")
        or frame.get("newly_added_part_visible")
        or frame.get("quality_check")
        or tags & {"visual_change", "focus_change", "tool"}
    )
    if exact_start <= timestamp_seconds <= exact_end:
        return 24
    if timestamp_seconds < exact_start:
        distance = exact_start - timestamp_seconds
        return -min(48, 10 * distance)
    distance = timestamp_seconds - exact_end
    if strong_after_state and distance <= 2.75:
        return 8 - min(18, 5 * distance)
    return -min(42, 6 * distance)


def score_frame_for_step(frame, step, target_time, window_start=None, window_end=None, current_context=None, competing_context=None, exact_start=None, exact_end=None):
    step_words = keywords_for_text(
        " ".join(
            [
                step.get("text") or "",
                " ".join(step.get("parts") or []),
                step.get("quality") or "",
            ]
        )
    )
    words = frame_words(frame)
    overlap = len(step_words & words)
    current_hits, competing_hits = frame_context_match(frame, current_context, competing_context)
    timestamp_seconds = float(frame.get("timestamp_seconds") or 0)
    delta = abs(timestamp_seconds - target_time)
    clarity = float(frame.get("clarity") or 0)
    relevance = float(frame.get("assembly_relevance") or 0)
    tags = set(frame.get("tags") or [])
    operator_image_cue = bool(tags & {"good_image", "operator_image_cue"})
    opencv = frame.get("opencv_metrics") or {}
    score = float(frame.get("instruction_value") or 0)
    score += overlap * 14
    score += min(30, len(current_hits) * 8)
    if competing_hits and len(competing_hits) > len(current_hits):
        score -= min(55, 22 + (len(competing_hits) - len(current_hits)) * 10)
    elif competing_hits and not current_hits:
        score -= 22
    score += clarity * 0.1
    score += relevance * 0.14
    if frame.get("pointing"):
        score += 24
    if frame.get("newly_added_part_visible"):
        score += 18
    if frame.get("after_state"):
        score += 18
    if frame.get("before_state") and not frame.get("after_state"):
        score -= 8
    if frame.get("quality_check") and re.search(r"\b(verify|inspect|confirm|quality|check|seated|flush)\b", step.get("quality") or step.get("text") or "", flags=re.IGNORECASE):
        score += 16
    if frame.get("tool_visible") and re.search(r"\b(tool|driver|screwdriver|wrench|torque)\b", step.get("tools") or step.get("text") or "", flags=re.IGNORECASE):
        score += 12
    if frame.get("fastener_visible") and re.search(r"\b(screw|bolt|pin|fastener|torque)\b", step.get("text") or "", flags=re.IGNORECASE):
        score += 12
    if operator_image_cue:
        score += 18
    if "sharp" in tags:
        score += 6
    if "focus_sharp" in tags:
        score += 10
    if "focus_change" in tags:
        score += 22 if window_start is not None and window_end is not None and window_start <= timestamp_seconds <= window_end else 12
    if "background_change" in tags:
        score -= 20
    focus_delta = opencv.get("focus_delta")
    if focus_delta is not None and focus_delta >= OPENCV_FOCUS_CHANGE_MIN:
        score += min(18, float(focus_delta) * 1.8)
    if "blurry" in tags:
        score -= 42
    if "focus_blurry" in tags:
        score -= 28
    if "dark" in tags:
        score -= 14
    if "near_duplicate" in tags:
        score -= 10
    if clarity < 35:
        score -= 35
    elif clarity < 50:
        score -= 14
    if relevance < 35:
        score -= 24
    if delta <= STEP_FRAME_TARGET_WINDOW_SECONDS:
        score += 14
    elif delta <= 20:
        score += max(0, 12 - (delta - STEP_FRAME_TARGET_WINDOW_SECONDS) * 0.8)
    else:
        score -= min(35, delta * 0.08)
    if window_start is not None and window_end is not None:
        if window_start <= timestamp_seconds <= window_end:
            score += 22
            if timestamp_seconds >= target_time:
                score += 4
        else:
            distance = min(abs(timestamp_seconds - window_start), abs(timestamp_seconds - window_end))
            score -= min(80, distance * 2.2)
    score += step_timing_alignment_adjustment(frame, target_time, exact_start, exact_end)
    if operator_image_cue and not frame.get("discard") and not frame.get("blocked_or_obstructed"):
        cue_floor = 86 if window_start is not None and window_end is not None and window_start <= timestamp_seconds <= window_end else 72
        score = max(score, cue_floor)
    if frame.get("discard") or frame.get("blocked_or_obstructed"):
        score -= 90
    return score


def frame_is_diverse_from_chosen(frame, chosen_frames):
    timestamp_seconds = float(frame.get("timestamp_seconds") or 0)
    tags = set(frame.get("tags") or [])
    strong_visual_cue = bool(tags & {"good_image", "visual_change", "quality", "tool"}) or bool(
        frame.get("pointing") or frame.get("after_state") or frame.get("newly_added_part_visible")
    )
    for chosen in chosen_frames:
        chosen_seconds = float(chosen.get("timestamp_seconds") or 0)
        if abs(timestamp_seconds - chosen_seconds) < 2.0 and not strong_visual_cue:
            return False
    return True


def step_media_window(step, step_index, total_steps, duration_seconds, previous_step=None, next_step=None):
    try:
        start = float(step.get("start_time"))
        end = float(step.get("end_time"))
        target = float(step.get("target_time", end))
    except Exception:
        start, end, target = fallback_step_timing(step_index, total_steps, duration_seconds)
    if end <= start:
        end = start + 6
    confidence = step.get("timing_confidence") or "low"
    before_buffer = 0.5 if confidence == "high" else 1.25
    after_buffer = 1.35 if confidence == "high" else 2.75
    available_duration = float(duration_seconds or 0)
    if available_duration > 0:
        duration = available_duration
        if start >= duration:
            original_span = max(0.5, end - start)
            tail_span = max(4.0, min(10.0, original_span + before_buffer + after_buffer))
            end = duration
            start = max(0, duration - tail_span)
            target = max(start, duration - min(0.5, tail_span * 0.2))
        else:
            end = min(end, duration)
            target = min(target, duration)
            if end <= start:
                tail_span = max(4.0, min(10.0, after_buffer + before_buffer + 0.5))
                start = max(0, duration - tail_span)
                end = duration
                target = max(start, duration - min(0.5, tail_span * 0.2))
    else:
        duration = max(float(duration_seconds or 0), end)
    next_start = timed_seconds((next_step or {}).get("start_time"))
    previous_end = timed_seconds((previous_step or {}).get("end_time"))
    if next_start is not None and next_start > start:
        gap_after_words = max(0, next_start - end)
        if gap_after_words >= 2.0:
            after_buffer = min(4.0, max(after_buffer, gap_after_words * 0.35))
            target = min(next_start - 0.35, max(target, end + min(1.25, gap_after_words * 0.3)))
        else:
            target = min(max(target, end), next_start)
    window_start = max(0, start - before_buffer)
    window_end = min(max(duration, end + after_buffer), max(end + after_buffer, start + 6))
    if previous_end is not None and previous_end <= start:
        window_start = max(window_start, previous_end)
    if next_start is not None and next_start > start:
        next_limit = next_start if next_start - end < 1.0 else next_start - 0.35
        window_end = min(window_end, max(window_start + 1.0, next_limit))
    if available_duration > 0:
        window_end = min(window_end, duration)
        if window_end <= window_start:
            window_start = max(0, duration - max(1.0, after_buffer + before_buffer))
            window_end = duration
    target = min(window_end, max(window_start, target))
    return window_start, window_end, target


def frame_has_print_value(frame):
    tags = set(frame.get("tags") or [])
    return bool(
        frame.get("pointing")
        or frame.get("after_state")
        or frame.get("newly_added_part_visible")
        or frame.get("quality_check")
        or tags & {"good_image", "visual_change", "tool", "quality", "sharp"}
    )


def media_decision_record(step_index, step, window_start, window_end, target_time, ranked, selected_paths, current_context=None, competing_context=None):
    selected = set(selected_paths or [])
    candidates = []
    for score, frame in ranked[:12]:
        path = frame.get("stored_path") or ""
        timestamp_seconds = round(float(frame.get("timestamp_seconds") or 0), 2)
        current_hits, competing_hits = frame_context_match(frame, current_context, competing_context)
        if path in selected:
            decision = "selected"
            decision_reason = "Selected for this step."
        elif competing_hits and len(competing_hits) > len(current_hits):
            decision = "rejected_other_section_context"
            decision_reason = "Frame matched neighboring/competing section context more strongly."
        elif score < 38:
            decision = "rejected_low_score"
            decision_reason = "Frame score was below the usable threshold."
        elif not frame_has_print_value(frame):
            decision = "rejected_low_print_value"
            decision_reason = "Frame lacked clear print value such as pointing, after-state, tool, quality, or sharp visual change."
        else:
            decision = "candidate_not_selected"
            decision_reason = "Viable alternate but not the best match."
        candidates.append(
            {
                "path": path,
                "timestamp_seconds": timestamp_seconds,
                "distance_from_target_seconds": round(abs(float(timestamp_seconds or 0) - float(target_time or 0)), 2),
                "score": round(float(score), 2),
                "decision": decision,
                "decision_reason": decision_reason,
                "pointing": bool(frame.get("pointing")),
                "after_state": bool(frame.get("after_state")),
                "newly_added_part_visible": bool(frame.get("newly_added_part_visible")),
                "clarity": frame.get("clarity"),
                "assembly_relevance": frame.get("assembly_relevance"),
                "tags": frame.get("tags") or [],
                "current_context_hits": current_hits[:8],
                "competing_context_hits": competing_hits[:8],
                "summary": frame.get("visual_summary") or frame.get("reason") or "",
            }
        )
    return {
        "step_index": step_index,
        "step_text": step.get("text") or "",
        "timing": {
            "start_time": step.get("start_time"),
            "end_time": step.get("end_time"),
            "target_time": round(float(target_time or 0), 2),
            "window_start": round(float(window_start or 0), 2),
            "window_end": round(float(window_end or 0), 2),
            "source": step.get("timing_source") or "",
            "confidence": step.get("timing_confidence") or "",
        },
        "selected_media": selected_paths or [],
        "candidates": candidates,
    }


def step_duration_seconds(step):
    start = timed_seconds(step.get("start_time"))
    end = timed_seconds(step.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    return end - start


def usable_fallback_frame(frame):
    return bool(frame.get("stored_path")) and not frame.get("discard") and not frame.get("blocked_or_obstructed")


def best_fallback_media_frame(ranked, frames, target_time, step, window_start, window_end, current_context, competing_context, exact_start=None, exact_end=None):
    target = float(target_time or 0)
    for score, frame in ranked:
        if not usable_fallback_frame(frame):
            continue
        if frame_matches_competing_context(frame, current_context, competing_context) and score < 35:
            continue
        distance = abs(float(frame.get("timestamp_seconds") or 0) - target)
        return score, frame, distance
    fallback_pool = []
    for frame in frames:
        if not usable_fallback_frame(frame):
            continue
        timestamp_seconds = float(frame.get("timestamp_seconds") or 0)
        distance = abs(timestamp_seconds - target)
        if distance > STEP_FALLBACK_GLOBAL_SECONDS:
            continue
        if exact_start is not None and exact_end is not None:
            if timestamp_seconds < exact_start - STEP_FALLBACK_NEAR_SECONDS or timestamp_seconds > exact_end + STEP_FALLBACK_NEAR_SECONDS:
                continue
        score = score_frame_for_step(
            frame,
            step,
            target_time,
            window_start,
            window_end,
            current_context,
            competing_context,
            exact_start,
            exact_end,
        )
        if distance > STEP_FALLBACK_NEAR_SECONDS:
            score -= min(30, (distance - STEP_FALLBACK_NEAR_SECONDS) * 5)
        fallback_pool.append((score, distance, frame))
    if fallback_pool:
        score, distance, frame = sorted(
            fallback_pool,
            key=lambda item: (item[1] > STEP_FALLBACK_NEAR_SECONDS, round(item[1] / 1.5), -item[0], item[1]),
        )[0]
        return score, frame, distance
    usable = [frame for frame in frames if usable_fallback_frame(frame)]
    if usable:
        frame = min(usable, key=lambda item: abs(float(item.get("timestamp_seconds") or 0) - target))
        distance = abs(float(frame.get("timestamp_seconds") or 0) - target)
        score = score_frame_for_step(frame, step, target_time, window_start, window_end, current_context, competing_context, exact_start, exact_end)
        return score, frame, distance
    return None, None, None


def merge_recommendation_for_step(step, previous_step, next_step, selected_score, fallback_used, has_media, fallback_distance=None):
    duration = step_duration_seconds(step)
    text = (step.get("text") or "").strip()
    reasons = []
    if duration is not None and duration <= 1.5:
        reasons.append("very short timed window")
    elif duration is not None and duration <= 2.5 and (fallback_used or selected_score < 42):
        reasons.append("short timed window with weak media match")
    if not has_media:
        reasons.append("no usable image found")
    elif fallback_used and selected_score < 38:
        reasons.append("only a low-confidence fallback image was found")
    if fallback_used and fallback_distance is not None and fallback_distance > STEP_FALLBACK_NEAR_SECONDS:
        reasons.append(f"fallback image is {fallback_distance:.1f}s from the target")
    if len(text) < 70 and (fallback_used or selected_score < 42):
        reasons.append("brief instruction text")
    if not reasons:
        return None
    neighbor = previous_step if previous_step else next_step
    direction = "previous" if previous_step else "next" if next_step else ""
    return {
        "recommended": True,
        "direction": direction,
        "reason": "; ".join(dict.fromkeys(reasons)),
        "neighbor_step": (neighbor.get("text") or "")[:140] if neighbor else "",
    }


def attach_best_media_to_drafts(job, sections):
    frames = frame_analysis_records(job)
    if not frames:
        return sections, {"steps": 0, "pointing_frames": 0, "scored_frames": 0}
    flat_steps = [
        step
        for section in sections
        for step in section.get("steps") or []
    ]
    total_steps = len(flat_steps)
    if not total_steps:
        return sections, {"steps": 0, "pointing_frames": 0, "scored_frames": len(frames)}
    duration = max([float(frame.get("timestamp_seconds") or 0) for frame in frames] or [0])
    chronological = sorted(
        enumerate(flat_steps),
        key=lambda item: (
            timed_seconds(item[1].get("start_time"), fallback_step_timing(item[0], total_steps, duration)[0]),
            item[0],
        ),
    )
    neighbor_map = {}
    for chronological_index, (original_index, chronological_step) in enumerate(chronological):
        previous_step = chronological[chronological_index - 1][1] if chronological_index > 0 else None
        next_step = chronological[chronological_index + 1][1] if chronological_index + 1 < len(chronological) else None
        neighbor_map[id(chronological_step)] = {
            "chronological_index": chronological_index,
            "previous_step": previous_step,
            "next_step": next_step,
            "original_index": original_index,
        }
    assigned = 0
    single_image_steps = 0
    double_image_steps = 0
    fallback_image_steps = 0
    merge_candidate_steps = 0
    step_index = 0
    decisions = []
    contexts = section_contexts(sections)
    for section_index, section in enumerate(sections):
        current_context = contexts[section_index] if section_index < len(contexts) else set()
        competing_context = competing_context_words(contexts, section_index)
        for step in section.get("steps") or []:
            neighbor_info = neighbor_map.get(id(step), {})
            chronological_index = neighbor_info.get("chronological_index", step_index)
            window_start, window_end, target_time = step_media_window(
                step,
                chronological_index,
                total_steps,
                duration,
                neighbor_info.get("previous_step"),
                neighbor_info.get("next_step"),
            )
            window_frames = [
                frame
                for frame in frames
                if window_start <= float(frame.get("timestamp_seconds") or 0) <= window_end
            ]
            exact_start, exact_end = step_exact_media_window(step, target_time, duration)
            scoring_frames = window_frames
            ranked = sorted(
                (
                    (score_frame_for_step(frame, step, target_time, window_start, window_end, current_context, competing_context, exact_start, exact_end), frame)
                    for frame in scoring_frames
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            chosen = []
            chosen_frames = []
            chosen_scores = []
            for score, frame in ranked:
                if len(chosen_frames) >= 2:
                    break
                if not chosen_frames:
                    if score < 42:
                        continue
                    if frame_matches_competing_context(frame, current_context, competing_context):
                        continue
                else:
                    if score < max(76, chosen_scores[0] - 4):
                        continue
                    if frame_matches_competing_context(frame, current_context, competing_context):
                        continue
                    if not frame_has_print_value(frame):
                        continue
                    if not (
                        frame.get("pointing")
                        or frame.get("after_state")
                        or frame.get("newly_added_part_visible")
                        or frame.get("quality_check")
                    ):
                        continue
                    if not frame_is_diverse_from_chosen(frame, chosen_frames):
                        continue
                    if abs(float(frame.get("timestamp_seconds") or 0) - float(chosen_frames[0].get("timestamp_seconds") or 0)) < 4:
                        continue
                if score < 30:
                    continue
                path = frame.get("stored_path")
                if path and path not in chosen:
                    chosen.append(path)
                    chosen_frames.append(frame)
                    chosen_scores.append(score)
            fallback_used = False
            fallback_distance = None
            if not chosen:
                fallback_score, fallback_frame, fallback_distance = best_fallback_media_frame(
                    ranked,
                    frames,
                    target_time,
                    step,
                    window_start,
                    window_end,
                    current_context,
                    competing_context,
                    exact_start,
                    exact_end,
                )
                fallback_path = fallback_frame.get("stored_path") if fallback_frame else ""
                if fallback_path:
                    chosen = [fallback_path]
                    chosen_frames = [fallback_frame]
                    chosen_scores = [fallback_score or 0]
                    fallback_used = True
            decision_ranked = list(ranked)
            if fallback_used and fallback_frame and all(item[1].get("stored_path") != fallback_frame.get("stored_path") for item in decision_ranked):
                decision_ranked.insert(0, (fallback_score or 0, fallback_frame))
            decision = media_decision_record(chronological_index, step, window_start, window_end, target_time, decision_ranked, chosen, current_context, competing_context)
            decisions.append(decision)
            step["media_candidates"] = decision.get("candidates", [])[:10]
            top_score = float(decision_ranked[0][0]) if decision_ranked else 0
            selected_score = max(chosen_scores) if chosen_scores else 0
            if fallback_used:
                step["media_match_confidence"] = "fallback"
                step["media_fallback"] = True
                step["media_fallback_distance_seconds"] = round(float(fallback_distance or 0), 2)
                if fallback_distance is not None and fallback_distance > STEP_FALLBACK_NEAR_SECONDS:
                    step["media_fallback_reason"] = f"No frame scored well inside the step window; nearest usable frame was {fallback_distance:.1f}s from the target. Review or merge this step."
                else:
                    step["media_fallback_reason"] = f"No high-confidence match was found; best nearby frame selected {float(fallback_distance or 0):.1f}s from the target."
                fallback_image_steps += 1
            elif chosen and selected_score >= 78:
                step["media_match_confidence"] = "high"
                step.pop("media_fallback", None)
                step.pop("media_fallback_reason", None)
                step.pop("media_fallback_distance_seconds", None)
            elif chosen and selected_score >= 52:
                step["media_match_confidence"] = "medium"
                step.pop("media_fallback", None)
                step.pop("media_fallback_reason", None)
                step.pop("media_fallback_distance_seconds", None)
            elif chosen:
                step["media_match_confidence"] = "low"
                step.pop("media_fallback", None)
                step.pop("media_fallback_reason", None)
                step.pop("media_fallback_distance_seconds", None)
            else:
                step["media_match_confidence"] = "missing"
                step.pop("media_fallback", None)
                step.pop("media_fallback_reason", None)
                step.pop("media_fallback_distance_seconds", None)
            step["media_match_score"] = round(selected_score or top_score, 2)
            if chosen:
                step["media"] = chosen
                assigned += 1
                if len(chosen) == 1:
                    single_image_steps += 1
                elif len(chosen) >= 2:
                    double_image_steps += 1
            else:
                step["media"] = []
            merge_recommendation = merge_recommendation_for_step(
                step,
                neighbor_info.get("previous_step"),
                neighbor_info.get("next_step"),
                selected_score,
                fallback_used,
                bool(chosen),
                fallback_distance,
            )
            if merge_recommendation:
                step["merge_candidate"] = True
                step["merge_recommendation"] = merge_recommendation
                merge_candidate_steps += 1
            else:
                step.pop("merge_candidate", None)
                step.pop("merge_recommendation", None)
            step_index += 1
    job["media_decisions"] = decisions
    return sections, {
        "steps": assigned,
        "single_image_steps": single_image_steps,
        "double_image_steps": double_image_steps,
        "fallback_image_steps": fallback_image_steps,
        "merge_candidate_steps": merge_candidate_steps,
        "zero_image_steps": total_steps - assigned,
        "pointing_frames": sum(1 for frame in frames if frame.get("pointing")),
        "scored_frames": len(frames),
    }


def raw_video_frame_sources(job):
    sources = []
    frames_root = ensure_job_dirs(job)["frames"]
    for file in active_job_files(job):
        if file.get("kind") != "video":
            continue
        digest = file.get("sha256") or hashlib.sha256(str(stored_path(file) or file.get("name") or "").encode("utf-8")).hexdigest()
        frame_dir = frames_root / digest[:12]
        timeline_offset_seconds = float(file.get("timeline_offset_seconds") or 0)
        for frame_path in extracted_frame_files(frame_dir):
            sampling = frame_sampling_metadata(frame_path)
            sources.append(
                {
                    "file": file,
                    "path": frame_path,
                    "timestamp_seconds": frame_timestamp_seconds(frame_path, file),
                    "source_timestamp_seconds": frame_source_timestamp_seconds(frame_path),
                    "timeline_offset_seconds": timeline_offset_seconds,
                    "timeline_source": file,
                    "sampling_reason": sampling.get("sampling_reason") or "",
                    "sampling_tier": sampling.get("sampling_tier") or "",
                    "sample_rate_hz": sampling.get("sample_rate_hz"),
                }
            )
    return sorted(sources, key=lambda item: item["timestamp_seconds"])


def supplemental_step_frame_candidates(job, sections):
    raw_sources = raw_video_frame_sources(job)
    if not raw_sources:
        return []
    config = frame_budget_config(job)
    per_step_limit = max(1, int(config.get("supplemental_per_weak_step") or 4))
    supplemental_max = max(1, int(config.get("supplemental_max_frames") or SUPPLEMENTAL_FRAME_ANALYSIS_LIMIT))
    existing_analysis_paths = {
        item.get("stored_path")
        for item in job.get("frame_analysis", [])
        if item.get("stored_path")
    }
    duration = max([item["timestamp_seconds"] for item in raw_sources] or [0])
    total_steps = sum(len(section.get("steps") or []) for section in sections)
    if not total_steps:
        return []
    flat_steps = [
        step
        for section in sections
        for step in section.get("steps") or []
    ]
    chronological = sorted(
        enumerate(flat_steps),
        key=lambda item: (
            timed_seconds(item[1].get("start_time"), fallback_step_timing(item[0], total_steps, duration)[0]),
            item[0],
        ),
    )
    neighbor_map = {}
    for chronological_index, (original_index, chronological_step) in enumerate(chronological):
        previous_step = chronological[chronological_index - 1][1] if chronological_index > 0 else None
        next_step = chronological[chronological_index + 1][1] if chronological_index + 1 < len(chronological) else None
        neighbor_map[id(chronological_step)] = {
            "chronological_index": chronological_index,
            "previous_step": previous_step,
            "next_step": next_step,
            "original_index": original_index,
        }
    cue_times = transcript_cue_times(job, duration)
    rough = []
    step_index = 0
    for section in sections:
        for step in section.get("steps") or []:
            weak_media = (
                not (step.get("media") or [])
                or bool(step.get("media_fallback"))
                or step.get("media_match_confidence") in {"missing", "fallback", "low"}
            )
            if not weak_media:
                step_index += 1
                continue
            neighbor_info = neighbor_map.get(id(step), {})
            chronological_index = neighbor_info.get("chronological_index", step_index)
            window_start, window_end, target_time = step_media_window(
                step,
                chronological_index,
                total_steps,
                duration,
                neighbor_info.get("previous_step"),
                neighbor_info.get("next_step"),
            )
            window_start = max(0, window_start - 0.75)
            window_end = min(max(duration, window_end + 0.75), window_end + 0.75)
            in_window = [
                source
                for source in raw_sources
                if window_start <= source["timestamp_seconds"] <= window_end
                and str(source["path"].relative_to(APP_DIR)) not in existing_analysis_paths
            ]
            if not in_window:
                in_window = [
                    source
                    for source in raw_sources
                    if abs(source["timestamp_seconds"] - target_time) <= STEP_FALLBACK_NEAR_SECONDS
                    and str(source["path"].relative_to(APP_DIR)) not in existing_analysis_paths
                ]
            if not in_window:
                in_window = [
                    source
                    for source in raw_sources
                    if abs(source["timestamp_seconds"] - target_time) <= STEP_FALLBACK_GLOBAL_SECONDS
                    and str(source["path"].relative_to(APP_DIR)) not in existing_analysis_paths
                ]
            if not in_window:
                step_index += 1
                continue
            cv_metrics, _ = opencv_frame_metrics([source["path"] for source in in_window], normalize_focus_area(job.get("focus_area")))
            ranked = []
            for source in in_window:
                meta = apply_opencv_scoring(
                    score_candidate_frame(source["path"], cue_times, duration, source.get("timeline_source") or source.get("timeline_offset_seconds") or 0),
                    cv_metrics.get(source["path"].name),
                )
                delta = abs(source["timestamp_seconds"] - target_time)
                meta["score"] = round(float(meta.get("score", 0)) + max(0, 30 - delta * 3), 2)
                tags = set(meta.get("tags") or [])
                tags.add("supplemental")
                tags.add("step_window" if window_start <= source["timestamp_seconds"] <= window_end else "near_step_target")
                meta["tags"] = sorted(tags)
                meta["reason"] = f"supplemental step media {delta:.1f}s from target; {meta.get('reason') or 'near unmatched step'}"
                ranked.append((meta["score"], source, meta))
            for _, source, meta in sorted(ranked, key=lambda item: item[0], reverse=True)[:per_step_limit]:
                rough.append(
                    {
                        "step_index": chronological_index,
                        "source": source,
                        "meta": meta,
                    }
                )
            step_index += 1
    selected = []
    seen_paths = set()
    for item in sorted(rough, key=lambda item: item["meta"].get("score", 0), reverse=True):
        rel_path = str(item["source"]["path"].relative_to(APP_DIR))
        seen_key = (item.get("step_index"), rel_path)
        if seen_key in seen_paths:
            continue
        selected.append(item)
        seen_paths.add(seen_key)
        if len(selected) >= supplemental_max:
            break
    return selected


def analyze_supplemental_step_frames(job_id, sections):
    with STATE_LOCK:
        job = JOBS[job_id]
        candidates = supplemental_step_frame_candidates(job, sections)
        if not candidates:
            return {"weak_steps": 0, "scored_frames": 0, "selected_frames": 0}
        frames = []
        for index, item in enumerate(candidates, start=1):
            source = item["source"]
            record = video_frame_record(
                job,
                source["file"],
                source["path"],
                item["meta"],
                source="supplemental step media",
            )
            record["frame_label"] = f"S{index:03d}"
            record["supplemental_step_index"] = item["step_index"]
            artifact = upsert_artifact(job, record)
            paths = source["file"].setdefault("frame_paths", [])
            if artifact["stored_path"] not in paths:
                paths.append(artifact["stored_path"])
            frames.append(artifact)
        weak_steps = len({item["step_index"] for item in candidates})
        job_snapshot = json.loads(json.dumps(job))
        config = frame_budget_config(job)
        add_log(job, f"Supplemental media pass: scoring {len(frames)} nearby frames for {weak_steps} steps without images.")

    failed = False
    saved_count = 0
    batches = [
        frames[start:start + FRAME_ANALYSIS_BATCH_SIZE]
        for start in range(0, len(frames), FRAME_ANALYSIS_BATCH_SIZE)
    ]

    def analyze_timed_supplemental(batch):
        started = epoch_now()
        results, error = analyze_frame_batch(job_snapshot, batch)
        return results, error, round(epoch_now() - started, 2)

    max_workers = max(1, min(int(config.get("parallel_batches") or 1), len(batches)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(analyze_timed_supplemental, batch): batch for batch in batches}
        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                results, error, batch_seconds = future.result()
            except Exception as exc:
                results, error, batch_seconds = [], f"supplemental_batch_exception: {exc}", 0
            if error:
                failed = True
                with STATE_LOCK:
                    add_log(JOBS[job_id], f"Supplemental frame analysis provider failed; using fallback scoring: {error}")
                results = fallback_frame_analysis(batch)
            with STATE_LOCK:
                job = JOBS[job_id]
                current_by_path = {
                    item.get("stored_path"): item
                    for item in job.get("frame_analysis", [])
                    if item.get("stored_path")
                }
                for result in results:
                    current_by_path[result["stored_path"]] = result
                job["frame_analysis"] = sorted(
                    current_by_path.values(),
                    key=lambda item: float(item.get("timestamp_seconds") or 0),
                )
                batch_by_path = {result["stored_path"]: result for result in results}
                for artifact in frame_artifacts(job):
                    analysis = batch_by_path.get(artifact.get("stored_path"))
                    if not analysis:
                        continue
                    artifact["analysis"] = analysis
                    artifact["processing_status"] = "scored keyframe"
                    artifact["instruction_value"] = analysis.get("instruction_value", 0)
                    artifact["pointing"] = analysis.get("pointing", False)
                    artifact["discard"] = analysis.get("discard", False)
                saved_count += len(results)
                add_log(job, f"Supplemental frame batch saved: {len(results)} frames in {batch_seconds}s.")

    with STATE_LOCK:
        job = JOBS[job_id]
        note = f"Supplemental media pass complete: {saved_count} frames scored."
        if failed:
            note += " Fallback scoring was used."
        add_log(job, note)
    return {
        "weak_steps": len({item["step_index"] for item in candidates}),
        "scored_frames": saved_count,
        "selected_frames": len(frames),
        "fallback_used": failed,
    }


def replace_document_evidence(job, source_id, records):
    job["document_evidence"] = [
        record
        for record in job.get("document_evidence", [])
        if record.get("source_id") != source_id
    ] + records


def clear_document_artifacts(job, source_file):
    job["artifacts"] = [
        artifact
        for artifact in job.get("artifacts", [])
        if not (
            artifact.get("source_file") == source_file
            and artifact.get("kind") in {"drawing", "drawing_page", "pdf_page", "document_text"}
        )
    ]


def document_evidence_record(file, page_number, text, stored_path="", extraction_method=""):
    return {
        "evidence_id": short_id("evidence"),
        "source_id": file.get("source_id") or "",
        "source_file": file.get("name") or "",
        "source_role": file.get("source_role") or "supporting_document",
        "authority_level": file.get("authority_level") or "supporting_context",
        "revision": file.get("revision") or "",
        "applicability": file.get("applicability") or "",
        "page_number": page_number,
        "text": re.sub(r"[ \t]+", " ", text or "").strip(),
        "stored_path": stored_path,
        "extraction_method": extraction_method,
    }


def process_text_document(job, file, path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return [], f"text read failed: {exc}"
    record = document_evidence_record(file, 1, text, extraction_method="text_file")
    output_dir = ensure_job_dirs(job)["ocr"] / (file.get("sha256") or file.get("source_id") or "text")[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "document.json"
    output_path.write_text(json.dumps({"source_file": file.get("name"), "pages": [record]}, indent=2), encoding="utf-8")
    file["ocr_path"] = str(output_path.relative_to(APP_DIR))
    file["document_status"] = "text extracted"
    file["document_page_count"] = 1
    file["document_text_chars"] = len(text)
    return [record], None


def process_pdf_document(job, file, path):
    pdftotext = shutil.which("pdftotext")
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftotext and not pdftoppm:
        return [], "PDF tools unavailable: install Poppler for text extraction and page rendering."

    source_key = (file.get("sha256") or file.get("source_id") or "pdf")[:12]
    output_dir = ensure_job_dirs(job)["ocr"] / source_key
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "document.txt"
    pages_text = []
    warnings = []

    if pdftotext:
        try:
            result = subprocess.run(
                [pdftotext, "-layout", str(path), str(text_path)],
                capture_output=True,
                text=True,
                timeout=DOCUMENT_TEXT_TIMEOUT_SECONDS,
            )
            if result.returncode == 0 and text_path.exists():
                pages_text = text_path.read_text(encoding="utf-8", errors="replace").split("\f")
                while pages_text and not pages_text[-1].strip():
                    pages_text.pop()
            else:
                warnings.append((result.stderr or result.stdout or "pdftotext failed")[-500:])
        except subprocess.TimeoutExpired:
            warnings.append("PDF text extraction timed out.")

    page_paths = []
    if pdftoppm:
        for stale in output_dir.glob("page-*.jpg"):
            stale.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                [pdftoppm, "-jpeg", "-r", "144", str(path), str(output_dir / "page")],
                capture_output=True,
                text=True,
                timeout=DOCUMENT_RENDER_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                page_paths = sorted(
                    output_dir.glob("page-*.jpg"),
                    key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1)) if re.search(r"(\d+)$", item.stem) else 0,
                )
            else:
                warnings.append((result.stderr or result.stdout or "PDF page rendering failed")[-500:])
        except subprocess.TimeoutExpired:
            warnings.append("PDF page rendering timed out.")

    page_count = max(len(pages_text), len(page_paths))
    records = []
    for index in range(page_count):
        page_text = pages_text[index].strip() if index < len(pages_text) else ""
        page_path = page_paths[index] if index < len(page_paths) else None
        method = "embedded_text" if page_text else ""
        if not page_text and page_path and tesseract:
            try:
                result = subprocess.run(
                    [tesseract, str(page_path), "stdout", "-l", "eng"],
                    capture_output=True,
                    text=True,
                    timeout=DOCUMENT_OCR_PAGE_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    page_text = (result.stdout or "").strip()
                    method = "tesseract_ocr"
                else:
                    warnings.append(f"OCR page {index + 1} failed: {(result.stderr or '')[-300:]}")
            except subprocess.TimeoutExpired:
                warnings.append(f"OCR page {index + 1} timed out.")
        stored_page = str(page_path.relative_to(APP_DIR)) if page_path else ""
        records.append(document_evidence_record(file, index + 1, page_text, stored_page, method or "render_only"))

    payload_path = output_dir / "document.json"
    payload = {
        "source_id": file.get("source_id") or "",
        "source_file": file.get("name") or "",
        "authority_level": file.get("authority_level") or "supporting_context",
        "revision": file.get("revision") or "",
        "applicability": file.get("applicability") or "",
        "pages": records,
        "warnings": warnings,
        "processed_at": datestamp(),
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    file["ocr_path"] = str(payload_path.relative_to(APP_DIR))
    file["document_status"] = "processed" if records else "processing failed"
    file["document_page_count"] = len(records)
    file["document_text_chars"] = sum(len(record.get("text") or "") for record in records)
    file["document_warnings"] = warnings[-10:]
    if not records:
        return [], "; ".join(warnings) or "No PDF pages or text could be extracted."
    return records, None


def document_context(job, max_chars=7000):
    blocks = []
    used = 0
    authority_order = {"authoritative": 0, "primary_evidence": 1, "supporting_context": 2, "reference_only": 3}
    evidence_records = sorted(
        job.get("document_evidence", []),
        key=lambda item: (
            authority_order.get(item.get("authority_level"), 4),
            item.get("source_file") or "",
            int(item.get("page_number") or 0),
        ),
    )
    for evidence in evidence_records:
        text = (evidence.get("text") or "").strip()
        if not text:
            continue
        heading = (
            f"Source: {evidence.get('source_file') or 'document'}"
            f" | authority: {evidence.get('authority_level') or 'supporting_context'}"
            f" | revision: {evidence.get('revision') or 'not provided'}"
            f" | page: {evidence.get('page_number') or 1}"
        )
        remaining = max_chars - used - len(heading) - 2
        if remaining <= 0:
            break
        block = f"{heading}\n{text[:remaining]}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def run_document_stage(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        normalize_job_sources(job)
        set_stage(job, "drawings", "running", 5, "Extracting document evidence")
        job["status"] = "Processing"
        add_log(job, "Document processing started.")
        sources = [
            file
            for file in active_job_files(job)
            if file.get("kind") in {"pdf", "text"} and stored_path(file)
        ]

    if not sources:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "drawings", "done", 100, "No PDF or text documents attached")
            add_log(job, "Document processing complete: no PDF or text documents attached.")
        return True

    processed = 0
    failed = 0
    page_count = 0
    text_chars = 0
    for index, file in enumerate(sources, start=1):
        with STATE_LOCK:
            job = JOBS[job_id]
            percent = 8 + round((index - 1) / max(1, len(sources)) * 82)
            set_stage(job, "drawings", "running", percent, f"Processing document {index}/{len(sources)}")
            clear_document_artifacts(job, file.get("name"))
        path = stored_path(file)
        if file.get("kind") == "pdf":
            records, error = process_pdf_document(job, file, path)
        else:
            records, error = process_text_document(job, file, path)
        with STATE_LOCK:
            job = JOBS[job_id]
            replace_document_evidence(job, file.get("source_id") or "", records)
            for record in records:
                if record.get("stored_path"):
                    page_path = APP_DIR / record["stored_path"]
                    upsert_artifact(
                        job,
                        {
                            "name": page_path.name,
                            "type": "image/jpeg",
                            "kind": "pdf_page",
                            "size": human_size(page_path.stat().st_size) if page_path.exists() else "",
                            "bytes": page_path.stat().st_size if page_path.exists() else 0,
                            "stored_path": record["stored_path"],
                            "source": "rendered document page",
                            "source_file": file.get("name"),
                            "source_id": file.get("source_id") or "",
                            "page_number": record.get("page_number"),
                            "authority_level": file.get("authority_level"),
                            "revision": file.get("revision"),
                            "applicability": file.get("applicability"),
                            "processing_status": record.get("extraction_method") or "rendered",
                            "summary": (record.get("text") or "")[:240],
                        },
                    )
            if error:
                failed += 1
                file["document_status"] = "processing failed"
                file["document_error"] = error
                add_log(job, f"Document processing failed for {file.get('name')}: {error}")
            else:
                processed += 1
                page_count += len(records)
                text_chars += sum(len(record.get("text") or "") for record in records)
                add_log(job, f"Document processed: {file.get('name')} · {len(records)} pages · {file.get('document_text_chars', 0)} text characters.")

    with STATE_LOCK:
        job = JOBS[job_id]
        note = f"{processed} documents · {page_count} pages · {text_chars} text characters"
        if failed:
            note += f" · {failed} failed"
            if "Document processing issue" not in job["flags"]:
                job["flags"].append("Document processing issue")
        set_stage(job, "drawings", "failed" if failed and not processed else "done", 100 if processed else 70, note)
        add_log(job, f"Document processing complete: {note}.")
    return True


def run_image_inventory_stage(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, "images", "running", 20, "Cataloging uploaded images")
        images = [file for file in active_job_files(job) if file.get("kind") == "image" and stored_path(file)]
        for file in images:
            file["image_status"] = "available for manual evidence selection"
        note = f"{len(images)} uploaded images cataloged"
        if images:
            note += " · semantic image analysis pending"
        set_stage(job, "images", "done", 100, note)
        add_log(job, f"Image inventory complete: {note}.")
    return True


def run_stage(job_id, key, note, seconds):
    steps = max(3, int(seconds * 2))
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, key, "running", 5, note)
        job["status"] = "Processing"
        add_log(job, f"{STAGES[[s[0] for s in STAGES].index(key)][1]} started.")
    for i in range(1, steps + 1):
        time.sleep(seconds / steps)
        with STATE_LOCK:
            job = JOBS[job_id]
            if job["status"] == "Cancelled":
                return False
            set_stage(job, key, "running", min(95, 5 + round((i / steps) * 88)), note)
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, key, "done", 100, "Complete")
        add_log(job, f"{STAGES[[s[0] for s in STAGES].index(key)][1]} complete.")
    return True


def transcript_is_ready_for_drafting(job):
    media_files = [file for file in drafting_source_files(job) if file.get("readable")]
    if not media_files:
        return True
    if not SETTINGS.get("transcription_enabled", True):
        return True
    return all(file.get("transcript_status") == "complete" for file in media_files)


def pause_for_transcript(job):
    job["status"] = "Needs Attention"
    if "Transcript required" not in job["flags"]:
        job["flags"].append("Transcript required")
    set_stage(job, "questions", "blocked", 0, "Waiting for transcript")
    set_stage(job, "draft_sections", "blocked", 0, "Waiting for transcript")
    refresh_processing_manifest(job, status="awaiting_input", outcome="Transcript required before drafting.")
    add_log(job, "Processing paused before questions: transcript text is not available.")


def run_video_stage(job_id):
    ffmpeg = shutil.which("ffmpeg")
    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, "video", "running", 5, "Extracting candidate keyframes")
        job["status"] = "Processing"
        add_log(job, "Video started.")

    if not ffmpeg:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "video", "failed", 0, "ffmpeg not installed")
            if "ffmpeg missing" not in job["flags"]:
                job["flags"].append("ffmpeg missing")
            add_log(job, "Video stage failed: ffmpeg not installed.")
        return False

    with STATE_LOCK:
        job = JOBS[job_id]
        video_files = [
            file
            for file in active_job_files(job)
            if file.get("kind") == "video" and stored_path(file)
        ]

    if not video_files:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "video", "done", 100, "No readable video sources")
            add_log(job, "Video complete: no readable video sources.")
        return True

    extracted_total = 0
    failed = 0
    reused_raw_frame_sets = 0
    for index, file in enumerate(video_files, start=1):
        input_path = stored_path(file)
        digest = file.get("sha256") or hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()
        frame_dir = ensure_job_dirs(JOBS[job_id])["frames"] / digest[:12]
        frame_dir.mkdir(parents=True, exist_ok=True)
        duration_seconds = probe_video_duration(input_path)
        schedule = build_adaptive_sampling_schedule(JOBS[job_id], duration_seconds or 0)
        if schedule.get("hash"):
            schedule = dict(schedule)
            schedule["hash"] = hashlib.sha256(
                json.dumps(
                    {
                        "schedule_hash": schedule.get("hash"),
                        "source_sha256": file.get("sha256") or "",
                        "timeline_offset_seconds": file.get("timeline_offset_seconds") or 0,
                        "timeline_scale": file.get("timeline_scale") or 1,
                        "timeline_sync": file.get("timeline_sync") or {},
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
        cached_frames = cached_video_frames(file, frame_dir, schedule.get("hash"))
        extraction_stats = {}
        with STATE_LOCK:
            job = JOBS[job_id]
            clear_frame_artifacts_for_file(job, file)
            percent = 10 + round((index - 1) / max(1, len(video_files)) * 75)
            if cached_frames:
                set_stage(job, "video", "running", percent, f"Reusing frames {index}/{len(video_files)}")
                add_log(job, f"Reusing raw sampled frames for {file.get('name')}; candidate selection will be recalculated.")
            else:
                set_stage(job, "video", "running", percent, f"Sampling frames {index}/{len(video_files)}")
                if schedule.get("mode") == "adaptive":
                    add_log(
                        job,
                        f"Adaptive sampling {file.get('name')}: base {schedule.get('base_rate_hz')} Hz · {len(schedule.get('windows') or [])} timestamp windows.",
                    )
                else:
                    add_log(job, f"Sampling frames at {VIDEO_FRAME_SAMPLE_RATE_HZ} Hz from {file.get('name')}.")

        if cached_frames:
            reused_raw_frame_sets += 1
            result = type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()
            frame_error = None
            frames = cached_frames
            manifest = load_frame_manifest(frame_dir)
            extraction_stats = {
                "schedule_windows": len(manifest.get("windows") or []),
                "base_rate_hz": manifest.get("base_rate_hz"),
                "duplicate_timestamps_removed": manifest.get("duplicate_timestamps_removed", 0),
                "window_error_count": len(manifest.get("window_errors") or []),
                "manifest_frames": len(manifest.get("frames") or []),
            }
        else:
            if schedule.get("mode") == "adaptive" and schedule.get("windows"):
                frames, extraction_stats, frame_error = extract_adaptive_video_frames(ffmpeg, input_path, frame_dir, file, schedule)
                result = type("Result", (), {"returncode": 0 if not frame_error else 1, "stderr": frame_error or "", "stdout": ""})()
            else:
                remove_frame_files(frame_dir)
                pattern = frame_dir / "frame_%04d.jpg"
                cmd = [
                    ffmpeg,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vf",
                    f"fps={VIDEO_FRAME_SAMPLE_RATE_HZ},scale=1280:-2",
                    "-q:v",
                    "3",
                    str(pattern),
                ]
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=VIDEO_EXTRACTION_TIMEOUT_SECONDS,
                    )
                    frame_error = None
                except subprocess.TimeoutExpired as exc:
                    result = None
                    frame_error = f"ffmpeg timed out after {VIDEO_EXTRACTION_TIMEOUT_SECONDS}s"
                    remove_frame_files(frame_dir)
                frames = extracted_frame_files(frame_dir)
        selected_frames = []
        candidate_meta = {}
        selection_stats = {"raw_frames": len(frames), "selected_frames": 0, "opencv": {"enabled": cv2 is not None}}
        if frames:
            selected_frames, candidate_meta, selection_stats = select_candidate_frames(JOBS[job_id], frames, file)

        with STATE_LOCK:
            job = JOBS[job_id]
            if frame_error or result.returncode != 0 or not selected_frames:
                failed += 1
                file["video_status"] = "frame extraction failed"
                file["video_error"] = frame_error or (result.stderr or result.stdout or "ffmpeg failed or no useful frames")[-1000:]
                if "Video frame issue" not in job["flags"]:
                    job["flags"].append("Video frame issue")
                add_log(job, f"Frame extraction failed for {file.get('name')}: {file['video_error']}")
                continue

            frame_records = record_video_frames(job, file, selected_frames, candidate_meta)
            file["raw_frame_count"] = selection_stats.get("raw_frames", len(frames))
            file["candidate_frame_count"] = len(frame_records)
            file["frame_selection"] = selection_stats
            file["frame_extraction"] = extraction_stats
            extracted_total += len(frame_records)
            opencv_note = "OpenCV filter active" if (selection_stats.get("opencv") or {}).get("enabled") else "OpenCV filter skipped"
            ceiling_note = "ceiling applied" if selection_stats.get("hard_ceiling_applied") else "no ceiling"
            adaptive_note = ""
            if extraction_stats:
                adaptive_note = f" · adaptive {extraction_stats.get('base_rate_hz') or '?'} Hz base · {extraction_stats.get('schedule_windows') or 0} windows · {extraction_stats.get('duplicate_timestamps_removed') or 0} timestamp duplicates removed"
                if extraction_stats.get("window_error_count"):
                    adaptive_note += f" · {extraction_stats.get('window_error_count')} window errors"
            add_log(
                job,
                f"Video frames extracted for {file.get('name')}: {selection_stats.get('raw_frames', len(frames))} raw · {len(frame_records)} candidates · {selection_stats.get('hard_cue_selected', 0)} hard cues · {selection_stats.get('step_window_selected', 0)} step windows · {selection_stats.get('soft_cue_selected', 0)} soft cues · {selection_stats.get('visual_change_selected', 0)} visual change · focus {selection_stats.get('focus_area', 'center')} · {selection_stats.get('blurry_rejected', 0)} blurry rejected · {selection_stats.get('duplicate_rejected', 0)} duplicate rejected · {ceiling_note} · {opencv_note}{adaptive_note}.",
            )

    if extracted_total:
        run_frame_analysis_stage(job_id)

    with STATE_LOCK:
        job = JOBS[job_id]
        if failed and not extracted_total:
            set_stage(job, "video", "failed", 70, f"{failed} video files failed")
        else:
            note = f"{extracted_total} candidate keyframes"
            analysis_count = len(job.get("frame_analysis") or [])
            if analysis_count:
                pointing_count = sum(1 for item in job.get("frame_analysis", []) if item.get("pointing"))
                note += f" · {analysis_count} scored"
                if pointing_count:
                    note += f" · {pointing_count} pointing"
            if failed:
                note += f" · {failed} failed"
            cache = job.setdefault("processing_cache", {})
            cache["video"] = {
                "raw_frame_sets_reused": reused_raw_frame_sets,
                "candidate_keyframes": extracted_total,
                "scored_frames": analysis_count,
                "updated_at": datestamp(),
            }
            set_stage(job, "video", "done", 100, note)
        add_log(job, f"Video complete: {extracted_total} candidate keyframes.")
    return not failed


def analyze_frame_batch(job, batch):
    transcript_excerpt = transcript_text(job)[:5000] or "No transcript text available."
    prompt = f"""
Analyze these sampled frames from a manufacturing/assembly video.

The operator may point at newly added parts after a step. Those frames are often high-value evidence.
If local cue tags include good_image/operator_image_cue or the local cue reason says the operator requested a shot
(for example "good shot", "show this", "take a picture", or "close-up"), treat a clear held completed
assembly/subassembly as useful after-state evidence even if there is no active assembly motion in that frame.

For each frame, identify whether it is useful for assembly instructions.

Return JSON only:
{{
  "frames": [
    {{
      "frame_label": "F001",
      "visual_summary": "short objective description",
      "visible_parts": ["part or component names visible if knowable"],
      "pointing": true,
      "pointing_target": "what the operator appears to indicate, or empty",
      "newly_added_part_visible": true,
      "tool_visible": true,
      "fastener_visible": true,
      "quality_check": true,
      "before_state": false,
      "after_state": true,
      "blocked_or_obstructed": false,
      "tags": ["pointing", "after_state"],
      "clarity": 0,
      "assembly_relevance": 0,
      "instruction_value": 0,
      "discard": false,
      "reason": "short reason"
    }}
  ]
}}

Scoring:
- clarity: 0 blurry/blocked, 100 clear and inspectable.
- assembly_relevance: 0 unrelated, 100 directly shows assembly state or part placement.
- instruction_value: 0 useless, 100 excellent step evidence.
- Give pointing/indicating frames a high instruction_value if the pointed part is visible.
- Tag frames that show tools, fasteners, before/after states, quality checks, or obstruction.
- Mark discard true for frames that are too blurry, obstructed, duplicate, or not useful.
- Prefer clear after-state and pointing frames for instruction media.
- Do not mark an operator-requested good shot low-value only because the operator is holding the completed state.
- Do not invent part names. Use generic names when needed.

Transcript excerpt for context:
{transcript_excerpt}
"""
    data, error = call_openai_frame_json(prompt, batch)
    if error:
        return [], error
    by_label = {}
    for item in (data or {}).get("frames", []):
        label = str(item.get("frame_label") or "").strip()
        if label:
            by_label[label] = item
    return [normalize_frame_analysis(frame, by_label.get(frame.get("frame_label"))) for frame in batch], None


def fallback_frame_analysis(frames):
    results = []
    for index, frame in enumerate(frames):
        results.append(
            normalize_frame_analysis(
                frame,
                {
                    "frame_label": frame.get("frame_label"),
                    "visual_summary": "Candidate assembly frame.",
                    "visible_parts": [],
                    "pointing": False,
                    "newly_added_part_visible": False,
                    "clarity": 50,
                    "assembly_relevance": 50,
                    "instruction_value": 45 + (index % 3) * 5,
                    "discard": False,
                    "reason": "Vision scoring unavailable; retained as fallback evidence.",
                },
            )
        )
    return results


def run_frame_analysis_stage(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        candidates = sorted(
            frame_artifacts(job),
            key=lambda frame: (
                float(frame.get("local_priority") or 0),
                float(frame.get("instruction_value") or 0),
            ),
            reverse=True,
        )
        frames, budget_stats = prioritize_frame_analysis_candidates(job, candidates)
        if budget_stats.get("ceiling_applied"):
            step_count = rough_step_count_for_limit(job)
            reason = f"expected/parsed {step_count} steps" if step_count else "candidate volume"
            add_log(
                job,
                (
                    f"Frame analysis budget applied: {len(candidates)} candidates available, scoring "
                    f"{len(frames)} targeted frames using {budget_stats.get('profile')} profile based on {reason}."
                ),
            )
            add_log(
                job,
                (
                    f"Frame budget detail: {budget_stats.get('rough_step_windows', 0)} timed windows · "
                    f"{budget_stats.get('frames_per_step', 0)} frames/step · "
                    f"{budget_stats.get('step_budgeted_frames', 0)} step picks · "
                    f"{budget_stats.get('hard_cue_frames', 0)} hard cues · "
                    f"{budget_stats.get('coverage_reserve_frames', 0)} coverage reserve."
                ),
            )
        for index, frame in enumerate(frames, start=1):
            frame["frame_label"] = f"F{index:03d}"
        existing_by_path = {
            item.get("stored_path"): item
            for item in job.get("frame_analysis", [])
            if item.get("stored_path")
        }
        job_snapshot = json.loads(json.dumps(job))
        config = frame_budget_config(job)

    if not frames:
        return

    all_results = [
        existing_by_path.get(frame.get("stored_path"))
        for frame in frames
        if existing_by_path.get(frame.get("stored_path"))
    ]
    reused_analysis_count = len(all_results)
    failed = False
    batch_durations = []
    batches = []
    for start in range(0, len(frames), FRAME_ANALYSIS_BATCH_SIZE):
        batch = frames[start:start + FRAME_ANALYSIS_BATCH_SIZE]
        if not all(frame.get("stored_path") in existing_by_path for frame in batch):
            batches.append((start, batch))

    if batches:
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "video", "running", 78, f"Scoring {len(frames)} targeted frames")
            add_log(
                job,
                (
                    f"Frame analysis queued: {len(batches)} batches · {len(frames)} target frames · "
                    f"{reused_analysis_count} reused · {config.get('parallel_batches', 1)} parallel workers."
                ),
            )
    else:
        with STATE_LOCK:
            job = JOBS[job_id]
            trimmed_results = [
                existing_by_path.get(frame.get("stored_path"))
                for frame in frames
                if existing_by_path.get(frame.get("stored_path"))
            ]
            job["frame_analysis"] = trimmed_results
            add_log(job, f"Frame analysis reused cached scoring for {len(trimmed_results)} targeted frames.")

    def analyze_timed_batch(batch):
        started = epoch_now()
        results, error = analyze_frame_batch(job_snapshot, batch)
        return results, error, round(epoch_now() - started, 2)

    if batches:
        max_workers = max(1, min(int(config.get("parallel_batches") or 1), len(batches)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(analyze_timed_batch, batch): (start, batch)
                for start, batch in batches
            }
            for future in as_completed(future_map):
                start, batch = future_map[future]
                try:
                    results, error, batch_seconds = future.result()
                except Exception as exc:
                    results, error, batch_seconds = [], f"frame_batch_exception: {exc}", 0
                batch_durations.append(
                    {
                        "start_frame": start + 1,
                        "end_frame": start + len(batch),
                        "seconds": batch_seconds,
                    }
                )
                if error:
                    failed = True
                    with STATE_LOCK:
                        add_log(JOBS[job_id], f"Frame analysis provider failed; using fallback scoring: {error}")
                    results = fallback_frame_analysis(batch)
                with STATE_LOCK:
                    job = JOBS[job_id]
                    current_by_path = {
                        item.get("stored_path"): item
                        for item in job.get("frame_analysis", [])
                        if item.get("stored_path")
                    }
                    for result in results:
                        current_by_path[result["stored_path"]] = result
                        existing_by_path[result["stored_path"]] = result
                    ordered_results = [
                        current_by_path.get(frame.get("stored_path"))
                        for frame in frames
                        if current_by_path.get(frame.get("stored_path"))
                    ]
                    job["frame_analysis"] = ordered_results
                    batch_by_path = {result["stored_path"]: result for result in results}
                    for artifact in frame_artifacts(job):
                        analysis = batch_by_path.get(artifact.get("stored_path"))
                        if not analysis:
                            continue
                        artifact["analysis"] = analysis
                        artifact["processing_status"] = "scored keyframe"
                        artifact["instruction_value"] = analysis.get("instruction_value", 0)
                        artifact["pointing"] = analysis.get("pointing", False)
                        artifact["discard"] = analysis.get("discard", False)
                    pointing_count = sum(1 for item in ordered_results if item.get("pointing"))
                    useful_count = sum(1 for item in ordered_results if item.get("instruction_value", 0) >= 65 and not item.get("discard"))
                    set_stage(
                        job,
                        "video",
                        "running",
                        min(96, 80 + round(len(ordered_results) / max(1, len(frames)) * 16)),
                        f"{len(ordered_results)}/{len(frames)} frames scored · {pointing_count} pointing",
                    )
                    add_log(
                        job,
                        (
                            f"Frame analysis saved: frames {start + 1}-{start + len(batch)} in {batch_seconds}s · "
                            f"{len(ordered_results)}/{len(frames)} scored · {useful_count} useful · {pointing_count} pointing."
                        ),
                    )

    with STATE_LOCK:
        job = JOBS[job_id]
        all_results = job.get("frame_analysis", [])
        pointing_count = sum(1 for item in all_results if item.get("pointing"))
        useful_count = sum(1 for item in all_results if item.get("instruction_value", 0) >= 65 and not item.get("discard"))
        note = f"{len(all_results)} scored · {useful_count} useful · {pointing_count} pointing"
        if len(all_results) < len(frames):
            note += f" · {len(frames) - len(all_results)} unscored"
        if failed:
            note += " · fallback used"
        cache = job.setdefault("processing_cache", {})
        cache["frame_analysis"] = {
            "analysis_reused": reused_analysis_count,
            "analysis_scored": len(all_results),
            "candidate_frames": len(frames),
            "available_candidate_frames": len(candidates),
            "budget": budget_stats,
            "parallel_batches": int(config.get("parallel_batches") or 1),
            "batch_durations": batch_durations,
            "updated_at": datestamp(),
        }
        add_log(job, f"Frame analysis complete: {note}.")


def has_processing_run(job):
    return bool(
        job.get("processing_started")
        or any(log.get("message") == "Processing queued." for log in job.get("logs", []))
        or any(stage.get("percent", 0) for stage in job.get("stages", {}).values())
    )


def running_stage_key(job):
    for key, stage in (job.get("stages") or {}).items():
        if stage.get("status") == "running":
            return key
    return ""


def job_thread_alive(job_id):
    thread = RUNNING_THREADS.get(job_id)
    return bool(thread and thread.is_alive())


def watch_processing_jobs():
    now = epoch_now()
    with STATE_LOCK:
        for job_id, job in JOBS.items():
            if job.get("deleted"):
                continue
            key = running_stage_key(job)
            processing_like = job.get("status") in {"Processing", "Drafting Sections"} or bool(key)
            if not processing_like:
                continue
            thread_alive = job_thread_alive(job_id)
            stage = job.get("stages", {}).get(key, {}) if key else {}
            last_epoch = max(
                float(job.get("last_activity_epoch") or 0),
                float(stage.get("updated_epoch") or 0),
            )
            idle_seconds = max(0, now - last_epoch) if last_epoch else 0
            job["watchdog"] = {
                "stage": key,
                "thread_alive": thread_alive,
                "idle_seconds": round(idle_seconds, 1),
                "threshold_seconds": PROCESSING_WATCHDOG_SECONDS,
                "checked_at": datestamp(),
            }
            if not thread_alive:
                if key and stage.get("status") == "running":
                    stage["status"] = "failed"
                    stage["note"] = "Worker stopped before this stage completed"
                job["status"] = "Needs Attention"
                flags = job.setdefault("flags", [])
                if "Processing interrupted" not in flags:
                    flags.append("Processing interrupted")
                recompute_progress(job)
                add_log(job, f"Watchdog recovered interrupted processing at stage: {key or 'unknown'}.")
                continue
            if idle_seconds >= PROCESSING_WATCHDOG_SECONDS:
                flags = job.setdefault("flags", [])
                if "Processing slow" not in flags:
                    flags.append("Processing slow")
                    add_log(job, f"Watchdog notice: {key or 'processing'} has had no visible progress for {round(idle_seconds / 60, 1)} minutes.")


def mark_job_thread_failed(job_id, exc):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        message = f"{type(exc).__name__}: {str(exc)[:180]}"
        had_running_stage = False
        for stage in job.get("stages", {}).values():
            if stage.get("status") == "running":
                stage["status"] = "failed"
                stage["note"] = f"Failed: {message}"
                stage["ended_at"] = datestamp()
                stage["ended_epoch"] = epoch_now()
                started = float(stage.get("started_epoch") or stage.get("updated_epoch") or epoch_now())
                stage["duration_seconds"] = round(max(0, epoch_now() - started), 2)
                had_running_stage = True
        if not had_running_stage:
            for stage in job.get("stages", {}).values():
                if stage.get("status") in {"idle", "blocked"}:
                    stage["status"] = "failed"
                    stage["note"] = f"Failed: {message}"
                    break
        job["status"] = "Needs Attention"
        flags = job.setdefault("flags", [])
        if "Processing error" not in flags:
            flags.append("Processing error")
        finish_job_processing_timing(job, status="failed", outcome=message)
        recompute_progress(job)
        add_log(job, f"Processing error: {message}")


def run_job_worker(job_id, worker):
    try:
        worker(job_id)
    except Exception as exc:
        mark_job_thread_failed(job_id, exc)


def start_processing(job_id, reset=True, first_run_only=False):
    with STATE_LOCK:
        if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
            return False
        job = JOBS[job_id]
        if job.get("deleted"):
            return False
        if first_run_only and has_processing_run(job) and not job.get("needs_processing"):
            return False
        if reset:
            job["stages"] = base_stages()
            job["questions"] = []
            job["question_review"] = {}
            job["draft_sections"] = []
            job["flags"] = []
            job["progress"] = 0
            job["media_selection"] = {}
            job["processing_started"] = True
            job["needs_processing"] = False
            start_job_processing_timing(job)
            add_log(job, "Processing queued.")
        thread = threading.Thread(target=run_job_worker, args=(job_id, process_job), daemon=True)
        RUNNING_THREADS[job_id] = thread
        thread.start()
        return True


def process_job(job_id):
    if not run_upload_check(job_id):
        with STATE_LOCK:
            finish_job_processing_timing(JOBS[job_id], status="failed", outcome="Upload validation failed.")
        return
    transcription_ready = run_transcription_stage(job_id)
    if not run_document_stage(job_id):
        with STATE_LOCK:
            finish_job_processing_timing(JOBS[job_id], status="failed", outcome="Document processing failed.")
        return
    if not run_image_inventory_stage(job_id):
        with STATE_LOCK:
            finish_job_processing_timing(JOBS[job_id], status="failed", outcome="Image inventory failed.")
        return
    if not run_video_stage(job_id):
        with STATE_LOCK:
            finish_job_processing_timing(JOBS[job_id], status="failed", outcome="Video processing failed.")
        return

    with STATE_LOCK:
        job = JOBS[job_id]
        job["cues"] = cue_summary(job)
        if not transcription_ready or not transcript_is_ready_for_drafting(job):
            pause_for_transcript(job)
            return

    with STATE_LOCK:
        job = JOBS[job_id]
        set_stage(job, "questions", "running", 10, "Creating decisions")
        add_log(job, "Question pass started.")
    time.sleep(1.2)
    with STATE_LOCK:
        job = JOBS[job_id]
        job["questions"] = transcript_questions_for(job)
        job["question_review"] = question_review_for(job, job["questions"])
        pending_count = len(unresolved_questions(job))
        set_question_flag(job, pending_count)
        if pending_count:
            set_stage(job, "questions", "blocked", 50, f"{pending_count} decisions pending")
            job["status"] = "Questions Pending"
            refresh_processing_manifest(job, status="awaiting_input", outcome="Clarification questions pending.")
            add_log(job, "Question pass paused for user decisions.")
        else:
            set_stage(job, "questions", "done", 100, "No decisions needed")
            review = job.get("question_review") or {}
            cleared = "; ".join(review.get("bypass_reasons") or [])
            add_log(job, f"Question pass complete: no decisions needed. {cleared[:500]}")
            should_draft = True
    if "should_draft" in locals():
        finish_drafting(job_id)


def reset_stages_from(job, start_key):
    if start_key not in stageKeys_py():
        return
    start_index = stageKeys_py().index(start_key)
    for key in stageKeys_py()[start_index:]:
        stage = job["stages"][key]
        stage["status"] = "idle"
        stage["percent"] = 0
        stage["note"] = ""
        stage["updated_at"] = datestamp()
        stage["updated_epoch"] = epoch_now()
        stage["started_at"] = ""
        stage["started_epoch"] = None
        stage["ended_at"] = ""
        stage["ended_epoch"] = None
        stage["duration_seconds"] = None
    recompute_progress(job)


def stageKeys_py():
    return [key for key, _label in STAGES]


def clear_video_scoring(job):
    job["frame_analysis"] = []
    for artifact in job.get("artifacts", []):
        if artifact.get("kind") == "frame":
            artifact.pop("analysis", None)
            artifact["processing_status"] = "ready for scoring"
            artifact.pop("instruction_value", None)
            artifact.pop("pointing", None)
            artifact.pop("discard", None)


def rerun_stage_worker(job_id, stage_key):
    with STATE_LOCK:
        job = JOBS[job_id]
        job["status"] = "Processing"
        flags = job.setdefault("flags", [])
        for flag in ["Processing interrupted", "Processing slow", "Processing error"]:
            if flag in flags:
                flags.remove(flag)
        add_log(job, f"Stage rerun queued: {stage_key}.")
    if stage_key == "transcription":
        with STATE_LOCK:
            reset_stages_from(JOBS[job_id], "transcription")
            JOBS[job_id]["questions"] = []
            JOBS[job_id]["question_review"] = {}
        run_transcription_stage(job_id)
        with STATE_LOCK:
            job = JOBS[job_id]
            job["status"] = "Needs Attention"
            if "Draft may need rerun" not in job["flags"]:
                job["flags"].append("Draft may need rerun")
            add_log(job, "Transcription rerun complete. Rerun Drafts if transcript content changed.")
        return
    if stage_key == "drawings":
        with STATE_LOCK:
            reset_stages_from(JOBS[job_id], "drawings")
        run_document_stage(job_id)
        with STATE_LOCK:
            job = JOBS[job_id]
            job["status"] = "Needs Attention"
            if "Draft may need rerun" not in job["flags"]:
                job["flags"].append("Draft may need rerun")
            add_log(job, "Document rerun complete. Rerun Drafts if supporting evidence changed.")
        return
    if stage_key == "video":
        with STATE_LOCK:
            job = JOBS[job_id]
            reset_stages_from(job, "video")
            clear_video_scoring(job)
            job["media_selection"] = {}
            add_log(job, "Video rerun will reuse raw sampled frames where possible and refresh frame scoring.")
        run_video_stage(job_id)
        with STATE_LOCK:
            has_drafts = bool(JOBS[job_id].get("draft_sections"))
        if has_drafts:
            run_media_assignment_stage(job_id)
        return
    if stage_key == "draft_sections":
        with STATE_LOCK:
            job = JOBS[job_id]
            reset_stages_from(job, "draft_sections")
            job["draft_sections"] = []
            job["media_selection"] = {}
            job["draft_validation"] = {}
            job["draft_audit"] = {}
        finish_drafting(job_id)
        return
    if stage_key == "media_assignment":
        run_media_assignment_stage(job_id)
        return
    with STATE_LOCK:
        job = JOBS[job_id]
        job["status"] = "Needs Attention"
        add_log(job, f"Unsupported stage rerun requested: {stage_key}.")


def rerun_job_stage(job_id, stage_key):
    allowed = {"transcription", "drawings", "video", "draft_sections", "media_assignment"}
    if stage_key not in allowed:
        return False
    with STATE_LOCK:
        if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
            return False
        job = JOBS.get(job_id)
        if not job or job.get("deleted"):
            return False
        thread = threading.Thread(target=run_job_worker, args=(job_id, lambda jid: rerun_stage_worker(jid, stage_key)), daemon=True)
        RUNNING_THREADS[job_id] = thread
        thread.start()
        return True


def continue_after_questions(job_id):
    with STATE_LOCK:
        if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
            return False
        thread = threading.Thread(target=run_job_worker, args=(job_id, finish_drafting), daemon=True)
        RUNNING_THREADS[job_id] = thread
        thread.start()
        return True


def validate_draft_outputs(job, draft_sections, timing_stats=None, media_stats=None, supplemental_stats=None):
    issues = []
    issue_counts = {}
    step_count = 0
    for section_index, section in enumerate(draft_sections or []):
        for step_index, step in enumerate(section.get("steps") or []):
            step_count += 1
            for issue in step_qa_issues(step, step_index):
                issue_id = issue.get("issue_id") or "unknown"
                issue_counts[issue_id] = issue_counts.get(issue_id, 0) + 1
                issues.append(
                    {
                        "section_index": section_index,
                        "section_title": section.get("title") or "",
                        "step_index": step_index,
                        "issue_id": issue_id,
                        "label": issue.get("label") or issue_id,
                        "step_text": (step.get("text") or "")[:180],
                    }
                )
    validation = {
        "section_count": len(draft_sections or []),
        "step_count": step_count,
        "issue_count": len(issues),
        "issue_counts": issue_counts,
        "issues": issues[:40],
        "validated_at": datestamp(),
    }
    transcript_sources = [
        {
            "name": file.get("name"),
            "status": file.get("transcript_status") or "",
            "words": file.get("timed_word_count") or 0,
        }
        for file in active_job_files(job)
        if file.get("kind") in {"video", "audio"}
    ]
    audit = {
        "created_at": datestamp(),
        "sections": validation["section_count"],
        "steps": validation["step_count"],
        "qa_issues": validation["issue_count"],
        "timing": timing_stats or job.get("step_timing") or {},
        "media": media_stats or job.get("media_selection") or {},
        "supplemental": supplemental_stats or {},
        "transcripts": transcript_sources,
        "cache": job.get("processing_cache") or {},
        "frame_analysis_count": len(job.get("frame_analysis") or []),
    }
    return validation, audit


def apply_draft_validation(job, draft_sections, timing_stats=None, media_stats=None, supplemental_stats=None):
    validation, audit = validate_draft_outputs(job, draft_sections, timing_stats, media_stats, supplemental_stats)
    job["draft_validation"] = validation
    job["draft_audit"] = audit
    flags = job.setdefault("flags", [])
    if validation.get("issue_count") and "Draft QA needed" not in flags:
        flags.append("Draft QA needed")
    if not validation.get("issue_count") and "Draft QA needed" in flags:
        flags.remove("Draft QA needed")
    return validation, audit


def run_media_assignment_stage(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        if not job.get("draft_sections"):
            set_stage(job, "draft_sections", "failed", 0, "No draft sections available")
            job["status"] = "Needs Attention"
            if "No draft sections" not in job["flags"]:
                job["flags"].append("No draft sections")
            add_log(job, "Media assignment skipped: no draft sections are available.")
            return False
        set_stage(job, "draft_sections", "running", 12, "Reassigning step media")
        job["status"] = "Drafting Sections"
        add_log(job, "Step media reassignment started.")
        draft_sections = json.loads(json.dumps(job.get("draft_sections") or []))
    timing_stats = assign_step_timings(JOBS[job_id], draft_sections)
    draft_sections, media_stats = attach_best_media_to_drafts(JOBS[job_id], draft_sections)
    supplemental_stats = {}
    if media_stats.get("zero_image_steps") or media_stats.get("fallback_image_steps"):
        supplemental_stats = analyze_supplemental_step_frames(job_id, draft_sections)
        if supplemental_stats.get("scored_frames"):
            draft_sections, media_stats = attach_best_media_to_drafts(JOBS[job_id], draft_sections)
            media_stats["supplemental_scored_frames"] = supplemental_stats.get("scored_frames", 0)
            media_stats["supplemental_weak_steps"] = supplemental_stats.get("weak_steps", 0)
    with STATE_LOCK:
        job = JOBS[job_id]
        job["draft_sections"] = draft_sections
        job["step_timing"] = timing_stats
        job["media_selection"] = media_stats
        validation, audit = apply_draft_validation(job, draft_sections, timing_stats, media_stats, supplemental_stats)
        note = "Ready for review"
        if validation.get("issue_count"):
            note += f" · {validation.get('issue_count')} QA items"
        set_stage(job, "draft_sections", "done", 100, note)
        job["status"] = "Draft Sections Ready"
        finish_job_processing_timing(job)
        add_log(
            job,
            f"Media reassignment complete: {media_stats.get('steps', 0)} steps matched · {media_stats.get('fallback_image_steps', 0)} fallback · {media_stats.get('zero_image_steps', 0)} no-image · {validation.get('issue_count', 0)} QA items.",
        )
    return True


def finish_drafting(job_id):
    with STATE_LOCK:
        job = JOBS[job_id]
        if not transcript_is_ready_for_drafting(job):
            pause_for_transcript(job)
            return
        set_stage(job, "questions", "done", 100, "Resolved or bypassed")
        set_stage(job, "draft_sections", "running", 5, "Drafting sections")
        job["status"] = "Drafting Sections"
        job["flags"] = []
        add_log(job, "All questions resolved or bypassed.")
        add_log(job, "Draft section generation started.")
    for i in range(1, 8):
        time.sleep(0.55)
        with STATE_LOCK:
            job = JOBS[job_id]
            set_stage(job, "draft_sections", "running", min(96, 5 + i * 13), "Drafting sections")
    with STATE_LOCK:
        job = JOBS[job_id]
    draft_sections = draft_sections_for(job)
    timing_stats = assign_step_timings(job, draft_sections)
    draft_sections, media_stats = attach_best_media_to_drafts(job, draft_sections)
    supplemental_stats = {}
    if media_stats.get("zero_image_steps") or media_stats.get("fallback_image_steps"):
        supplemental_stats = analyze_supplemental_step_frames(job_id, draft_sections)
        if supplemental_stats.get("scored_frames"):
            with STATE_LOCK:
                job = JOBS[job_id]
            draft_sections, media_stats = attach_best_media_to_drafts(job, draft_sections)
            media_stats["supplemental_scored_frames"] = supplemental_stats.get("scored_frames", 0)
            media_stats["supplemental_weak_steps"] = supplemental_stats.get("weak_steps", 0)
    with STATE_LOCK:
        job = JOBS[job_id]
        job["draft_sections"] = draft_sections
        job["step_timing"] = timing_stats
        job["media_selection"] = media_stats
        validation, audit = apply_draft_validation(job, draft_sections, timing_stats, media_stats, supplemental_stats)
        draft_note = "Ready for review"
        if validation.get("issue_count"):
            draft_note += f" · {validation.get('issue_count')} QA items"
        set_stage(job, "draft_sections", "done", 100, draft_note)
        job["status"] = "Draft Sections Ready"
        finish_job_processing_timing(job)
        if timing_stats.get("steps"):
            add_log(
                job,
                f"Step timing aligned: {timing_stats.get('matched_steps', 0)}/{timing_stats.get('steps', 0)} steps matched across {timing_stats.get('segments', 0)} transcript segments.",
            )
        if media_stats.get("scored_frames"):
            add_log(
                job,
                f"Step media selected: {media_stats.get('steps', 0)} steps matched from {media_stats.get('scored_frames', 0)} scored frames · {media_stats.get('single_image_steps', 0)} single-image · {media_stats.get('double_image_steps', 0)} two-image · {media_stats.get('fallback_image_steps', 0)} fallback · {media_stats.get('zero_image_steps', 0)} no-image · {media_stats.get('merge_candidate_steps', 0)} merge candidates · {media_stats.get('pointing_frames', 0)} pointing frames.",
            )
        if supplemental_stats.get("scored_frames"):
            add_log(
                job,
                f"Supplemental media improved candidate pool: {supplemental_stats.get('scored_frames', 0)} frames scored for {supplemental_stats.get('weak_steps', 0)} weak steps.",
            )
        add_log(
            job,
            f"Draft audit: {audit.get('sections', 0)} sections · {audit.get('steps', 0)} steps · {audit.get('qa_issues', 0)} QA items · {audit.get('frame_analysis_count', 0)} scored frames.",
        )
        add_log(job, "Draft sections ready for review.")


def promote_job_sections(job):
    created = []
    for draft in job["draft_sections"]:
        section_id = short_id("sec")
        section = {
            "id": section_id,
            "title": draft["title"],
            "product": draft["product"],
            "work_area": draft["work_area"],
            "station": draft["station"],
            "status": "Needs Edit",
            "tags": draft["tags"],
            "source_job_id": job["id"],
            "source_job": f"{job['job_no']} - {job['name']}",
            "steps": draft["steps"],
        }
        SECTIONS[section_id] = section
        created.append(section_id)
    return created


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def normalize_media_settings(value, media_paths=None):
    if not isinstance(value, dict):
        return {}
    allowed_paths = set(media_paths or [])
    fit_values = {"contain", "cover"}
    aspect_values = {"auto", "portrait", "landscape", "square"}
    position_values = {"center", "top", "bottom", "left", "right"}
    settings = {}
    for raw_path, raw_setting in value.items():
        path = str(raw_path or "").strip()
        if not path or (allowed_paths and path not in allowed_paths) or not isinstance(raw_setting, dict):
            continue
        fit = str(raw_setting.get("fit") or "contain").strip().lower()
        if fit == "fill":
            fit = "cover"
        aspect = str(raw_setting.get("aspect") or "auto").strip().lower()
        crop_position = str(raw_setting.get("crop_position") or raw_setting.get("position") or "center").strip().lower()
        settings[path] = {
            "fit": fit if fit in fit_values else "contain",
            "aspect": aspect if aspect in aspect_values else "auto",
            "crop_position": crop_position if crop_position in position_values else "center",
        }
    return settings


def clean_step(step):
    cleaned = {
        "step_id": step.get("step_id") or short_id("step"),
        "text": step.get("text") or "",
        "tools": step.get("tools") or "",
        "torque": step.get("torque") or "",
        "quality": step.get("quality") or "",
        "safety": step.get("safety") or "",
        "parts": normalize_list(step.get("parts")),
        "media": normalize_list(step.get("media")),
    }
    media_settings = normalize_media_settings(step.get("media_settings"), cleaned["media"])
    if media_settings:
        cleaned["media_settings"] = media_settings
    for key in ["start_time", "end_time", "target_time"]:
        if step.get(key) is not None:
            try:
                cleaned[key] = round(float(step.get(key)), 2)
            except Exception:
                pass
    for key in ["timing_source", "timing_confidence"]:
        if step.get(key):
            cleaned[key] = step.get(key)
    for key in ["source_step_id", "raw_transcript", "boundary_reason", "transcript_confidence", "media_match_confidence", "media_fallback_reason"]:
        if step.get(key):
            cleaned[key] = step.get(key)
    for key in ["media_fallback", "merge_candidate"]:
        if step.get(key) is not None:
            cleaned[key] = bool(step.get(key))
    for key in ["source_step_start", "source_step_end", "media_match_score", "timing_word_matches", "media_fallback_distance_seconds"]:
        if step.get(key) is not None:
            try:
                cleaned[key] = round(float(step.get(key)), 2)
            except Exception:
                cleaned[key] = step.get(key)
    if isinstance(step.get("media_candidates"), list):
        cleaned["media_candidates"] = step.get("media_candidates")[:12]
    if isinstance(step.get("qa_resolutions"), dict):
        resolutions = {}
        for key, value in step.get("qa_resolutions", {}).items():
            if not isinstance(value, dict):
                continue
            status = value.get("status") or ""
            if status not in {"bypassed", "resolved"}:
                continue
            resolutions[str(key)] = {
                "status": status,
                "note": str(value.get("note") or "")[:240],
                "at": str(value.get("at") or "")[:80],
            }
        if resolutions:
            cleaned["qa_resolutions"] = resolutions
    if isinstance(step.get("merge_recommendation"), dict):
        recommendation = step.get("merge_recommendation") or {}
        cleaned["merge_recommendation"] = {
            "recommended": bool(recommendation.get("recommended")),
            "direction": recommendation.get("direction") or "",
            "reason": recommendation.get("reason") or "",
            "neighbor_step": recommendation.get("neighbor_step") or "",
        }
    if isinstance(step.get("evidence_review"), dict):
        review = step.get("evidence_review") or {}
        cleaned["evidence_review"] = {
            "media_reviewed": bool(review.get("media_reviewed")),
            "image_manually_selected": bool(review.get("image_manually_selected")),
            "timing_adjusted": bool(review.get("timing_adjusted")),
            "needs_more_media": bool(review.get("needs_more_media")),
            "note": str(review.get("note") or "")[:240],
            "updated_at": str(review.get("updated_at") or "")[:80],
        }
    return cleaned


def step_resolution_handled(step, issue_id):
    resolution = (step.get("qa_resolutions") or {}).get(issue_id) or {}
    return resolution.get("status") in {"bypassed", "resolved"}


def step_qa_issues(step, step_index=0):
    text = (step.get("text") or "").strip()
    lower = text.lower()
    parts = normalize_list(step.get("parts"))
    media = normalize_list(step.get("media"))
    torque = (step.get("torque") or "").strip().lower()
    issues = []

    def add(issue_id, label):
        if not step_resolution_handled(step, issue_id):
            issues.append(
                {
                    "issue_id": issue_id,
                    "step_id": step.get("step_id") or "",
                    "step_index": step_index,
                    "label": label,
                }
            )

    if not text:
        add("blank_text", "Blank instruction")
    if not media:
        add("missing_media", "Missing media")
    elif step.get("media_fallback") or step.get("media_match_confidence") in {"fallback", "low", "missing"}:
        add("fallback_media", "Low-confidence media")
    if step.get("merge_candidate"):
        add("merge_candidate", "Merge candidate")
    if lower and not parts and re.search(r"\b(attach|install|insert|place|apply|assemble|add|mount|connect|bolt|screw|fasten|position|weld)\b", lower):
        add("missing_parts", "Missing parts")
    if re.search(r"\b(bolt|screw|fastener|nut|tighten|torque)\b", lower) and (not torque or torque in {"none", "not required", "n/a", "na"}):
        add("missing_torque", "Missing torque")
    if not (step.get("quality") or "").strip() and re.search(r"\b(check|verify|confirm|inspect|test|measure|flush|seated|alignment|align)\b", lower):
        add("quality_cue", "Quality cue")
    if not (step.get("safety") or "").strip() and re.search(r"\b(sharp|hot|pinch|ppe|glove|weld|paint|fume|disconnect|lockout|lift|heavy|pressure|energized)\b", lower):
        add("safety_cue", "Safety cue")
    return issues


def section_qa_issues(section):
    issues = []
    for index, step in enumerate(section.get("steps") or []):
        issues.extend(step_qa_issues(step, index))
    return issues


def section_snapshot(section):
    return {
        "id": section.get("id"),
        "title": section.get("title") or "Untitled Section",
        "product": section.get("product") or "",
        "work_area": section.get("work_area") or "",
        "station": section.get("station") or "",
        "status": section.get("status") or "Needs Edit",
        "tags": normalize_list(section.get("tags")),
        "source_job_id": section.get("source_job_id") or "",
        "source_job": section.get("source_job") or "",
        "steps": [clean_step(step) for step in section.get("steps") or []],
    }


def source_job_label(job):
    return f"{job.get('job_no')} - {job.get('name')}"


def source_job_for_section(section):
    source_job_id = section.get("source_job_id")
    if source_job_id and source_job_id in JOBS:
        return JOBS[source_job_id]
    source_label = section.get("source_job") or ""
    for job in JOBS.values():
        if source_job_label(job) == source_label:
            return job
    return None


def previewable_media_path(path):
    return bool(re.search(r"\.(jpg|jpeg|png|webp|gif)$", str(path or ""), flags=re.IGNORECASE))


def section_current_media(section):
    used = set()
    for step in section.get("steps") or []:
        used.update(normalize_list(step.get("media")))
    return used


def media_option_record(path, kind, source_file="", timestamp_seconds=None, label="", tags=None, score=0, reason="", summary="", used=None):
    return {
        "path": path,
        "kind": kind,
        "source_file": source_file or "",
        "timestamp_seconds": timestamp_seconds,
        "label": label or Path(path or "").name,
        "tags": sorted(set(tags or [])),
        "score": score or 0,
        "reason": reason or "",
        "summary": summary or "",
        "is_currently_used": path in (used or set()),
        "previewable": previewable_media_path(path),
    }


def section_media_options(section):
    used = section_current_media(section)
    options = {}
    job = source_job_for_section(section)
    if job:
        analysis_by_path = {
            item.get("stored_path"): item
            for item in job.get("frame_analysis", [])
            if item.get("stored_path")
        }
        for artifact in frame_artifacts(job):
            path = artifact.get("stored_path")
            if not path:
                continue
            analysis = analysis_by_path.get(path) or artifact.get("analysis") or {}
            tags = [
                *(artifact.get("candidate_tags") or []),
                *(analysis.get("tags") or []),
                "frame",
            ]
            for tag in [artifact.get("sampling_tier"), analysis.get("sampling_tier")]:
                if tag:
                    tags.append(tag)
            if artifact.get("sampling_reason") or analysis.get("sampling_reason"):
                tags.append("adaptive_sample")
            if analysis.get("pointing") or artifact.get("pointing"):
                tags.append("pointing")
            score = analysis.get("instruction_value", artifact.get("instruction_value", artifact.get("local_priority", 0)))
            options[path] = media_option_record(
                path,
                "frame",
                artifact.get("source_file") or "",
                artifact.get("timestamp_seconds", analysis.get("timestamp_seconds")),
                artifact.get("timestamp") or Path(path).name,
                tags,
                score,
                analysis.get("reason") or artifact.get("candidate_reason") or "",
                analysis.get("visual_summary") or artifact.get("candidate_reason") or "",
                used,
            )
        for file in active_job_files(job):
            path = file.get("stored_path")
            if file.get("kind") == "image" and path:
                options[path] = media_option_record(
                    path,
                    "image",
                    file.get("name") or "",
                    None,
                    file.get("name") or Path(path).name,
                    ["uploaded", "image"],
                    50,
                    file.get("processing_status") or "",
                    file.get("name") or "",
                    used,
                )
        for artifact in job.get("artifacts", []):
            path = artifact.get("stored_path")
            if artifact.get("kind") in {"drawing", "drawing_page", "pdf_page", "image"} and path:
                options[path] = media_option_record(
                    path,
                    artifact.get("kind"),
                    artifact.get("source_file") or "",
                    artifact.get("timestamp_seconds"),
                    artifact.get("name") or Path(path).name,
                    [artifact.get("kind")],
                    artifact.get("instruction_value", 0),
                    artifact.get("candidate_reason") or artifact.get("processing_status") or "",
                    artifact.get("summary") or artifact.get("processing_status") or "",
                    used,
                )
    for media in section.get("media_uploads") or []:
        path = media.get("stored_path")
        if path:
            options[path] = media_option_record(
                path,
                "section_image",
                media.get("name") or "",
                None,
                media.get("name") or Path(path).name,
                ["manual_upload", "image", "section"],
                100,
                "Manually uploaded in the section editor.",
                media.get("name") or "",
                used,
            )
    for path in used:
        if path not in options:
            options[path] = media_option_record(path, "existing", "", None, Path(path).name, ["existing"], 0, "Already attached to this section.", "", used)
    return sorted(
        options.values(),
        key=lambda item: (
            item.get("kind") != "frame",
            item.get("timestamp_seconds") is None,
            float(item.get("timestamp_seconds") or 0),
            item.get("label") or "",
        ),
    )


def package_step_count(package):
    return sum(len(section.get("steps") or []) for section in package.get("sections") or [])


def package_summary(package):
    return {
        "id": package["id"],
        "title": package.get("title") or "Instruction Package",
        "product": package.get("product") or "",
        "work_area": package.get("work_area") or "",
        "station": package.get("station") or "",
        "status": package.get("status") or "Draft",
        "section_count": len(package.get("section_ids") or []),
        "step_count": package_step_count(package),
        "created_at": package.get("created_at") or "",
        "updated_at": package.get("updated_at") or "",
    }


def ordered_package_snapshots(package, section_ids):
    frozen = {
        section.get("id"): section
        for section in package.get("sections") or []
        if section.get("id")
    }
    snapshots = []
    for section_id in section_ids:
        if section_id in frozen:
            snapshots.append(frozen[section_id])
        elif section_id in SECTIONS:
            snapshots.append(section_snapshot(SECTIONS[section_id]))
    return snapshots


def make_package(data):
    raw_ids = data.get("section_ids") or []
    section_ids = [section_id for section_id in raw_ids if section_id in SECTIONS]
    first_section = SECTIONS.get(section_ids[0]) if section_ids else {}
    now = datestamp()
    package_id = short_id("pkg")
    package = {
        "id": package_id,
        "title": data.get("title") or "Instruction Package",
        "product": data.get("product") or first_section.get("product") or "",
        "work_area": data.get("work_area") or first_section.get("work_area") or "",
        "station": data.get("station") or first_section.get("station") or "",
        "status": data.get("status") or "Draft",
        "section_ids": section_ids,
        "sections": [section_snapshot(SECTIONS[section_id]) for section_id in section_ids],
        "created_at": now,
        "updated_at": now,
        "deleted": False,
    }
    PACKAGES[package_id] = package
    return package


def update_package(package, data):
    for field in ["title", "product", "work_area", "station", "status"]:
        if field in data:
            package[field] = data.get(field) or ""
    if "section_ids" in data:
        section_ids = [section_id for section_id in data.get("section_ids") or [] if section_id in SECTIONS]
        package["section_ids"] = section_ids
        package["sections"] = ordered_package_snapshots(package, section_ids)
    package["updated_at"] = datestamp()
    return package


def refresh_package_snapshot(package):
    section_ids = [section_id for section_id in package.get("section_ids") or [] if section_id in SECTIONS]
    package["section_ids"] = section_ids
    package["sections"] = [section_snapshot(SECTIONS[section_id]) for section_id in section_ids]
    package["updated_at"] = datestamp()
    return package


def job_processing_health(job):
    transcript_sources = [
        file
        for file in active_job_files(job)
        if file.get("kind") in {"video", "audio"}
    ]
    timed_words = sum(int(file.get("timed_word_count") or 0) for file in transcript_sources)
    complete_transcripts = sum(1 for file in transcript_sources if file.get("transcript_status") == "complete")
    step_timing = job.get("step_timing") or {}
    media_selection = job.get("media_selection") or {}
    warnings = []
    if transcript_sources and timed_words <= 0:
        warnings.append("No word-level transcript timing")
    if step_timing.get("steps") and step_timing.get("matched_steps", 0) < step_timing.get("steps", 0):
        warnings.append("Some steps did not align to transcript timing")
    if media_selection.get("zero_image_steps"):
        warnings.append(f"{media_selection.get('zero_image_steps')} steps have no selected image")
    if media_selection.get("fallback_image_steps"):
        warnings.append(f"{media_selection.get('fallback_image_steps')} steps use fallback media")
    if job.get("watchdog", {}).get("stage"):
        warnings.append("Watchdog has recent processing state")
    return {
        "transcript_sources": len(transcript_sources),
        "complete_transcripts": complete_transcripts,
        "timed_words": timed_words,
        "rough_steps": (job.get("step_parse") or {}).get("rough_step_count") or 0,
        "expected_steps": job.get("expected_step_count") or 0,
        "timing_matched_steps": step_timing.get("matched_steps") or 0,
        "timing_steps": step_timing.get("steps") or 0,
        "zero_image_steps": media_selection.get("zero_image_steps") or 0,
        "fallback_image_steps": media_selection.get("fallback_image_steps") or 0,
        "warnings": warnings,
    }


def job_summary(job):
    latest_run = (job.get("processing_runs") or [{}])[-1]
    return {
        "id": job["id"],
        "job_no": job["job_no"],
        "name": job["name"],
        "product": job["product"],
        "work_area": job["work_area"],
        "station": job["station"],
        "profile": job["profile"],
        "expected_step_count": job.get("expected_step_count"),
        "focus_area": job.get("focus_area") or "center",
        "status": job["status"],
        "active": job["active"],
        "deleted": bool(job.get("deleted")),
        "processing_started": has_processing_run(job),
        "needs_processing": bool(job.get("needs_processing")),
        "progress": job["progress"],
        "flags": job["flags"],
        "file_count": len(active_job_files(job)),
        "questions_pending": len(unresolved_questions(job)),
        "draft_count": len(job["draft_sections"]),
        "last_activity": job["last_activity"],
        "watchdog": job.get("watchdog") or {},
        "draft_validation": job.get("draft_validation") or {},
        "draft_audit": job.get("draft_audit") or {},
        "processing_cache": job.get("processing_cache") or {},
        "processing_timing": job.get("processing_timing") or {},
        "latest_processing_run": {
            "run_id": latest_run.get("run_id") or "",
            "status": latest_run.get("status") or "",
            "engine": latest_run.get("engine") or {},
            "started_at": latest_run.get("started_at") or "",
            "duration_seconds": latest_run.get("duration_seconds"),
        },
        "processing_health": job_processing_health(job),
        "stages": job["stages"],
    }


def transcript_results(job):
    normalize_job_sources(job)
    results = []
    for file in active_job_files(job):
        transcript_rel = file.get("transcript_path")
        if not transcript_rel:
            continue
        transcript_file = APP_DIR / transcript_rel
        payload = {
            "source_file": file.get("name"),
            "path": transcript_rel,
            "status": file.get("transcript_status") or "unknown",
            "text": "",
            "error": None,
            "segments": [],
            "words": [],
            "chunks": [],
            "timing": {},
            "timeline_offset_seconds": float(file.get("timeline_offset_seconds") or 0),
            "timeline_scale": float(file.get("timeline_scale") or 1),
            "timeline_sync": file.get("timeline_sync") or {},
            "timeline_alignment": file.get("timeline_alignment") or {},
            "source_id": file.get("source_id") or "",
            "source_role": file.get("source_role") or "",
            "source_label": file.get("source_label") or "",
            "use_for_drafting": bool(file.get("use_for_drafting")),
            "timeline_reference": bool(file.get("timeline_reference")),
        }
        if transcript_file.exists():
            try:
                data = json.loads(transcript_file.read_text(encoding="utf-8"))
                payload.update(
                    {
                        "status": data.get("status") or payload["status"],
                        "text": data.get("text") or "",
                        "error": data.get("error"),
                        "segments": data.get("segments") or [],
                        "words": data.get("words") or [],
                        "chunks": data.get("chunks") or [],
                        "timing": data.get("timing") or {},
                        "timeline_offset_seconds": float(file.get("timeline_offset_seconds") or data.get("timeline_offset_seconds") or 0),
                        "timeline_scale": float(file.get("timeline_scale") or data.get("timeline_scale") or 1),
                        "timeline_sync": file.get("timeline_sync") or data.get("timeline_sync") or {},
                        "timeline_alignment": file.get("timeline_alignment") or data.get("timeline_alignment") or {},
                        "note": data.get("note") or "",
                    }
                )
            except Exception as exc:
                payload["status"] = "read_error"
                payload["error"] = str(exc)
        else:
            payload["status"] = "missing_file"
            payload["error"] = "Transcript file is missing."
        results.append(payload)
    return results


def processing_export():
    jobs = []
    for job in JOBS.values():
        item = json.loads(json.dumps(job))
        item["transcripts"] = transcript_results(job)
        item["summary"] = job_summary(job)
        jobs.append(item)
    return {
        "export_type": "assembly_assistant_processing_log",
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": {
            "storage_root": str(STORAGE_DIR.relative_to(APP_DIR)),
            "stages": [{"key": key, "label": label} for key, label in STAGES],
        },
        "settings": {
            "has_openai_api_key": bool(SETTINGS.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")),
            "openai_api_key": "[redacted]",
            "transcription_backend": SETTINGS.get("transcription_backend"),
            "transcription_model": SETTINGS.get("transcription_model"),
            "local_whisper_cli_path": SETTINGS.get("local_whisper_cli_path"),
            "local_whisper_model_path": SETTINGS.get("local_whisper_model_path"),
            "local_whisper_no_gpu": SETTINGS.get("local_whisper_no_gpu"),
            "instruction_model": SETTINGS.get("instruction_model"),
            "transcription_enabled": SETTINGS.get("transcription_enabled"),
            "default_processing": SETTINGS.get("default_processing"),
        },
        "jobs": jobs,
        "sections": list(SECTIONS.values()),
        "packages": [
            package
            for package in PACKAGES.values()
            if not package.get("deleted")
        ],
    }


def clear_processing_runtime_logs():
    for job in JOBS.values():
        job["logs"] = []
    SETTINGS["last_processing_log_exported_at"] = datestamp()
    save_state()




STATE_INITIALIZED = False


def init_app_state():
    global STATE_INITIALIZED
    if STATE_INITIALIZED:
        return
    init_db()
    if load_state():
        print(f"Loaded persisted data from {DB_PATH}")
    else:
        initialize_empty_state()
        print(f"Initialized empty data store at {DB_PATH}")
    STATE_INITIALIZED = True


def json_response(payload, status=200):
    return jsonify(payload), status


def download_json_response(payload, filename):
    return Response(
        json.dumps(payload, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def request_json():
    return request.get_json(silent=True) or {}


def request_uploads():
    body = request.get_data(cache=False) or b""
    return parse_multipart_files(request.headers.get("Content-Type", ""), body)


def request_query_lists():
    return {key: request.args.getlist(key) for key in request.args.keys()}


def storage_file_response(rel_path):
    try:
        target = (APP_DIR / (rel_path or "")).resolve()
        storage_root = STORAGE_DIR.resolve()
        try:
            target.relative_to(storage_root)
        except ValueError:
            return json_response({"error": "Forbidden"}, 403)
        if not target.exists() or not target.is_file():
            return json_response({"error": "File not found"}, 404)
        return send_file(target, mimetype=mimetypes.guess_type(target.name)[0] or "application/octet-stream")
    except Exception as exc:
        return json_response({"error": str(exc)}, 500)


def create_app():
    init_app_state()
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_DIR / "templates"),
        static_folder=str(PACKAGE_DIR / "static"),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = None

    @app.get("/")
    @app.get("/jobs")
    @app.get("/sections")
    @app.get("/package")
    @app.get("/settings")
    def index_page():
        return render_template("index.html")

    @app.get("/package/<package_id>")
    @app.get("/package/<package_id>/<mode>")
    def package_page(package_id, mode=None):
        if mode and mode not in {"print", "operator", "edit"}:
            return json_response({"error": "Not found"}, 404)
        return render_template("index.html")

    @app.get("/media")
    def media_file():
        return storage_file_response(request.args.get("path", ""))

    @app.get("/api/jobs")
    def api_jobs():
        watch_processing_jobs()
        with STATE_LOCK:
            return json_response(
                {
                    "jobs": [
                        job_summary(job)
                        for job in JOBS.values()
                        if not job.get("deleted")
                    ]
                }
            )

    @app.get("/api/jobs/<job_id>")
    def api_job_detail(job_id):
        watch_processing_jobs()
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("deleted"):
                return json_response({"error": "Job not found"}, 404)
            normalize_job_sources(job)
            payload = dict(job)
            payload["questions_pending"] = len(unresolved_questions(job))
            payload["processing_started"] = has_processing_run(job)
            payload["needs_processing"] = bool(job.get("needs_processing"))
            payload["processing_health"] = job_processing_health(job)
            return json_response({"job": payload})

    @app.get("/api/jobs/<job_id>/transcripts")
    def api_job_transcripts(job_id):
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("deleted"):
                return json_response({"error": "Job not found"}, 404)
            return json_response({"transcripts": transcript_results(job)})

    @app.get("/api/sections")
    def api_sections():
        with STATE_LOCK:
            return json_response(
                {
                    "sections": [
                        section
                        for section in SECTIONS.values()
                        if not section.get("deleted")
                    ]
                }
            )

    @app.get("/api/sections/<section_id>/media-options")
    def api_section_media_options(section_id):
        with STATE_LOCK:
            section = SECTIONS.get(section_id)
            if not section or section.get("deleted"):
                return json_response({"error": "Section not found"}, 404)
            return json_response({"media": section_media_options(section)})

    @app.get("/api/packages")
    def api_packages():
        with STATE_LOCK:
            return json_response(
                {
                    "packages": [
                        package_summary(package)
                        for package in PACKAGES.values()
                        if not package.get("deleted")
                    ]
                }
            )

    @app.get("/api/packages/<package_id>")
    def api_package_detail(package_id):
        with STATE_LOCK:
            package = PACKAGES.get(package_id)
            if not package or package.get("deleted"):
                return json_response({"error": "Package not found"}, 404)
            return json_response({"package": package})

    @app.get("/api/settings")
    def api_settings():
        with STATE_LOCK:
            return json_response({"settings": public_settings()})

    @app.get("/api/export/processing-log")
    def api_export_processing_log():
        with STATE_LOCK:
            return download_json_response(
                processing_export(),
                f"assembly_assistant_processing_log_{time.strftime('%Y%m%d_%H%M%S')}.json",
            )

    @app.post("/api/export/processing-log/clear")
    def api_clear_processing_log():
        with STATE_LOCK:
            clear_processing_runtime_logs()
            return json_response({"ok": True, "cleared_at": SETTINGS.get("last_processing_log_exported_at")})

    @app.post("/api/jobs")
    def api_create_job():
        data = request_json()
        with STATE_LOCK:
            job = make_job(
                data.get("name") or "Untitled Job",
                data.get("product") or "Unassigned Product",
                data.get("work_area") or "Unassigned Area",
                data.get("station") or "Unassigned Station",
                data.get("profile") or "Standard",
                data.get("files") or [],
                data.get("expected_step_count"),
                data.get("focus_area") or "center",
            )
            return json_response({"job": job_summary(job)})

    @app.post("/api/settings")
    def api_update_settings():
        data = request_json()
        with STATE_LOCK:
            key = (data.get("openai_api_key") or "").strip()
            if key:
                SETTINGS["openai_api_key"] = key
            SETTINGS["transcription_backend"] = data.get("transcription_backend") or "auto"
            SETTINGS["transcription_model"] = data.get("transcription_model") or TIMESTAMPED_TRANSCRIPTION_MODEL
            SETTINGS["local_whisper_cli_path"] = data.get("local_whisper_cli_path") or ""
            SETTINGS["local_whisper_model_path"] = data.get("local_whisper_model_path") or ""
            SETTINGS["local_whisper_no_gpu"] = bool(data.get("local_whisper_no_gpu", True))
            SETTINGS["instruction_model"] = data.get("instruction_model") or SETTINGS.get("instruction_model") or "gpt-4o-mini"
            SETTINGS["transcription_enabled"] = bool(data.get("transcription_enabled", True))
            SETTINGS["default_processing"] = data.get("default_processing") or "Standard"
            save_state()
            return json_response({"ok": True, "settings": public_settings()})

    @app.post("/api/packages")
    def api_create_package():
        data = request_json()
        with STATE_LOCK:
            package = make_package(data)
            save_state()
            return json_response({"ok": True, "package": package})

    @app.post("/api/packages/<package_id>/<action>")
    def api_package_action(package_id, action):
        if action not in {"update", "snapshot", "delete"}:
            return json_response({"error": "Not found"}, 404)
        with STATE_LOCK:
            package = PACKAGES.get(package_id)
            if not package or package.get("deleted"):
                return json_response({"error": "Package not found"}, 404)
            if action == "update":
                package = update_package(package, request_json())
                save_state()
                return json_response({"ok": True, "package": package})
            if action == "snapshot":
                package = refresh_package_snapshot(package)
                save_state()
                return json_response({"ok": True, "package": package})
            package["deleted"] = True
            package["status"] = "Deleted"
            package["updated_at"] = datestamp()
            save_state()
            return json_response({"ok": True})

    @app.post("/api/sections/<section_id>/upload-media")
    def api_section_upload_media(section_id):
        uploads = request_uploads()
        with STATE_LOCK:
            section = SECTIONS.get(section_id)
            if not section or section.get("deleted"):
                return json_response({"error": "Section not found"}, 404)
            added, skipped = store_section_media(section, uploads)
            return json_response({"ok": True, "added": added, "skipped": skipped})

    @app.post("/api/jobs/<job_id>/upload-chunk")
    def api_job_upload_chunk(job_id):
        body = request.get_data(cache=False) or b""
        params = request_query_lists()
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return json_response({"error": "Job not found"}, 404)
            payload, status = store_chunked_job_upload(job, params, body)
            if status < 400:
                save_state()
            return json_response(payload, status)

    @app.post("/api/jobs/<job_id>/upload")
    def api_job_upload(job_id):
        uploads = request_uploads()
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return json_response({"error": "Job not found"}, 404)
            if not uploads:
                add_log(job, "Upload failed: no files were received or parsed from the request.")
                return json_response({"error": "No files were received by the upload endpoint."}, 400)
            added, skipped = store_uploaded_files(job, uploads)
            if not added and not skipped:
                add_log(job, "Upload failed: files were received but none were accepted.")
                return json_response({"error": "No uploaded files were accepted."}, 400)
            return json_response({"ok": True, "added": added, "skipped": skipped})

    @app.post("/api/questions/<qid>/upload")
    def api_question_upload(qid):
        uploads = request_uploads()
        with STATE_LOCK:
            target_job, target_question = find_question(qid)
            if not target_question:
                return json_response({"error": "Question not found"}, 404)
            added, skipped = store_uploaded_files(
                target_job,
                uploads,
                source="question clarification",
                question_id=qid,
                clarification_for=target_question["title"],
            )
            target_question["attachment_files"] = added + skipped
        return json_response({"ok": True, "added": added, "skipped": skipped})

    @app.post("/api/jobs/<job_id>/rerun-stage")
    def api_job_rerun_stage(job_id):
        data = request_json()
        stage_key = data.get("stage") or ""
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("deleted"):
                return json_response({"error": "Job not found"}, 404)
        ok = rerun_job_stage(job_id, stage_key)
        return json_response({"ok": ok})

    @app.post("/api/jobs/<job_id>/source")
    def api_job_source(job_id):
        data = request_json()
        identity = data.get("identity") or data.get("source_id") or ""
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("deleted"):
                return json_response({"error": "Job not found"}, 404)
            if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
                return json_response({"error": "Source setup cannot change while processing is running."}, 409)
            source, error = update_job_source(job, identity, data)
            if error:
                return json_response({"error": error}, 400)
            return json_response({"ok": True, "source": source})

    @app.post("/api/jobs/<job_id>/<action>")
    def api_job_action(job_id, action):
        if action not in {"start", "rerun", "complete", "delete", "files", "remove-file"}:
            return json_response({"error": "Not found"}, 404)
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("deleted"):
                return json_response({"error": "Job not found"}, 404)

        if action == "start":
            with STATE_LOCK:
                if not active_job_files(job):
                    add_log(job, "Processing blocked: no uploaded files are attached.")
                    return json_response({"ok": False, "error": "Upload at least one file before starting processing."}, 400)
            ok = start_processing(job_id, reset=True, first_run_only=True)
            return json_response({"ok": ok})

        if action == "rerun":
            with STATE_LOCK:
                if not active_job_files(job):
                    add_log(job, "Rerun blocked: no uploaded files are attached.")
                    return json_response({"ok": False, "error": "Upload at least one file before rerunning processing."}, 400)
            ok = start_processing(job_id, reset=True)
            return json_response({"ok": ok})

        if action == "complete":
            with STATE_LOCK:
                created = promote_job_sections(job)
                job["status"] = "Complete"
                job["active"] = False
                job["flags"] = []
                add_log(job, f"Job complete. {len(created)} sections moved to library.")
                return json_response({"ok": True, "section_ids": created})

        if action == "delete":
            with STATE_LOCK:
                if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
                    return json_response({"error": "Cannot delete while processing"}, 409)
                job["deleted"] = True
                job["active"] = False
                job["status"] = "Deleted"
                add_log(job, "Job deleted from active job list. Stored media was retained.")
                save_state()
                return json_response({"ok": True})

        if action == "files":
            data = request_json()
            incoming = data.get("files") or []
            with STATE_LOCK:
                existing = {f"{f.get('name')}:{f.get('size')}" for f in job["files"]}
                added = 0
                skipped = 0
                for file in incoming:
                    key = f"{file.get('name')}:{file.get('size')}"
                    if key in existing:
                        skipped += 1
                        continue
                    job["files"].append(file)
                    existing.add(key)
                    added += 1
                add_log(job, f"Added {added} files. Skipped {skipped} duplicates.")
                return json_response({"ok": True, "added": added, "skipped": skipped})

        data = request_json()
        identity = data.get("identity") or data.get("sha256") or data.get("stored_path") or ""
        with STATE_LOCK:
            removed = remove_job_upload(job, identity)
            if not removed:
                return json_response({"error": "File not found"}, 404)
            return json_response({"ok": True, "removed": removed})

    @app.post("/api/questions/<qid>/answer")
    def api_question_answer(qid):
        data = request_json()
        with STATE_LOCK:
            target_job, target_question = find_question(qid)
            if not target_question:
                return json_response({"error": "Question not found"}, 404)
            target_question["status"] = data.get("status") or "answered"
            target_question["answer"] = data.get("answer") or ""
            target_question["attachment_note"] = data.get("attachment_note") or ""
            target_question["attachment_files"] = data.get("attachment_files") or []
            metadata_only_files = [
                file
                for file in target_question["attachment_files"]
                if not file.get("sha256")
            ]
            if metadata_only_files:
                existing = {f"{f.get('name')}:{f.get('size')}" for f in target_job["files"]}
                added = 0
                for file in metadata_only_files:
                    enriched = dict(file)
                    enriched["question_id"] = qid
                    enriched["clarification_for"] = target_question["title"]
                    key = f"{enriched.get('name')}:{enriched.get('size')}"
                    if key in existing:
                        continue
                    target_job["files"].append(enriched)
                    existing.add(key)
                    added += 1
                add_log(target_job, f"Attached {added} clarification files to question: {target_question['title']}.")
            elif target_question["attachment_files"]:
                add_log(target_job, f"Linked {len(target_question['attachment_files'])} stored clarification files to question: {target_question['title']}.")
            pending = unresolved_questions(target_job)
            set_question_flag(target_job, len(pending))
            if pending:
                set_stage(
                    target_job,
                    "questions",
                    "blocked",
                    50,
                    f"{len(pending)} decisions pending",
                )
            add_log(target_job, f"Question {target_question['status']}: {target_question['title']}.")
            should_continue = not pending and target_job["status"] == "Questions Pending"
        if should_continue:
            continue_after_questions(target_job["id"])
        return json_response({"ok": True})

    @app.post("/api/sections/<section_id>/<action>")
    def api_section_action(section_id, action):
        if action not in {"duplicate", "complete", "update", "delete"}:
            return json_response({"error": "Not found"}, 404)
        with STATE_LOCK:
            section = SECTIONS.get(section_id)
            if not section or (section.get("deleted") and action != "delete"):
                return json_response({"error": "Section not found"}, 404)
            if action == "delete":
                section["deleted"] = True
                section["status"] = "Deleted"
                section["updated_at"] = datestamp()
                save_state()
                return json_response({"ok": True})
            if action == "duplicate":
                new_id = short_id("sec")
                clone = json.loads(json.dumps(section))
                clone["id"] = new_id
                clone["title"] = f"{section['title']} Copy"
                clone["status"] = "Needs Edit"
                clone.pop("deleted", None)
                SECTIONS[new_id] = clone
                save_state()
                return json_response({"ok": True, "section": clone})
            if action == "complete":
                unresolved = section_qa_issues(section)
                if unresolved:
                    return json_response({"error": "Section has unresolved QA items", "qa_issues": unresolved[:12]}, 409)
                section["status"] = "Complete"
                save_state()
                return json_response({"ok": True, "section": section})

            data = request_json()
            incoming = data.get("section") or {}
            clean_steps = []
            for step in incoming.get("steps") or []:
                clean_steps.append(clean_step(step))
            if not clean_steps:
                clean_steps = section["steps"]
            requested_status = incoming.get("status") or "Needs Edit"
            candidate_section = dict(section)
            candidate_section["steps"] = clean_steps
            unresolved = section_qa_issues(candidate_section) if requested_status == "Complete" else []
            final_status = "Needs Edit" if unresolved else requested_status
            section.update(
                {
                    "title": incoming.get("title") or section["title"],
                    "product": incoming.get("product") or "",
                    "work_area": incoming.get("work_area") or "",
                    "station": incoming.get("station") or "",
                    "status": final_status,
                    "tags": incoming.get("tags") or [],
                    "steps": clean_steps,
                }
            )
            save_state()
            return json_response({"ok": True, "section": section, "blocked_qa": unresolved[:12]})

    @app.errorhandler(404)
    def not_found(_error):
        return json_response({"error": "Not found"}, 404)

    return app


def main(host="127.0.0.1", port=8765):
    app = create_app()
    print(f"Assembly Assistant running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
