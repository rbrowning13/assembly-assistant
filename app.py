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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qs, urlparse

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
APP_DIR = Path(__file__).resolve().parent
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
FRAME_MANIFEST_NAME = "frame_manifest.json"
FRAME_MANIFEST_VERSION = 2
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
    return removed


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


def align_transcript_timelines(job):
    sources = []
    for file in active_job_files(job):
        if file.get("transcript_status") != "complete":
            continue
        payload = transcript_payload_for_file(file)
        if not payload:
            continue
        if len(payload_word_tokens(payload, max_words=60)) < 20:
            continue
        sources.append((file, payload))

    if not sources:
        return []

    reference_file, reference_payload = sources[0]
    reference_file["timeline_offset_seconds"] = 0
    reference_file["timeline_alignment"] = {
        "method": "reference_audio",
        "confidence": "reference",
        "reference_file": reference_file.get("name"),
    }

    alignments = []
    for file, payload in sources[1:]:
        estimate = estimate_transcript_offset(reference_payload, payload)
        if estimate:
            file["timeline_offset_seconds"] = estimate["offset_seconds"]
            file["timeline_alignment"] = {
                "method": "transcript_word_match",
                "reference_file": reference_file.get("name"),
                **estimate,
            }
            alignments.append({"source_file": file.get("name"), **estimate})
        else:
            file["timeline_offset_seconds"] = float(file.get("timeline_offset_seconds") or 0)
            file["timeline_alignment"] = {
                "method": "not_aligned",
                "reference_file": reference_file.get("name"),
                "confidence": "low",
                "reason": "Not enough shared timed words to estimate an offset.",
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
                        if "OpenAI key missing" not in job["flags"]:
                            job["flags"].append("OpenAI key missing")
                        add_log(job, "Transcript provider skipped: OpenAI API key is not set.")
                    else:
                        transcription_failed += 1
                        if "Transcription issue" not in job["flags"]:
                            job["flags"].append("Transcription issue")
                        add_log(job, f"Transcript provider failed for {file.get('name')}: {error}")
    with STATE_LOCK:
        job = JOBS[job_id]
        alignments = align_transcript_timelines(job)
        aligned_sources = len(alignments)
        if alignments:
            for alignment in alignments:
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
        if extraction_failed or transcription_failed:
            status = "failed"
        elif pending_provider:
            status = "blocked"
        set_stage(job, "transcription", status, 100 if not extraction_failed else 75, note)
        add_log(job, f"Transcript stage complete: {note}.")
    return not extraction_failed and not transcription_failed and not pending_provider


def base_stages():
    return {
        key: {
            "key": key,
            "label": label,
            "status": "idle",
            "percent": 0,
            "note": "",
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
    stage["status"] = status
    stage["updated_at"] = datestamp()
    stage["updated_epoch"] = epoch_now()
    if percent is not None:
        stage["percent"] = percent
    if note is not None:
        stage["note"] = note
    recompute_progress(job)
    job["last_activity"] = timestamp()
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
    data["_timeline_alignment"] = file.get("timeline_alignment") or data.get("timeline_alignment") or {}
    return data


def transcript_payloads(job):
    payloads = []
    for file in active_job_files(job):
        payload = transcript_payload_for_file(file)
        if payload:
            payloads.append(payload)
    return payloads


def transcript_text(job):
    texts = []
    for payload in transcript_payloads(job):
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
    for payload in transcript_payloads(job):
        offset = float(payload.get("_timeline_offset_seconds") or 0)
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
            start = max(0, start + offset)
            end = max(start, (end if end is not None else start) + offset)
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
    for payload in transcript_payloads(job):
        offset = float(payload.get("_timeline_offset_seconds") or 0)
        source_file = payload.get("_source_file") or payload.get("source_file")
        for segment in payload.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            text = re.sub(r"\s+", " ", segment.get("text") or "").strip()
            try:
                start = float(segment.get("start")) + offset
                end = float(segment.get("end")) + offset
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


def frame_timestamp_seconds(frame_path, timeline_offset_seconds=0):
    record = frame_manifest_record(frame_path)
    if record:
        try:
            if record.get("timestamp_seconds") is not None:
                return max(0, float(record.get("timestamp_seconds")))
        except Exception:
            pass
        try:
            return max(0, float(record.get("source_timestamp_seconds") or 0) + float(timeline_offset_seconds or record.get("timeline_offset_seconds") or 0))
        except Exception:
            pass
    return max(0, frame_source_timestamp_seconds(frame_path) + float(timeline_offset_seconds or 0))


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
            return sorted(expected_paths, key=lambda path: frame_timestamp_seconds(path, float(file.get("timeline_offset_seconds") or 0)))
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
                "timestamp_seconds": round(max(0, source_timestamp + timeline_offset_seconds), 3),
                "source_timestamp_seconds": round(max(0, source_timestamp), 3),
                "timeline_offset_seconds": round(timeline_offset_seconds, 3),
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


def score_candidate_frame(frame_path, cue_times, duration_seconds, timeline_offset_seconds=0):
    timestamp_seconds = frame_timestamp_seconds(frame_path, timeline_offset_seconds)
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
    strong_cue = bool(tags & {"good_image", "step", "quality", "torque", "tool", "parts"})

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
    timeline_offset_seconds = float((file or {}).get("timeline_offset_seconds") or 0)
    focus_area = normalize_focus_area(job.get("focus_area"))
    duration_seconds = max(frame_timestamp_seconds(frame, timeline_offset_seconds) for frame in frames)
    cue_times = transcript_cue_times(job, duration_seconds)
    cv_metrics, cv_stats = opencv_frame_metrics(frames, focus_area)
    scored = [
        (
            apply_opencv_scoring(
                score_candidate_frame(frame, cue_times, duration_seconds, timeline_offset_seconds),
                cv_metrics.get(frame.name),
            ),
            frame,
        )
        for frame in frames
    ]
    selected = {}
    selection_sources = {}
    content_cue_tags = {"good_image", "step", "parts", "tool", "torque", "threadlocker", "quality", "safety", "fixture"}
    rejected_blurry = 0
    rejected_duplicate = 0

    def keep(frame, meta, source):
        selected[frame] = meta
        selection_sources.setdefault(frame, set()).add(source)

    for meta, frame in scored:
        timestamp_seconds = frame_timestamp_seconds(frame, timeline_offset_seconds)
        tags = set(meta.get("tags") or [])
        strong_cue = bool(tags & {"good_image", "step", "quality", "tool", "parts"})
        endpoint_coverage = timestamp_seconds <= 2 or (duration_seconds and abs(duration_seconds - timestamp_seconds) <= 5)
        if endpoint_coverage:
            keep(frame, meta, "coverage")
            continue
        if "blurry" in tags and not strong_cue:
            rejected_blurry += 1
            continue
        if "near_duplicate" in tags and not (strong_cue or "visual_change" in tags):
            rejected_duplicate += 1
            continue
        if tags & content_cue_tags and meta["score"] >= 24:
            keep(frame, meta, "cue")
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
            if "blurry" in tags and not (tags & {"good_image", "step", "quality"}):
                continue
            if "near_duplicate" in tags and "visual_change" not in tags:
                continue
            keep(frame, meta, "duration_baseline")
            if len(selected) >= duration_baseline:
                break

    candidate_limit = dynamic_candidate_limit(job, len(frames), duration_seconds)
    hard_ceiling_applied = False
    if len(selected) > candidate_limit:
        protected_sources = {"coverage", "cue"}
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

    ordered_frames = sorted(selected, key=lambda frame: frame_timestamp_seconds(frame, timeline_offset_seconds))
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
        "cue_selected": source_counts.get("cue", 0),
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
    duration_bonus = int(max(0, float(duration_seconds or 0)) // 45) * 8
    desired = max(220, step_count * 24 + 96 + duration_bonus)
    return max(1, min(int(raw_count or desired), desired, VIDEO_CANDIDATE_FRAME_LIMIT))


def dynamic_frame_analysis_limit(job, candidate_count):
    step_count = rough_step_count_for_limit(job)
    if not step_count:
        step_count = max(8, int(candidate_count // 24) + 1)
    desired = max(220, step_count * 22 + 96)
    return max(1, min(int(candidate_count or desired), desired, FRAME_ANALYSIS_MAX_FRAMES))


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
    source_timestamp_seconds = frame_source_timestamp_seconds(frame_path)
    timestamp_seconds = frame_timestamp_seconds(frame_path, timeline_offset_seconds)
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
        "transcript_step_slices": [],
        "step_parse": {},
        "draft_sections": [],
        "logs": [],
        "created_at": timestamp(),
        "last_activity": timestamp(),
    }
    JOBS[job_id] = job
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
    job.setdefault("draft_validation", {})
    job.setdefault("draft_audit", {})
    job.setdefault("processing_cache", {})
    job.setdefault("watchdog", {})
    job.setdefault("last_activity_epoch", epoch_now())
    job.setdefault("expected_step_count", None)
    job.setdefault("focus_area", "center")
    job.setdefault("transcript_step_slices", [])
    job.setdefault("step_parse", {})
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
                    "timestamp_seconds": frame_timestamp_seconds(frame_path, timeline_offset_seconds),
                    "source_timestamp_seconds": frame_source_timestamp_seconds(frame_path),
                    "timeline_offset_seconds": timeline_offset_seconds,
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
                    score_candidate_frame(source["path"], cue_times, duration, source.get("timeline_offset_seconds") or 0),
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
            for _, source, meta in sorted(ranked, key=lambda item: item[0], reverse=True)[:4]:
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
        if len(selected) >= SUPPLEMENTAL_FRAME_ANALYSIS_LIMIT:
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
        add_log(job, f"Supplemental media pass: scoring {len(frames)} nearby frames for {weak_steps} steps without images.")

    failed = False
    saved_count = 0
    for start in range(0, len(frames), FRAME_ANALYSIS_BATCH_SIZE):
        batch = frames[start:start + FRAME_ANALYSIS_BATCH_SIZE]
        results, error = analyze_frame_batch(JOBS[job_id], batch)
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
    media_files = [
        file
        for file in active_job_files(job)
        if file.get("kind") in {"video", "audio"} and file.get("readable")
    ]
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
                f"Video frames extracted for {file.get('name')}: {selection_stats.get('raw_frames', len(frames))} raw · {len(frame_records)} candidates · {selection_stats.get('cue_selected', 0)} cue · {selection_stats.get('visual_change_selected', 0)} visual change · focus {selection_stats.get('focus_area', 'center')} · {selection_stats.get('blurry_rejected', 0)} blurry rejected · {selection_stats.get('duplicate_rejected', 0)} duplicate rejected · {ceiling_note} · {opencv_note}{adaptive_note}.",
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
        analysis_limit = dynamic_frame_analysis_limit(job, len(candidates))
        ceiling_applied = len(candidates) > analysis_limit
        prioritized = candidates[:analysis_limit]
        if ceiling_applied:
            step_count = rough_step_count_for_limit(job)
            reason = f"expected/parsed {step_count} steps" if step_count else "candidate volume"
            add_log(job, f"Frame analysis dynamic guardrail applied: {len(candidates)} candidates available, scoring top {analysis_limit} based on {reason}.")
        frames = sorted(prioritized, key=lambda frame: frame.get("timestamp_seconds", 0))
        for index, frame in enumerate(frames, start=1):
            frame["frame_label"] = f"F{index:03d}"
        existing_by_path = {
            item.get("stored_path"): item
            for item in job.get("frame_analysis", [])
            if item.get("stored_path")
        }

    if not frames:
        return

    all_results = [
        existing_by_path.get(frame.get("stored_path"))
        for frame in frames
        if existing_by_path.get(frame.get("stored_path"))
    ]
    reused_analysis_count = len(all_results)
    failed = False
    for start in range(0, len(frames), FRAME_ANALYSIS_BATCH_SIZE):
        batch = frames[start:start + FRAME_ANALYSIS_BATCH_SIZE]
        if all(frame.get("stored_path") in existing_by_path for frame in batch):
            continue
        with STATE_LOCK:
            job = JOBS[job_id]
            percent = 78 + round((start / max(1, len(frames))) * 17)
            set_stage(job, "video", "running", percent, f"Scoring frames {start + 1}-{start + len(batch)}")
            add_log(job, f"Analyzing video frames {start + 1}-{start + len(batch)} for clarity, part placement, and pointing.")
        results, error = analyze_frame_batch(job, batch)
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
            add_log(job, f"Frame analysis saved: {len(ordered_results)}/{len(frames)} scored · {useful_count} useful · {pointing_count} pointing.")

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
            add_log(job, "Processing queued.")
        thread = threading.Thread(target=run_job_worker, args=(job_id, process_job), daemon=True)
        RUNNING_THREADS[job_id] = thread
        thread.start()
        return True


def process_job(job_id):
    if not run_upload_check(job_id):
        return
    transcription_ready = run_transcription_stage(job_id)
    stages = [
        ("drawings", "Reading PDFs, OCR text, title blocks, and BOM-like tables.", 2.2),
        ("images", "Scoring still images for clarity and build context.", 1.8),
    ]
    for key, note, seconds in stages:
        if not run_stage(job_id, key, note, seconds):
            return
    if not run_video_stage(job_id):
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
    allowed = {"transcription", "video", "draft_sections", "media_assignment"}
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
        "processing_health": job_processing_health(job),
        "stages": job["stages"],
    }


def transcript_results(job):
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
            "timeline_alignment": file.get("timeline_alignment") or {},
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


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Assembly Assistant</title>
  <style>
    :root {
      --bg: #cfcfca;
      --panel: #ffffff;
      --panel-2: #f0efea;
      --ink: #1f2328;
      --muted: #697079;
      --line: #d9d6cc;
      --brand: #0f766e;
      --brand-2: #155e75;
      --warn: #a15c07;
      --bad: #b42318;
      --good: #16703a;
      --shadow: 0 10px 26px rgba(31, 35, 40, .08);
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      min-height: 36px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary { background: var(--brand); color: white; border-color: var(--brand); }
    button.ghost { background: transparent; }
    button.danger { color: var(--bad); border-color: rgba(180,35,24,.32); background: rgba(180,35,24,.06); }
    button.icon { width: 36px; padding: 0; display: inline-grid; place-items: center; }
    button:disabled { opacity: .48; cursor: not-allowed; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      min-height: 38px;
      padding: 8px 10px;
      color: var(--ink);
    }
    textarea { min-height: 78px; resize: vertical; }
    textarea.transcript-box {
      min-height: 240px;
      max-height: 560px;
      overflow: auto;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 228px 1fr;
    }
    .app.nav-closed { grid-template-columns: 56px 1fr; }
    .nav {
      border-right: 1px solid var(--line);
      background: #deded8;
      padding: 14px 10px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      position: sticky;
      top: 0;
      height: 100vh;
      align-self: start;
      overflow-y: auto;
      z-index: 4;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      padding: 0 4px;
      font-weight: 720;
      white-space: nowrap;
      overflow: hidden;
    }
    .brand-mark {
      width: 30px;
      height: 30px;
      border-radius: 7px;
      background: var(--ink);
      color: white;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
    }
    .nav-items { display: grid; gap: 6px; }
    .nav button {
      justify-content: flex-start;
      display: flex;
      align-items: center;
      gap: 10px;
      border: 0;
      background: transparent;
      text-align: left;
    }
    .nav button.active { background: white; box-shadow: inset 0 0 0 1px var(--line); }
    .nav-closed .nav-label, .nav-closed .brand-text { display: none; }
    .main { min-width: 0; display: flex; flex-direction: column; }
    .topbar {
      height: 58px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: rgba(207,207,202,.9);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    .topbar h1 { margin: 0; font-size: 20px; line-height: 1.1; }
    .top-actions { display: flex; align-items: center; gap: 8px; }
    .content { padding: 18px 20px 28px; min-width: 0; }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .toolbar-left, .toolbar-right { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .search { width: min(360px, 72vw); }
    .table-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: auto;
      box-shadow: var(--shadow);
    }
    table { width: 100%; border-collapse: collapse; min-width: 960px; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; text-align: left; }
    th {
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      background: #fbfaf7;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    tr:hover td { background: #fbfaf7; }
    td.main-cell { min-width: 230px; }
    .subtle { color: var(--muted); font-size: 12px; }
    .strong { font-weight: 720; }
    .chip-row { display: flex; gap: 5px; align-items: center; flex-wrap: wrap; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: #f8f7f3;
      color: var(--muted);
      white-space: nowrap;
    }
    .chip.good { color: var(--good); border-color: rgba(22,112,58,.25); background: rgba(22,112,58,.08); }
    .chip.warn { color: var(--warn); border-color: rgba(161,92,7,.28); background: rgba(161,92,7,.08); }
    .chip.bad { color: var(--bad); border-color: rgba(180,35,24,.28); background: rgba(180,35,24,.08); }
    .chip.brand { color: var(--brand-2); border-color: rgba(21,94,117,.24); background: rgba(21,94,117,.08); font-weight: 640; }
    .stage-dots { display: grid; grid-template-columns: repeat(7, 24px); gap: 5px; }
    .dot {
      width: 24px;
      height: 24px;
      border-radius: 999px;
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      font-size: 12px;
      background: var(--panel-2);
      color: var(--muted);
    }
    .dot.done { background: rgba(22,112,58,.12); color: var(--good); border-color: rgba(22,112,58,.25); }
    .dot.running { background: rgba(21,94,117,.13); color: var(--brand-2); border-color: rgba(21,94,117,.25); }
    .dot.blocked { background: rgba(161,92,7,.12); color: var(--warn); border-color: rgba(161,92,7,.28); }
    .dot.failed { background: rgba(180,35,24,.1); color: var(--bad); border-color: rgba(180,35,24,.28); }
    .bar {
      height: 8px;
      background: #e7e4dc;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar > span { display: block; height: 100%; width: 0; background: var(--brand); transition: width .22s ease; }
    .grid-2 {
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      gap: 14px;
      align-items: start;
      min-width: 0;
    }
    .grid-2 > * { min-width: 0; }
    .stage-review-grid {
      display: grid;
      grid-template-columns: minmax(300px, 420px) 1fr;
      gap: 14px;
      align-items: stretch;
      margin-bottom: 14px;
      min-width: 0;
    }
    .stage-review-grid > * { min-width: 0; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: hidden;
    }
    .panel-head {
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .panel-head h2, .panel-head h3 { margin: 0; font-size: 15px; }
    .panel-body { padding: 14px; min-width: 0; overflow-wrap: anywhere; }
    .stack { display: grid; gap: 12px; min-width: 0; }
    .stage-list { display: grid; gap: 8px; }
    .stage-item {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      background: #fbfaf7;
      min-width: 0;
    }
    .stage-item.selectable { cursor: pointer; }
    .stage-item.active { border-color: var(--brand); background: rgba(15,118,110,.06); }
    .stage-detail-body {
      max-height: calc(100vh - 235px);
      min-height: 420px;
      overflow: auto;
    }
    .stage-detail-list {
      display: grid;
      gap: 8px;
    }
    .frame-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 10px;
      max-height: 620px;
      overflow: auto;
      padding-right: 4px;
    }
    .frame-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      background: #fbfaf7;
      min-width: 0;
    }
    .frame-card.clickable, .thumb.clickable, .media-option-preview.clickable { cursor: zoom-in; }
    .frame-card img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      background: #ece9e0;
    }
    .frame-card .subtle { padding: 6px; overflow-wrap: anywhere; }
    .draft-browser {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      gap: 12px;
      height: clamp(540px, calc(100vh - 290px), 760px);
      min-height: 0;
    }
    .draft-browser-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .draft-step-view {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: white;
      padding: 14px;
      display: grid;
      align-content: start;
      gap: 12px;
      min-height: 0;
      max-height: 100%;
      overflow-x: hidden;
      overflow-y: auto;
    }
    .draft-step-text {
      font-size: 20px;
      line-height: 1.22;
      font-weight: 680;
    }
    .draft-nav {
      display: grid;
      grid-template-columns: 140px minmax(0, 1fr) 140px;
      gap: 10px;
      align-items: center;
      min-height: 44px;
    }
    .draft-nav > button {
      width: 100%;
      white-space: nowrap;
    }
    .draft-nav .chip-row {
      justify-content: center;
      flex-wrap: nowrap;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    .draft-media {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 10px;
    }
    .draft-media .thumb {
      width: 100%;
      height: auto;
      aspect-ratio: 16 / 9;
    }
    .question-card, .section-card, .step-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: white;
      padding: 12px;
    }
    .question-card.pending { border-left: 4px solid var(--warn); }
    .question-card.answered { border-left: 4px solid var(--good); }
    .question-card.bypassed { border-left: 4px solid var(--muted); }
    .section-qa-panel .panel-body {
      gap: 8px;
    }
    .qa-list {
      display: grid;
      gap: 7px;
    }
    .qa-item {
      width: 100%;
      text-align: left;
      justify-content: flex-start;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      padding: 8px;
      min-height: auto;
    }
    .qa-item.active {
      border-color: rgba(15,118,110,.45);
      background: rgba(15,118,110,.06);
    }
    .qa-item small {
      display: block;
      color: var(--muted);
      font-weight: 500;
      line-height: 1.25;
      margin-top: 2px;
    }
    .qa-attention-row {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
      padding: 9px;
      display: grid;
      gap: 8px;
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      margin-bottom: 8px;
    }
    .source-tile {
      min-height: 96px;
      border: 1px dashed #c9c4b8;
      border-radius: 7px;
      background: linear-gradient(135deg, #efeee7, #faf9f5);
      display: grid;
      place-items: center;
      color: var(--muted);
      margin: 10px 0;
      text-align: center;
      padding: 10px;
    }
    .choice-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
    .choice-row button.selected { background: rgba(15,118,110,.1); border-color: var(--brand); color: var(--brand); }
    .split-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
    .log {
      background: #171a1d;
      color: #e7ebe8;
      border-radius: 7px;
      padding: 12px;
      height: 210px;
      overflow: auto;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .log-line { display: grid; grid-template-columns: 82px minmax(0, 1fr); gap: 8px; margin-bottom: 6px; }
    .log-line span { overflow-wrap: anywhere; }
    .file-list { display: grid; gap: 8px; }
    .file-row {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
      padding: 9px;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .file-row > * { min-width: 0; }
    details {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: white;
      min-width: 0;
      overflow: hidden;
    }
    summary { cursor: pointer; padding: 12px 14px; font-weight: 700; }
    details .details-body { padding: 0 14px 14px; min-width: 0; }
    .section-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }
    .section-card { display: grid; gap: 10px; }
    .section-table td { vertical-align: top; }
    .section-title-cell { min-width: 260px; }
    .section-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
      min-width: 300px;
    }
    .section-workspace {
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
      min-width: 0;
    }
    .editor-step-list {
      display: grid;
      gap: 7px;
      max-height: calc(100vh - 440px);
      min-height: 140px;
      overflow: auto;
      padding-right: 4px;
    }
    .editor-step-row {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
      padding: 9px;
      cursor: pointer;
      display: grid;
      gap: 3px;
    }
    .editor-step-row.active {
      border-color: var(--brand);
      background: rgba(15,118,110,.06);
    }
    .editor-step-row .preview-line {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .token-field {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      min-height: 38px;
      padding: 5px;
      display: flex;
      align-items: center;
      gap: 5px;
      flex-wrap: wrap;
    }
    .token-field:focus-within {
      border-color: rgba(15,118,110,.55);
      box-shadow: 0 0 0 3px rgba(15,118,110,.1);
    }
    .token-badge {
      min-height: 28px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: rgba(21,94,117,.08);
      border-color: rgba(21,94,117,.22);
      color: var(--brand-2);
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .token-badge .token-remove {
      color: var(--muted);
      font-weight: 800;
      line-height: 1;
    }
    .token-list {
      display: contents;
    }
    .token-input {
      flex: 1 1 130px;
      min-width: 110px;
      border: 0;
      min-height: 28px;
      padding: 2px 4px;
      background: transparent;
      box-shadow: none;
    }
    .token-input:focus {
      outline: none;
    }
    .selected-step-media {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 360px));
      justify-content: start;
      align-items: start;
      gap: 12px;
    }
    .step-media-manager {
      display: grid;
      gap: 10px;
    }
    .step-media-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .step-media-toolbar .split-actions {
      margin: 0;
    }
    .step-media-upload {
      position: relative;
      overflow: hidden;
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      padding: 8px 13px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      cursor: pointer;
      font-weight: 650;
    }
    .step-media-upload input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }
    .step-media-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
      overflow: hidden;
      display: grid;
      align-self: start;
      box-shadow: 0 8px 18px rgba(31, 35, 40, .06);
    }
    .step-media-card.primary, .step-media-card.hero {
      border-color: rgba(15,118,110,.45);
      background: rgba(15,118,110,.05);
    }
    .media-card-preview {
      width: 100%;
      min-height: 0;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: #e9e7df;
      cursor: zoom-in;
      position: relative;
      display: block;
      overflow: hidden;
    }
    .media-display {
      width: 100%;
      background: #e9e7df;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .media-display.aspect-auto { aspect-ratio: 16 / 9; }
    .media-display.aspect-portrait { aspect-ratio: 3 / 4; }
    .media-display.aspect-landscape { aspect-ratio: 16 / 9; }
    .media-display.aspect-square { aspect-ratio: 1 / 1; }
    .media-display img {
      width: 100%;
      height: 100%;
      display: block;
      background: #e9e7df;
    }
    .media-display.fit-contain img { object-fit: contain; }
    .media-display.fit-cover img { object-fit: cover; }
    .media-display.position-center img { object-position: center center; }
    .media-display.position-top img { object-position: center top; }
    .media-display.position-bottom img { object-position: center bottom; }
    .media-display.position-left img { object-position: left center; }
    .media-display.position-right img { object-position: right center; }
    .media-card-preview .media-display.aspect-auto {
      aspect-ratio: auto;
      min-height: 180px;
    }
    .media-card-preview .media-display.aspect-auto img {
      height: auto;
      max-height: 440px;
      object-fit: contain;
    }
    .media-card-preview .media-display.aspect-auto.fit-cover {
      aspect-ratio: 16 / 9;
      min-height: 0;
    }
    .media-card-preview .media-display.aspect-auto.fit-cover img {
      height: 100%;
      max-height: none;
      object-fit: cover;
    }
    .media-card-preview img {
      width: 100%;
      display: block;
    }
    .media-setting-controls {
      display: grid;
      gap: 6px;
      border-top: 1px solid var(--line);
      padding-top: 7px;
    }
    .media-control-row {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .media-control-row .subtle {
      width: 54px;
      flex: 0 0 auto;
    }
    .segmented {
      display: inline-flex;
      gap: 3px;
      flex-wrap: wrap;
    }
    .segmented button {
      min-height: 28px;
      padding: 4px 8px;
      font-size: 12px;
      background: white;
    }
    .segmented button.selected {
      border-color: var(--brand);
      color: var(--brand);
      background: rgba(15,118,110,.08);
    }
    .media-role-badge, .media-zoom-hint {
      position: absolute;
      background: rgba(31,35,40,.82);
      color: white;
      border-radius: 999px;
      font-size: 11px;
      padding: 3px 7px;
    }
    .media-role-badge { left: 8px; top: 8px; }
    .media-zoom-hint { right: 8px; bottom: 8px; opacity: 0; transition: opacity .16s ease; }
    .media-card-preview:hover .media-zoom-hint { opacity: 1; }
    .media-card-body {
      display: grid;
      gap: 7px;
      padding: 9px;
    }
    .media-card-title {
      font-weight: 720;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .media-card-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .media-card-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .evidence-workbench {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfaf7;
      display: grid;
      gap: 10px;
      padding: 10px;
    }
    .evidence-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .evidence-grid {
      display: grid;
      grid-template-columns: minmax(240px, .9fr) minmax(0, 1.4fr);
      gap: 10px;
      align-items: start;
    }
    .evidence-panel {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      padding: 9px;
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .transcript-slice {
      font-size: 13px;
      line-height: 1.35;
      max-height: 150px;
      overflow: auto;
      color: var(--ink);
      white-space: pre-wrap;
    }
    .evidence-candidates {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 8px;
      max-height: 410px;
      overflow: auto;
      padding-right: 3px;
    }
    .evidence-candidate {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      overflow: hidden;
      display: grid;
      min-width: 0;
    }
    .evidence-candidate.used {
      border-color: rgba(15,118,110,.5);
      background: rgba(15,118,110,.04);
    }
    .evidence-candidate img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      background: #e9e7df;
      cursor: zoom-in;
    }
    .evidence-candidate-body {
      display: grid;
      gap: 5px;
      padding: 7px;
    }
    .evidence-candidate-note {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .media-picker {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfaf7;
      display: grid;
      gap: 10px;
      padding: 10px;
    }
    .media-picker-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(252px, 1fr));
      gap: 10px;
      max-height: 620px;
      overflow: auto;
      padding-right: 4px;
    }
    .media-option {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      padding: 7px;
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .media-option.used {
      border-color: rgba(15,118,110,.5);
      background: rgba(15,118,110,.05);
    }
    .media-option img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border-radius: 5px;
      background: #e9e7df;
      display: block;
    }
    .media-option-title {
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .selected-step-media .thumb {
      width: 100%;
      height: auto;
      min-height: 132px;
      aspect-ratio: 16 / 9;
    }
    .media-strip { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; }
    .thumb {
      width: 156px;
      height: 116px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #e9e7df;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      padding: 4px;
    }
    .thumb img {
      width: 100%;
      height: 100%;
      border-radius: 5px;
    }
    .thumb.fit-contain img { object-fit: contain; }
    .thumb.fit-cover img { object-fit: cover; }
    .thumb.position-center img { object-position: center center; }
    .thumb.position-top img { object-position: center top; }
    .thumb.position-bottom img { object-position: center bottom; }
    .thumb.position-left img { object-position: left center; }
    .thumb.position-right img { object-position: right center; }
    .builder {
      display: grid;
      grid-template-columns: minmax(280px, 38%) 1fr;
      gap: 14px;
      min-height: calc(100vh - 132px);
    }
    .package-meta-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .package-list {
      display: grid;
      gap: 8px;
      max-height: 220px;
      overflow: auto;
      padding-right: 4px;
    }
    .package-editor {
      display: grid;
      grid-template-columns: minmax(340px, 46%) 1fr;
      gap: 14px;
      align-items: start;
    }
    .section-result-list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 330px);
      overflow: auto;
      padding-right: 4px;
    }
    .section-result {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .section-result-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
    }
    .section-preview-steps {
      display: grid;
      gap: 8px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .section-preview-step {
      display: grid;
      gap: 6px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
    }
    .section-preview-step .selected-step-media {
      grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
    }
    .package-outline {
      display: grid;
      gap: 8px;
    }
    .package-editor .drop-list {
      max-height: calc(100vh - 430px);
      overflow: auto;
      padding-right: 4px;
    }
    .drop-list { display: grid; gap: 8px; }
    .drop-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      padding: 10px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .package-preview {
      display: grid;
      gap: 14px;
      max-width: 980px;
      margin: 0 auto;
    }
    .preview-sheet {
      background: white;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .preview-heading {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .preview-section {
      border-top: 2px solid var(--ink);
      padding-top: 12px;
      display: grid;
      gap: 10px;
    }
    .preview-step {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      display: grid;
      gap: 8px;
      break-inside: avoid;
    }
    .preview-step-text {
      font-size: 17px;
      line-height: 1.28;
      font-weight: 650;
    }
    .media-reference-list {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .print-preview {
      display: grid;
      gap: 18px;
      max-width: 11in;
      margin: 0 auto;
    }
    .print-page {
      width: 11in;
      min-height: 8.5in;
      background: white;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: .34in;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
      break-after: page;
      page-break-after: always;
    }
    .print-page:last-child { break-after: auto; page-break-after: auto; }
    .print-page-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    .print-page-title {
      font-size: 20px;
      font-weight: 760;
      line-height: 1.1;
      margin-top: 2px;
    }
    .print-page-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 11px;
    }
    .print-steps {
      display: grid;
      gap: 10px;
      min-height: 0;
      align-items: start;
    }
    .print-steps.steps-1 { grid-template-columns: minmax(0, .68fr); }
    .print-steps.steps-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .print-steps.steps-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .print-step-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      display: grid;
      grid-template-rows: auto auto auto auto;
      gap: 6px;
      min-width: 0;
      min-height: 0;
      align-self: start;
      break-inside: avoid;
      break-inside: avoid-page;
      page-break-inside: avoid;
      -webkit-column-break-inside: avoid;
    }
    .print-step-kicker {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: flex-start;
      color: var(--muted);
      font-size: 11px;
    }
    .print-step-text {
      font-size: 16px;
      line-height: 1.22;
      font-weight: 700;
    }
    .print-media {
      display: grid;
      grid-template-columns: 2.1fr 1fr;
      gap: 7px;
      min-height: 150px;
      align-items: stretch;
      break-inside: avoid;
      page-break-inside: avoid;
      margin: 0;
    }
    .steps-2 .print-media { min-height: 225px; }
    .steps-1 .print-media { min-height: 360px; }
    .print-media.single { grid-template-columns: 1fr; }
    .print-media-slot {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #eeece5;
      min-height: 0;
      overflow: hidden;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      position: relative;
    }
    .print-media-slot img {
      width: 100%;
      height: 100%;
      display: block;
    }
    .print-media-slot.fit-contain img { object-fit: contain; }
    .print-media-slot.fit-cover img { object-fit: cover; }
    .print-media-slot.position-center img { object-position: center center; }
    .print-media-slot.position-top img { object-position: center top; }
    .print-media-slot.position-bottom img { object-position: center bottom; }
    .print-media-slot.position-left img { object-position: left center; }
    .print-media-slot.position-right img { object-position: right center; }
    .print-media-fallback { display: none; padding: 12px; }
    .print-media-slot.missing .print-media-fallback { display: grid; }
    .print-callout-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      align-self: start;
      break-inside: avoid;
      page-break-inside: avoid;
    }
    .print-callout {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 7px;
      min-width: 0;
      background: #fbfaf7;
    }
    .print-callout h4 {
      margin: 0 0 3px;
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .print-callout div {
      font-size: 11px;
      line-height: 1.22;
      overflow-wrap: anywhere;
    }
    .print-pill-list {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
    }
    .print-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 6px;
      background: white;
      color: var(--ink);
      font-size: 10px;
      line-height: 1.2;
    }
    .print-pill.muted {
      color: var(--muted);
      background: #f7f5ef;
    }
    .print-pill.warn {
      color: var(--warn);
      border-color: rgba(161,92,7,.32);
      background: rgba(161,92,7,.06);
    }
    .print-callout.safety { border-color: rgba(161,92,7,.35); background: rgba(161,92,7,.06); }
    .print-callout.quality { border-color: rgba(22,112,58,.28); background: rgba(22,112,58,.06); }
    .print-callout.torque-required { border-color: rgba(161,92,7,.35); }
    .print-muted { color: var(--muted); }
    .print-page-foot {
      border-top: 1px solid var(--line);
      padding-top: 7px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
    }
    .operator-shell {
      min-height: calc(100vh - 116px);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 12px;
    }
    .operator-stage {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr);
      gap: 14px;
      min-height: 0;
    }
    .operator-media {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      background: #e9e7df;
      min-height: 420px;
      display: grid;
      place-items: center;
    }
    .operator-media img {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 230px);
      object-fit: contain;
      background: #111;
      display: block;
    }
    .operator-step-panel {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: white;
      padding: 16px;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
    }
    .operator-step-text {
      font-size: 28px;
      line-height: 1.18;
      font-weight: 760;
    }
    .operator-callouts {
      display: grid;
      gap: 8px;
      overflow: auto;
    }
    .operator-callout {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfaf7;
      padding: 9px;
      display: grid;
      gap: 5px;
    }
    .operator-callout.safety { border-color: rgba(161,92,7,.35); background: rgba(161,92,7,.07); }
    .operator-callout.quality { border-color: rgba(22,112,58,.28); background: rgba(22,112,58,.06); }
    .operator-nav {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr) 150px;
      align-items: center;
      gap: 10px;
    }
    .operator-progress {
      text-align: center;
      color: var(--muted);
      font-size: 13px;
    }
    .empty {
      border: 1px dashed #bcb6aa;
      border-radius: var(--radius);
      min-height: 180px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      padding: 18px;
      background: #fbfaf7;
    }
    .empty.compact {
      min-height: 0;
      padding: 10px;
      font-size: 13px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .field { display: grid; gap: 5px; }
    .field label { color: var(--muted); font-size: 12px; font-weight: 700; }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(31,35,40,.35);
      z-index: 20;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .modal-backdrop.open { display: flex; }
    .modal {
      width: min(760px, 100%);
      max-height: 90vh;
      overflow: auto;
      background: white;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      box-shadow: 0 24px 64px rgba(0,0,0,.22);
    }
    .modal-head, .modal-foot {
      padding: 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .modal-foot { border-top: 1px solid var(--line); border-bottom: 0; justify-content: flex-end; }
    .modal-body { padding: 14px; }
    .image-zoom-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, .82);
      z-index: 40;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 22px;
      cursor: zoom-out;
    }
    .image-zoom-backdrop.open { display: flex; }
    .image-zoom-panel {
      max-width: min(1100px, 96vw);
      max-height: 92vh;
      display: grid;
      gap: 10px;
      color: white;
      text-align: center;
    }
    .image-zoom-panel img {
      max-width: 100%;
      max-height: 84vh;
      object-fit: contain;
      border-radius: 7px;
      background: #111;
    }
    .file-box {
      border: 1px dashed #bcb6aa;
      border-radius: var(--radius);
      background: #fbfaf7;
      padding: 14px;
      display: grid;
      gap: 8px;
    }
    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      background: #1f2328;
      color: white;
      padding: 10px 12px;
      border-radius: 7px;
      display: none;
      z-index: 30;
      box-shadow: var(--shadow);
    }
    .toast.show { display: block; }
    @media (max-width: 980px) {
      .app, .app.nav-closed { grid-template-columns: 1fr; }
      .nav { position: sticky; top: 0; z-index: 4; border-right: 0; border-bottom: 1px solid var(--line); flex-direction: row; align-items: center; overflow-x: auto; }
      .nav-items { display: flex; }
      .brand-text { display: inline; }
      .grid-2, .builder, .stage-review-grid, .section-workspace, .package-editor { grid-template-columns: 1fr; }
      .evidence-grid, .operator-stage { grid-template-columns: 1fr; }
      .operator-media { min-height: 260px; }
      .operator-step-text { font-size: 22px; }
      .package-meta-grid { grid-template-columns: 1fr; }
      .form-grid { grid-template-columns: 1fr; }
      .topbar { position: static; }
    }
    @media print {
      @page { size: letter landscape; margin: 0; }
      body { background: white; }
      .nav, .topbar, .toolbar, .top-actions, button { display: none !important; }
      .app, .app.nav-closed { display: block; }
      .content { padding: 0; }
      .preview-sheet { border: 0; box-shadow: none; padding: 0; }
      .print-preview { max-width: none; gap: 0; margin: 0; }
      .print-page {
        width: 11in;
        height: 8.5in;
        min-height: 8.5in;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        padding: .34in;
      }
      .print-page-title { font-size: 19px; }
      .print-step-text { font-size: 15px; }
      .print-steps,
      .print-step-card,
      .print-media,
      .print-media-slot,
      .print-callout-grid,
      .print-callout {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
        -webkit-column-break-inside: avoid !important;
      }
      .print-step-card {
        display: inline-grid;
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app" id="appShell">
    <aside class="nav">
      <div class="brand">
        <div class="brand-mark">A</div>
        <div class="brand-text">Assembly Assistant</div>
      </div>
      <div class="nav-items">
        <button data-page="jobs" onclick="go('jobs')"><span>▦</span><span class="nav-label">Jobs</span></button>
        <button data-page="sections" onclick="go('sections')"><span>▣</span><span class="nav-label">Sections</span></button>
        <button data-page="package" onclick="go('package')"><span>↕</span><span class="nav-label">Package</span></button>
        <button data-page="settings" onclick="go('settings')"><span>⚙</span><span class="nav-label">Settings</span></button>
      </div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="toolbar-left">
          <button class="icon" onclick="toggleNav()" title="Toggle navigation">☰</button>
          <h1 id="pageTitle">Jobs</h1>
        </div>
        <div class="top-actions" id="topActions"></div>
      </div>
      <div class="content" id="root"></div>
    </main>
  </div>

  <div class="modal-backdrop" id="jobModal">
    <div class="modal">
      <div class="modal-head">
        <strong>Start Job</strong>
        <button class="icon" onclick="closeJobModal()">×</button>
      </div>
      <div class="modal-body">
        <div class="form-grid">
          <div class="field"><label>Job name</label><input id="newName" value="Lego SF-24 cockpit module"></div>
          <div class="field"><label>Product</label><input id="newProduct" value="F1 SF-24"></div>
          <div class="field"><label>Work area</label><input id="newArea" value="Prototype bench"></div>
          <div class="field"><label>Station</label><input id="newStation" value="Station B"></div>
          <div class="field"><label>Expected steps</label><input id="newExpectedSteps" type="number" min="1" max="200" placeholder="Optional"></div>
          <div class="field"><label>Frame focus</label><select id="newFocusArea"><option value="center">Center work area</option><option value="full">Full frame</option></select></div>
          <div class="field"><label>Processing</label><select id="newProfile"><option>Standard</option><option>Fast</option><option>Deep</option></select></div>
        </div>
        <div style="height:12px"></div>
        <div class="file-box">
          <strong>Files</strong>
          <input id="newFiles" type="file" multiple>
          <div class="subtle">Video, audio, images, PDFs, and scans can be added to the processing set.</div>
        </div>
      </div>
      <div class="modal-foot">
        <button onclick="closeJobModal()">Cancel</button>
        <button class="primary" onclick="createJob()">Create and Go</button>
      </div>
    </div>
  </div>
  <div class="image-zoom-backdrop" id="imageZoom" onclick="closeImageZoom()">
    <div class="image-zoom-panel">
      <img id="imageZoomImg" alt="">
      <div id="imageZoomLabel" class="subtle"></div>
    </div>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const state = {
      jobs: [],
      sections: [],
      packages: [],
      transcripts: [],
      settings: {
        has_openai_api_key: false,
        transcription_model: 'whisper-1',
        transcription_enabled: true,
        default_processing: 'Standard',
        storage_root: 'storage'
      },
      currentJob: null,
      currentPackage: null,
      currentPage: 'jobs',
      showArchived: false,
      search: '',
      sectionSearch: '',
      packageListSearch: '',
      packageSearch: '',
      packageDraft: [],
      packageDraftId: null,
      packageDraftMeta: {
        title: '',
        product: '',
        work_area: '',
        station: '',
        status: 'Draft'
      },
      currentSection: null,
      sectionDraftId: null,
      sectionDraft: null,
      sectionMediaOptions: {},
      sectionMediaPicker: {
        open: false,
        replaceIndex: null,
        mode: 'append',
        filter: 'near',
        search: ''
      },
      sectionStepIndexes: {},
      sectionQaFocus: null,
      currentPackageData: null,
      advancedOpenJobs: {},
      expandedPackageSections: {},
      scrollPositions: {},
      selectedChoices: {},
      questionDrafts: {},
      questionAttachmentNotes: {},
      questionFiles: {},
      questionFileObjects: {},
      selectedStages: {},
      draftSlideIndexes: {},
      operatorStepIndexes: {},
      renderFingerprint: '',
      lastScrollAt: 0,
      poller: null
    };

    const stageKeys = ['upload_check', 'transcription', 'drawings', 'images', 'video', 'questions', 'draft_sections'];
    const stageInitials = {upload_check:'U', transcription:'T', drawings:'D', images:'I', video:'V', questions:'Q', draft_sections:'S'};

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
    }

    function mediaUrl(path) {
      return `/media?path=${encodeURIComponent(path || '')}`;
    }

    function defaultMediaSetting() {
      return {fit: 'contain', aspect: 'auto', crop_position: 'center'};
    }

    function normalizeMediaSetting(setting = {}) {
      const fit = ['contain', 'cover', 'fill'].includes(setting.fit) ? setting.fit : 'contain';
      const aspect = ['auto', 'portrait', 'landscape', 'square'].includes(setting.aspect) ? setting.aspect : 'auto';
      const cropPosition = ['center', 'top', 'bottom', 'left', 'right'].includes(setting.crop_position || setting.position)
        ? (setting.crop_position || setting.position)
        : 'center';
      return {
        fit: fit === 'fill' ? 'cover' : fit,
        aspect,
        crop_position: cropPosition
      };
    }

    function normalizeStepMediaSettings(step) {
      const media = normalizeMediaList(step?.media || []);
      const raw = step?.media_settings && typeof step.media_settings === 'object' && !Array.isArray(step.media_settings)
        ? step.media_settings
        : {};
      const allowed = new Set(media);
      const settings = {};
      Object.entries(raw).forEach(([path, setting]) => {
        if (!allowed.has(path) || !setting || typeof setting !== 'object') return;
        settings[path] = normalizeMediaSetting(setting);
      });
      return settings;
    }

    function mediaSettingFor(step, path) {
      const settings = normalizeStepMediaSettings(step || {});
      return {...defaultMediaSetting(), ...(settings[String(path || '')] || {})};
    }

    function mediaClassNames(setting) {
      const clean = normalizeMediaSetting(setting || {});
      return `fit-${clean.fit} aspect-${clean.aspect} position-${clean.crop_position}`;
    }

    function openImageZoom(path, label = '') {
      if (!path) return;
      const zoom = document.getElementById('imageZoom');
      const img = document.getElementById('imageZoomImg');
      const title = document.getElementById('imageZoomLabel');
      img.src = mediaUrl(path);
      img.alt = label || path;
      title.textContent = label || path;
      zoom.classList.add('open');
    }

    function closeImageZoom() {
      const zoom = document.getElementById('imageZoom');
      const img = document.getElementById('imageZoomImg');
      zoom.classList.remove('open');
      img.src = '';
    }

    function isBrowserPreviewImage(file) {
      const type = String(file?.type || '').toLowerCase();
      const name = String(file?.name || file?.stored_path || '').toLowerCase();
      return type.startsWith('image/') && /\.(jpe?g|png|webp|gif)$/i.test(name);
    }

    function jsSingleQuoted(value) {
      return String(value ?? '')
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/\r/g, '\\r')
        .replace(/\n/g, '\\n');
    }

    function uploadIdentity(file) {
      return file?.sha256 || file?.stored_path || `${file?.name || ''}:${file?.bytes || file?.size || ''}`;
    }

    function compactArtifacts(artifacts) {
      const transcribedAudioSources = new Set(
        (artifacts || [])
          .filter(file => file.kind === 'audio' && file.processing_status === 'transcribed')
          .map(file => file.source_file)
      );
      const seen = new Set();
      return (artifacts || []).filter(file => {
        const key = `${file.kind || ''}:${file.stored_path || file.name || ''}`;
        if (seen.has(key)) return false;
        seen.add(key);
        if (file.kind === 'audio' && String(file.processing_status || '').includes('failed') && transcribedAudioSources.has(file.source_file)) {
          return false;
        }
        return true;
      });
    }

    async function api(path, options = {}) {
      const headers = options.body instanceof FormData ? {} : {'Content-Type': 'application/json'};
      const res = await fetch(path, {
        headers,
        ...options
      });
      if (!res.ok) {
        const text = await res.text();
        try {
          const parsed = JSON.parse(text);
          throw new Error(parsed.error || text);
        } catch (err) {
          if (err instanceof SyntaxError) throw new Error(text);
          throw err;
        }
      }
      return res.json();
    }

    function toast(message) {
      const el = document.getElementById('toast');
      el.textContent = message;
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 2200);
    }

    function setTitle(title) {
      document.getElementById('pageTitle').textContent = title;
      document.querySelectorAll('.nav button').forEach(btn => {
        const activePage = state.currentPage === 'job' ? 'jobs' : state.currentPage === 'section' ? 'sections' : ['package_edit', 'package_view', 'package_print', 'operator_view'].includes(state.currentPage) ? 'package' : state.currentPage;
        btn.classList.toggle('active', btn.dataset.page === activePage);
      });
    }

    function setTopActions(html) {
      document.getElementById('topActions').innerHTML = html || '';
    }

    function invalidateRender() {
      state.renderFingerprint = '';
    }

    function go(page, id = null) {
      invalidateRender();
      state.currentPage = page;
      if (page === 'job' && id) {
        location.hash = `job/${id}`;
      } else if (page === 'section' && id) {
        location.hash = `section/${id}`;
      } else if (page === 'package_view' && id) {
        location.hash = `package/${id}`;
      } else if (page === 'package_edit' && id) {
        location.hash = `package/${id}/edit`;
      } else if (page === 'package_print' && id) {
        location.hash = `package/${id}/print`;
      } else if (page === 'operator_view' && id) {
        location.hash = `package/${id}/operator`;
      } else {
        location.hash = page;
      }
      render();
    }

    function readHash() {
      const pathRoute = location.pathname.replace(/^\/+|\/+$/g, '');
      const hash = location.hash.replace(/^#/, '') || (pathRoute.startsWith('package/') ? pathRoute : pathRoute);
      if (!hash || hash === 'jobs') {
        state.currentPage = 'jobs';
        state.currentJob = null;
        state.currentSection = null;
        state.currentPackage = null;
      } else if (hash.startsWith('job/')) {
        state.currentPage = 'job';
        state.currentJob = hash.split('/')[1];
        state.currentSection = null;
        state.currentPackage = null;
      } else if (hash.startsWith('section/')) {
        state.currentPage = 'section';
        state.currentSection = hash.split('/')[1];
        state.currentJob = null;
        state.currentPackage = null;
      } else if (hash.startsWith('package/')) {
        const parts = hash.split('/');
        state.currentPage = parts[2] === 'print' ? 'package_print' : parts[2] === 'edit' ? 'package_edit' : parts[2] === 'operator' ? 'operator_view' : 'package_view';
        state.currentPackage = parts[1];
        state.currentJob = null;
        state.currentSection = null;
      } else {
        state.currentPage = hash;
        state.currentJob = null;
        state.currentSection = null;
        state.currentPackage = null;
      }
    }

    function toggleNav() {
      document.getElementById('appShell').classList.toggle('nav-closed');
    }

    function isEditingField() {
      const el = document.activeElement;
      if (!el) return false;
      return ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName);
    }

    function isActivelyScrolling() {
      return Date.now() - (state.lastScrollAt || 0) < 900;
    }

    function scrollKey(label) {
      const id = state.currentJob || state.currentSection || state.currentPage;
      return `${state.currentPage}:${id}:${label}`;
    }

    function elementScrollLabel(el, index) {
      if (el.dataset.scrollKey) return el.dataset.scrollKey;
      if (el.id) return `id:${el.id}`;
      if (el.className && typeof el.className === 'string') return `class:${el.className.split(/\s+/).filter(Boolean).join('.')}:${index}`;
      return `node:${el.tagName}:${index}`;
    }

    function scrollableElements() {
      return Array.from(document.querySelectorAll('textarea, .table-wrap, .log, [data-scroll-key], .stage-detail-body, .frame-grid, .media-strip, .panel-body, .details-body')).filter((el, index, arr) => {
        if (arr.indexOf(el) !== index) return false;
        return el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1 || el.dataset.scrollKey;
      });
    }

    function saveScrollPositions() {
      scrollableElements().forEach((el, index) => {
        state.scrollPositions[scrollKey(elementScrollLabel(el, index))] = {
          left: el.scrollLeft,
          top: el.scrollTop
        };
      });
      const page = document.scrollingElement || document.documentElement;
      if (page) state.scrollPositions[scrollKey('page')] = {left: page.scrollLeft, top: page.scrollTop};
    }

    function restoreScrollPositions() {
      const applyRestore = () => {
        autosizeTextareas();
        scrollableElements().forEach((el, index) => {
          const pos = state.scrollPositions[scrollKey(elementScrollLabel(el, index))];
          if (pos) {
            el.scrollLeft = pos.left;
            el.scrollTop = pos.top;
          }
        });
        const page = document.scrollingElement || document.documentElement;
        const pagePos = state.scrollPositions[scrollKey('page')];
        if (page && pagePos) {
          page.scrollLeft = pagePos.left;
          page.scrollTop = pagePos.top;
        }
      };
      requestAnimationFrame(() => {
        applyRestore();
        requestAnimationFrame(applyRestore);
        setTimeout(applyRestore, 80);
      });
    }

    async function loadData() {
      const [jobs, sections, packagesData, settings] = await Promise.all([api('/api/jobs'), api('/api/sections'), api('/api/packages'), api('/api/settings')]);
      state.jobs = jobs.jobs;
      state.sections = sections.sections;
      state.packages = packagesData.packages;
      state.settings = settings.settings;
      if (state.currentJob) {
        const [detail, transcriptData] = await Promise.all([
          api(`/api/jobs/${state.currentJob}`),
          api(`/api/jobs/${state.currentJob}/transcripts`)
        ]);
        state.currentJobData = detail.job;
        state.transcripts = transcriptData.transcripts;
      } else {
        state.transcripts = [];
      }
      if (state.currentPackage) {
        const detail = await api(`/api/packages/${state.currentPackage}`);
        state.currentPackageData = detail.package;
      } else {
        state.currentPackageData = null;
      }
      if (state.currentSection) {
        try {
          const mediaData = await api(`/api/sections/${state.currentSection}/media-options`);
          state.sectionMediaOptions[state.currentSection] = mediaData.media || [];
        } catch (err) {
          state.sectionMediaOptions[state.currentSection] = [];
        }
      }
    }

    function startPolling() {
      if (state.poller) clearInterval(state.poller);
      state.poller = setInterval(async () => {
        if (isEditingField() || isActivelyScrolling()) return;
        saveScrollPositions();
        await loadData();
        if (isEditingField() || isActivelyScrolling()) return;
        if (state.currentPage === 'jobs' || state.currentPage === 'job' || state.currentPage === 'sections' || state.currentPage === 'package' || state.currentPage === 'package_edit' || state.currentPage === 'package_view' || state.currentPage === 'package_print' || state.currentPage === 'operator_view') {
          render(false);
        }
      }, 2500);
    }

    function statusChip(status) {
      let cls = 'chip';
      if (['Draft Sections Ready', 'Complete', 'Ready'].includes(status)) cls += ' good';
      if (['Questions Pending', 'Processing', 'Drafting Sections'].includes(status)) cls += ' warn';
      if (['Failed', 'Cancelled', 'Needs Attention'].includes(status)) cls += ' bad';
      return `<span class="${cls}">${esc(status)}</span>`;
    }

    function dot(stage) {
      return `<span class="dot ${esc(stage.status)}" title="${esc(stage.label)}: ${esc(stage.status)}">${stageInitials[stage.key]}</span>`;
    }

    function stageDots(job) {
      return `<div class="stage-dots">${stageKeys.map(k => dot(job.stages[k])).join('')}</div>`;
    }

    function stageCell(job, key) {
      const s = job.stages[key];
      return `<div class="chip-row">${dot(s)}<span class="subtle">${s.percent}%</span></div>`;
    }

    function autosizeTextareas() {
      document.querySelectorAll('textarea[data-autogrow]').forEach(el => {
        const top = el.scrollTop;
        el.style.height = 'auto';
        const maxHeight = Number(el.dataset.maxHeight || 560);
        el.style.height = `${Math.min(el.scrollHeight + 2, maxHeight)}px`;
        el.scrollTop = top;
      });
    }

    function pageFingerprint() {
      if (state.currentPage === 'job' && state.currentJobData) {
        const job = state.currentJobData;
        return JSON.stringify({
          page: state.currentPage,
          job: job.id,
          status: job.status,
          progress: job.progress,
          selectedStage: state.selectedStages[job.id] || activeStageKey(job),
          advanced: !!state.advancedOpenJobs[job.id],
          stageNotes: stageKeys.map(key => [key, job.stages[key]?.status, job.stages[key]?.percent, job.stages[key]?.note]),
          questions: (job.questions || []).map(q => [q.id, q.status, q.answer, q.attachment_note, (q.attachment_files || []).length]),
          questionReview: job.question_review || {},
          drafts: (job.draft_sections || []).map(section => [section.id, section.title, (section.steps || []).length]),
          draftSlide: state.draftSlideIndexes[job.id] || 0,
          files: (job.files || []).map(file => [file.sha256, file.name, file.processing_status, file.video_status]),
          artifacts: (job.artifacts || []).length,
          transcripts: (state.transcripts || []).map(item => [item.source_file, item.status, item.error, (item.text || '').length]),
          logs: [job.logs?.length || 0, job.logs?.[job.logs.length - 1]?.message || ''],
          needsProcessing: !!job.needs_processing,
          watchdog: job.watchdog || {},
          draftValidation: job.draft_validation || {},
          draftAudit: job.draft_audit || {},
          processingCache: job.processing_cache || {},
          processingHealth: job.processing_health || {}
        });
      }
      if (state.currentPage === 'jobs') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.search,
          showArchived: state.showArchived,
          jobs: state.jobs.map(job => [job.id, job.status, job.progress, job.last_activity, job.file_count, job.questions_pending, job.draft_count, job.needs_processing, job.expected_step_count, job.focus_area])
        });
      }
      if (state.currentPage === 'sections') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.sectionSearch,
          sections: state.sections.map(section => [section.id, section.title, section.status, (section.steps || []).length])
        });
      }
      if (state.currentPage === 'section') {
        const draft = state.sectionDraft;
        return JSON.stringify({
          page: state.currentPage,
          section: state.currentSection,
          draftId: state.sectionDraftId,
          selectedStep: state.sectionStepIndexes[state.currentSection] || 0,
          qaFocus: state.sectionQaFocus,
          draft: draft ? [draft.id, draft.title, draft.status, draft.product, draft.work_area, draft.station, (draft.tags || []).join('|'), (draft.steps || []).map(step => [
            step.step_id,
            (step.media || []).join('|'),
            step.text,
            step.tools,
            step.torque,
            (step.parts || []).join('|'),
            step.quality,
            step.safety,
            step.media_match_confidence,
            !!step.media_fallback,
            !!step.merge_candidate,
            JSON.stringify(step.media_settings || {}),
            JSON.stringify(step.qa_resolutions || {}),
            JSON.stringify(step.evidence_review || {}),
            step.start_time,
            step.end_time,
            step.target_time
          ]).join('||')] : null,
          mediaOptions: (state.sectionMediaOptions[state.currentSection] || []).map(item => [item.path, item.score, item.is_currently_used]),
          mediaPicker: state.sectionMediaPicker
        });
      }
      if (state.currentPage === 'package') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.packageListSearch,
          packages: state.packages.map(packageItem => [packageItem.id, packageItem.title, packageItem.product, packageItem.work_area, packageItem.station, packageItem.status, packageItem.section_count, packageItem.step_count, packageItem.updated_at])
        });
      }
      if (state.currentPage === 'package_edit') {
        const packageData = state.currentPackageData;
        return JSON.stringify({
          page: state.currentPage,
          packageId: state.currentPackage,
          packageData: packageData ? [packageData.id, packageData.title, packageData.status, packageData.updated_at] : null,
          search: state.packageSearch,
          draftId: state.packageDraftId,
          draft: state.packageDraft,
          meta: state.packageDraftMeta,
          expanded: state.expandedPackageSections,
          sections: state.sections.map(section => [section.id, section.title, section.status, (section.steps || []).length]),
        });
      }
      if (state.currentPage === 'package_view' || state.currentPage === 'package_print' || state.currentPage === 'operator_view') {
        const packageData = state.currentPackageData;
        return JSON.stringify({
          page: state.currentPage,
          packageId: state.currentPackage,
          operatorStep: state.operatorStepIndexes[state.currentPackage] || 0,
          packageData: packageData ? [packageData.id, packageData.title, packageData.status, packageData.updated_at, (packageData.sections || []).map(section => [section.id, section.title, (section.steps || []).length])] : null
        });
      }
      return JSON.stringify({page: state.currentPage});
    }

    function renderJobs() {
      setTitle('Jobs');
      setTopActions(`<button class="primary" onclick="openJobModal()">Start Job</button>`);
      const rows = state.jobs
        .filter(j => state.showArchived || j.active)
        .filter(j => {
          const hay = `${j.job_no} ${j.name} ${j.product} ${j.work_area} ${j.station}`.toLowerCase();
          return hay.includes(state.search.toLowerCase());
        });
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left">
            <input class="search" placeholder="Search jobs" value="${esc(state.search)}" oninput="state.search=this.value" onblur="renderJobs()">
            <button class="ghost" onclick="state.showArchived=!state.showArchived; renderJobs()">${state.showArchived ? 'Hide Complete' : 'Show Complete'}</button>
          </div>
          <div class="toolbar-right"><span class="subtle">${rows.length} shown</span></div>
        </div>
        <div class="table-wrap" data-scroll-key="jobs-table">
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>Context</th>
                <th>Files</th>
                <th>Uploads</th>
                <th>Transcript</th>
                <th>Drawings</th>
                <th>Images</th>
                <th>Video</th>
                <th>Questions</th>
                <th>Drafts</th>
                <th>Overall</th>
                <th>Flags</th>
                <th>Updated</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(job => `
                <tr>
                  <td class="main-cell">
                    <div class="strong">${esc(job.job_no)} · ${esc(job.name)}</div>
                    <div class="subtle">${statusChip(job.status)} <span class="chip">${esc(job.profile)}</span>${job.needs_processing ? ' <span class="chip warn">Needs processing</span>' : ''}</div>
                  </td>
                  <td>
                    <div>${esc(job.product)}</div>
                    <div class="subtle">${esc(job.work_area)} · ${esc(job.station)}${job.expected_step_count ? ' · ~' + esc(job.expected_step_count) + ' steps' : ''} · ${job.focus_area === 'full' ? 'full frame' : 'center focus'}</div>
                  </td>
                  <td>${job.file_count}</td>
                  <td>${stageCell(job, 'upload_check')}</td>
                  <td>${stageCell(job, 'transcription')}</td>
                  <td>${stageCell(job, 'drawings')}</td>
                  <td>${stageCell(job, 'images')}</td>
                  <td>${stageCell(job, 'video')}</td>
                  <td>${job.questions_pending ? `<span class="chip warn">${job.questions_pending} pending</span>` : stageCell(job, 'questions')}</td>
                  <td>${job.draft_count ? `<span class="chip brand">${job.draft_count} ready</span>` : stageCell(job, 'draft_sections')}</td>
                  <td style="min-width:150px"><div class="bar"><span style="width:${job.progress}%"></span></div><div class="subtle">${job.progress}%</div></td>
                  <td><div class="chip-row">${job.flags.length ? job.flags.map(f => `<span class="chip warn">${esc(f)}</span>`).join('') : '<span class="chip good">None</span>'}</div></td>
                  <td><span class="subtle">${esc(job.last_activity)}</span></td>
                  <td><button onclick="go('job','${job.id}')">Open</button></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderStageList(job) {
      return `<div class="stage-list">
        ${stageKeys.map(k => {
          const s = job.stages[k];
          const selected = (state.selectedStages[job.id] || activeStageKey(job)) === k;
          return `<div class="stage-item selectable ${selected ? 'active' : ''}" onclick="selectStage('${job.id}', '${k}')">
            ${dot(s)}
            <div>
              <div class="strong">${esc(s.label)}</div>
              <div class="subtle">${esc(s.note || s.status)}</div>
            </div>
            <div class="subtle">${s.percent}%</div>
          </div>`;
        }).join('')}
      </div>`;
    }

    function activeStageKey(job) {
      return stageKeys.find(k => job.stages[k]?.status === 'running')
        || stageKeys.find(k => job.stages[k]?.status === 'blocked' || job.stages[k]?.status === 'failed')
        || (job.draft_sections?.length ? 'draft_sections' : null)
        || (job.questions?.length ? 'questions' : null)
        || 'transcription';
    }

    function selectStage(jobId, key) {
      invalidateRender();
      state.selectedStages[jobId] = key;
      render(false);
    }

    function stageArtifacts(job, kind) {
      return compactArtifacts(job.artifacts || []).filter(file => file.kind === kind);
    }

    function renderFrameGrid(job) {
      const frames = stageArtifacts(job, 'frame');
      if (!frames.length) return `<div class="empty">No frames extracted yet.</div>`;
      return `<div class="frame-grid" data-scroll-key="stage-video-frames">
        ${frames.map(file => {
          const analysis = file.analysis || {};
          const score = analysis.instruction_value ?? file.instruction_value;
          const pointing = analysis.pointing || file.pointing;
          const summary = analysis.visual_summary || analysis.reason || file.processing_status || '';
          const tags = [...new Set([...(file.candidate_tags || []), ...(analysis.tags || [])])].slice(0, 5);
          return `
          <div class="frame-card clickable" onclick="openImageZoom('${jsSingleQuoted(file.stored_path)}', '${jsSingleQuoted(file.timestamp || file.name)}')">
            <img src="${mediaUrl(file.stored_path)}" alt="${esc(file.name)}" loading="lazy">
            <div class="subtle">${esc(file.timestamp || '')} · ${esc(file.name)}</div>
            <div class="chip-row" style="padding:0 6px 6px">
              ${score !== undefined ? `<span class="chip ${score >= 70 ? 'good' : score < 35 ? 'bad' : ''}">Score ${esc(score)}</span>` : ''}
              ${file.local_priority ? `<span class="chip">Local ${esc(file.local_priority)}</span>` : ''}
              ${pointing ? '<span class="chip warn">Pointing</span>' : ''}
              ${tags.map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
            </div>
            ${summary ? `<div class="subtle" style="padding:0 6px 8px">${esc(summary)}</div>` : ''}
          </div>
        `;
        }).join('')}
      </div>`;
    }

    function renderUploadedImages(job) {
      const images = (job.files || []).filter(file => file.kind === 'image' && !file.removed);
      if (!images.length) {
        return `<div class="empty">No uploaded images yet.</div>`;
      }
      return `<div class="frame-grid" data-scroll-key="stage-uploaded-images">
        ${images.map(file => `
          <div class="frame-card ${isBrowserPreviewImage(file) && file.stored_path ? 'clickable' : ''}" ${isBrowserPreviewImage(file) && file.stored_path ? `onclick="openImageZoom('${jsSingleQuoted(file.stored_path)}', '${jsSingleQuoted(file.name)}')"` : ''}>
            ${isBrowserPreviewImage(file) && file.stored_path
              ? `<img src="${mediaUrl(file.stored_path)}" alt="${esc(file.name)}" loading="lazy">`
              : `<div class="source-tile">${esc((file.name || '').split('.').pop() || 'image')}<br><span class="subtle">Preview unavailable</span></div>`}
            <div class="subtle">${esc(file.name)}</div>
            <div class="chip-row" style="padding:0 6px 8px">
              <span class="chip brand">image</span>
              <span class="chip">${esc(file.size || '')}</span>
              <span class="chip">${esc(file.processing_status || 'stored')}</span>
            </div>
          </div>
        `).join('')}
      </div>`;
    }

    function renderStageDetails(job) {
      const key = state.selectedStages[job.id] || activeStageKey(job);
      const stage = job.stages[key];
      const audio = stageArtifacts(job, 'audio');
      const frames = stageArtifacts(job, 'frame');
      const cueRows = (job.cues || []).map(cue => `<span class="chip">${esc(cue.label)}: ${cue.count}</span>`).join('');
      let body = '';
      if (key === 'upload_check') {
        body = `<div class="stage-detail-list">
          <div class="split-actions" style="justify-content:flex-start;margin-top:0">
            <button class="primary" onclick="openAddFiles('${job.id}')">Add Files</button>
            ${job.needs_processing ? '<span class="chip warn">New files need processing</span>' : '<span class="chip good">Uploads current</span>'}
          </div>
          ${renderUploadedSources(job)}
        </div>`;
      } else if (key === 'transcription') {
        body = `<div class="stage-detail-list">
          <div class="chip-row">
            <span class="chip ${stage.status === 'done' ? 'good' : stage.status === 'failed' ? 'bad' : 'warn'}">${esc(stage.status)}</span>
            <span class="chip">${esc(stage.note || '')}</span>
          </div>
          <div class="chip-row">${cueRows || '<span class="chip">No spoken cues detected yet</span>'}</div>
          <div class="file-list">${audio.map(file => `
            <div class="file-row">
              <div>
                <div class="strong">${esc(file.name)}</div>
                <div class="subtle">${esc(file.type || 'audio')} · ${esc(file.size || '')}</div>
              </div>
              <div class="chip-row"><span class="chip">${esc(file.processing_status || '')}</span></div>
            </div>`).join('') || '<div class="empty">No audio artifact yet.</div>'}</div>
          ${renderTranscripts()}
        </div>`;
      } else if (key === 'video') {
        body = renderFrameGrid(job);
      } else if (key === 'images') {
        const imageCount = (job.files || []).filter(file => file.kind === 'image' && !file.removed).length;
        body = `<div class="stage-detail-list"><span class="chip">${imageCount} uploaded images</span>${renderUploadedImages(job)}</div>`;
      } else if (key === 'drawings') {
        const drawingCount = (job.files || []).filter(file => file.kind === 'pdf' && !file.removed).length;
        body = `<div class="stage-detail-list"><span class="chip">${drawingCount} PDF drawings</span><div class="subtle">Drawing OCR and BOM extraction are still pending implementation.</div></div>`;
      } else if (key === 'questions') {
        body = renderQuestions(job);
      } else if (key === 'draft_sections') {
        body = renderDraftBrowser(job);
      }
      return `<div class="panel stage-detail-card stage-detail-${esc(key)}">
        <div class="panel-head"><h2>${esc(stage.label)} Details</h2><span class="chip">${esc(stage.status)}</span></div>
        <div class="panel-body stage-detail-body" data-scroll-key="stage-detail">${body}</div>
      </div>`;
    }

    function mediaThumb(path, step = null) {
      const value = String(path || '');
      const setting = mediaSettingFor(step || {}, value);
      if (value.match(/\\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/')) {
        return `<div class="thumb clickable ${mediaClassNames(setting)}" onclick="openImageZoom('${jsSingleQuoted(value)}', '${jsSingleQuoted(value)}')"><img src="${mediaUrl(value)}" alt="${esc(value)}" loading="lazy"></div>`;
      }
      return `<div class="thumb">${esc(value)}</div>`;
    }

    function mediaDisplayName(path) {
      const value = String(path || '');
      const name = value.split('/').filter(Boolean).pop() || value || 'Media';
      return name.replace(/^[a-f0-9]{8,}_/i, '');
    }

    function mediaCandidateFor(step, path) {
      const value = String(path || '');
      return (step.media_candidates || []).find(item => String(item.path || '') === value) || null;
    }

    function mediaSettingButton(path, key, value, label, selectedValue) {
      const selected = selectedValue === value || (key === 'fit' && value === 'cover' && selectedValue === 'fill');
      return `<button class="${selected ? 'selected' : ''}" onclick="setStepMediaSetting('${jsSingleQuoted(path)}', '${key}', '${value}')">${label}</button>`;
    }

    function renderMediaSettingControls(path, setting) {
      const clean = normalizeMediaSetting(setting);
      return `<div class="media-setting-controls">
        <div class="media-control-row">
          <span class="subtle">Shape</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'aspect', 'auto', 'Auto', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'portrait', 'Portrait', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'landscape', 'Landscape', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'square', 'Square', clean.aspect)}
          </span>
        </div>
        <div class="media-control-row">
          <span class="subtle">Fit</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'fit', 'contain', 'Full', clean.fit)}
            ${mediaSettingButton(path, 'fit', 'cover', 'Fill', clean.fit)}
          </span>
        </div>
        <div class="media-control-row">
          <span class="subtle">Crop</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'crop_position', 'center', 'Center', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'top', 'Top', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'bottom', 'Bottom', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'left', 'Left', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'right', 'Right', clean.crop_position)}
          </span>
        </div>
      </div>`;
    }

    function mediaCard(step, path, idx, mediaCount) {
      const value = String(path || '');
      const candidate = mediaCandidateFor(step, value);
      const setting = mediaSettingFor(step, value);
      const label = mediaDisplayName(value);
      const previewable = value.match(/\\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/');
      const role = idx === 0 ? 'Primary image' : `Support ${idx}`;
      const note = candidate?.summary || candidate?.reason || (idx === 0 ? step.media_fallback_reason : '');
      const confidence = idx === 0 ? step.media_match_confidence : '';
      const score = candidate?.score || (idx === 0 ? step.media_match_score : '');
      const timestamp = candidate?.timestamp_seconds ?? candidate?.timestamp ?? null;
      return `<div class="step-media-card ${idx === 0 ? 'primary' : ''}">
        ${previewable ? `
          <button class="media-card-preview" type="button" onclick="openImageZoom('${jsSingleQuoted(value)}', '${jsSingleQuoted(label)}')">
            <span class="media-display ${mediaClassNames(setting)}"><img src="${mediaUrl(value)}" alt="${esc(label)}" loading="lazy"></span>
            <span class="media-role-badge">${esc(role)}</span>
            <span class="media-zoom-hint">Zoom</span>
          </button>
        ` : `<div class="source-tile">${esc(value)}<br><span class="subtle">Preview unavailable</span></div>`}
        <div class="media-card-body">
          <div class="media-card-title" title="${esc(value)}">${esc(label)}</div>
          <div class="chip-row">
            <span class="chip ${idx === 0 ? 'good' : ''}">${esc(role)}</span>
            ${timestamp !== null && timestamp !== undefined && timestamp !== '' ? `<span class="chip">~${esc(timestamp)}s</span>` : ''}
            ${confidence ? `<span class="chip ${confidence === 'manual' || confidence === 'high' ? 'good' : confidence === 'fallback' || confidence === 'low' ? 'warn' : ''}">${esc(confidence)}</span>` : ''}
            ${score ? `<span class="chip">Score ${esc(score)}</span>` : ''}
          </div>
          ${note ? `<div class="media-card-note">${esc(note)}</div>` : ''}
          ${previewable ? renderMediaSettingControls(value, setting) : ''}
          <div class="media-card-actions">
            <button class="icon" onclick="moveStepMedia(${idx}, -1)" ${idx <= 0 ? 'disabled' : ''} title="Move left">←</button>
            <button class="icon" onclick="moveStepMedia(${idx}, 1)" ${idx >= mediaCount - 1 ? 'disabled' : ''} title="Move right">→</button>
            <button onclick="openSectionMediaPicker(${idx}, 'replace')">Replace</button>
            <button class="danger" onclick="removeStepMedia(${idx})">Remove</button>
          </div>
        </div>
      </div>`;
    }

    function renderQuestionReview(job) {
      const review = job.question_review || {};
      const checks = review.checks || [];
      if (!checks.length) {
        return `<div class="empty compact">Question checks will appear after processing reaches this stage.</div>`;
      }
      return `<div class="question-card ${review.generated_questions ? 'pending' : 'bypassed'}">
        <div class="card-head">
          <div>
            <div class="strong">Question Review</div>
            <div class="subtle">${esc(review.confidence_note || '')}</div>
          </div>
          <span class="chip ${review.generated_questions ? 'warn' : 'good'}">${esc(review.generated_questions || 0)} generated</span>
        </div>
        <div class="stage-detail-list">
          ${checks.map(check => `
            <div class="file-row">
              <div>
                <div class="strong">${esc(check.label)}</div>
                <div class="subtle">${esc(check.detail)}</div>
              </div>
              <span class="chip ${check.result === 'question' ? 'warn' : 'good'}">${esc(check.result)}</span>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    function renderQuestions(job) {
      const review = renderQuestionReview(job);
      if (!job.questions.length) return `<div class="stack">${review}</div>`;
      return `<div class="stack">
        ${review}
        ${job.questions.map(q => {
          const selected = state.selectedChoices[q.id] ?? q.answer;
          const draftAnswer = state.questionDrafts[q.id] ?? q.answer ?? '';
          const draftNote = state.questionAttachmentNotes[q.id] ?? q.attachment_note ?? '';
          const savedFiles = q.attachment_files || [];
          const draftFiles = state.questionFiles[q.id] || [];
          const shownFiles = draftFiles.length ? draftFiles : savedFiles;
          return `<div class="question-card ${esc(q.status)}">
            <div class="card-head">
              <div>
                <div class="strong">${esc(q.title)}</div>
                <div class="subtle">${esc(q.source)}</div>
              </div>
              <span class="chip ${q.status === 'pending' ? 'warn' : 'good'}">${esc(q.status)}</span>
            </div>
            <div>${esc(q.prompt)}</div>
            <div class="source-tile">${esc(q.source)}</div>
            <div class="subtle">${esc(q.why)}</div>
            <div class="choice-row">
              ${q.choices.map(choice => `<button class="${selected === choice ? 'selected' : ''}" onclick="selectQuestionChoice('${q.id}', '${esc(choice)}')">${esc(choice)}</button>`).join('')}
            </div>
            <textarea id="answer-${q.id}" placeholder="Answer or add context" oninput="state.questionDrafts['${q.id}']=this.value">${esc(draftAnswer)}</textarea>
            <input id="attach-${q.id}" placeholder="Attach note or reference file name" value="${esc(draftNote)}" oninput="state.questionAttachmentNotes['${q.id}']=this.value">
            <input type="file" multiple onchange="captureQuestionFiles('${q.id}', this.files)">
            <div class="chip-row">${shownFiles.length ? shownFiles.map(file => `<span class="chip">${esc(file.name)}</span>`).join('') : '<span class="chip">No clarification files</span>'}</div>
            <div class="split-actions">
              <button onclick="answerQuestion('${q.id}', 'bypassed')">Bypass</button>
              <button class="primary" onclick="answerQuestion('${q.id}', 'answered')">Save Answer</button>
            </div>
          </div>`;
        }).join('')}
      </div>`;
    }

    function renderUploadedSources(job) {
      const files = (job.files || []).filter(file => !file.removed);
      if (!files.length) {
        return `<div class="empty">No uploaded sources yet.</div>`;
      }
      return `<div class="file-list">
        ${files.map(file => `
          <div class="file-row">
            <div>
              <div class="strong">${esc(file.name)}</div>
              <div class="subtle">${esc(file.type || 'unknown')} · ${esc(file.size || '')}</div>
              ${file.stored_path ? `<div class="subtle">${esc(file.stored_path)}</div>` : ''}
            </div>
            <div class="chip-row">
              ${file.kind ? `<span class="chip brand">${esc(file.kind)}</span>` : ''}
              ${file.processing_status ? `<span class="chip">${esc(file.processing_status)}</span>` : ''}
              ${file.video_status ? `<span class="chip">${esc(file.video_status)}</span>` : ''}
              <span class="chip">${esc(file.source || 'seeded')}</span>
              ${file.clarification_for ? `<span class="chip warn">${esc(file.clarification_for)}</span>` : ''}
            </div>
            <div class="split-actions">
              <button class="danger" onclick="removeUpload('${job.id}', '${jsSingleQuoted(uploadIdentity(file))}')">Remove</button>
            </div>
          </div>
        `).join('')}
      </div>`;
    }

    function transcriptStatusClass(status) {
      if (status === 'complete') return 'good';
      if (status === 'transcription_failed' || status === 'read_error' || status === 'missing_file') return 'bad';
      return 'warn';
    }

    function renderTranscripts() {
      const transcripts = state.transcripts || [];
      if (!transcripts.length) {
        return `<div class="empty">Transcripts will appear after audio is extracted and sent to the transcription provider.</div>`;
      }
      const combinedText = transcripts.map(item => item.text || '').filter(Boolean).join('\\n\\n');
      return `<div class="step-card">
        <div class="card-head">
          <div>
            <div class="strong">Combined Transcript</div>
            <div class="subtle">${transcripts.length} source${transcripts.length === 1 ? '' : 's'} compiled for this job</div>
          </div>
          <span class="chip ${transcripts.every(item => item.status === 'complete') ? 'good' : 'warn'}">${transcripts.filter(item => item.status === 'complete').length}/${transcripts.length} complete</span>
        </div>
        <div class="chip-row" style="margin-bottom:8px">
          ${transcripts.map(item => `<span class="chip ${transcriptStatusClass(item.status)}">${esc(item.source_file)}: ${esc(item.status)}</span>`).join('')}
        </div>
        <div class="chip-row" style="margin-bottom:8px">
          ${transcripts.map(item => `<span class="chip">${esc(item.source_file)}: ${esc((item.words || []).length)} timed words</span>`).join('')}
          ${state.currentJobData?.step_parse?.rough_step_count ? `<span class="chip good">${esc(state.currentJobData.step_parse.rough_step_count)} rough steps</span>` : ''}
          ${state.currentJobData?.step_parse?.expected_step_count ? `<span class="chip">Expected ${esc(state.currentJobData.step_parse.expected_step_count)}</span>` : ''}
        </div>
        ${transcripts.some(item => item.error) ? `<div class="chip-row" style="margin-bottom:8px">${transcripts.filter(item => item.error).map(item => `<span class="chip bad" style="white-space:normal">${esc(item.source_file)}: ${esc(item.error)}</span>`).join('')}</div>` : ''}
        ${transcripts.flatMap(item => item.chunks || []).length ? `<div class="chip-row" style="margin-bottom:8px">${transcripts.flatMap(item => item.chunks || []).map((chunk, idx) => `<span class="chip">Chunk ${idx + 1}: ${esc(chunk.size || '')}</span>`).join('')}</div>` : ''}
        <textarea readonly class="transcript-box" data-autogrow="true" data-max-height="560" data-scroll-key="combined-transcript">${esc(combinedText || 'No transcript text yet.')}</textarea>
      </div>`;
    }

    function draftSlides(job) {
      const slides = [];
      (job.draft_sections || []).forEach((section, sectionIndex) => {
        (section.steps || []).forEach((step, stepIndex) => {
          slides.push({section, sectionIndex, step, stepIndex});
        });
      });
      return slides;
    }

    function setDraftSlide(jobId, index) {
      const job = state.currentJobData;
      const total = draftSlides(job).length;
      if (!total) return;
      invalidateRender();
      state.draftSlideIndexes[jobId] = Math.max(0, Math.min(total - 1, index));
      render(false);
    }

    function renderDraftAudit(job) {
      const validation = job.draft_validation || {};
      const audit = job.draft_audit || {};
      if (!audit.steps && !validation.issue_count) return '';
      return `<div class="draft-audit chip-row">
        ${audit.steps ? `<span class="chip">Audit: ${esc(audit.sections || 0)} sections · ${esc(audit.steps || 0)} steps</span>` : ''}
        <span class="chip ${validation.issue_count ? 'warn' : 'good'}">${esc(validation.issue_count || 0)} QA items</span>
        ${(job.media_selection?.fallback_image_steps || 0) ? `<span class="chip warn">${esc(job.media_selection.fallback_image_steps)} fallback images</span>` : ''}
        ${(job.media_selection?.zero_image_steps || 0) ? `<span class="chip bad">${esc(job.media_selection.zero_image_steps)} no image</span>` : ''}
        ${(job.media_selection?.merge_candidate_steps || 0) ? `<span class="chip warn">${esc(job.media_selection.merge_candidate_steps)} merge candidates</span>` : ''}
      </div>`;
    }

    function moveDraftSlide(jobId, delta) {
      const current = state.draftSlideIndexes[jobId] || 0;
      setDraftSlide(jobId, current + delta);
    }

    function renderDraftBrowser(job) {
      const slides = draftSlides(job);
      if (!slides.length) return `<div class="empty">Draft sections will appear after questions are resolved.</div>`;
      const index = Math.max(0, Math.min(slides.length - 1, state.draftSlideIndexes[job.id] || 0));
      state.draftSlideIndexes[job.id] = index;
      const slide = slides[index];
      const section = slide.section;
      const step = slide.step;
      const sectionStepCount = section.steps.length;
      return `<div class="draft-browser" data-scroll-key="draft-browser">
        <div class="draft-browser-head">
          <div>
            <div class="strong">${esc(section.title)}</div>
            <div class="subtle">Section ${slide.sectionIndex + 1} of ${job.draft_sections.length} · Step ${slide.stepIndex + 1} of ${sectionStepCount}</div>
          </div>
          <div class="chip-row">
            <span class="chip">${index + 1}/${slides.length}</span>
            <span class="chip">${esc(job.draft_source || 'draft')}</span>
            ${job.step_parse?.rough_step_count ? `<span class="chip">${esc(job.step_parse.rough_step_count)} parsed steps</span>` : ''}
            ${job.media_selection?.scored_frames ? `<span class="chip good">${esc(job.media_selection.steps || 0)} media matches</span>` : ''}
            ${job.media_selection?.pointing_frames ? `<span class="chip warn">${esc(job.media_selection.pointing_frames)} pointing frames</span>` : ''}
          </div>
        </div>
        ${renderDraftAudit(job)}
        <div class="draft-step-view">
          <div class="draft-step-text">${esc(step.text)}</div>
          <div class="chip-row">
            <span class="chip">Tool: ${esc(step.tools || 'None')}</span>
            <span class="chip ${step.torque && step.torque !== 'Not required' ? 'warn' : ''}">Torque: ${esc(step.torque || 'None')}</span>
            ${step.timing_confidence ? `<span class="chip">Timing: ${esc(step.timing_confidence)}</span>` : ''}
            ${step.media_match_confidence ? `<span class="chip ${step.media_match_confidence === 'high' ? 'good' : step.media_match_confidence === 'missing' ? 'bad' : ''}">Media: ${esc(step.media_match_confidence)}</span>` : ''}
            ${(step.media_candidates || []).length ? `<span class="chip">${esc((step.media_candidates || []).length)} alternates</span>` : ''}
            ${step.quality ? `<span class="chip good">QC: ${esc(step.quality)}</span>` : ''}
            ${step.safety ? `<span class="chip warn">Safety: ${esc(step.safety)}</span>` : ''}
          </div>
          ${(step.parts || []).length ? `<div class="chip-row">${step.parts.map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}</div>` : ''}
          <div class="draft-media">${(step.media || []).map(m => mediaThumb(m, step)).join('') || '<div class="thumb">No media</div>'}</div>
        </div>
        <div class="draft-nav">
          <button onclick="moveDraftSlide('${job.id}', -1)" ${index <= 0 ? 'disabled' : ''}>&larr; Previous</button>
          <div class="chip-row">
            ${job.draft_sections.map((draftSection, sectionIndex) => {
              const startIndex = slides.findIndex(item => item.sectionIndex === sectionIndex);
              return `<button class="${sectionIndex === slide.sectionIndex ? 'primary' : ''}" onclick="setDraftSlide('${job.id}', ${startIndex})">${esc(draftSection.title)}</button>`;
            }).join('')}
          </div>
          <button onclick="moveDraftSlide('${job.id}', 1)" ${index >= slides.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
        </div>
      </div>`;
    }

    function renderLog(job) {
      return `<div class="log">${job.logs.slice().reverse().map(line => `<div class="log-line"><span>${esc(line.time)}</span><span>${esc(line.message)}</span></div>`).join('')}</div>`;
    }

    function renderJobReliability(job) {
      const validation = job.draft_validation || {};
      const audit = job.draft_audit || {};
      const cache = job.processing_cache || {};
      const watchdog = job.watchdog || {};
      const health = job.processing_health || {};
      return `<div class="step-card">
        <div class="card-head">
          <div>
            <div class="strong">Reliability</div>
            <div class="subtle">Watchdog, timing health, cache reuse, and draft audit</div>
          </div>
          <span class="chip ${(validation.issue_count || (health.warnings || []).length) ? 'warn' : 'good'}">${esc((validation.issue_count || 0) + ((health.warnings || []).length || 0))} items</span>
        </div>
        <div class="chip-row">
          ${watchdog.stage ? `<span class="chip ${watchdog.thread_alive ? '' : 'warn'}">Watchdog: ${esc(watchdog.stage)} · ${esc(Math.round((watchdog.idle_seconds || 0) / 60))}m idle</span>` : '<span class="chip good">Watchdog clear</span>'}
          <span class="chip ${health.timed_words ? 'good' : (health.transcript_sources ? 'warn' : '')}">${esc(health.timed_words || 0)} timed words</span>
          ${health.timing_steps ? `<span class="chip ${health.timing_matched_steps === health.timing_steps ? 'good' : 'warn'}">Timing ${esc(health.timing_matched_steps || 0)}/${esc(health.timing_steps || 0)}</span>` : ''}
          ${health.rough_steps ? `<span class="chip">Parsed ${esc(health.rough_steps)} steps</span>` : ''}
          ${health.zero_image_steps ? `<span class="chip bad">${esc(health.zero_image_steps)} no image</span>` : ''}
          ${health.fallback_image_steps ? `<span class="chip warn">${esc(health.fallback_image_steps)} fallback media</span>` : ''}
          ${cache.transcription ? `<span class="chip">Transcript cache: ${esc(cache.transcription.transcripts_reused || 0)} reused</span>` : ''}
          ${cache.video ? `<span class="chip">Video cache: ${esc(cache.video.raw_frame_sets_reused || 0)} frame sets reused</span>` : ''}
          ${cache.frame_analysis ? `<span class="chip">Frame analysis: ${esc(cache.frame_analysis.analysis_reused || 0)} reused</span>` : ''}
          ${audit.steps ? `<span class="chip">Audit: ${esc(audit.sections || 0)} sections · ${esc(audit.steps || 0)} steps</span>` : ''}
        </div>
        ${(health.warnings || []).length ? `<div class="chip-row" style="margin-top:8px">${health.warnings.map(item => `<span class="chip warn">${esc(item)}</span>`).join('')}</div>` : ''}
        ${validation.issue_counts ? `<div class="chip-row" style="margin-top:8px">${Object.entries(validation.issue_counts).map(([key, count]) => `<span class="chip ${count ? 'warn' : ''}">${esc(key)}: ${esc(count)}</span>`).join('')}</div>` : ''}
      </div>`;
    }

    function renderJob() {
      const job = state.currentJobData;
      if (!job) {
        document.getElementById('root').innerHTML = `<div class="empty">Job not found.</div>`;
        return;
      }
      setTitle('Job Review');
      const processingActive = ['Processing','Questions Pending','Drafting Sections'].includes(job.status);
      const canStartProcessing = (!job.processing_started || job.needs_processing) && !processingActive;
      setTopActions(`
        <button onclick="go('jobs')">Back</button>
        <button onclick="startJob('${job.id}')" ${canStartProcessing ? '' : 'disabled'}>Start Processing</button>
        <button class="primary" onclick="completeJob('${job.id}')" ${job.draft_sections.length ? '' : 'disabled'}>Mark Complete</button>
      `);
      const advancedOpen = Object.prototype.hasOwnProperty.call(state.advancedOpenJobs, job.id)
        ? state.advancedOpenJobs[job.id]
        : ['Processing', 'Questions Pending', 'Drafting Sections'].includes(job.status);
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div>
            <div class="strong">${esc(job.job_no)} · ${esc(job.name)}</div>
            <div class="subtle">${esc(job.product)} · ${esc(job.work_area)} · ${esc(job.station)}${job.expected_step_count ? ' · ~' + esc(job.expected_step_count) + ' expected steps' : ''} · ${job.focus_area === 'full' ? 'full frame' : 'center focus'}</div>
          </div>
          <div class="chip-row">${statusChip(job.status)}<span class="chip">${esc(job.profile)}</span>${job.flags.map(f => `<span class="chip warn">${esc(f)}</span>`).join('')}</div>
        </div>
        <div class="stage-review-grid">
          <div class="panel">
            <div class="panel-head"><h2>Progress</h2><span class="subtle">${job.progress}%</span></div>
            <div class="panel-body stack" data-scroll-key="section-library">
              <div class="bar"><span style="width:${job.progress}%"></span></div>
              ${renderStageList(job)}
            </div>
          </div>
          ${renderStageDetails(job)}
        </div>
        <details ${advancedOpen ? 'open' : ''} ontoggle="state.advancedOpenJobs['${job.id}']=this.open">
          <summary>Advanced</summary>
          <div class="details-body stack">
            <div class="split-actions" style="justify-content:flex-start">
              <button onclick="openAddFiles('${job.id}')">Add Files</button>
              <button onclick="resumeCurrentStage('${job.id}')" ${processingActive ? 'disabled' : ''}>Resume Current</button>
              <button onclick="rerunJob('${job.id}')" ${processingActive ? 'disabled' : ''}>Rerun Job</button>
              <button onclick="rerunStage('${job.id}', 'transcription')" ${processingActive ? 'disabled' : ''}>Rerun Transcript</button>
              <button onclick="rerunStage('${job.id}', 'video')" ${processingActive ? 'disabled' : ''}>Rerun Video</button>
              <button onclick="rerunStage('${job.id}', 'draft_sections')" ${processingActive ? 'disabled' : ''}>Rerun Drafts</button>
              <button onclick="rerunStage('${job.id}', 'media_assignment')" ${processingActive || !job.draft_sections.length ? 'disabled' : ''}>Rerun Media</button>
              <button class="danger" onclick="deleteJob('${job.id}')" ${processingActive ? 'disabled' : ''}>Delete Job</button>
            </div>
            ${renderJobReliability(job)}
            ${renderLog(job)}
          </div>
        </details>
      `;
    }

    function sectionSearchText(section) {
      const steps = (section.steps || []).map(step => [
        step.text,
        step.tools,
        step.torque,
        step.quality,
        step.safety,
        (step.parts || []).join(' '),
        (step.media || []).join(' ')
      ].join(' ')).join(' ');
      return [
        section.title,
        section.product,
        section.work_area,
        section.station,
        (section.tags || []).join(' '),
        section.source_job,
        steps
      ].join(' ').toLowerCase();
    }

    function filterSectionRows() {
      const query = String(state.sectionSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-section-row]').forEach(row => {
        const match = (row.dataset.sectionSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('section-count');
      if (count) count.textContent = `${shown} sections`;
      const empty = document.getElementById('section-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function renderSections() {
      setTitle('Sections');
      setTopActions('');
      const rows = state.sections;
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left"><input class="search" placeholder="Search sections" value="${esc(state.sectionSearch)}" oninput="state.sectionSearch=this.value; filterSectionRows()"></div>
          <div class="toolbar-right"><span class="subtle" id="section-count">${rows.length} sections</span></div>
        </div>
        <div class="table-wrap" data-scroll-key="sections-table">
          <table class="section-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Title</th>
                <th>Product</th>
                <th>Work Area</th>
                <th>Station</th>
                <th>Tags</th>
                <th>Steps</th>
                <th>Source Job</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(s => `
                <tr data-section-row data-section-search="${esc(sectionSearchText(s))}">
                  <td>${statusChip(s.status || 'Needs Edit')}</td>
                  <td class="section-title-cell">
                    <div class="strong">${esc(s.title)}</div>
                    <div class="subtle">${esc((s.steps || [])[0]?.text || '')}</div>
                  </td>
                  <td>${esc(s.product || '')}</td>
                  <td>${esc(s.work_area || '')}</td>
                  <td>${esc(s.station || '')}</td>
                  <td><div class="chip-row">${(s.tags || []).map(t => `<span class="chip">${esc(t)}</span>`).join('') || '<span class="subtle">None</span>'}</div></td>
                  <td>${(s.steps || []).length}</td>
                  <td><span class="subtle">${esc(s.source_job || '')}</span></td>
                  <td>
                    <div class="section-actions">
                      <button onclick="go('section','${s.id}')">Edit</button>
                      <button onclick="duplicateSection('${s.id}')">Duplicate</button>
                      <button onclick="markSectionComplete('${s.id}')" ${s.status === 'Complete' ? 'disabled' : ''}>Mark Complete</button>
                      <button class="primary" onclick="addToPackage('${s.id}')" ${s.status === 'Complete' ? '' : 'disabled'}>Add to Package</button>
                      <button class="danger" onclick="deleteSection('${s.id}')">Delete</button>
                    </div>
                  </td>
                </tr>
              `).join('')}
              <tr id="section-empty-row" style="display:none"><td colspan="9"><div class="empty">No matching sections.</div></td></tr>
            </tbody>
          </table>
        </div>
      `;
      filterSectionRows();
    }

    function cloneSection(section) {
      return JSON.parse(JSON.stringify(section));
    }

    function newStepId() {
      if (window.crypto?.randomUUID) return `step-${window.crypto.randomUUID().slice(0, 8)}`;
      return `step-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    }

    function blankStep() {
      return {
        step_id: newStepId(),
        text: '',
        tools: '',
        torque: '',
        quality: '',
        safety: '',
        parts: [],
        media: [],
        media_settings: {},
        qa_resolutions: {}
      };
    }

    function ensureStepIdentity(step) {
      if (!step.step_id) step.step_id = newStepId();
      if (!step.qa_resolutions || typeof step.qa_resolutions !== 'object' || Array.isArray(step.qa_resolutions)) {
        step.qa_resolutions = {};
      }
      step.parts = normalizeMediaList(step.parts || []);
      step.media = normalizeMediaList(step.media || []);
      step.media_settings = normalizeStepMediaSettings(step);
      return step;
    }

    function ensureDraftStepIds(draft) {
      if (!draft) return draft;
      if (!draft.steps || !draft.steps.length) draft.steps = [blankStep()];
      draft.steps.forEach(ensureStepIdentity);
      return draft;
    }

    function ensureSectionDraft(section) {
      if (!section) return null;
      if (state.sectionDraftId !== section.id || !state.sectionDraft) {
        state.sectionDraftId = section.id;
        state.sectionDraft = cloneSection(section);
        if (!state.sectionDraft.steps || !state.sectionDraft.steps.length) {
          state.sectionDraft.steps = [blankStep()];
        }
        ensureDraftStepIds(state.sectionDraft);
        state.sectionStepIndexes[section.id] = 0;
      }
      return state.sectionDraft;
    }

    function commaList(value) {
      return String(value || '').split(',').map(item => item.trim()).filter(Boolean);
    }

    function tokenValues(value) {
      return uniqueList(Array.isArray(value) ? value : commaList(value));
    }

    function tokenValueText(values) {
      return tokenValues(values).join(', ');
    }

    function renderTokenBadges(id, values) {
      return tokenValues(values).map((value, idx) => `
        <button type="button" class="token-badge" onclick="removeTokenValue('${id}', ${idx})" title="Remove ${esc(value)}">
          <span>${esc(value)}</span><span class="token-remove">&times;</span>
        </button>
      `).join('');
    }

    function tokenField(id, value, placeholder = 'Add item') {
      const values = tokenValues(value);
      return `<div class="token-field" data-token-field="${id}">
        <input type="hidden" id="${id}" value="${esc(tokenValueText(values))}">
        <span class="token-list" id="${id}-tokens">${renderTokenBadges(id, values)}</span>
        <input class="token-input" id="${id}-entry" placeholder="${esc(placeholder)}" onkeydown="handleTokenFieldKey(event, '${id}')" onblur="commitTokenFieldInput('${id}')">
      </div>`;
    }

    function getTokenValues(id) {
      return tokenValues(document.getElementById(id)?.value || '');
    }

    function setTokenValues(id, values) {
      const clean = tokenValues(values);
      const hidden = document.getElementById(id);
      if (hidden) hidden.value = tokenValueText(clean);
      const tokens = document.getElementById(`${id}-tokens`);
      if (tokens) tokens.innerHTML = renderTokenBadges(id, clean);
    }

    function commitTokenFieldInput(id) {
      const entry = document.getElementById(`${id}-entry`);
      if (!entry) return;
      const additions = commaList(entry.value);
      if (!additions.length) return;
      setTokenValues(id, [...getTokenValues(id), ...additions]);
      entry.value = '';
    }

    function handleTokenFieldKey(event, id) {
      if (event.key === 'Enter' || event.key === ',') {
        event.preventDefault();
        commitTokenFieldInput(id);
        return;
      }
      if (event.key === 'Backspace') {
        const entry = document.getElementById(`${id}-entry`);
        if (entry && !entry.value) {
          const values = getTokenValues(id);
          values.pop();
          setTokenValues(id, values);
        }
      }
    }

    function removeTokenValue(id, index) {
      const values = getTokenValues(id);
      values.splice(index, 1);
      setTokenValues(id, values);
      const entry = document.getElementById(`${id}-entry`);
      if (entry) entry.focus();
    }

    function normalizeMediaList(value) {
      if (Array.isArray(value)) return value.map(item => String(item || '').trim()).filter(Boolean);
      return commaList(value);
    }

    function uniqueList(items) {
      const seen = new Set();
      return (items || []).map(item => String(item || '').trim()).filter(Boolean).filter(item => {
        const key = item.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }

    function combineTextValues(values, separator = ' ') {
      return uniqueList(values).join(separator).replace(/\s+/g, ' ').trim();
    }

    function qaResolutionHandled(step, issueId) {
      const status = step?.qa_resolutions?.[issueId]?.status;
      return status === 'bypassed' || status === 'resolved';
    }

    function setQaResolution(step, issueId, status = 'bypassed', note = '') {
      ensureStepIdentity(step);
      step.qa_resolutions[issueId] = {
        status,
        note,
        at: new Date().toISOString()
      };
    }

    function qaIssueDefinition(issueId) {
      return {
        blank_text: {label: 'Blank instruction', detail: 'Instruction text is empty.', target: 'step-text'},
        missing_media: {label: 'Missing media', detail: 'No image is attached to this step.', target: 'media'},
        fallback_media: {label: 'Low-confidence media', detail: 'Image was selected as fallback or scored low.', target: 'media'},
        merge_candidate: {label: 'Merge candidate', detail: 'This step may be too small to stand alone.', target: 'merge'},
        missing_parts: {label: 'Missing parts', detail: 'Step wording suggests parts or consumables, but none are listed.', target: 'step-parts'},
        missing_torque: {label: 'Missing torque', detail: 'Fastener wording suggests torque may matter.', target: 'step-torque'},
        quality_cue: {label: 'Quality cue', detail: 'Step wording suggests a check, but QC is empty.', target: 'step-quality'},
        safety_cue: {label: 'Safety cue', detail: 'Step wording suggests a hazard, but safety is empty.', target: 'step-safety'}
      }[issueId] || {label: issueId, detail: '', target: ''};
    }

    function stepQaIssues(step, idx) {
      ensureStepIdentity(step);
      const text = String(step.text || '').trim();
      const lower = text.toLowerCase();
      const media = normalizeMediaList(step.media);
      const parts = normalizeMediaList(step.parts);
      const torque = String(step.torque || '').trim().toLowerCase();
      const issues = [];
      const add = issueId => {
        if (qaResolutionHandled(step, issueId)) return;
        const def = qaIssueDefinition(issueId);
        issues.push({issueId, stepIndex: idx, stepId: step.step_id, label: def.label, detail: def.detail, target: def.target});
      };
      if (!text) add('blank_text');
      if (!media.length) add('missing_media');
      else if (step.media_fallback || ['fallback', 'low', 'missing'].includes(step.media_match_confidence)) add('fallback_media');
      if (step.merge_candidate) add('merge_candidate');
      if (lower && !parts.length && /\b(attach|install|insert|place|apply|assemble|add|mount|connect|bolt|screw|fasten|position|weld)\b/.test(lower)) add('missing_parts');
      if (/\b(bolt|screw|fastener|nut|tighten|torque)\b/.test(lower) && (!torque || ['none', 'not required', 'n/a', 'na'].includes(torque))) add('missing_torque');
      if (!String(step.quality || '').trim() && /\b(check|verify|confirm|inspect|test|measure|flush|seated|alignment|align)\b/.test(lower)) add('quality_cue');
      if (!String(step.safety || '').trim() && /\b(sharp|hot|pinch|ppe|glove|weld|paint|fume|disconnect|lockout|lift|heavy|pressure|energized)\b/.test(lower)) add('safety_cue');
      return issues;
    }

    function sectionQaIssues(draft) {
      ensureDraftStepIds(draft);
      return (draft.steps || []).flatMap((step, idx) => stepQaIssues(step, idx));
    }

    function selectedSectionStepIndex(draft) {
      const count = Math.max((draft?.steps || []).length, 1);
      const raw = state.sectionStepIndexes[draft.id] ?? 0;
      const index = Math.min(Math.max(Number(raw) || 0, 0), count - 1);
      state.sectionStepIndexes[draft.id] = index;
      return index;
    }

    function selectedSectionStep() {
      const draft = collectSectionDraft();
      if (!draft) return null;
      return draft.steps[selectedSectionStepIndex(draft)] || null;
    }

    function openSectionMediaPicker(replaceIndex = null, mode = 'append') {
      const draft = collectSectionDraft();
      if (!draft) return;
      state.sectionMediaPicker = {
        open: true,
        replaceIndex,
        mode,
        filter: state.sectionMediaPicker.filter || 'near',
        search: state.sectionMediaPicker.search || ''
      };
      renderSectionEditor();
    }

    function closeSectionMediaPicker() {
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
    }

    function setSectionMediaPickerFilter(filter) {
      state.sectionMediaPicker.filter = filter;
      renderSectionEditor();
    }

    function setSectionMediaPickerSearch(value) {
      state.sectionMediaPicker.search = value;
      filterSectionMediaOptionCards();
    }

    function filterSectionMediaOptionCards() {
      const query = String(state.sectionMediaPicker.search || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-media-option]').forEach(card => {
        const match = (card.dataset.mediaSearch || '').includes(query);
        card.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('media-picker-count');
      if (count) count.textContent = `${shown} options shown`;
      const empty = document.getElementById('media-picker-empty');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function clearStepMediaFallback(step, manual = true) {
      step.media_fallback = false;
      step.media_fallback_reason = '';
      if (manual) step.media_match_confidence = 'manual';
      step.evidence_review = {
        ...(step.evidence_review || {}),
        media_reviewed: true,
        image_manually_selected: manual ? true : !!step.evidence_review?.image_manually_selected,
        updated_at: new Date().toISOString()
      };
      setQaResolution(step, 'fallback_media', 'resolved', manual ? 'Manual image selection accepted.' : 'Selected image accepted.');
      if (step.qa_resolutions?.missing_media) {
        setQaResolution(step, 'missing_media', 'resolved', manual ? 'Manual image attached.' : 'Selected image accepted.');
      }
    }

    function addStepMedia(path, mode = null) {
      const draft = collectSectionDraft();
      if (!draft || !path) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const replaceIndex = state.sectionMediaPicker.replaceIndex;
      const action = mode || state.sectionMediaPicker.mode || (replaceIndex !== null ? 'replace' : 'append');
      if (action === 'primary' || action === 'hero') {
        step.media = [path, ...step.media.filter(item => item !== path)];
        clearStepMediaFallback(step);
      } else if (replaceIndex !== null && replaceIndex >= 0) {
        step.media[replaceIndex] = path;
        if (replaceIndex === 0) clearStepMediaFallback(step);
      } else if (!step.media.includes(path)) {
        step.media.push(path);
      }
      step.media_settings = normalizeStepMediaSettings(step);
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
    }

    function secondsLabel(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return '';
      const minutes = Math.floor(number / 60);
      const seconds = Math.max(0, number - minutes * 60);
      return `${minutes}:${String(seconds.toFixed(seconds % 1 ? 1 : 0)).padStart(2, '0')}`;
    }

    function stepEvidenceReview(step) {
      step.evidence_review = step.evidence_review || {};
      return step.evidence_review;
    }

    function setEvidenceReview(step, patch = {}) {
      const review = stepEvidenceReview(step);
      Object.assign(review, patch, {updated_at: new Date().toISOString()});
      return review;
    }

    function selectedEvidenceStep() {
      const draft = collectSectionDraft();
      if (!draft) return {};
      const index = selectedSectionStepIndex(draft);
      const step = draft.steps[index];
      ensureStepIdentity(step);
      return {draft, step, index};
    }

    function useEvidenceCandidate(path, mode) {
      if (!path) return;
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      step.media = normalizeMediaList(step.media);
      const primaryMode = mode === 'primary' || mode === 'hero';
      if (primaryMode) {
        step.media = [path, ...step.media.filter(item => item !== path)];
        clearStepMediaFallback(step);
      } else if (!step.media.includes(path)) {
        step.media.push(path);
      }
      step.media_settings = normalizeStepMediaSettings(step);
      setEvidenceReview(step, {
        media_reviewed: true,
        image_manually_selected: true,
        needs_more_media: false,
        note: primaryMode ? 'Primary image manually selected from evidence workbench.' : 'Support image added from evidence workbench.'
      });
      renderSectionEditor();
      toast(primaryMode ? 'Primary image updated.' : 'Support image added.');
    }

    function markEvidenceReviewed() {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      setEvidenceReview(step, {media_reviewed: true, needs_more_media: false, note: 'Evidence accepted by user.'});
      if (normalizeMediaList(step.media).length) clearStepMediaFallback(step, false);
      renderSectionEditor();
      toast('Evidence accepted.');
    }

    function markEvidenceNeedsMoreMedia() {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      setEvidenceReview(step, {needs_more_media: true, media_reviewed: false, note: 'User requested more media near this step.'});
      state.sectionMediaPicker.open = true;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      state.sectionMediaPicker.filter = 'near';
      renderSectionEditor();
      toast('Showing nearby media options.');
    }

    function nudgeEvidenceTiming(direction) {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      const delta = direction === 'earlier' ? -2 : 2;
      const target = Number.isFinite(Number(step.target_time)) ? Number(step.target_time) : Number(step.end_time || step.start_time || 0);
      const nextTarget = Math.max(0, target + delta);
      step.target_time = Number(nextTarget.toFixed(2));
      if (Number.isFinite(Number(step.start_time))) step.start_time = Number(Math.max(0, Number(step.start_time) + delta).toFixed(2));
      if (Number.isFinite(Number(step.end_time))) step.end_time = Number(Math.max(step.start_time || 0, Number(step.end_time) + delta).toFixed(2));
      setEvidenceReview(step, {
        timing_adjusted: true,
        media_reviewed: false,
        needs_more_media: true,
        note: `Timing nudged ${direction}; find a better nearby frame.`
      });
      state.sectionMediaPicker.open = true;
      state.sectionMediaPicker.replaceIndex = 0;
      state.sectionMediaPicker.mode = 'replace';
      state.sectionMediaPicker.filter = 'near';
      renderSectionEditor();
      toast(direction === 'earlier' ? 'Timing nudged earlier.' : 'Timing nudged later.');
    }

    async function uploadSectionStepMedia(files) {
      const fileList = Array.from(files || []);
      const draft = collectSectionDraft();
      if (!draft || !state.currentSection || !fileList.length) return;
      const selected = selectedSectionStepIndex(draft);
      const form = new FormData();
      fileList.forEach(file => form.append('files', file));
      try {
        const result = await api(`/api/sections/${state.currentSection}/upload-media`, {method:'POST', body: form});
        const step = draft.steps[selected];
        ensureStepIdentity(step);
        step.media = normalizeMediaList(step.media);
        const existingMediaCount = step.media.length;
        const attachedPaths = [];
        [...(result.added || []), ...(result.skipped || [])].forEach(file => {
          if (file.stored_path && !step.media.includes(file.stored_path)) {
            step.media.push(file.stored_path);
            attachedPaths.push(file.stored_path);
          }
        });
        if (attachedPaths.length && (!existingMediaCount || step.media_fallback)) {
          step.media = [attachedPaths[0], ...step.media.filter(path => path !== attachedPaths[0])];
          clearStepMediaFallback(step);
        }
        step.media_settings = normalizeStepMediaSettings(step);
        const mediaData = await api(`/api/sections/${state.currentSection}/media-options`);
        state.sectionMediaOptions[state.currentSection] = mediaData.media;
        state.sectionDraft = draft;
        state.sectionDraftId = draft.id;
        state.sectionStepIndexes[draft.id] = selected;
        state.sectionMediaPicker.open = false;
        state.sectionMediaPicker.replaceIndex = null;
        state.sectionMediaPicker.mode = 'append';
        renderSectionEditor();
        toast(`${(result.added || []).length} images attached. ${(result.skipped || []).length} skipped.`);
      } catch (err) {
        toast(`Image upload failed: ${err.message}`);
      }
    }

    function removeStepMedia(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const removed = step.media.splice(idx, 1)[0];
      if (removed && step.media_settings) delete step.media_settings[removed];
      step.media_settings = normalizeStepMediaSettings(step);
      renderSectionEditor();
    }

    function moveStepMedia(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const next = idx + dir;
      if (next < 0 || next >= step.media.length) return;
      const [item] = step.media.splice(idx, 1);
      step.media.splice(next, 0, item);
      step.media_settings = normalizeStepMediaSettings(step);
      renderSectionEditor();
    }

    function setStepMediaSetting(path, key, value) {
      const draft = collectSectionDraft();
      if (!draft || !path) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      if (!step.media.includes(path)) return;
      const current = mediaSettingFor(step, path);
      if (key === 'fit') current.fit = value === 'cover' || value === 'fill' ? 'cover' : 'contain';
      if (key === 'aspect') current.aspect = ['auto', 'portrait', 'landscape', 'square'].includes(value) ? value : 'auto';
      if (key === 'crop_position') current.crop_position = ['center', 'top', 'bottom', 'left', 'right'].includes(value) ? value : 'center';
      step.media_settings[path] = normalizeMediaSetting(current);
      renderSectionEditor();
    }

    function mediaOptionSearchText(option) {
      return [
        option.path,
        option.kind,
        option.source_file,
        option.label,
        option.reason,
        option.summary,
        ...(option.tags || [])
      ].join(' ').toLowerCase();
    }

    function mediaOptionNearStep(option, step) {
      if (option.timestamp_seconds === null || option.timestamp_seconds === undefined) return false;
      const start = Number(step.start_time ?? step.target_time ?? 0);
      const end = Number(step.end_time ?? step.target_time ?? start + 8);
      const ts = Number(option.timestamp_seconds || 0);
      return ts >= Math.max(0, start - 8) && ts <= end + 10;
    }

    function stepMediaCandidateMap(step) {
      return new Map((step.media_candidates || []).map(candidate => [candidate.path, candidate]));
    }

    function filteredSectionMediaOptions(step) {
      const options = state.sectionMediaOptions[state.currentSection] || [];
      const filter = state.sectionMediaPicker.filter || 'near';
      const query = String(state.sectionMediaPicker.search || '').toLowerCase();
      const candidates = stepMediaCandidateMap(step);
      return options.filter(option => {
        const tags = new Set(option.tags || []);
        const tagText = Array.from(tags).join(' ').toLowerCase();
        if (query && !mediaOptionSearchText(option).includes(query)) return false;
        if (filter === 'near') return mediaOptionNearStep(option, step);
        if (filter === 'pointing') return tags.has('pointing') || tagText.includes('pointing');
        if (filter === 'visual_change') return tags.has('visual_change') || tagText.includes('visual_change');
        if (filter === 'sharp') return tags.has('sharp') || tagText.includes('sharp');
        if (filter === 'tool') return tags.has('tool') || tags.has('tool_visible') || tagText.includes('tool');
        if (filter === 'quality') return tags.has('quality') || tags.has('quality_check') || tagText.includes('quality');
        if (filter === 'safety') return tags.has('safety') || tagText.includes('safety');
        return true;
      }).sort((a, b) => {
        const aCandidate = candidates.get(a.path);
        const bCandidate = candidates.get(b.path);
        if (!!aCandidate !== !!bCandidate) return aCandidate ? -1 : 1;
        if (a.is_currently_used !== b.is_currently_used) return a.is_currently_used ? -1 : 1;
        return Number(bCandidate?.score || b.score || 0) - Number(aCandidate?.score || a.score || 0);
      });
    }

    function mediaOptionImage(option) {
      if (option.previewable) {
        return `<img class="media-option-preview clickable" src="${mediaUrl(option.path)}" alt="${esc(option.label || option.path)}" loading="lazy" onclick="openImageZoom('${jsSingleQuoted(option.path)}', '${jsSingleQuoted(option.label || option.path)}')">`;
      }
      return `<div class="source-tile">${esc(option.kind || 'media')}<br><span class="subtle">Preview unavailable</span></div>`;
    }

    function renderEvidenceCandidate(step, candidate) {
      const path = String(candidate.path || '').trim();
      if (!path) return '';
      const used = normalizeMediaList(step.media).includes(path);
      const label = mediaDisplayName(path);
      const tags = candidate.tags || [];
      return `<div class="evidence-candidate ${used ? 'used' : ''}">
        <img src="${mediaUrl(path)}" alt="${esc(label)}" loading="lazy" onclick="openImageZoom('${jsSingleQuoted(path)}', '${jsSingleQuoted(label)}')">
        <div class="evidence-candidate-body">
          <div class="strong media-card-title" title="${esc(path)}">${esc(label)}</div>
          <div class="chip-row">
            ${candidate.timestamp_seconds !== null && candidate.timestamp_seconds !== undefined ? `<span class="chip">~${esc(candidate.timestamp_seconds)}s</span>` : ''}
            ${candidate.score !== null && candidate.score !== undefined ? `<span class="chip">Score ${esc(candidate.score)}</span>` : ''}
            ${candidate.pointing ? '<span class="chip warn">Pointing</span>' : ''}
            ${candidate.after_state ? '<span class="chip good">After-state</span>' : ''}
            ${used ? '<span class="chip good">Used</span>' : ''}
          </div>
          ${tags.length ? `<div class="chip-row">${tags.slice(0, 4).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>` : ''}
          ${candidate.summary || candidate.reason || candidate.decision ? `<div class="evidence-candidate-note">${esc(candidate.summary || candidate.reason || candidate.decision)}</div>` : ''}
          <div class="media-card-actions">
            <button class="primary" onclick="useEvidenceCandidate('${jsSingleQuoted(path)}', 'primary')">Use as Primary</button>
            <button onclick="useEvidenceCandidate('${jsSingleQuoted(path)}', 'support')">Use as Support</button>
          </div>
        </div>
      </div>`;
    }

    function renderEvidenceWorkbench(draft, step, selected) {
      const review = step.evidence_review || {};
      const candidates = (step.media_candidates || []).filter(item => item.path).slice(0, 12);
      const previous = draft.steps[selected - 1] || null;
      const next = draft.steps[selected + 1] || null;
      const timing = [
        step.start_time !== undefined ? `Start ${secondsLabel(step.start_time)}` : '',
        step.target_time !== undefined ? `Target ${secondsLabel(step.target_time)}` : '',
        step.end_time !== undefined ? `End ${secondsLabel(step.end_time)}` : '',
      ].filter(Boolean).join(' · ') || 'No timing data';
      const neighborTiming = [
        previous ? `Prev ends ${secondsLabel(previous.end_time) || 'unknown'}` : '',
        next ? `Next starts ${secondsLabel(next.start_time) || 'unknown'}` : '',
      ].filter(Boolean).join(' · ') || 'No neighboring timing';
      const transcript = step.raw_transcript || 'No transcript slice captured for this step.';
      return `<div class="evidence-workbench">
        <div class="evidence-head">
          <div>
            <div class="strong">Evidence Workbench</div>
            <div class="subtle">Review timing, transcript, and candidate frames for this step.</div>
          </div>
          <div class="chip-row">
            <span class="chip ${review.media_reviewed ? 'good' : 'warn'}">${review.media_reviewed ? 'Evidence accepted' : 'Needs review'}</span>
            ${review.image_manually_selected ? '<span class="chip good">Manual image</span>' : ''}
            ${review.timing_adjusted ? '<span class="chip warn">Timing adjusted</span>' : ''}
            ${review.needs_more_media ? '<span class="chip warn">Needs media</span>' : ''}
          </div>
        </div>
        <div class="evidence-grid">
          <div class="evidence-panel">
            <div>
              <div class="subtle">Step Timing</div>
              <div class="strong">${esc(timing)}</div>
              <div class="subtle">${esc(neighborTiming)}</div>
            </div>
            <div class="chip-row">
              <button onclick="nudgeEvidenceTiming('earlier')">Too Late</button>
              <button onclick="nudgeEvidenceTiming('later')">Too Early</button>
              <button onclick="markEvidenceNeedsMoreMedia()">Find More Near This Step</button>
              <button class="primary" onclick="markEvidenceReviewed()">Accept Image</button>
            </div>
            <div>
              <div class="subtle">Transcript Slice</div>
              <div class="transcript-slice">${esc(transcript)}</div>
            </div>
            ${review.note ? `<div class="subtle">Review note: ${esc(review.note)}</div>` : ''}
          </div>
          <div class="evidence-panel">
            <div class="card-head">
              <div>
                <div class="strong">Candidates</div>
                <div class="subtle">${candidates.length} scored alternates attached to this step</div>
              </div>
              <span class="chip">${normalizeMediaList(step.media).length} selected</span>
            </div>
            <div class="evidence-candidates">
              ${candidates.map(candidate => renderEvidenceCandidate(step, candidate)).join('') || '<div class="empty compact">No scored candidates for this step. Use Find More Near This Step.</div>'}
            </div>
          </div>
        </div>
      </div>`;
    }

    function renderStepMediaManager(step) {
      const media = normalizeMediaList(step.media);
      const suggestedCount = (step.media_candidates || []).filter(item => item.path).length;
      return `<div class="step-media-manager">
        <div class="step-media-toolbar">
          <div class="chip-row">
            <span class="chip ${media.length ? 'good' : 'warn'}">${esc(media.length)} selected</span>
            ${suggestedCount ? `<span class="chip">${esc(suggestedCount)} alternates</span>` : ''}
            ${step.media_fallback ? '<span class="chip warn">Fallback media</span>' : ''}
            ${step.merge_candidate ? '<span class="chip warn">Merge candidate</span>' : ''}
          </div>
          <div class="split-actions">
            <button onclick="openSectionMediaPicker(null, 'append')">Add Media</button>
            <label class="step-media-upload">Upload Image<input type="file" accept="image/*" multiple onchange="uploadSectionStepMedia(this.files); this.value='';"></label>
            ${media.length ? `<button onclick="openSectionMediaPicker(0, 'replace')">Replace Primary</button>` : ''}
          </div>
        </div>
        ${step.media_fallback_reason ? `<div class="subtle">${esc(step.media_fallback_reason)}</div>` : ''}
        ${step.merge_candidate ? `<div class="question-card pending"><div class="strong">Possible merge</div><div class="subtle">${esc(step.merge_recommendation?.reason || 'This step may be too small to stand alone.')} ${step.merge_recommendation?.direction ? `Consider merging with the ${esc(step.merge_recommendation.direction)} step.` : ''}</div></div>` : ''}
        <div class="selected-step-media">
          ${media.map((path, idx) => mediaCard(step, path, idx, media.length)).join('') || '<div class="empty compact">No media selected for this step.</div>'}
        </div>
        ${renderSectionMediaPicker(step)}
      </div>`;
    }

    function renderSectionMediaPicker(step) {
      if (!state.sectionMediaPicker.open) return '';
      const filters = [
        ['near', 'Near step'],
        ['all', 'All'],
        ['pointing', 'Pointing'],
        ['visual_change', 'Visual change'],
        ['sharp', 'Sharp'],
        ['tool', 'Tool'],
        ['quality', 'Quality'],
        ['safety', 'Safety']
      ];
      const options = filteredSectionMediaOptions(step);
      const candidateMap = stepMediaCandidateMap(step);
      const replacing = state.sectionMediaPicker.replaceIndex !== null;
      return `<div class="media-picker">
        <div class="split-actions">
          <div>
            <div class="strong">${replacing ? 'Replace media' : 'Add media'}</div>
            <div class="subtle" id="media-picker-count">${options.length} options shown</div>
          </div>
          <button onclick="closeSectionMediaPicker()">Close</button>
        </div>
        <div class="chip-row">
          ${filters.map(([value, label]) => `<button class="${state.sectionMediaPicker.filter === value ? 'selected' : ''}" onclick="setSectionMediaPickerFilter('${value}')">${label}</button>`).join('')}
        </div>
        <input class="search" placeholder="Search media" value="${esc(state.sectionMediaPicker.search || '')}" oninput="setSectionMediaPickerSearch(this.value)">
        <div class="media-picker-grid" data-scroll-key="section-media-picker">
          ${options.map(option => `
            <div class="media-option ${option.is_currently_used ? 'used' : ''}" data-media-option data-media-search="${esc(mediaOptionSearchText(option))}">
              ${mediaOptionImage(option)}
              <div class="media-option-title strong">${esc(option.label || option.path)}</div>
              <div class="subtle">${option.timestamp_seconds !== null && option.timestamp_seconds !== undefined ? `~${esc(option.timestamp_seconds)}s · ` : ''}${esc(option.kind || '')}</div>
              <div class="chip-row">
                ${candidateMap.has(option.path) ? `<span class="chip good">Suggested</span>` : ''}
                ${(candidateMap.get(option.path)?.score || option.score) ? `<span class="chip">Score ${esc(candidateMap.get(option.path)?.score || option.score)}</span>` : ''}
                ${option.is_currently_used ? '<span class="chip good">Used</span>' : ''}
                ${(option.tags || []).slice(0, 3).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
              </div>
              ${option.summary || option.reason ? `<div class="subtle">${esc(option.summary || option.reason)}</div>` : ''}
              <div class="split-actions">
                ${option.previewable ? `<button onclick="openImageZoom('${jsSingleQuoted(option.path)}', '${jsSingleQuoted(option.label || option.path)}')">Preview</button>` : ''}
                ${replacing ? `<button class="primary" onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'replace')">Use Replacement</button>` : `
                  <button onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'support')">Use as Support</button>
                  <button class="primary" onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'primary')">Use as Primary</button>
                `}
              </div>
            </div>
          `).join('')}<div class="empty" id="media-picker-empty" style="display:${options.length ? 'none' : ''}">No media matches this filter.</div>
        </div>
      </div>`;
    }

    function collectSectionDraft() {
      const draft = state.sectionDraft;
      if (!draft) return null;
      if (!draft.steps || !draft.steps.length) draft.steps = [blankStep()];
      draft.title = document.getElementById('section-title')?.value || draft.title;
      draft.product = document.getElementById('section-product')?.value || '';
      draft.work_area = document.getElementById('section-area')?.value || '';
      draft.station = document.getElementById('section-station')?.value || '';
      draft.status = document.getElementById('section-status')?.value || draft.status;
      draft.tags = commaList(document.getElementById('section-tags')?.value || '');
      const selected = selectedSectionStepIndex(draft);
      if (document.getElementById('step-text')) {
        ['step-tools', 'step-torque', 'step-parts', 'step-safety'].forEach(commitTokenFieldInput);
        const existingStep = draft.steps[selected] || {};
        ensureStepIdentity(existingStep);
        draft.steps[selected] = {
          step_id: existingStep.step_id,
          text: document.getElementById('step-text')?.value || '',
          tools: document.getElementById('step-tools')?.value || '',
          torque: document.getElementById('step-torque')?.value || '',
          quality: document.getElementById('step-quality')?.value || '',
          safety: document.getElementById('step-safety')?.value || '',
          parts: commaList(document.getElementById('step-parts')?.value || ''),
          media: normalizeMediaList(existingStep.media || []),
          media_settings: normalizeStepMediaSettings(existingStep),
          start_time: existingStep.start_time,
          end_time: existingStep.end_time,
          target_time: existingStep.target_time,
          timing_source: existingStep.timing_source,
          timing_confidence: existingStep.timing_confidence,
          timing_word_matches: existingStep.timing_word_matches,
          source_step_id: existingStep.source_step_id,
          source_step_start: existingStep.source_step_start,
          source_step_end: existingStep.source_step_end,
          raw_transcript: existingStep.raw_transcript,
          boundary_reason: existingStep.boundary_reason,
          transcript_confidence: existingStep.transcript_confidence,
          media_candidates: existingStep.media_candidates || [],
          media_match_confidence: existingStep.media_match_confidence,
          media_match_score: existingStep.media_match_score,
          media_fallback: existingStep.media_fallback,
          media_fallback_reason: existingStep.media_fallback_reason,
          merge_candidate: existingStep.merge_candidate,
          merge_recommendation: existingStep.merge_recommendation,
          evidence_review: existingStep.evidence_review || {},
          qa_resolutions: existingStep.qa_resolutions || {}
        };
      }
      ensureDraftStepIds(draft);
      return draft;
    }

    function selectQaIssue(stepIndex, issueId) {
      const draft = collectSectionDraft();
      if (!draft) return;
      state.sectionStepIndexes[draft.id] = Math.min(Math.max(Number(stepIndex) || 0, 0), draft.steps.length - 1);
      state.sectionQaFocus = issueId;
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
      requestAnimationFrame(() => focusQaIssue(issueId));
    }

    function focusQaIssue(issueId) {
      const def = qaIssueDefinition(issueId);
      if (def.target === 'media') {
        openSectionMediaPicker(null, 'append');
        return;
      }
      const target = document.getElementById(def.target);
      const tokenEntry = document.getElementById(`${def.target}-entry`);
      if (tokenEntry) {
        tokenEntry.focus();
      } else if (target) {
        target.focus();
        if (target.select) target.select();
      }
    }

    function bypassQaIssue(issueId) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      setQaResolution(step, issueId, 'bypassed', 'User bypassed this QA item.');
      if (issueId === 'merge_candidate') {
        step.merge_candidate = false;
        step.merge_recommendation = null;
      }
      renderSectionEditor();
      toast('QA item bypassed.');
    }

    function resolveQaIssue(issueId, note = 'User resolved this QA item.') {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      setQaResolution(step, issueId, 'resolved', note);
      renderSectionEditor();
    }

    function acceptStepImage() {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      if (!normalizeMediaList(step.media).length) {
        toast('Attach an image first.');
        return;
      }
      clearStepMediaFallback(step);
      renderSectionEditor();
      toast('Image accepted.');
    }

    function keepStepSeparate() {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      step.merge_candidate = false;
      step.merge_recommendation = null;
      setQaResolution(step, 'merge_candidate', 'bypassed', 'User chose to keep this as a separate step.');
      renderSectionEditor();
      toast('Step kept separate.');
    }

    function renderSectionQaPanel(draft, selected) {
      const issues = sectionQaIssues(draft);
      const selectedIssues = issues.filter(issue => issue.stepIndex === selected);
      const handledCount = (draft.steps || []).reduce((total, step) => {
        return total + Object.values(step.qa_resolutions || {}).filter(item => ['bypassed', 'resolved'].includes(item.status)).length;
      }, 0);
      return `<div class="panel section-qa-panel">
        <div class="panel-head">
          <h2>Section QA</h2>
          <span class="chip ${issues.length ? 'warn' : 'good'}">${issues.length ? `${issues.length} open` : 'Clear'}</span>
        </div>
        <div class="panel-body stack">
          ${issues.length ? `
            <div class="qa-list">
              ${issues.slice(0, 12).map(issue => `
                <button class="qa-item ${issue.stepIndex === selected ? 'active' : ''}" onclick="selectQaIssue(${issue.stepIndex}, '${issue.issueId}')">
                  <span class="chip">${issue.stepIndex + 1}</span>
                  <span><strong>${esc(issue.label)}</strong><small>${esc(issue.detail)}</small></span>
                </button>
              `).join('')}
            </div>
          ` : '<div class="empty compact">No open QA items.</div>'}
          <div class="subtle">${selectedIssues.length ? `${selectedIssues.length} on selected step` : 'Selected step clear'}${handledCount ? ` · ${handledCount} handled` : ''}</div>
        </div>
      </div>`;
    }

    function renderSelectedStepQa(draft, step, selected) {
      const issues = stepQaIssues(step, selected);
      if (!issues.length) return '';
      return `<div class="question-card pending step-attention">
        <div class="card-head">
          <div>
            <div class="strong">Needs attention</div>
            <div class="subtle">${issues.length} item${issues.length === 1 ? '' : 's'} on this step</div>
          </div>
          <span class="chip warn">${issues.length}</span>
        </div>
        <div class="qa-list">
          ${issues.map(issue => `
            <div class="qa-attention-row">
              <div>
                <div class="strong">${esc(issue.label)}</div>
                <div class="subtle">${esc(issue.detail)}</div>
              </div>
              <div class="chip-row">
                ${issue.issueId === 'fallback_media' ? '<button onclick="acceptStepImage()">Accept Image</button>' : ''}
                ${issue.issueId === 'merge_candidate' ? `
                  <button onclick="mergeEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''}>Merge Previous</button>
                  <button onclick="mergeEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''}>Merge Next</button>
                  <button onclick="keepStepSeparate()">Keep Separate</button>
                ` : ''}
                ${issue.target && issue.target !== 'merge' ? `<button onclick="focusQaIssue('${issue.issueId}')">Fix</button>` : ''}
                <button onclick="bypassQaIssue('${issue.issueId}')">Bypass</button>
              </div>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    function renderSectionEditor() {
      const section = state.sections.find(s => s.id === state.currentSection);
      const draft = ensureSectionDraft(section);
      if (!draft) {
        setTitle('Section Editor');
        setTopActions(`<button onclick="go('sections')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Section not found.</div>`;
        return;
      }
      setTitle('Section Editor');
      setTopActions(`
        <button onclick="go('sections')">Back</button>
        <button class="danger" onclick="deleteSection('${draft.id}')">Delete</button>
        <button onclick="saveSection()">Save</button>
        <button class="primary" onclick="saveSection('Complete')">Mark Complete</button>
      `);
      const selected = selectedSectionStepIndex(draft);
      const step = draft.steps[selected] || blankStep();
      const qaIssues = sectionQaIssues(draft);
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div>
            <div class="strong">${esc(draft.title)}</div>
            <div class="subtle">${esc(draft.product)} · ${esc(draft.station)} · Source: ${esc(draft.source_job)}</div>
          </div>
          <div class="chip-row"><span class="chip ${draft.status === 'Complete' ? 'good' : 'warn'}">${esc(draft.status)}</span><span class="chip">${draft.steps.length} steps</span><span class="chip ${qaIssues.length ? 'warn' : 'good'}">${qaIssues.length ? `${qaIssues.length} QA` : 'QA clear'}</span></div>
        </div>
        <div class="section-workspace">
          <div class="stack">
            <div class="panel">
              <div class="panel-head"><h2>Section</h2></div>
              <div class="panel-body stack">
                <div class="field"><label>Title</label><input id="section-title" value="${esc(draft.title)}"></div>
                <div class="field"><label>Product</label><input id="section-product" value="${esc(draft.product)}"></div>
                <div class="form-grid">
                  <div class="field"><label>Work Area</label><input id="section-area" value="${esc(draft.work_area)}"></div>
                  <div class="field"><label>Station</label><input id="section-station" value="${esc(draft.station)}"></div>
                </div>
                <div class="form-grid">
                  <div class="field"><label>Status</label><select id="section-status">
                    <option ${draft.status === 'Needs Edit' ? 'selected' : ''}>Needs Edit</option>
                    <option ${draft.status === 'Complete' ? 'selected' : ''}>Complete</option>
                  </select></div>
                  <div class="field"><label>Source Job</label><input value="${esc(draft.source_job || '')}" disabled></div>
                </div>
                <div class="field"><label>Tags</label><input id="section-tags" value="${esc((draft.tags || []).join(', '))}"></div>
              </div>
            </div>
            ${renderSectionQaPanel(draft, selected)}
            <div class="panel">
              <div class="panel-head">
                <h2>Steps</h2>
                <button onclick="addEditorStep()">Add</button>
              </div>
              <div class="panel-body">
                <div class="editor-step-list" data-scroll-key="section-step-list">
                  ${draft.steps.map((item, idx) => `
                    <div class="editor-step-row ${idx === selected ? 'active' : ''}" onclick="selectEditorStep(${idx})">
                      <div class="chip-row">
                        <span class="chip">${idx + 1}</span>
                        ${(item.parts || []).length ? `<span class="chip brand">${(item.parts || []).length} parts</span>` : ''}
                        ${(item.media || []).length ? `<span class="chip">${(item.media || []).length} media</span>` : ''}
                        ${stepQaIssues(item, idx).length ? `<span class="chip warn">${stepQaIssues(item, idx).length} QA</span>` : ''}
                        ${item.media_fallback ? '<span class="chip warn">fallback</span>' : ''}
                        ${item.merge_candidate ? '<span class="chip warn">merge?</span>' : ''}
                      </div>
                      <div class="preview-line strong">${esc(item.text || 'Untitled step')}</div>
                      <div class="preview-line subtle">${esc(item.tools || 'No tools')} · ${esc(item.torque || 'No torque')}</div>
                    </div>
                  `).join('')}
                </div>
              </div>
            </div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <h2>Step ${selected + 1}</h2>
              <div class="chip-row">
                <button class="icon" onclick="moveEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''} title="Move up">↑</button>
                <button class="icon" onclick="moveEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''} title="Move down">↓</button>
                <button onclick="mergeEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''}>Merge Previous</button>
                <button onclick="mergeEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''}>Merge Next</button>
                <button onclick="splitEditorStep(${selected})">Split</button>
                <button class="icon" onclick="duplicateEditorStep(${selected})" title="Duplicate">⧉</button>
                <button class="icon danger" onclick="removeEditorStep(${selected})" title="Delete">×</button>
              </div>
            </div>
            <div class="panel-body stack">
              ${renderSelectedStepQa(draft, step, selected)}
              <div class="field"><label>Instruction</label><textarea id="step-text" data-autogrow data-max-height="420">${esc(step.text)}</textarea></div>
              <div class="form-grid">
                <div class="field"><label>Tools</label>${tokenField('step-tools', step.tools, 'Add tool')}</div>
                <div class="field"><label>Torque</label>${tokenField('step-torque', step.torque, 'Add torque')}</div>
              </div>
              <div class="field"><label>Parts / Consumables</label>${tokenField('step-parts', step.parts || [], 'Add part or consumable')}</div>
              <div class="form-grid">
                <div class="field"><label>Quality Check</label><textarea id="step-quality" data-autogrow data-max-height="260">${esc(step.quality)}</textarea></div>
                <div class="field"><label>Safety</label>${tokenField('step-safety', step.safety, 'Add safety note')}</div>
              </div>
              <div class="field"><label>Step Media</label>${renderStepMediaManager(step)}</div>
              ${renderEvidenceWorkbench(draft, step, selected)}
            </div>
          </div>
        </div>
      `;
    }

    function packageSearchText(section) {
      return sectionSearchText(section);
    }

    function packageListSearchText(packageItem) {
      return [
        packageItem.title,
        packageItem.product,
        packageItem.work_area,
        packageItem.station,
        packageItem.status,
        packageItem.created_at,
        packageItem.updated_at
      ].join(' ').toLowerCase();
    }

    function filterPackageListRows() {
      const query = String(state.packageListSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-package-list-row]').forEach(row => {
        const match = (row.dataset.packageSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('package-list-count');
      if (count) count.textContent = `${shown} packages`;
      const empty = document.getElementById('package-list-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function filterPackageRows() {
      const query = String(state.packageSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-package-section-row]').forEach(row => {
        const match = (row.dataset.packageSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('package-section-count');
      if (count) count.textContent = `${shown} complete sections`;
      const empty = document.getElementById('package-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function collectPackageDraftMeta() {
      const packageTitle = document.getElementById('package-title');
      const packageProduct = document.getElementById('package-product');
      const packageArea = document.getElementById('package-area');
      const packageStation = document.getElementById('package-station');
      const packageStatus = document.getElementById('package-status');
      state.packageDraftMeta = {
        title: packageTitle ? packageTitle.value : state.packageDraftMeta.title || '',
        product: packageProduct ? packageProduct.value : state.packageDraftMeta.product || '',
        work_area: packageArea ? packageArea.value : state.packageDraftMeta.work_area || '',
        station: packageStation ? packageStation.value : state.packageDraftMeta.station || '',
        status: packageStatus ? packageStatus.value : state.packageDraftMeta.status || 'Draft'
      };
      return state.packageDraftMeta;
    }

    function draftPackageSections() {
      return state.packageDraft
        .map(id => state.sections.find(section => section.id === id))
        .filter(Boolean);
    }

    function ensurePackageDraft(packageData) {
      if (!packageData || state.packageDraftId === packageData.id) return;
      state.packageDraftId = packageData.id;
      state.packageDraft = [...(packageData.section_ids || [])];
      state.packageDraftMeta = {
        title: packageData.title || '',
        product: packageData.product || '',
        work_area: packageData.work_area || '',
        station: packageData.station || '',
        status: packageData.status || 'Draft'
      };
    }

    function renderPackage() {
      setTitle('Packages');
      setTopActions(`<button class="primary" onclick="createPackageDraft()">New Package</button>`);
      const rows = state.packages || [];
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left">
            <input class="search" placeholder="Search packages" value="${esc(state.packageListSearch)}" oninput="state.packageListSearch=this.value; filterPackageListRows()">
          </div>
          <div class="toolbar-right">
            <span class="subtle" id="package-list-count">${rows.length} packages</span>
          </div>
        </div>
        <div class="table-wrap" data-scroll-key="packages-table">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Title</th>
                <th>Product</th>
                <th>Work Area / Station</th>
                <th>Sections</th>
                <th>Steps</th>
                <th>Updated</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(packageItem => `
                <tr data-package-list-row data-package-search="${esc(packageListSearchText(packageItem))}">
                  <td>${statusChip(packageItem.status || 'Draft')}</td>
                  <td>
                    <div class="strong">${esc(packageItem.title || 'Instruction Package')}</div>
                    <div class="subtle">Created ${esc(packageItem.created_at || '')}</div>
                  </td>
                  <td>${esc(packageItem.product || '')}</td>
                  <td>${esc(packageItem.work_area || '')}${packageItem.station ? ' / ' + esc(packageItem.station) : ''}</td>
                  <td>${packageItem.section_count || 0}</td>
                  <td>${packageItem.step_count || 0}</td>
                  <td><span class="subtle">${esc(packageItem.updated_at || '')}</span></td>
                  <td>
                    <div class="section-actions">
                      <button onclick="loadPackageDraft('${packageItem.id}')">Edit</button>
                      <button onclick="go('package_view','${packageItem.id}')">Preview</button>
                      <button onclick="go('operator_view','${packageItem.id}')">Operator</button>
                      <button onclick="go('package_print','${packageItem.id}')">Print</button>
                      <button class="danger" onclick="deletePackage('${packageItem.id}')">Delete</button>
                    </div>
                  </td>
                </tr>
              `).join('')}
              <tr id="package-list-empty-row" style="display:none"><td colspan="8"><div class="empty">No matching packages.</div></td></tr>
            </tbody>
          </table>
        </div>
      `;
      filterPackageListRows();
    }

    function renderSectionPackagePreview(section) {
      const steps = (section.steps || []).slice(0, 5);
      const hiddenCount = Math.max(0, (section.steps || []).length - steps.length);
      if (!steps.length) return `<div class="empty compact">No steps in this section.</div>`;
      return `<div class="section-preview-steps">
        ${steps.map((step, idx) => `
          <div class="section-preview-step">
            <div>
              <div class="subtle">Step ${idx + 1}</div>
              <div>${esc(step.text || '')}</div>
            </div>
            <div class="chip-row">
              ${step.tools ? `<span class="chip">Tool: ${esc(step.tools)}</span>` : ''}
              ${step.torque ? `<span class="chip ${step.torque === 'Not required' ? '' : 'warn'}">Torque: ${esc(step.torque)}</span>` : ''}
              ${(step.parts || []).slice(0, 4).map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}
              ${step.quality ? '<span class="chip good">QC</span>' : ''}
              ${step.safety ? '<span class="chip warn">Safety</span>' : ''}
            </div>
            <div class="selected-step-media">${(step.media || []).slice(0, 2).map(m => mediaThumb(m, step)).join('') || '<div class="thumb">No media</div>'}</div>
          </div>
        `).join('')}
        ${hiddenCount ? `<div class="subtle">${hiddenCount} more steps hidden.</div>` : ''}
      </div>`;
    }

    function renderPackageEditor() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Package Editor');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      ensurePackageDraft(packageData);
      const available = state.sections.filter(s => s.status === 'Complete');
      const draftSections = draftPackageSections();
      const meta = state.packageDraftMeta;
      setTitle('Package Editor');
      setTopActions(`
        <button onclick="go('package')">Back</button>
        <button onclick="savePackage()">Save Package</button>
        <button onclick="savePackage('preview')">Preview</button>
        <button class="primary" onclick="savePackage('print')">Print Preview</button>
      `);
      document.getElementById('root').innerHTML = `
        <div class="panel">
          <div class="panel-head">
            <h2>Package Details</h2>
            <span class="chip">${draftSections.length} sections</span>
          </div>
          <div class="panel-body">
            <div class="package-meta-grid">
              <div class="field"><label>Title</label><input id="package-title" value="${esc(meta.title || '')}" placeholder="Instruction package title"></div>
              <div class="field"><label>Product</label><input id="package-product" value="${esc(meta.product || '')}"></div>
              <div class="field"><label>Work Area</label><input id="package-area" value="${esc(meta.work_area || '')}"></div>
              <div class="field"><label>Station</label><input id="package-station" value="${esc(meta.station || '')}"></div>
              <div class="field"><label>Status</label><select id="package-status">
                <option ${meta.status === 'Draft' ? 'selected' : ''}>Draft</option>
                <option ${meta.status === 'Ready for Review' ? 'selected' : ''}>Ready for Review</option>
                <option ${meta.status === 'Published' ? 'selected' : ''}>Published</option>
              </select></div>
            </div>
          </div>
        </div>
        <div style="height:14px"></div>
        <div class="package-editor">
          <div class="panel">
            <div class="panel-head">
              <h2>Complete Sections</h2>
              <span class="chip" id="package-section-count">${available.length} complete sections</span>
            </div>
            <div class="panel-body">
              <div class="toolbar" style="padding:0 0 10px">
                <div class="toolbar-left"><input class="search" placeholder="Search completed sections" value="${esc(state.packageSearch)}" oninput="state.packageSearch=this.value; filterPackageRows()"></div>
              </div>
              <div class="section-result-list" data-scroll-key="package-library">
                ${available.map(s => {
                  const expanded = !!state.expandedPackageSections[s.id];
                  const alreadyAdded = state.packageDraft.includes(s.id);
                  return `
                    <div class="section-result" data-package-section-row data-package-search="${esc(packageSearchText(s))}">
                      <div class="section-result-head">
                        <div>
                          <div class="strong">${esc(s.title)}</div>
                          <div class="subtle">${esc(s.product || '')} · ${esc(s.work_area || '')}${s.station ? ' · ' + esc(s.station) : ''}</div>
                          <div class="chip-row">
                            <span class="chip">${(s.steps || []).length} steps</span>
                            ${(s.tags || []).slice(0, 5).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
                          </div>
                        </div>
                        <div class="chip-row">
                          <button onclick="togglePackageSectionPreview('${s.id}')">${expanded ? 'Collapse' : 'Expand'}</button>
                          <button class="primary" onclick="addToPackage('${s.id}')" ${alreadyAdded ? 'disabled' : ''}>${alreadyAdded ? 'Added' : 'Add'}</button>
                        </div>
                      </div>
                      ${expanded ? renderSectionPackagePreview(s) : ''}
                    </div>
                  `;
                }).join('')}
                <div class="empty" id="package-empty-row" style="display:none">No matching complete sections.</div>
              </div>
            </div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <h2>Package Outline</h2>
              <span class="chip">${draftSections.length} sections</span>
            </div>
            <div class="panel-body" data-scroll-key="package-draft">
              <div class="package-outline">
                ${draftSections.map((s, idx) => `
                  <div class="drop-item">
                    <div>
                      <div class="subtle">Section ${idx + 1}</div>
                      <div class="strong">${esc(s.title)}</div>
                      <div class="subtle">${(s.steps || []).length} steps · ${esc(s.source_job || '')}</div>
                      <div class="chip-row">${(s.tags || []).slice(0, 4).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>
                    </div>
                    <div class="chip-row">
                      <button class="icon" onclick="movePackage(${idx}, -1)" ${idx <= 0 ? 'disabled' : ''}>↑</button>
                      <button class="icon" onclick="movePackage(${idx}, 1)" ${idx >= draftSections.length - 1 ? 'disabled' : ''}>↓</button>
                      <button class="icon danger" onclick="removePackage(${idx})">×</button>
                    </div>
                  </div>
                `).join('') || '<div class="empty">Add complete sections to build this package.</div>'}
              </div>
            </div>
          </div>
        </div>
      `;
      filterPackageRows();
    }

    function renderPackagePreview() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Package Preview');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const sections = packageData.sections || [];
      const stepCount = sections.reduce((total, section) => total + (section.steps || []).length, 0);
      setTitle('Package Preview');
      setTopActions(`
        <button onclick="go('package')">Back</button>
        <button onclick="loadPackageDraft('${packageData.id}')">Edit Package</button>
        <button onclick="snapshotPackage('${packageData.id}')">Snapshot From Sections</button>
        <button onclick="go('operator_view','${packageData.id}')">Operator View</button>
        <button class="primary" onclick="go('package_print','${packageData.id}')">Print Preview</button>
      `);
      document.getElementById('root').innerHTML = `
        <div class="package-preview">
          <div class="preview-sheet">
            <div class="preview-heading">
              <div>
                <div class="subtle">Instruction Package</div>
                <h2 style="margin:4px 0 0">${esc(packageData.title)}</h2>
              </div>
              <div class="chip-row">
                ${statusChip(packageData.status || 'Draft')}
                <span class="chip">${sections.length} sections</span>
                <span class="chip">${stepCount} steps</span>
              </div>
            </div>
            <div class="form-grid">
              <div><div class="subtle">Product</div><div class="strong">${esc(packageData.product || '')}</div></div>
              <div><div class="subtle">Work Area / Station</div><div class="strong">${esc(packageData.work_area || '')} ${packageData.station ? '· ' + esc(packageData.station) : ''}</div></div>
              <div><div class="subtle">Created</div><div class="strong">${esc(packageData.created_at || '')}</div></div>
              <div><div class="subtle">Updated</div><div class="strong">${esc(packageData.updated_at || '')}</div></div>
            </div>
            ${sections.map((section, sectionIndex) => `
              <div class="preview-section">
                <div>
                  <div class="subtle">Section ${sectionIndex + 1}</div>
                  <h3 style="margin:2px 0 4px">${esc(section.title)}</h3>
                  <div class="subtle">${esc(section.product || '')} · ${esc(section.work_area || '')} · ${esc(section.station || '')} · Source: ${esc(section.source_job || '')}</div>
                  <div class="chip-row">${(section.tags || []).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>
                </div>
                ${(section.steps || []).map((step, stepIndex) => `
                  <div class="preview-step">
                    <div class="subtle">Step ${stepIndex + 1}</div>
                    <div class="preview-step-text">${esc(step.text || '')}</div>
                    <div class="chip-row">
                      ${step.tools ? `<span class="chip">Tool: ${esc(step.tools)}</span>` : ''}
                      ${step.torque ? `<span class="chip ${step.torque === 'Not required' ? '' : 'warn'}">Torque: ${esc(step.torque)}</span>` : ''}
                      ${step.quality ? `<span class="chip good">QC</span>` : ''}
                      ${step.safety ? `<span class="chip warn">Safety</span>` : ''}
                    </div>
                    ${(step.parts || []).length ? `<div class="chip-row">${(step.parts || []).map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}</div>` : ''}
                    ${step.quality ? `<div><div class="subtle">Quality Check</div><div>${esc(step.quality)}</div></div>` : ''}
                    ${step.safety ? `<div><div class="subtle">Safety</div><div>${esc(step.safety)}</div></div>` : ''}
                    ${(step.media || []).length ? `
                      <div>
                        <div class="subtle">Media</div>
                        <div class="selected-step-media">${(step.media || []).map(path => mediaThumb(path, step)).join('')}</div>
                        <div class="media-reference-list">${(step.media || []).map(path => `<div>${esc(path)}</div>`).join('')}</div>
                      </div>` : ''}
                  </div>
                `).join('')}
              </div>
            `).join('') || '<div class="empty">No sections saved in this package.</div>'}
          </div>
        </div>
      `;
    }

    function printablePackageSteps(packageData) {
      const items = [];
      (packageData.sections || []).forEach((section, sectionIndex) => {
        (section.steps || []).forEach((step, stepIndex) => {
          items.push({
            globalStep: items.length + 1,
            sectionIndex,
            stepIndex,
            sectionTitle: section.title || `Section ${sectionIndex + 1}`,
            sectionMeta: [section.product, section.work_area, section.station].filter(Boolean).join(' · '),
            step
          });
        });
      });
      return items;
    }

    function estimatePrintStepWeight(item) {
      const step = item.step || {};
      const textLength = String(step.text || '').length;
      const mediaCount = (step.media || []).filter(Boolean).length;
      const partsCount = (step.parts || []).length;
      let weight = 1;
      if (textLength > 160) weight += .7;
      else if (textLength > 105) weight += .35;
      if (mediaCount > 1) weight += .55;
      if (step.quality) weight += .45;
      if (step.safety) weight += .45;
      if (partsCount > 5) weight += .35;
      if (`${step.tools || ''} ${step.torque || ''}`.length > 70) weight += .25;
      return weight;
    }

    function canUseThreePrintSteps(items) {
      if (items.length !== 3) return false;
      const weights = items.map(estimatePrintStepWeight);
      const hasHeavyDetail = items.some(item => {
        const step = item.step || {};
        return String(step.text || '').length > 120
          || (step.media || []).filter(Boolean).length > 1
          || !!step.quality
          || !!step.safety
          || (step.parts || []).length > 5;
      });
      return !hasHeavyDetail && weights.every(weight => weight <= 1.45) && weights.reduce((sum, weight) => sum + weight, 0) <= 3.75;
    }

    function shouldUseSinglePrintStep(item) {
      const step = item.step || {};
      const weight = estimatePrintStepWeight(item);
      return weight >= 2.15
        || String(step.text || '').length > 230
        || ((step.media || []).filter(Boolean).length > 1 && (!!step.quality || !!step.safety || (step.parts || []).length > 6));
    }

    function chunkPrintableSteps(items) {
      const pages = [];
      let index = 0;
      while (index < items.length) {
        const remaining = items.length - index;
        let count = 1;
        if (remaining >= 3 && canUseThreePrintSteps(items.slice(index, index + 3))) {
          count = 3;
        } else if (remaining >= 2 && !shouldUseSinglePrintStep(items[index])) {
          count = 2;
        }
        pages.push(items.slice(index, index + count));
        index += count;
      }
      return pages;
    }

    function printMediaSlot(path, role, step = null) {
      const value = String(path || '').trim();
      const label = role === 'support' ? 'Support image' : 'Primary image';
      const fallback = role === 'support' ? 'Support image unavailable' : 'No image selected';
      const setting = mediaSettingFor(step || {}, value);
      if (!value || !(value.match(/\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/'))) {
        return `<div class="print-media-slot ${role} missing"><div class="print-media-fallback">${fallback}</div></div>`;
      }
      return `<div class="print-media-slot ${role} ${mediaClassNames(setting)}">
        <img src="${mediaUrl(value)}" alt="${esc(label)}" loading="lazy" onerror="this.parentElement.classList.add('missing'); this.remove();">
        <div class="print-media-fallback">${role === 'support' ? 'Support image unavailable' : 'Image unavailable'}</div>
      </div>`;
    }

    function renderPrintMedia(step) {
      const media = (step.media || []).filter(Boolean).slice(0, 2);
      if (!media.length) {
        return `<div class="print-media single">${printMediaSlot('', 'primary', step)}</div>`;
      }
      return `<div class="print-media ${media.length === 1 ? 'single' : ''}">
        ${printMediaSlot(media[0], 'primary', step)}
        ${media[1] ? printMediaSlot(media[1], 'support', step) : ''}
      </div>`;
    }

    function renderPrintCallout(title, content, cls = '') {
      return `<div class="print-callout ${cls}">
        <h4>${esc(title)}</h4>
        <div>${content}</div>
      </div>`;
    }

    function printPills(values, emptyText = 'None listed', cls = '') {
      const items = tokenValues(values);
      if (!items.length) return `<span class="print-muted">${esc(emptyText)}</span>`;
      return `<div class="print-pill-list">${items.map(item => `<span class="print-pill ${cls}">${esc(item)}</span>`).join('')}</div>`;
    }

    function renderPrintStep(item) {
      const step = item.step || {};
      const parts = printPills(step.parts || [], 'None listed');
      const torque = step.torque || 'Not required';
      const torqueRequired = torque && torque !== 'Not required';
      const tools = step.tools || 'None listed';
      const toolsTorque = `${printPills(tools, 'None listed')}<div style="margin-top:4px">${printPills(torque, 'Not required', torqueRequired ? 'warn' : 'muted')}</div>`;
      const callouts = [
        renderPrintCallout('Parts / Consumables', parts),
        renderPrintCallout('Tools & Torque', toolsTorque, torqueRequired ? 'torque-required' : ''),
        step.quality ? renderPrintCallout('Quality', esc(step.quality), 'quality') : '',
        step.safety ? renderPrintCallout('Safety', printPills(step.safety, '', 'warn'), 'safety') : ''
      ].filter(Boolean).join('');
      return `<article class="print-step-card">
        <div class="print-step-kicker">
          <span>Step ${item.globalStep}</span>
          <span>${esc(item.sectionTitle)}</span>
        </div>
        <div class="print-step-text">${esc(step.text || 'No instruction text.')}</div>
        ${renderPrintMedia(step)}
        <div class="print-callout-grid">${callouts}</div>
      </article>`;
    }

    function renderPackagePrint() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Print Preview');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const steps = printablePackageSteps(packageData);
      const pages = chunkPrintableSteps(steps);
      setTitle('Print Preview');
      setTopActions(`
        <button onclick="go('package_view','${packageData.id}')">Back</button>
        <button class="primary" onclick="window.print()">Print</button>
      `);
      if (!steps.length) {
        document.getElementById('root').innerHTML = `
          <div class="print-preview">
            <section class="print-page">
              <div class="print-page-head">
                <div>
                  <div class="subtle">Instruction Package</div>
                  <div class="print-page-title">${esc(packageData.title || 'Untitled Package')}</div>
                </div>
                <div class="print-page-meta"><span>No steps</span>${statusChip(packageData.status || 'Draft')}</div>
              </div>
              <div class="empty">No sections or steps saved in this package.</div>
              <div class="print-page-foot"><span>${esc(packageData.product || '')}</span><span>Page 1 of 1</span></div>
            </section>
          </div>
        `;
        return;
      }
      const pageTotal = pages.length;
      document.getElementById('root').innerHTML = `
        <div class="print-preview">
          ${pages.map((pageSteps, pageIndex) => `
            <section class="print-page">
              <div class="print-page-head">
                <div>
                  <div class="subtle">Instruction Package</div>
                  <div class="print-page-title">${esc(packageData.title || 'Untitled Package')}</div>
                  <div class="subtle">${esc(packageData.product || '')}${packageData.work_area ? ' · ' + esc(packageData.work_area) : ''}${packageData.station ? ' · ' + esc(packageData.station) : ''}</div>
                </div>
                <div class="print-page-meta">
                  ${statusChip(packageData.status || 'Draft')}
                  <span>${steps.length} steps</span>
                  <span>Page ${pageIndex + 1} of ${pageTotal}</span>
                </div>
              </div>
              <div class="print-steps steps-${pageSteps.length}">
                ${pageSteps.map(renderPrintStep).join('')}
              </div>
              <div class="print-page-foot">
                <span>Snapshot updated ${esc(packageData.updated_at || '')}</span>
                <span>Page ${pageIndex + 1} of ${pageTotal}</span>
              </div>
            </section>
          `).join('')}
        </div>
      `;
    }

    function setOperatorStep(packageId, index) {
      const packageData = state.currentPackageData;
      const steps = printablePackageSteps(packageData || {});
      if (!steps.length) return;
      state.operatorStepIndexes[packageId] = Math.max(0, Math.min(steps.length - 1, Number(index) || 0));
      renderOperatorView();
    }

    function moveOperatorStep(packageId, delta) {
      const current = state.operatorStepIndexes[packageId] || 0;
      setOperatorStep(packageId, current + delta);
    }

    function renderOperatorCallout(title, content, cls = '') {
      if (!content) return '';
      return `<div class="operator-callout ${cls}">
        <div class="subtle">${esc(title)}</div>
        <div>${content}</div>
      </div>`;
    }

    function renderOperatorView() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Operator View');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const steps = printablePackageSteps(packageData);
      const index = Math.max(0, Math.min(steps.length - 1, state.operatorStepIndexes[packageData.id] || 0));
      state.operatorStepIndexes[packageData.id] = index;
      setTitle('Operator View');
      setTopActions(`
        <button onclick="go('package_view','${packageData.id}')">Back</button>
        <button onclick="toggleNav()">Toggle Nav</button>
        <button onclick="go('package_print','${packageData.id}')">Print</button>
      `);
      if (!steps.length) {
        document.getElementById('root').innerHTML = `<div class="empty">No steps saved in this package.</div>`;
        return;
      }
      const item = steps[index];
      const step = item.step || {};
      const media = normalizeMediaList(step.media);
      const primaryImage = media[0] || '';
      const parts = printPills(step.parts || [], 'None listed');
      const tools = printPills(step.tools || '', 'None listed');
      const torque = printPills(step.torque || 'Not required', 'Not required', step.torque && step.torque !== 'Not required' ? 'warn' : 'muted');
      document.getElementById('root').innerHTML = `
        <div class="operator-shell">
          <div class="toolbar">
            <div>
              <div class="strong">${esc(packageData.title || 'Instruction Package')}</div>
              <div class="subtle">${esc(packageData.product || '')}${packageData.work_area ? ' · ' + esc(packageData.work_area) : ''}${packageData.station ? ' · ' + esc(packageData.station) : ''}</div>
            </div>
            <div class="chip-row">
              ${statusChip(packageData.status || 'Draft')}
              <span class="chip">${index + 1}/${steps.length}</span>
              <span class="chip">${esc(item.sectionTitle)}</span>
            </div>
          </div>
          <div class="operator-stage">
            <div class="operator-media">
              ${primaryImage ? `<img src="${mediaUrl(primaryImage)}" alt="${esc(item.sectionTitle)} step ${index + 1}" onclick="openImageZoom('${jsSingleQuoted(primaryImage)}', '${jsSingleQuoted(item.sectionTitle)} step ${index + 1}')">` : '<div class="empty compact">No image selected for this step.</div>'}
            </div>
            <div class="operator-step-panel">
              <div>
                <div class="subtle">Step ${item.globalStep} · ${esc(item.sectionTitle)}</div>
                <div class="operator-step-text">${esc(step.text || 'No instruction text.')}</div>
              </div>
              <div class="chip-row">
                ${(step.media || []).length ? `<span class="chip good">${(step.media || []).length} media</span>` : '<span class="chip warn">No media</span>'}
                ${step.media_match_confidence ? `<span class="chip">${esc(step.media_match_confidence)}</span>` : ''}
                ${step.timing_confidence ? `<span class="chip">Timing ${esc(step.timing_confidence)}</span>` : ''}
              </div>
              <div class="operator-callouts">
                ${renderOperatorCallout('Parts / Consumables', parts)}
                ${renderOperatorCallout('Tools', tools)}
                ${renderOperatorCallout('Torque', torque, step.torque && step.torque !== 'Not required' ? 'safety' : '')}
                ${step.quality ? renderOperatorCallout('Quality Check', esc(step.quality), 'quality') : ''}
                ${step.safety ? renderOperatorCallout('Safety', printPills(step.safety, '', 'warn'), 'safety') : ''}
              </div>
            </div>
          </div>
          <div class="operator-nav">
            <button onclick="moveOperatorStep('${packageData.id}', -1)" ${index <= 0 ? 'disabled' : ''}>Previous</button>
            <div class="operator-progress">
              <div class="bar"><span style="width:${Math.round(((index + 1) / steps.length) * 100)}%"></span></div>
              <div>Step ${index + 1} of ${steps.length}</div>
            </div>
            <button class="primary" onclick="moveOperatorStep('${packageData.id}', 1)" ${index >= steps.length - 1 ? 'disabled' : ''}>Next</button>
          </div>
        </div>
      `;
    }

    function renderSettings() {
      setTitle('Settings');
      setTopActions(`<button class="primary" onclick="saveSettings()">Save</button>`);
      const settings = state.settings || {};
      document.getElementById('root').innerHTML = `
        <div class="panel" style="max-width:760px">
          <div class="panel-head"><h2>Settings</h2></div>
          <div class="panel-body stack">
            <div class="field">
              <label>OpenAI API key</label>
              <input id="settings-openai-key" type="password" placeholder="${settings.has_openai_api_key ? 'Key saved. Leave blank to keep it.' : 'Paste API key'}">
            </div>
            <div class="form-grid">
              <div class="field"><label>Transcription backend</label><select id="settings-transcription-backend">
                <option value="auto" ${settings.transcription_backend === 'auto' ? 'selected' : ''}>Auto - local if configured</option>
                <option value="local_whisper" ${settings.transcription_backend === 'local_whisper' ? 'selected' : ''}>Local whisper.cpp</option>
                <option value="openai" ${settings.transcription_backend === 'openai' ? 'selected' : ''}>OpenAI API</option>
              </select></div>
              <div class="field"><label>Transcription model</label><select id="settings-transcription-model">
                <option ${settings.transcription_model === 'gpt-4o-mini-transcribe' ? 'selected' : ''}>gpt-4o-mini-transcribe</option>
                <option ${settings.transcription_model === 'gpt-4o-transcribe' ? 'selected' : ''}>gpt-4o-transcribe</option>
                <option ${settings.transcription_model === 'whisper-1' ? 'selected' : ''}>whisper-1</option>
              </select></div>
              <div class="field"><label>Local whisper CLI</label><input id="settings-local-whisper-cli" value="${esc(settings.local_whisper_cli_path || '')}" placeholder="/opt/homebrew/bin/whisper-cli"></div>
              <div class="field"><label>Local whisper model</label><input id="settings-local-whisper-model" value="${esc(settings.local_whisper_model_path || '')}" placeholder="/path/to/ggml-small.en.bin"></div>
              <div class="field"><label>Default processing</label><select id="settings-default-processing">
                <option ${settings.default_processing === 'Fast' ? 'selected' : ''}>Fast</option>
                <option ${settings.default_processing === 'Standard' ? 'selected' : ''}>Standard</option>
                <option ${settings.default_processing === 'Deep' ? 'selected' : ''}>Deep</option>
              </select></div>
            </div>
            <div class="chip-row">
              <span class="chip ${settings.has_local_whisper_cli ? 'good' : 'warn'}">CLI: ${settings.has_local_whisper_cli ? 'found' : 'missing'}</span>
              <span class="chip ${settings.has_local_whisper_model ? 'good' : 'warn'}">Local model: ${settings.has_local_whisper_model ? 'found' : 'missing'}</span>
              <span class="chip">Effective: ${esc(settings.effective_transcription_backend || 'openai')}</span>
            </div>
            <label class="chip-row" style="justify-content:flex-start">
              <input id="settings-local-whisper-no-gpu" type="checkbox" style="width:auto; min-height:auto" ${settings.local_whisper_no_gpu ? 'checked' : ''}>
              <span>Run local Whisper CPU-only</span>
            </label>
            <label class="chip-row" style="justify-content:flex-start">
              <input id="settings-transcription-enabled" type="checkbox" style="width:auto; min-height:auto" ${settings.transcription_enabled ? 'checked' : ''}>
              <span>Run transcription when processing jobs</span>
            </label>
            <div class="field"><label>Storage root</label><input value="${esc(settings.storage_root || 'storage')}" disabled></div>
            <div class="split-actions" style="justify-content:flex-start">
              <button onclick="exportProcessingLog()">Export Processing Log</button>
            </div>
            <div class="field">
              <label>Operator phrases</label>
              <div class="stage-detail-list">
                ${(settings.operator_phrase_guide || []).map(group => `
                  <div class="file-row">
                    <div>
                      <div class="strong">${esc(group.label)}</div>
                      <div class="subtle">${esc(group.purpose)}</div>
                      <div class="chip-row" style="margin-top:6px">${(group.phrases || []).map(phrase => `<span class="chip">${esc(phrase)}</span>`).join('')}</div>
                    </div>
                  </div>
                `).join('')}
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function render(refreshData = true) {
      saveScrollPositions();
      readHash();
      if (refreshData) {
        loadData().then(() => render(false)).catch(err => toast(err.message));
        return;
      }
      const fingerprint = pageFingerprint();
      if (fingerprint === state.renderFingerprint) {
        return;
      }
      state.renderFingerprint = fingerprint;
      if (state.currentPage === 'jobs') renderJobs();
      if (state.currentPage === 'job') renderJob();
      if (state.currentPage === 'sections') renderSections();
      if (state.currentPage === 'section') renderSectionEditor();
      if (state.currentPage === 'package') renderPackage();
      if (state.currentPage === 'package_edit') renderPackageEditor();
      if (state.currentPage === 'package_view') renderPackagePreview();
      if (state.currentPage === 'package_print') renderPackagePrint();
      if (state.currentPage === 'operator_view') renderOperatorView();
      if (state.currentPage === 'settings') renderSettings();
      restoreScrollPositions();
    }

    function openJobModal() {
      document.getElementById('jobModal').classList.add('open');
    }

    function closeJobModal() {
      document.getElementById('jobModal').classList.remove('open');
    }

    async function createJob() {
      const input = document.getElementById('newFiles');
      const files = Array.from(input.files || []);
      if (!files.length) {
        toast('Choose at least one file before starting a job.');
        return;
      }
      const payload = {
        name: document.getElementById('newName').value,
        product: document.getElementById('newProduct').value,
        work_area: document.getElementById('newArea').value,
        station: document.getElementById('newStation').value,
        profile: document.getElementById('newProfile').value,
        expected_step_count: document.getElementById('newExpectedSteps').value || null,
        focus_area: document.getElementById('newFocusArea').value || 'center',
        files: []
      };
      try {
        const res = await api('/api/jobs', {method: 'POST', body: JSON.stringify(payload)});
        const uploadResult = await uploadJobFiles(res.job.id, files);
        const uploadedCount = (uploadResult.added || []).length + (uploadResult.skipped || []).length;
        if (!uploadedCount) throw new Error('No files were accepted by the upload parser.');
        closeJobModal();
        await api(`/api/jobs/${res.job.id}/start`, {method: 'POST', body: '{}'});
        go('job', res.job.id);
      } catch (err) {
        toast(`Job upload failed: ${err.message}`);
        await loadData();
        render(false);
      }
    }

    async function saveSettings() {
      const payload = {
        openai_api_key: document.getElementById('settings-openai-key')?.value || '',
        transcription_backend: document.getElementById('settings-transcription-backend')?.value || 'auto',
        transcription_model: document.getElementById('settings-transcription-model')?.value || 'whisper-1',
        local_whisper_cli_path: document.getElementById('settings-local-whisper-cli')?.value || '',
        local_whisper_model_path: document.getElementById('settings-local-whisper-model')?.value || '',
        local_whisper_no_gpu: !!document.getElementById('settings-local-whisper-no-gpu')?.checked,
        transcription_enabled: !!document.getElementById('settings-transcription-enabled')?.checked,
        default_processing: document.getElementById('settings-default-processing')?.value || 'Standard'
      };
      const result = await api('/api/settings', {method: 'POST', body: JSON.stringify(payload)});
      state.settings = result.settings;
      toast('Settings saved.');
      render(false);
    }

    async function exportProcessingLog() {
      const res = await fetch('/api/export/processing-log');
      if (!res.ok) {
        toast('Export failed.');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = /filename="([^"]+)"/.exec(disposition);
      link.href = url;
      link.download = match ? match[1] : 'assembly_assistant_processing_log.json';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      try {
        const clearRes = await fetch('/api/export/processing-log/clear', {method: 'POST', body: '{}'});
        if (clearRes.ok) {
          await loadData();
          invalidateRender();
          render(false);
          toast('Processing log exported and cleared.');
          return;
        }
      } catch (err) {
        console.warn('Processing log clear failed', err);
      }
      toast('Processing log exported. Clear failed.');
    }

    async function uploadLargeJobFile(id, file) {
      const chunkSize = 16 * 1024 * 1024;
      const uploadId = `${Date.now()}-${Math.random().toString(16).slice(2)}-${file.name}`.replace(/[^A-Za-z0-9._-]+/g, '_');
      let offset = 0;
      let finalResult = {added: [], skipped: []};
      while (offset < file.size) {
        const end = Math.min(file.size, offset + chunkSize);
        const params = new URLSearchParams({
          upload_id: uploadId,
          name: file.name || 'upload.bin',
          type: file.type || 'application/octet-stream',
          size: String(file.size || 0),
          offset: String(offset),
          final: end >= file.size ? '1' : '0'
        });
        const res = await fetch(`/api/jobs/${id}/upload-chunk?${params.toString()}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/octet-stream'},
          body: file.slice(offset, end)
        });
        if (!res.ok) throw new Error(await res.text());
        const result = await res.json();
        if (result.complete) finalResult = result;
        offset = end;
      }
      return finalResult;
    }

    async function uploadJobFiles(id, files) {
      const fileList = Array.from(files || []);
      if (!fileList.length) return {added: [], skipped: []};
      const largeThreshold = 100 * 1024 * 1024;
      const smallFiles = fileList.filter(file => (file.size || 0) <= largeThreshold);
      const largeFiles = fileList.filter(file => (file.size || 0) > largeThreshold);
      const added = [];
      const skipped = [];
      if (largeFiles.length) toast(`Uploading ${largeFiles.length} large file${largeFiles.length === 1 ? '' : 's'} in chunks...`);
      if (smallFiles.length) {
        const form = new FormData();
        smallFiles.forEach(file => form.append('files', file));
        const result = await api(`/api/jobs/${id}/upload`, {method: 'POST', body: form});
        added.push(...(result.added || []));
        skipped.push(...(result.skipped || []));
      }
      for (const file of largeFiles) {
        const result = await uploadLargeJobFile(id, file);
        added.push(...(result.added || []));
        skipped.push(...(result.skipped || []));
      }
      return {added, skipped};
    }

    async function uploadQuestionFiles(id, files) {
      const fileList = Array.from(files || []);
      if (!fileList.length) return {added: [], skipped: []};
      const form = new FormData();
      fileList.forEach(file => form.append('files', file));
      return api(`/api/questions/${id}/upload`, {method: 'POST', body: form});
    }

    async function startJob(id) {
      let result = null;
      try {
        result = await api(`/api/jobs/${id}/start`, {method: 'POST', body: '{}'});
      } catch (err) {
        toast(`Processing did not start: ${err.message}`);
        await loadData();
        render();
        return;
      }
      if (!result.ok) {
        toast(result.error || 'Use Advanced to rerun this job.');
        render();
        return;
      }
      toast('Processing started.');
      render();
    }

    async function rerunJob(id) {
      try {
        await api(`/api/jobs/${id}/rerun`, {method: 'POST', body: '{}'});
        toast('Rerun started.');
        render();
      } catch (err) {
        toast(`Rerun did not start: ${err.message}`);
        await loadData();
        render();
      }
    }

    async function rerunStage(id, stage) {
      const result = await api(`/api/jobs/${id}/rerun-stage`, {
        method: 'POST',
        body: JSON.stringify({stage})
      });
      if (!result.ok) {
        toast('Stage rerun is not available right now.');
        render();
        return;
      }
      toast('Stage rerun started.');
      render();
    }

    async function resumeCurrentStage(id) {
      const job = state.currentJobData || state.jobs.find(item => item.id === id);
      const stage = job?.watchdog?.stage
        || stageKeys.find(key => job?.stages?.[key]?.status === 'failed')
        || stageKeys.find(key => job?.stages?.[key]?.status === 'running')
        || activeStageKey(job || {});
      const supported = {
        transcription: 'transcription',
        video: 'video',
        draft_sections: 'draft_sections'
      };
      if (supported[stage]) {
        await rerunStage(id, supported[stage]);
        return;
      }
      if (job?.draft_sections?.length) {
        await rerunStage(id, 'media_assignment');
        return;
      }
      await rerunJob(id);
    }

    async function completeJob(id) {
      await api(`/api/jobs/${id}/complete`, {method: 'POST', body: '{}'});
      toast('Sections moved to library.');
      go('sections');
    }

    async function deleteJob(id) {
      if (!confirm('Delete this job from the active job list? Stored media will be retained.')) return;
      await api(`/api/jobs/${id}/delete`, {method: 'POST', body: '{}'});
      toast('Job deleted.');
      go('jobs');
    }

    async function removeUpload(jobId, identity) {
      if (!confirm('Remove this upload from the processing set? Stored media will be retained.')) return;
      await api(`/api/jobs/${jobId}/remove-file`, {
        method: 'POST',
        body: JSON.stringify({identity})
      });
      toast('Upload removed. Run processing again when ready.');
      await loadData();
      invalidateRender();
      render(false);
    }

    function openAddFiles(id) {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.onchange = async () => {
        try {
          const result = await uploadJobFiles(id, input.files);
          toast(`${result.added.length} files added. ${result.skipped.length} duplicates skipped.`);
          await loadData();
          render();
        } catch (err) {
          toast(`Upload failed: ${err.message}`);
          await loadData();
          render();
        }
      };
      input.click();
    }

    function selectQuestionChoice(id, choice) {
      state.selectedChoices[id] = choice;
      state.questionDrafts[id] = choice;
      renderJob();
    }

    function captureQuestionFiles(id, fileList) {
      const files = Array.from(fileList || []);
      state.questionFileObjects[id] = files;
      state.questionFiles[id] = files.map(file => ({
        name: file.name,
        type: file.type || 'unknown',
        size: `${Math.round(file.size / 1024)} KB`
      }));
      renderJob();
    }

    async function answerQuestion(id, status) {
      const typed = document.getElementById(`answer-${id}`)?.value || '';
      const selected = state.selectedChoices[id] || '';
      const attachment = document.getElementById(`attach-${id}`)?.value || '';
      const uploadResult = await uploadQuestionFiles(id, state.questionFileObjects[id] || []);
      const uploadedFiles = [...(uploadResult.added || []), ...(uploadResult.skipped || [])];
      const attachmentFiles = uploadedFiles.length ? uploadedFiles : (state.questionFiles[id] || []);
      await api(`/api/questions/${id}/answer`, {
        method: 'POST',
        body: JSON.stringify({status, answer: typed || selected, attachment_note: attachment, attachment_files: attachmentFiles})
      });
      delete state.questionDrafts[id];
      delete state.questionAttachmentNotes[id];
      delete state.questionFiles[id];
      delete state.questionFileObjects[id];
      delete state.selectedChoices[id];
      toast(status === 'bypassed' ? 'Question bypassed.' : 'Answer saved.');
      render();
    }

    async function duplicateSection(id) {
      await api(`/api/sections/${id}/duplicate`, {method:'POST', body:'{}'});
      toast('Section duplicated.');
      render();
    }

    async function markSectionComplete(id) {
      const section = state.sections.find(item => item.id === id);
      const draft = cloneSection(section || {});
      ensureDraftStepIds(draft);
      const issues = sectionQaIssues(draft);
      if (issues.length) {
        state.currentSection = id;
        state.sectionDraftId = id;
        state.sectionDraft = draft;
        state.sectionStepIndexes[id] = issues[0].stepIndex;
        state.sectionQaFocus = issues[0].issueId;
        toast(`${issues.length} QA items must be resolved or bypassed.`);
        go('section', id);
        return;
      }
      try {
        await api(`/api/sections/${id}/complete`, {method:'POST', body:'{}'});
        toast('Section complete.');
        render();
      } catch (err) {
        toast('Section has unresolved QA items.');
        go('section', id);
      }
    }

    async function deleteSection(id) {
      const section = state.sections.find(item => item.id === id);
      const label = section?.title || 'this section';
      if (!confirm(`Delete ${label}? Packages already built from snapshots will not be changed.`)) return;
      await api(`/api/sections/${id}/delete`, {method:'POST', body:'{}'});
      delete state.sectionMediaOptions[id];
      if (state.currentSection === id) {
        state.currentSection = null;
        state.sectionDraft = null;
        state.sectionDraftId = null;
        await loadData();
        go('sections');
      } else {
        await loadData();
        toast('Section deleted.');
        render(false);
      }
    }

    async function saveSection(statusOverride = null) {
      const draft = collectSectionDraft();
      if (!draft) return;
      ensureDraftStepIds(draft);
      if (statusOverride === 'Complete') {
        const issues = sectionQaIssues(draft);
        if (issues.length) {
          draft.status = 'Needs Edit';
          state.sectionStepIndexes[draft.id] = issues[0].stepIndex;
          state.sectionQaFocus = issues[0].issueId;
          await api(`/api/sections/${draft.id}/update`, {
            method: 'POST',
            body: JSON.stringify({section: draft})
          });
          state.sectionDraft = cloneSection(draft);
          toast(`${issues.length} QA items must be resolved or bypassed.`);
          await loadData();
          render(false);
          return;
        }
      }
      if (statusOverride) draft.status = statusOverride;
      const result = await api(`/api/sections/${draft.id}/update`, {
        method: 'POST',
        body: JSON.stringify({section: draft})
      });
      if (result.blocked_qa && result.blocked_qa.length) {
        draft.status = 'Needs Edit';
        state.sectionStepIndexes[draft.id] = result.blocked_qa[0].step_index || 0;
        state.sectionQaFocus = result.blocked_qa[0].issue_id || null;
        toast(`${result.blocked_qa.length} QA items must be resolved or bypassed.`);
      } else {
        toast(statusOverride === 'Complete' ? 'Section completed.' : 'Section saved.');
      }
      state.sectionDraft = cloneSection(result.section || draft);
      await loadData();
      render(false);
    }

    function selectEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const next = Math.min(Math.max(Number(idx) || 0, 0), draft.steps.length - 1);
      state.sectionStepIndexes[draft.id] = next;
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
      autosizeTextareas();
    }

    function addEditorStep() {
      const draft = collectSectionDraft();
      if (!draft) return;
      draft.steps.push(blankStep());
      state.sectionStepIndexes[draft.id] = draft.steps.length - 1;
      renderSectionEditor();
      autosizeTextareas();
    }

    function removeEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft || draft.steps.length <= 1) {
        toast('A section needs at least one step.');
        return;
      }
      draft.steps.splice(idx, 1);
      state.sectionStepIndexes[draft.id] = Math.min(idx, draft.steps.length - 1);
      renderSectionEditor();
      autosizeTextareas();
    }

    function duplicateEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const clone = cloneSection(draft.steps[idx]);
      clone.step_id = newStepId();
      clone.qa_resolutions = {};
      draft.steps.splice(idx + 1, 0, clone);
      state.sectionStepIndexes[draft.id] = idx + 1;
      renderSectionEditor();
      autosizeTextareas();
    }

    function moveEditorStep(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const next = idx + dir;
      if (next < 0 || next >= draft.steps.length) return;
      const [item] = draft.steps.splice(idx, 1);
      draft.steps.splice(next, 0, item);
      state.sectionStepIndexes[draft.id] = next;
      renderSectionEditor();
      autosizeTextareas();
    }

    function nonEmptyValue(value) {
      const text = String(value || '').trim();
      if (!text || ['none', 'not required', 'n/a', 'na'].includes(text.toLowerCase())) return '';
      return text;
    }

    function combineTorque(first, second) {
      return nonEmptyValue(first.torque) || nonEmptyValue(second.torque) || first.torque || second.torque || '';
    }

    function mergeStepTiming(combined, first, second) {
      const numberValues = values => values.filter(value => value !== null && value !== undefined && value !== '').map(Number).filter(Number.isFinite);
      const starts = numberValues([first.start_time, second.start_time]);
      const ends = numberValues([first.end_time, second.end_time]);
      const targets = numberValues([first.target_time, second.target_time]);
      if (starts.length) combined.start_time = Math.min(...starts);
      if (ends.length) combined.end_time = Math.max(...ends);
      if (targets.length) combined.target_time = Math.max(...targets);
      combined.timing_confidence = first.timing_confidence || second.timing_confidence;
      combined.timing_source = first.timing_source || second.timing_source;
    }

    function combineSteps(first, second) {
      ensureStepIdentity(first);
      ensureStepIdentity(second);
      const combined = cloneSection(first);
      combined.text = combineTextValues([first.text, second.text], ' ');
      combined.tools = combineTextValues([first.tools, second.tools], ', ');
      combined.torque = combineTorque(first, second);
      combined.quality = combineTextValues([first.quality, second.quality], ' ');
      combined.safety = combineTextValues([first.safety, second.safety], ' ');
      combined.parts = uniqueList([...(first.parts || []), ...(second.parts || [])]);
      combined.media = uniqueList([...(first.media || []), ...(second.media || [])]);
      combined.media_settings = normalizeStepMediaSettings({
        media: combined.media,
        media_settings: {
          ...(first.media_settings || {}),
          ...(second.media_settings || {})
        }
      });
      combined.media_candidates = [...(first.media_candidates || []), ...(second.media_candidates || [])].slice(0, 12);
      combined.qa_resolutions = {...(first.qa_resolutions || {}), ...(second.qa_resolutions || {})};
      delete combined.qa_resolutions.merge_candidate;
      combined.merge_candidate = false;
      combined.merge_recommendation = null;
      const confidenceRank = {manual: 5, high: 4, medium: 3, low: 2, fallback: 1, missing: 0};
      combined.media_match_confidence = [first.media_match_confidence, second.media_match_confidence].sort((a, b) => (confidenceRank[b] || 0) - (confidenceRank[a] || 0))[0] || '';
      combined.media_match_score = Math.max(Number(first.media_match_score || 0), Number(second.media_match_score || 0));
      combined.media_fallback = combined.media_match_confidence === 'fallback' || (first.media_fallback && second.media_fallback && combined.media.length <= 1);
      if (!combined.media_fallback) combined.media_fallback_reason = '';
      mergeStepTiming(combined, first, second);
      return combined;
    }

    function mergeEditorStep(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const targetIndex = dir < 0 ? idx - 1 : idx;
      const sourceIndex = dir < 0 ? idx : idx + 1;
      if (targetIndex < 0 || sourceIndex >= draft.steps.length) return;
      const combined = combineSteps(draft.steps[targetIndex], draft.steps[sourceIndex]);
      draft.steps.splice(targetIndex, 2, combined);
      state.sectionStepIndexes[draft.id] = targetIndex;
      state.sectionQaFocus = null;
      renderSectionEditor();
      autosizeTextareas();
      toast('Steps merged.');
    }

    function splitPointForText(text, cursor) {
      const cleaned = String(text || '');
      if (cursor && cursor > 8 && cursor < cleaned.length - 8) return cursor;
      const midpoint = Math.floor(cleaned.length / 2);
      const matches = [...cleaned.matchAll(/[.!?]\s+/g)].map(match => match.index + match[0].length);
      if (matches.length) return matches.sort((a, b) => Math.abs(a - midpoint) - Math.abs(b - midpoint))[0];
      const before = cleaned.lastIndexOf(' ', midpoint);
      const after = cleaned.indexOf(' ', midpoint);
      if (before > 8) return before;
      if (after > 8 && after < cleaned.length - 8) return after;
      return midpoint;
    }

    function splitEditorStep(idx) {
      const textarea = document.getElementById('step-text');
      const cursor = textarea ? textarea.selectionStart : null;
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[idx];
      ensureStepIdentity(step);
      const text = String(step.text || '');
      if (text.trim().length < 18) {
        toast('Step text is too short to split cleanly.');
        return;
      }
      const splitAt = splitPointForText(text, cursor);
      const firstText = text.slice(0, splitAt).trim();
      const secondText = text.slice(splitAt).trim();
      if (!firstText || !secondText) {
        toast('Could not find a clean split point.');
        return;
      }
      const media = normalizeMediaList(step.media);
      const nextStep = {
        ...blankStep(),
        text: secondText,
        tools: step.tools || '',
        torque: step.torque || '',
        media: media.slice(1),
        media_settings: normalizeStepMediaSettings({media: media.slice(1), media_settings: step.media_settings || {}}),
        timing_source: step.timing_source,
        timing_confidence: step.timing_confidence,
        source_step_id: step.source_step_id,
        raw_transcript: step.raw_transcript,
        boundary_reason: 'manual split',
        transcript_confidence: step.transcript_confidence
      };
      step.text = firstText;
      step.media = media.slice(0, 1);
      step.media_settings = normalizeStepMediaSettings(step);
      step.qa_resolutions = {};
      step.merge_candidate = false;
      step.merge_recommendation = null;
      if (Number.isFinite(Number(step.start_time)) && Number.isFinite(Number(step.end_time)) && Number(step.end_time) > Number(step.start_time)) {
        const midpoint = Number(step.start_time) + (Number(step.end_time) - Number(step.start_time)) / 2;
        nextStep.start_time = Number(midpoint.toFixed(2));
        nextStep.end_time = step.end_time;
        nextStep.target_time = nextStep.end_time;
        step.end_time = Number(midpoint.toFixed(2));
        step.target_time = step.end_time;
      }
      draft.steps.splice(idx + 1, 0, nextStep);
      state.sectionStepIndexes[draft.id] = idx + 1;
      state.sectionQaFocus = null;
      renderSectionEditor();
      autosizeTextareas();
      toast('Step split.');
    }

    async function createPackageDraft(sectionIds = []) {
      const firstSection = state.sections.find(section => section.id === sectionIds[0]) || {};
      const payload = {
        title: firstSection.product ? `${firstSection.product} Instructions` : 'Instruction Package',
        product: firstSection.product || '',
        work_area: firstSection.work_area || '',
        station: firstSection.station || '',
        status: 'Draft',
        section_ids: sectionIds
      };
      const result = await api('/api/packages', {method:'POST', body: JSON.stringify(payload)});
      state.currentPackageData = result.package;
      state.packageDraftId = result.package.id;
      state.packageDraft = [...(result.package.section_ids || [])];
      state.packageDraftMeta = {
        title: result.package.title || '',
        product: result.package.product || '',
        work_area: result.package.work_area || '',
        station: result.package.station || '',
        status: result.package.status || 'Draft'
      };
      toast('Package draft created.');
      await loadData();
      go('package_edit', result.package.id);
    }

    function newPackageDraft() {
      createPackageDraft();
    }

    async function loadPackageDraft(id) {
      const result = await api(`/api/packages/${id}`);
      const packageData = result.package;
      state.currentPackageData = packageData;
      state.packageDraftId = packageData.id;
      state.packageDraft = [...(packageData.section_ids || [])];
      state.packageDraftMeta = {
        title: packageData.title || '',
        product: packageData.product || '',
        work_area: packageData.work_area || '',
        station: packageData.station || '',
        status: packageData.status || 'Draft'
      };
      go('package_edit', packageData.id);
    }

    function togglePackageSectionPreview(id) {
      collectPackageDraftMeta();
      state.expandedPackageSections[id] = !state.expandedPackageSections[id];
      renderPackageEditor();
    }

    async function addToPackage(id) {
      const section = state.sections.find(s => s.id === id);
      if (!section || section.status !== 'Complete') {
        toast('Only complete sections can be added to packages.');
        return;
      }
      if (state.currentPage !== 'package_edit' || !state.packageDraftId) {
        await createPackageDraft([id]);
        return;
      }
      collectPackageDraftMeta();
      if (state.packageDraft.includes(id)) {
        toast('Section is already in this package.');
        renderPackageEditor();
        return;
      }
      state.packageDraft.push(id);
      if (!state.packageDraftMeta.product) state.packageDraftMeta.product = section.product || '';
      if (!state.packageDraftMeta.work_area) state.packageDraftMeta.work_area = section.work_area || '';
      if (!state.packageDraftMeta.station) state.packageDraftMeta.station = section.station || '';
      if (!state.packageDraftMeta.title) state.packageDraftMeta.title = section.product ? `${section.product} Instructions` : 'Instruction Package';
      renderPackageEditor();
    }

    function removePackage(idx) {
      collectPackageDraftMeta();
      state.packageDraft.splice(idx, 1);
      renderPackageEditor();
    }

    function movePackage(idx, dir) {
      collectPackageDraftMeta();
      const next = idx + dir;
      if (next < 0 || next >= state.packageDraft.length) return;
      const [item] = state.packageDraft.splice(idx, 1);
      state.packageDraft.splice(next, 0, item);
      renderPackageEditor();
    }

    async function savePackage(next = 'edit') {
      const meta = collectPackageDraftMeta();
      const payload = {
        ...meta,
        title: meta.title || 'Instruction Package',
        section_ids: state.packageDraft
      };
      const result = state.packageDraftId
        ? await api(`/api/packages/${state.packageDraftId}/update`, {method:'POST', body: JSON.stringify(payload)})
        : await api('/api/packages', {method:'POST', body: JSON.stringify(payload)});
      state.packageDraftId = result.package.id;
      toast('Package saved.');
      await loadData();
      if (next === 'preview') {
        go('package_view', result.package.id);
      } else if (next === 'print') {
        go('package_print', result.package.id);
      } else {
        go('package_edit', result.package.id);
      }
    }

    async function snapshotPackage(id) {
      await api(`/api/packages/${id}/snapshot`, {method:'POST', body:'{}'});
      toast('Package snapshot refreshed from current sections.');
      await loadData();
      render(false);
    }

    async function deletePackage(id) {
      if (!confirm('Delete this package? Sections and media will be retained.')) return;
      await api(`/api/packages/${id}/delete`, {method:'POST', body:'{}'});
      toast('Package deleted.');
      state.packageDraftId = state.packageDraftId === id ? null : state.packageDraftId;
      if (state.currentPackage === id || state.currentPage === 'package_view' || state.currentPage === 'package_edit' || state.currentPage === 'package_print') {
        go('package');
      } else {
        await loadData();
        render(false);
      }
    }

    document.addEventListener('scroll', () => {
      state.lastScrollAt = Date.now();
    }, true);

    window.addEventListener('hashchange', () => {
      invalidateRender();
      render();
    });
    seed().catch(err => {
      const root = document.getElementById('root');
      if (root) {
        root.innerHTML = `<div class="empty">App failed to load: ${esc(err.message || err)}</div>`;
      }
      console.error(err);
    });

    async function seed() {
      readHash();
      await loadData();
      startPolling();
      render(false);
    }
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_download_json(self, payload, filename):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_storage_file(self, rel_path):
        try:
            target = (APP_DIR / rel_path).resolve()
            storage_root = STORAGE_DIR.resolve()
            if not str(target).startswith(str(storage_root) + os.sep):
                self.send_json({"error": "Forbidden"}, 403)
                return
            if not target.exists() or not target.is_file():
                self.send_json({"error": "File not found"}, 404)
                return
            body = target.read_bytes()
            mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path in ["/jobs", "/sections", "/package", "/settings"] or re.fullmatch(r"/package/[^/]+(?:/(print|operator|edit))?", path):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/media":
            rel_path = (parse_qs(parsed.query).get("path") or [""])[0]
            self.send_storage_file(rel_path)
            return

        if path == "/api/jobs":
            watch_processing_jobs()
            with STATE_LOCK:
                self.send_json(
                    {
                        "jobs": [
                            job_summary(job)
                            for job in JOBS.values()
                            if not job.get("deleted")
                        ]
                    }
                )
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)", path)
        if match:
            job_id = match.group(1)
            watch_processing_jobs()
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("deleted"):
                    self.send_json({"error": "Job not found"}, 404)
                    return
                payload = dict(job)
                payload["questions_pending"] = len(unresolved_questions(job))
                payload["processing_started"] = has_processing_run(job)
                payload["needs_processing"] = bool(job.get("needs_processing"))
                payload["processing_health"] = job_processing_health(job)
                self.send_json({"job": payload})
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/transcripts", path)
        if match:
            job_id = match.group(1)
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("deleted"):
                    self.send_json({"error": "Job not found"}, 404)
                    return
                self.send_json({"transcripts": transcript_results(job)})
            return

        if path == "/api/sections":
            with STATE_LOCK:
                self.send_json(
                    {
                        "sections": [
                            section
                            for section in SECTIONS.values()
                            if not section.get("deleted")
                        ]
                    }
                )
            return

        match = re.fullmatch(r"/api/sections/([^/]+)/media-options", path)
        if match:
            section_id = match.group(1)
            with STATE_LOCK:
                section = SECTIONS.get(section_id)
                if not section or section.get("deleted"):
                    self.send_json({"error": "Section not found"}, 404)
                    return
                self.send_json({"media": section_media_options(section)})
            return

        if path == "/api/packages":
            with STATE_LOCK:
                self.send_json(
                    {
                        "packages": [
                            package_summary(package)
                            for package in PACKAGES.values()
                            if not package.get("deleted")
                        ]
                    }
                )
            return

        match = re.fullmatch(r"/api/packages/([^/]+)", path)
        if match:
            package_id = match.group(1)
            with STATE_LOCK:
                package = PACKAGES.get(package_id)
                if not package or package.get("deleted"):
                    self.send_json({"error": "Package not found"}, 404)
                    return
                self.send_json({"package": package})
            return

        if path == "/api/settings":
            with STATE_LOCK:
                self.send_json({"settings": public_settings()})
            return

        if path == "/api/export/processing-log":
            with STATE_LOCK:
                self.send_download_json(
                    processing_export(),
                    f"assembly_assistant_processing_log_{time.strftime('%Y%m%d_%H%M%S')}.json",
                )
            return

        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/export/processing-log/clear":
            with STATE_LOCK:
                clear_processing_runtime_logs()
                self.send_json({"ok": True, "cleared_at": SETTINGS.get("last_processing_log_exported_at")})
            return

        if path == "/api/jobs":
            data = self.read_json()
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
                self.send_json({"job": job_summary(job)})
            return

        if path == "/api/settings":
            data = self.read_json()
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
                self.send_json({"ok": True, "settings": public_settings()})
            return

        if path == "/api/packages":
            data = self.read_json()
            with STATE_LOCK:
                package = make_package(data)
                save_state()
                self.send_json({"ok": True, "package": package})
            return

        match = re.fullmatch(r"/api/packages/([^/]+)/(update|snapshot|delete)", path)
        if match:
            package_id, action = match.groups()
            with STATE_LOCK:
                package = PACKAGES.get(package_id)
                if not package or package.get("deleted"):
                    self.send_json({"error": "Package not found"}, 404)
                    return
                if action == "update":
                    package = update_package(package, self.read_json())
                    save_state()
                    self.send_json({"ok": True, "package": package})
                    return
                if action == "snapshot":
                    package = refresh_package_snapshot(package)
                    save_state()
                    self.send_json({"ok": True, "package": package})
                    return
                if action == "delete":
                    package["deleted"] = True
                    package["status"] = "Deleted"
                    package["updated_at"] = datestamp()
                    save_state()
                    self.send_json({"ok": True})
                    return

        match = re.fullmatch(r"/api/sections/([^/]+)/upload-media", path)
        if match:
            section_id = match.group(1)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            uploads = parse_multipart_files(self.headers.get("Content-Type", ""), body)
            with STATE_LOCK:
                section = SECTIONS.get(section_id)
                if not section or section.get("deleted"):
                    self.send_json({"error": "Section not found"}, 404)
                    return
                added, skipped = store_section_media(section, uploads)
                self.send_json({"ok": True, "added": added, "skipped": skipped})
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/upload-chunk", path)
        if match:
            job_id = match.group(1)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            params = parse_qs(parsed.query)
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self.send_json({"error": "Job not found"}, 404)
                    return
                payload, status = store_chunked_job_upload(job, params, body)
                if status < 400:
                    save_state()
                self.send_json(payload, status)
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/upload", path)
        if match:
            job_id = match.group(1)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            uploads = parse_multipart_files(self.headers.get("Content-Type", ""), body)
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self.send_json({"error": "Job not found"}, 404)
                    return
                if not uploads:
                    add_log(job, "Upload failed: no files were received or parsed from the request.")
                    self.send_json({"error": "No files were received by the upload endpoint."}, 400)
                    return
                added, skipped = store_uploaded_files(job, uploads)
                if not added and not skipped:
                    add_log(job, "Upload failed: files were received but none were accepted.")
                    self.send_json({"error": "No uploaded files were accepted."}, 400)
                    return
                self.send_json({"ok": True, "added": added, "skipped": skipped})
            return

        match = re.fullmatch(r"/api/questions/([^/]+)/upload", path)
        if match:
            qid = match.group(1)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            uploads = parse_multipart_files(self.headers.get("Content-Type", ""), body)
            with STATE_LOCK:
                target_job, target_question = find_question(qid)
                if not target_question:
                    self.send_json({"error": "Question not found"}, 404)
                    return
                added, skipped = store_uploaded_files(
                    target_job,
                    uploads,
                    source="question clarification",
                    question_id=qid,
                    clarification_for=target_question["title"],
                )
                target_question["attachment_files"] = added + skipped
            self.send_json({"ok": True, "added": added, "skipped": skipped})
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/rerun-stage", path)
        if match:
            job_id = match.group(1)
            data = self.read_json()
            stage_key = data.get("stage") or ""
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("deleted"):
                    self.send_json({"error": "Job not found"}, 404)
                    return
            ok = rerun_job_stage(job_id, stage_key)
            self.send_json({"ok": ok})
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/(start|rerun|complete|delete|files|remove-file)", path)
        if match:
            job_id, action = match.groups()
            with STATE_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("deleted"):
                    self.send_json({"error": "Job not found"}, 404)
                    return

            if action == "start":
                with STATE_LOCK:
                    if not active_job_files(job):
                        add_log(job, "Processing blocked: no uploaded files are attached.")
                        self.send_json({"ok": False, "error": "Upload at least one file before starting processing."}, 400)
                        return
                ok = start_processing(job_id, reset=True, first_run_only=True)
                self.send_json({"ok": ok})
                return

            if action == "rerun":
                with STATE_LOCK:
                    if not active_job_files(job):
                        add_log(job, "Rerun blocked: no uploaded files are attached.")
                        self.send_json({"ok": False, "error": "Upload at least one file before rerunning processing."}, 400)
                        return
                ok = start_processing(job_id, reset=True)
                self.send_json({"ok": ok})
                return

            if action == "complete":
                with STATE_LOCK:
                    created = promote_job_sections(job)
                    job["status"] = "Complete"
                    job["active"] = False
                    job["flags"] = []
                    add_log(job, f"Job complete. {len(created)} sections moved to library.")
                    self.send_json({"ok": True, "section_ids": created})
                return

            if action == "delete":
                with STATE_LOCK:
                    if job_id in RUNNING_THREADS and RUNNING_THREADS[job_id].is_alive():
                        self.send_json({"error": "Cannot delete while processing"}, 409)
                        return
                    job["deleted"] = True
                    job["active"] = False
                    job["status"] = "Deleted"
                    add_log(job, "Job deleted from active job list. Stored media was retained.")
                    save_state()
                    self.send_json({"ok": True})
                return

            if action == "files":
                data = self.read_json()
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
                    self.send_json({"ok": True, "added": added, "skipped": skipped})
                return

            if action == "remove-file":
                data = self.read_json()
                identity = data.get("identity") or data.get("sha256") or data.get("stored_path") or ""
                with STATE_LOCK:
                    removed = remove_job_upload(job, identity)
                    if not removed:
                        self.send_json({"error": "File not found"}, 404)
                        return
                    self.send_json({"ok": True, "removed": removed})
                return

        match = re.fullmatch(r"/api/questions/([^/]+)/answer", path)
        if match:
            qid = match.group(1)
            data = self.read_json()
            with STATE_LOCK:
                target_job, target_question = find_question(qid)
                if not target_question:
                    self.send_json({"error": "Question not found"}, 404)
                    return
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
            self.send_json({"ok": True})
            return

        match = re.fullmatch(r"/api/sections/([^/]+)/(duplicate|complete|update|delete)", path)
        if match:
            section_id, action = match.groups()
            with STATE_LOCK:
                section = SECTIONS.get(section_id)
                if not section or (section.get("deleted") and action != "delete"):
                    self.send_json({"error": "Section not found"}, 404)
                    return
                if action == "delete":
                    section["deleted"] = True
                    section["status"] = "Deleted"
                    section["updated_at"] = datestamp()
                    save_state()
                    self.send_json({"ok": True})
                    return
                if action == "duplicate":
                    new_id = short_id("sec")
                    clone = json.loads(json.dumps(section))
                    clone["id"] = new_id
                    clone["title"] = f"{section['title']} Copy"
                    clone["status"] = "Needs Edit"
                    clone.pop("deleted", None)
                    SECTIONS[new_id] = clone
                    save_state()
                    self.send_json({"ok": True, "section": clone})
                    return
                if action == "complete":
                    unresolved = section_qa_issues(section)
                    if unresolved:
                        self.send_json({"error": "Section has unresolved QA items", "qa_issues": unresolved[:12]}, 409)
                        return
                    section["status"] = "Complete"
                    save_state()
                    self.send_json({"ok": True, "section": section})
                    return
                if action == "update":
                    data = self.read_json()
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
                    self.send_json({"ok": True, "section": section, "blocked_qa": unresolved[:12]})
                    return

        self.send_json({"error": "Not found"}, 404)


def main():
    init_db()
    if load_state():
        print(f"Loaded persisted data from {DB_PATH}")
    else:
        initialize_empty_state()
        print(f"Initialized empty data store at {DB_PATH}")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Assembly Assistant running at http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
