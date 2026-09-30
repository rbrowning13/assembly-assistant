import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from assembly_assistant import core
from assembly_assistant.sync import estimate_envelope_offset, estimate_waveform_alignment
from tools.generate_sync_fixtures import generate_fixtures


class ProcessingFoundationTests(unittest.TestCase):
    def test_envelope_alignment_recovers_known_offset(self):
        rng = np.random.default_rng(42)
        master = rng.normal(size=1600).astype(np.float32)
        reference = master[:1200]
        candidate = master[140:1340]

        result = estimate_envelope_offset(reference, candidate, 40, max_offset_seconds=10)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["offset_seconds"], 3.5, delta=0.05)
        self.assertEqual(result["confidence"], "high")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools unavailable")
    def test_generated_multisource_fixture_meets_alignment_and_orientation_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = generate_fixtures(root, duration=10.0, offset=1.75, scale=1.0)

            alignment = estimate_waveform_alignment(
                root / manifest["reference"]["path"],
                root / manifest["alternate"]["path"],
            )
            self.assertEqual(alignment["confidence"], "high")
            self.assertAlmostEqual(alignment["offset_seconds"], 1.75, delta=0.5)

            probe = subprocess.run(
                [
                    shutil.which("ffprobe"),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(root / manifest["alternate"]["path"]),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertGreater(stream["height"], stream["width"])
            self.assertTrue(manifest["scripted_steps"])

    def test_source_alignment_prefers_waveform_evidence(self):
        job = {
            "id": "job-sync",
            "files": [
                {"name": "reference.mov", "type": "video/quicktime", "sha256": "a" * 64},
                {"name": "alternate.mov", "type": "video/quicktime", "sha256": "b" * 64},
            ],
        }
        core.normalize_job_sources(job)
        with patch.object(core, "alignment_audio_path", return_value=Path("audio.wav")), patch.object(
            core,
            "waveform_backend_available",
            return_value=True,
        ), patch.object(
            core,
            "estimate_waveform_alignment",
            return_value={
                "method": "audio_waveform",
                "offset_seconds": 2.25,
                "scale": 1.0,
                "correlation": 0.81,
                "peak_separation": 0.2,
                "confidence": "high",
            },
        ):
            alignments = core.align_transcript_timelines(job)

        self.assertEqual(len(alignments), 1)
        self.assertEqual(job["files"][1]["timeline_alignment"]["method"], "audio_waveform")
        self.assertEqual(job["files"][1]["timeline_offset_seconds"], 2.25)

    def test_processing_manifest_is_versioned_redacted_and_measured(self):
        job = {
            "id": "job-run",
            "profile": "Standard",
            "focus_area": "center",
            "files": [
                {
                    "name": "reference.mov",
                    "type": "video/quicktime",
                    "sha256": "a" * 64,
                    "bytes": 1024,
                }
            ],
            "stages": core.base_stages(),
            "document_evidence": [{"text": "released note"}],
            "draft_sections": [{"title": "Section", "steps": [{"instruction": "Install part."}]}],
            "questions": [],
            "frame_analysis": [{"stored_path": "storage/frame.jpg"}],
            "processing_cache": {"video": {"candidate_keyframes": 4}},
            "draft_validation": {"issue_count": 0},
        }
        with patch.dict(core.SETTINGS, {"openai_api_key": "secret-value"}, clear=False):
            core.start_job_processing_timing(job)
            core.finish_job_processing_timing(job)

        run = job["processing_runs"][-1]
        encoded = json.dumps(run)
        self.assertEqual(run["manifest_version"], core.PROCESSING_MANIFEST_VERSION)
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["metrics"]["steps"], 1)
        self.assertIn("profile_version", run["engine"])
        self.assertNotIn("secret-value", encoded)
        self.assertEqual(job["active_processing_run_id"], "")


if __name__ == "__main__":
    unittest.main()
