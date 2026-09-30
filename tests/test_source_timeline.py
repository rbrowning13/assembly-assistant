import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assembly_assistant import core


class SourceTimelineTests(unittest.TestCase):
    def make_job(self):
        return {
            "id": "job-test",
            "files": [
                {"name": "overhead.mov", "type": "video/quicktime", "sha256": "a" * 64},
                {"name": "left.mov", "type": "video/quicktime", "sha256": "b" * 64},
                {"name": "manual.pdf", "type": "application/pdf", "sha256": "c" * 64},
            ],
        }

    def test_source_defaults_select_one_reference_and_narration(self):
        job = self.make_job()

        core.normalize_job_sources(job)

        overhead, left, manual = job["files"]
        self.assertEqual(overhead["source_role"], "primary_video")
        self.assertTrue(overhead["timeline_reference"])
        self.assertTrue(overhead["use_for_drafting"])
        self.assertEqual(left["source_role"], "alternate_angle")
        self.assertFalse(left["timeline_reference"])
        self.assertFalse(left["use_for_drafting"])
        self.assertEqual(manual["source_role"], "supporting_document")
        self.assertEqual(manual["authority_level"], "supporting_context")

    def test_manual_two_anchor_transform_corrects_drift(self):
        source = {
            "timeline_sync": {
                "mode": "manual",
                "start_source_seconds": 10,
                "start_timeline_seconds": 0,
                "end_source_seconds": 110,
                "end_timeline_seconds": 99,
            }
        }

        transform = core.source_timeline_transform(source)

        self.assertAlmostEqual(transform["scale"], 0.99)
        self.assertAlmostEqual(core.apply_source_timeline(60, source), 49.5)
        self.assertTrue(transform["has_drift_anchor"])

    def test_manual_one_anchor_applies_fixed_offset(self):
        source = {
            "timeline_sync": {
                "mode": "manual",
                "start_source_seconds": 12.5,
                "start_timeline_seconds": 2.0,
                "end_source_seconds": None,
                "end_timeline_seconds": None,
            }
        }

        self.assertAlmostEqual(core.apply_source_timeline(20, source), 9.5)

    def test_transcript_word_matching_estimates_source_offset(self):
        tokens = [f"instruction{index}" for index in range(30)]
        reference = {
            "words": [
                {"word": token, "start": 15 + index * 0.5}
                for index, token in enumerate(tokens)
            ]
        }
        candidate = {
            "words": [
                {"word": token, "start": 3 + index * 0.5}
                for index, token in enumerate(tokens)
            ]
        }

        result = core.estimate_transcript_offset(reference, candidate)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["offset_seconds"], 12.0)

    def test_only_selected_transcript_is_used_for_drafting(self):
        payloads = [
            {"text": "primary narration", "_use_for_drafting": True},
            {"text": "duplicate alternate narration", "_use_for_drafting": False},
        ]
        with patch.object(core, "transcript_payloads", return_value=payloads):
            selected = core.drafting_transcript_payloads({"files": []})

        self.assertEqual([item["text"] for item in selected], ["primary narration"])

    def test_payload_timeline_metadata_applies_manual_drift(self):
        payload = {
            "_timeline_sync": {
                "mode": "manual",
                "start_source_seconds": 5,
                "start_timeline_seconds": 1,
                "end_source_seconds": 105,
                "end_timeline_seconds": 100,
            }
        }

        self.assertAlmostEqual(core.apply_source_timeline(55, payload), 50.5)

    def test_document_context_preserves_source_authority(self):
        job = {
            "document_evidence": [
                {
                    "source_file": "service_manual.pdf",
                    "authority_level": "authoritative",
                    "revision": "C",
                    "page_number": 12,
                    "text": "Torque the mounting fastener to the released specification.",
                }
            ]
        }

        context = core.document_context(job)

        self.assertIn("service_manual.pdf", context)
        self.assertIn("authority: authoritative", context)
        self.assertIn("revision: C", context)
        self.assertIn("page: 12", context)

    def test_text_document_creates_traceable_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "setup.txt"
            source.write_text("Verify the guard is installed before operation.", encoding="utf-8")
            file = {
                "name": "setup.txt",
                "source_id": "src-text",
                "source_role": "supporting_document",
                "authority_level": "authoritative",
                "revision": "2",
                "applicability": "Model A",
                "sha256": "d" * 64,
            }
            (root / "ocr").mkdir()
            with patch.object(core, "APP_DIR", root), patch.object(
                core,
                "ensure_job_dirs",
                return_value={"ocr": root / "ocr"},
            ):
                records, error = core.process_text_document({"id": "job-test"}, file, source)

            self.assertIsNone(error)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["authority_level"], "authoritative")
            self.assertIn("guard is installed", records[0]["text"])
            self.assertTrue(file["ocr_path"].endswith("document.json"))

    def test_invalid_manual_sync_does_not_change_source(self):
        job = self.make_job()
        core.normalize_job_sources(job)
        target = job["files"][1]
        original_sync = dict(target["timeline_sync"])

        updated, error = core.update_job_source(
            job,
            target["source_id"],
            {
                "source_role": "alternate_angle",
                "authority_level": "primary_evidence",
                "sync_mode": "manual",
                "start_source_seconds": "",
                "start_timeline_seconds": "",
            },
        )

        self.assertIsNone(updated)
        self.assertIn("valid start anchor", error)
        self.assertEqual(target["timeline_sync"], original_sync)


if __name__ == "__main__":
    unittest.main()
