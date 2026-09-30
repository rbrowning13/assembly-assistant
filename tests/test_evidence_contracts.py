import unittest
from pathlib import Path
from unittest.mock import patch

from assembly_assistant import core
from assembly_assistant.contracts import (
    CONTRACT_VERSION,
    applicability_review,
    detect_conflicts,
    evidence_item,
    source_contract,
)


class EvidenceContractTests(unittest.TestCase):
    def source(self, source_id, name, authority="authoritative", revision="A", applicability="Model X"):
        return source_contract(
            {
                "source_id": source_id,
                "name": name,
                "kind": "text",
                "type": "text/plain",
                "sha256": source_id * 8,
                "bytes": 100,
                "source_role": "supporting_document",
                "authority_level": authority,
                "revision": revision,
                "applicability": applicability,
            }
        )

    def test_source_contract_is_neutral_and_versioned(self):
        source = self.source("src-a", "manual-a.txt")

        self.assertEqual(source["contract"], "Source")
        self.assertEqual(source["contract_version"], CONTRACT_VERSION)
        self.assertEqual(source["authority_level"], "authoritative")
        self.assertIn("timeline", source)
        self.assertIn("governance", source)

    def test_authoritative_source_without_revision_requires_review(self):
        review = applicability_review("Model X", "Model X", "authoritative", "")

        self.assertEqual(review["status"], "review_required")
        self.assertTrue(any("revision" in reason.lower() for reason in review["reasons"]))

    def test_conflicting_torque_values_are_detected_without_selecting_winner(self):
        first = evidence_item(
            evidence_type="document_page",
            source=self.source("src-a", "manual-a.txt"),
            locator={"page_number": 1},
            text="Tighten the mounting bolt to 25 N-m.",
            extraction_method="text_file",
            product="Model X",
        )
        second = evidence_item(
            evidence_type="document_page",
            source=self.source("src-b", "manual-b.txt", authority="supporting_context"),
            locator={"page_number": 2},
            text="Tighten the mounting bolt to 30 N-m.",
            extraction_method="text_file",
            product="Model X",
        )

        conflicts, facts = detect_conflicts([first, second])

        self.assertEqual(len(facts), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["fact_type"], "torque")
        self.assertEqual(conflicts[0]["status"], "open")
        self.assertTrue(conflicts[0]["requires_user_review"])
        self.assertNotIn("selected_value", conflicts[0])

    def test_different_fastener_subjects_do_not_conflict(self):
        source_a = self.source("src-a", "manual-a.txt")
        source_b = self.source("src-b", "manual-b.txt")
        first = evidence_item(
            evidence_type="document_page",
            source=source_a,
            locator={"page_number": 1},
            text="Tighten the mounting bolt to 25 N-m.",
            product="Model X",
        )
        second = evidence_item(
            evidence_type="document_page",
            source=source_b,
            locator={"page_number": 1},
            text="Tighten the drain plug to 30 N-m.",
            product="Model X",
        )

        conflicts, _facts = detect_conflicts([first, second])

        self.assertEqual(conflicts, [])

    def test_job_evidence_refresh_is_stable_and_creates_conflict_question(self):
        job = {
            "id": "job-evidence",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-a",
                    "name": "manual-a.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "a" * 64,
                    "authority_level": "authoritative",
                    "revision": "A",
                    "applicability": "Model X",
                },
                {
                    "source_id": "src-b",
                    "name": "manual-b.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "b" * 64,
                    "authority_level": "supporting_context",
                    "revision": "draft",
                    "applicability": "Model X",
                },
            ],
            "document_evidence": [
                {
                    "source_id": "src-a",
                    "source_file": "manual-a.txt",
                    "page_number": 1,
                    "text": "Tighten the mounting bolt to 25 N-m.",
                    "extraction_method": "text_file",
                },
                {
                    "source_id": "src-b",
                    "source_file": "manual-b.txt",
                    "page_number": 1,
                    "text": "Tighten the mounting bolt to 30 N-m.",
                    "extraction_method": "text_file",
                },
            ],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
        }

        first_summary = core.refresh_job_evidence(job)
        first_ids = [item["evidence_id"] for item in job["evidence_items"]]
        second_summary = core.refresh_job_evidence(job)
        second_ids = [item["evidence_id"] for item in job["evidence_items"]]
        questions = core.transcript_questions_for(job)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_summary["open_conflicts"], 1)
        self.assertEqual(second_summary["open_conflicts"], 1)
        self.assertEqual(questions[0]["title"], "Resolve source conflict")
        self.assertEqual(questions[0]["conflict_id"], job["evidence_conflicts"][0]["conflict_id"])

    def test_unresolved_document_applicability_creates_question_and_blocks_context(self):
        job = {
            "id": "job-applicability",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-a",
                    "name": "manual-a.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "a" * 64,
                    "authority_level": "authoritative",
                    "revision": "",
                    "applicability": "",
                }
            ],
            "document_evidence": [
                {
                    "source_id": "src-a",
                    "source_file": "manual-a.txt",
                    "page_number": 1,
                    "text": "Tighten the mounting bolt to 25 N-m.",
                    "extraction_method": "text_file",
                }
            ],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
            "evidence_applicability_resolutions": {},
        }

        core.refresh_job_evidence(job)
        questions = core.transcript_questions_for(job)

        self.assertEqual(questions[0]["title"], "Confirm source applicability")
        self.assertEqual(questions[0]["applicability_source_id"], "src-a")
        self.assertEqual(core.document_context(job), "")

    def test_applicability_resolution_controls_document_context(self):
        job = {
            "id": "job-applicability-resolution",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-a",
                    "name": "manual-a.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "a" * 64,
                    "authority_level": "authoritative",
                    "revision": "",
                    "applicability": "",
                }
            ],
            "document_evidence": [
                {
                    "source_id": "src-a",
                    "source_file": "manual-a.txt",
                    "page_number": 1,
                    "text": "Tighten the mounting bolt to 25 N-m.",
                    "extraction_method": "text_file",
                }
            ],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
            "evidence_applicability_resolutions": {
                "src-a": {"status": "supporting_only", "answer": "Supporting context only"}
            },
        }

        core.refresh_job_evidence(job)
        included_context = core.document_context(job)
        job["evidence_applicability_resolutions"]["src-a"] = {
            "status": "not_applicable",
            "answer": "Does not apply to this job",
        }
        core.refresh_job_evidence(job)

        self.assertIn("25 N-m", included_context)
        self.assertIn("applicability review: supporting_only", included_context)
        self.assertEqual(core.document_context(job), "")

    def test_conflict_resolution_survives_evidence_refresh(self):
        job = {
            "id": "job-conflict-resolution",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-a",
                    "name": "manual-a.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "a" * 64,
                    "authority_level": "authoritative",
                    "revision": "A",
                    "applicability": "Model X",
                },
                {
                    "source_id": "src-b",
                    "name": "manual-b.txt",
                    "type": "text/plain",
                    "kind": "text",
                    "sha256": "b" * 64,
                    "authority_level": "supporting_context",
                    "revision": "draft",
                    "applicability": "Model X",
                },
            ],
            "document_evidence": [
                {"source_id": "src-a", "source_file": "manual-a.txt", "page_number": 1, "text": "Tighten the mounting bolt to 25 N-m."},
                {"source_id": "src-b", "source_file": "manual-b.txt", "page_number": 1, "text": "Tighten the mounting bolt to 30 N-m."},
            ],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
        }

        core.refresh_job_evidence(job)
        job["evidence_conflicts"][0]["status"] = "resolved"
        job["evidence_conflicts"][0]["resolution"] = {"answer": "Use manual-a.txt: 25 n-m"}
        core.refresh_job_evidence(job)

        self.assertEqual(job["evidence_conflicts"][0]["status"], "resolved")
        self.assertEqual(job["evidence_conflicts"][0]["resolution"]["answer"], "Use manual-a.txt: 25 n-m")

    def test_semantic_targets_include_uploads_and_prioritize_document_pages(self):
        job = {
            "id": "job-semantic-targets",
            "files": [
                {
                    "source_id": "src-image",
                    "name": "reference.jpg",
                    "kind": "image",
                    "stored_path": "storage/reference.jpg",
                }
            ],
            "document_evidence": [
                {
                    "source_id": "src-doc",
                    "source_file": "drawing.pdf",
                    "page_number": index,
                    "stored_path": f"storage/page-{index}.jpg",
                    "text": "Released drawing revision note" if index == 90 else "General page",
                }
                for index in range(1, 91)
            ],
        }

        with patch.object(core, "stored_path", return_value=Path("/tmp/source.jpg")), patch.object(
            core,
            "stored_relative_path",
            return_value=Path("/tmp/page.jpg"),
        ):
            targets = core.semantic_analysis_targets(job)

        self.assertEqual(len(targets), 1 + core.SEMANTIC_DOCUMENT_PAGE_LIMIT)
        self.assertEqual(targets[0]["evidence_type"], "uploaded_image")
        self.assertTrue(any(target.get("page_number") == 90 for target in targets))

    def test_semantic_analysis_normalization_is_conservative_and_structured(self):
        target = {
            "analysis_key": "sem-1",
            "source_id": "src-image",
            "source_file": "reference.jpg",
            "stored_path": "storage/reference.jpg",
            "preview_path": "storage/reference-preview.jpg",
            "evidence_type": "uploaded_image",
            "local_metrics": {"clarity": 72, "local_tags": ["landscape"]},
        }
        analysis = core.normalize_semantic_analysis(
            target,
            {
                "visual_summary": "Top view of an assembled housing.",
                "observed_items": ["housing", "fastener"],
                "tags": ["reference photo"],
                "clarity": 80,
                "instruction_value": 65,
                "document": {"title_block_present": True, "page_type": "technical_drawing"},
                "candidate_identifiers": [
                    {"label": "drawing number", "value": "ABC-123", "confidence": "medium"}
                ],
            },
        )

        self.assertEqual(analysis["status"], "complete")
        self.assertEqual(analysis["clarity"], 80)
        self.assertIn("reference_photo", analysis["tags"])
        self.assertIn("title_block", analysis["tags"])
        self.assertEqual(analysis["candidate_identifiers"][0]["value"], "ABC-123")

    def test_semantic_image_analysis_enriches_evidence_without_creating_technical_facts(self):
        job = {
            "id": "job-semantic-evidence",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-image",
                    "name": "reference.jpg",
                    "type": "image/jpeg",
                    "kind": "image",
                    "sha256": "c" * 64,
                    "stored_path": "storage/reference.jpg",
                }
            ],
            "document_evidence": [],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
            "image_analysis": [
                {
                    "analysis_key": "sem-image",
                    "analysis_version": core.SEMANTIC_ANALYSIS_VERSION,
                    "source_id": "src-image",
                    "source_file": "reference.jpg",
                    "stored_path": "storage/reference.jpg",
                    "evidence_type": "uploaded_image",
                    "status": "complete",
                    "visual_summary": "A mounting bolt is visible beside the housing; label appears to show 25 N-m.",
                    "observed_items": ["housing", "mounting bolt"],
                    "tags": ["detail_view"],
                    "confidence": "medium",
                }
            ],
        }

        core.refresh_job_evidence(job)
        image_evidence = next(item for item in job["evidence_items"] if item["evidence_type"] == "uploaded_image")

        self.assertEqual(image_evidence["summary"], job["image_analysis"][0]["visual_summary"])
        self.assertIn("semantic_analysis_complete", image_evidence["tags"])
        self.assertEqual(job["evidence_facts"], [])
        self.assertIn("mounting bolt", core.supporting_visual_context(job))

    def test_unresolved_document_visual_analysis_is_not_sent_to_drafting(self):
        job = {
            "id": "job-document-visual",
            "product": "Model X",
            "files": [
                {
                    "source_id": "src-doc",
                    "name": "drawing.pdf",
                    "type": "application/pdf",
                    "kind": "pdf",
                    "sha256": "d" * 64,
                    "authority_level": "authoritative",
                    "revision": "",
                    "applicability": "",
                }
            ],
            "document_evidence": [
                {
                    "source_id": "src-doc",
                    "source_file": "drawing.pdf",
                    "page_number": 1,
                    "stored_path": "storage/page-1.jpg",
                    "text": "Drawing notes.",
                    "extraction_method": "embedded_text",
                }
            ],
            "frame_analysis": [],
            "artifacts": [],
            "evidence_conflicts": [],
            "image_analysis": [
                {
                    "analysis_key": "sem-doc",
                    "analysis_version": core.SEMANTIC_ANALYSIS_VERSION,
                    "source_id": "src-doc",
                    "source_file": "drawing.pdf",
                    "stored_path": "storage/page-1.jpg",
                    "evidence_type": "document_page",
                    "page_number": 1,
                    "status": "complete",
                    "visual_summary": "A technical drawing with a title block and callouts.",
                    "observed_items": ["title block", "callouts"],
                    "tags": ["technical_drawing"],
                    "confidence": "medium",
                }
            ],
            "evidence_applicability_resolutions": {},
        }

        core.refresh_job_evidence(job)
        self.assertEqual(core.supporting_visual_context(job), "")
        job["evidence_applicability_resolutions"]["src-doc"] = {
            "status": "supporting_only",
            "answer": "Supporting context only",
        }
        core.refresh_job_evidence(job)

        self.assertIn("technical drawing", core.supporting_visual_context(job))


if __name__ == "__main__":
    unittest.main()
