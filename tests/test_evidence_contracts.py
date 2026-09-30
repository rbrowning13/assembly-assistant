import unittest

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


if __name__ == "__main__":
    unittest.main()
