# Procedure Intelligence Platform Architecture

## Purpose

This document defines how the processing capability currently inside Assembly Assistant can serve multiple products without turning Assembly Assistant into a generic, overloaded application.

## Core Decision

The processing engine should not be copied from one application to another. It should be extracted incrementally into a versioned, headless Python package with neutral input and output contracts.

Initially, every product can run that package in its own local process:

```text
Assembly UX -----------\
Software UX ------------> Procedure Engine Package ---> Evidence/Media Store
Field Service UX -------/
Customer Publishing UX /
```

When multi-user load, centralized data, or operational reliability justify it, the same package can be hosted behind a versioned API and durable job queue:

```text
Product UIs ---> Versioned API ---> Durable Queue ---> Procedure Engine ---> Evidence Store
```

This sequence avoids premature service complexity while preventing permanent code duplication.

## Responsibility Boundaries

### Shared Engine

The engine owns technical processing that should improve once for every product:

- Source identity, hashing, deduplication, and storage references.
- Media probing and metadata normalization.
- Audio extraction, cleanup, channel selection, and chunking.
- Provider-neutral timestamped transcription.
- Voiceover and multi-source timeline alignment.
- Document rendering, OCR, tables, regions, and drawing metadata.
- Image normalization, orientation, quality analysis, and visual evidence.
- Adaptive video sampling, OpenCV scoring, semantic scoring, and media assignment.
- Neutral action/step-boundary evidence.
- Provenance, source authority, conflicts, confidence, and processing diagnostics.
- Domain-profile execution and versioned output.
- Deterministic validation of neutral contracts.

The engine does not decide what an assembly package, service approval, software guide, or customer publication should look like.

### Domain Profile

A domain profile converts neutral evidence into domain-specific draft behavior. It owns:

- Vocabulary and spoken cue phrases.
- Step-boundary guidance.
- Frame-relevance features.
- Prompt templates and structured output extensions.
- Clarification-question policy.
- QA checks and completion rules.
- Source-precedence additions for the domain.
- Output mapping into the product's editor fields.
- Profile and prompt version identifiers.

Examples are `assembly`, `general_physical`, `software`, and `field_service`.

### Product Application

Each product owns the human workflow:

- Navigation and page structure.
- Job setup and product-specific metadata.
- Review and correction interfaces.
- Permissions, approval, publication, and retention policy.
- Editors, viewers, print layouts, and exports.
- Search, packaging, and release behavior.

This is why Software Instructions and Field Service Authoring deserve separate UXs even when they use the same transcription and evidence pipeline.

## Target Package Shape

The following is a target boundary, not a requirement for a single large refactor:

```text
procedure_engine/
  contracts.py
  orchestration.py
  processing_runs.py
  sources/
    uploads.py
    documents.py
    audio.py
    video.py
    images.py
  transcription/
    providers.py
    alignment.py
  evidence/
    records.py
    precedence.py
    conflicts.py
  vision/
    sampling.py
    opencv_scoring.py
    semantic_scoring.py
    media_assignment.py
  profiles/
    base.py
    assembly.py
    general_physical.py
    software.py
    field_service.py
  evaluation/
    fixtures.py
    metrics.py
```

Functions should move only after their current behavior has baseline coverage. Assembly Assistant can continue importing them from the package in the same process.

## Neutral Contracts

### Source

A source identifies uploaded or generated material independently of a particular product UI.

Required concepts:

- Stable source ID.
- Source kind: video, audio, voiceover, image, PDF, drawing, text, report, or generated derivative.
- Filename, media type, hash, size, and storage reference.
- Parent source and derivation details.
- Authority level, revision, applicability, and effective date when available.
- Timeline relationship for timed sources.
- Access and retention classification.

### Evidence Item

An evidence item is an addressable observation or extracted fact:

- Transcript word, phrase, or segment with start/end time.
- Video frame or range with timestamp and camera/source ID.
- Document page region, drawing callout, table row, or OCR span.
- Uploaded image or selected crop/display treatment.
- Detected cue, object, action, measurement, warning, or contradiction.
- Confidence, extraction method, and provider/model version.

Evidence items point back to immutable sources. Edited instruction text does not overwrite evidence.

### Procedure Draft

A neutral procedure draft contains:

- Procedure ID, title, profile ID, and profile version.
- Applicability and prerequisite data.
- Ordered sections and steps.
- Open questions, conflicts, and confidence summary.
- Processing-run ID and source set.

### Step Draft

A neutral step supports common fields without requiring assembly terminology:

- Stable step ID.
- Instruction and optional rationale.
- Start/end time and sequence confidence.
- Objects/resources, tools, parameters, hazards, verification, and expected result.
- Primary and supporting media.
- Evidence links and unsupported-claim indicators.
- Domain-specific data under a versioned `profile_data` object.

For example, Assembly can store torque and part details while Software stores navigation and expected-screen state. Neither extension becomes mandatory for the other.

## Profile Interface

The first profile interface should remain small and explicit:

- `profile_id` and `profile_version`.
- `normalize_job_context()`.
- `cue_vocabulary()`.
- `detect_candidate_boundaries()`.
- `rank_domain_media()`.
- `build_clarification_questions()`.
- `draft_procedure()`.
- `validate_draft()`.
- `map_to_product()`.

The orchestration layer calls the interface; it should not contain `if assembly`, `if software`, or `if service` branches throughout the pipeline.

## Supporting Documents

Supporting files should not be flattened into one text prompt. Ingestion should preserve structure and authority.

### PDFs and Drawings

- Render each page at a known scale.
- Retain OCR text with page and bounding coordinates.
- Detect title blocks, drawing number, revision, notes, tables, BOMs, and callouts where practical.
- Link a generated claim to the exact page/region used.
- Treat released technical values as authoritative only when revision and applicability are known.

### 3D Renders and Images

- Preserve the original file and orientation.
- Extract visual summaries, visible components, viewpoints, labels, and quality metrics.
- Mark renders as supporting context unless the organization explicitly designates them as controlled technical sources.
- Allow a reviewer to select them as primary or supporting media without changing the original.

### Reports and Prior Instructions

- Preserve author, date, product/model, status, and approval metadata.
- Distinguish observed field evidence from an approved technical conclusion.
- Do not allow an unapproved report to override a released manual or bulletin silently.

## Source Precedence and Conflicts

The engine stores evidence; the domain profile applies source policy. A default order is:

1. Released technical source applicable to the exact product/revision.
2. Approved organization procedure or bulletin.
3. Direct timed evidence from the current job.
4. Curated supporting context.
5. Unapproved reference material.

Precedence is not permission to hide conflicts. If a video demonstrates an action that contradicts an applicable manual, the result must be a visible conflict requiring review.

## Voiceover and Timeline Alignment

A voiceover is another timed source, not replacement text pasted over the original transcript.

### Alignment Order

1. Use a shared recording timeline when voiceover was captured while the source played.
2. Apply user-supplied start offset and synchronization anchors.
3. Match distinctive cue phrases and pauses.
4. Use audio-feature alignment when both tracks contain common sound.
5. Use semantic alignment between narration spans and visible action.
6. Require review when drift or ambiguity exceeds the configured threshold.

Both original and voiceover transcripts remain available. The product selects a preferred narration source for drafting, while the engine preserves provenance for every word span.

## Product Knowledge Assistant Boundary

The Product Knowledge Assistant is not another procedure profile. It is a retrieval-and-answer product built on shared sources and evidence.

Its processing flow is:

```text
Approved Sources
  -> parse/OCR/version/applicability metadata
  -> evidence chunks and structured facts
  -> lexical + semantic + structured retrieval
  -> applicability/conflict/safety checks
  -> cited answer or escalation
```

It may use approved procedures produced by the authoring products, but it needs separate capabilities:

- Corpus curation and source lifecycle.
- Product/model/serial/revision filtering.
- Hybrid retrieval and reranking.
- Citation-complete answers.
- Diagnostic state and follow-up questions.
- Role-aware safety and action limits.
- Feedback routed to expert review rather than silently becoming truth.

The authoring engine creates and maintains procedures. The knowledge assistant finds approved knowledge and answers questions. Sharing evidence infrastructure does not make those the same workflow.

## Compatibility Strategy

Extraction should be incremental:

1. Select representative existing assembly jobs and record baseline outputs.
2. Add neutral contracts alongside the current saved structures.
3. Write adapters from current job/media/transcript data into neutral sources and evidence.
4. Wrap current prompts, cues, QA, and mappings as the Assembly profile.
5. Route one processing stage at a time through the new package.
6. Compare timing, steps, fields, media, questions, and saved outputs against the baseline.
7. Keep existing API and saved-package behavior until compatibility is proven.
8. Build the second product against the package boundary instead of importing Assembly Assistant application code.

This is how the engine moves between applications: product applications depend on the package or its later API. They do not copy `app.py`, share browser state, or read another application's internal database.

## Safety and Governance

Before field-service, customer, or knowledge-assistant deployment, define:

- Approved and prohibited source classes.
- External-AI versus local-only data rules.
- Product, model, serial, and revision applicability requirements.
- Safety-critical values that require direct authoritative citations.
- Technician-only and customer-permitted operations.
- Reviewer qualifications and release authority.
- Source retention, access control, audit, and deletion policy.
- How superseded documents and instructions are withdrawn.

The system should make uncertainty and missing evidence visible. It should not manufacture certainty to produce a complete-looking document.

## What Not to Do

- Do not add a single "instruction type" dropdown to the current Assembly Assistant and conditionally hide dozens of fields.
- Do not create a universal prompt that attempts to cover every domain.
- Do not copy the processing functions into multiple repositories.
- Do not use field reports as unqualified authoritative truth.
- Do not publish AI-generated customer or safety-critical instructions without approval.
- Do not build a distributed service before local contracts, profiles, and evaluation are stable.
