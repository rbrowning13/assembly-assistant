# Procedure Intelligence Platform Roadmap

## Direction

Assembly Assistant remains a focused manufacturing-authoring product. It should not become a generic interface with assembly, software, service, and customer workflows mixed together.

The broader direction is a shared procedure-processing platform with separate products:

- **Assembly Assistant** - manufacturing assembly and production work instructions.
- **Software Instructions** - procedures created from screen recordings and supporting documents.
- **Field Service Authoring** - diagnostic, repair, test, and return-to-service procedures.
- **Customer Instructions** - approved, customer-safe procedures derived from controlled content.
- **Product Knowledge Assistant** - cited answers and troubleshooting guidance from an approved product corpus.

The shared core handles media and evidence. Each product owns its terminology, decisions, review rules, and user experience.

The platform claim is not "any video becomes perfect instructions." The defensible claim is:

> Narrated procedural media and supporting documents can be converted into evidence-linked draft instructions that a qualified person can review and approve.

## Product Boundaries

### Assembly Assistant

Assembly Assistant continues to own:

- Product, work area, station, assembly, and source-job metadata.
- Parts and consumables.
- Tools and torque.
- Quality checks and safety callouts.
- Manufacturing sections, packages, print output, and the future operator viewer.
- Manufacturing review and approval rules.

The existing assembly UX and package model are protected product behavior, not a temporary generic prototype.

### Software Instructions

Software instruction authoring gets its own UX and domain profile. It should understand:

- Application, environment, version, and user role.
- Access and prerequisite state.
- Navigation paths, menus, tabs, buttons, fields, and entered values.
- Expected screen results and verification.
- Before/after screenshots and sensitive-data redaction.
- Error recovery, exception paths, and escalation.

It should not expose assembly-only fields such as torque, work cells, or parts unless a specific workflow extension requires them.

### Field Service Authoring

Field service authoring gets a technician-focused UX and profile. It should understand:

- Machine family, model, serial range, configuration, and revision applicability.
- Reported symptom, diagnostic path, root cause, corrective action, and verification.
- Machine state, prerequisites, PPE, lockout/tagout, stored energy, and access hazards.
- Tools, service parts, consumables, measurements, specifications, and tolerances.
- Return-to-service criteria and escalation limits.

Technical claims should be traceable to authoritative manuals, bulletins, drawings, specifications, or approved service knowledge.

### Customer Instructions

Customer instructions should be a controlled publication surface, not an unrestricted AI output mode. It should reuse approved field-service content while adding:

- Customer-permitted versus technician-only work boundaries.
- Reading-level and terminology controls.
- Strong warning, prerequisite, and stop-work presentation.
- Model and serial applicability.
- Escalation and contact guidance.
- Approval, revision, and publication controls.

No generated procedure should publish directly to customers without a qualified approval step.

### Product Knowledge Assistant

The Product Knowledge Assistant is a related but separate product. It answers product, diagnostic, and troubleshooting questions using an approved corpus such as:

- Owner and service manuals.
- Service bulletins and technical specifications.
- Approved work instructions.
- Drawings, parts information, and revision records.
- Curated field-service reports and resolved cases.

It shares ingestion, OCR, evidence records, source versioning, permissions, and storage with the procedure platform. It needs a separate retrieval and answer service with:

- Citations for claims and recommended actions.
- Model, serial, configuration, and revision applicability.
- Source authority and conflict handling.
- Safety and role boundaries.
- A clear refusal or escalation path when evidence is insufficient.

It does not belong inside the Assembly Assistant job or package workflow.

## Shared Platform

See [Platform Architecture](docs/PLATFORM_ARCHITECTURE.md) for the extraction and integration design.

### Shared Responsibilities

- Upload intake, hashing, deduplication, and source identity.
- Audio extraction and timestamped transcription.
- Voiceover ingestion, selection, and timeline alignment.
- Video probing, adaptive frame sampling, OpenCV scoring, and media selection.
- Image normalization and analysis.
- PDF/document rendering, OCR, tables, drawing callouts, and page coordinates.
- Evidence records, source precedence, conflict detection, and provenance.
- Neutral procedure, section, step, and processing-run contracts.
- Provider adapters, retry behavior, logs, timings, cost metrics, and diagnostics.
- Domain-profile execution and version tracking.

### Product-Specific Responsibilities

- Navigation, pages, terminology, and job setup.
- Domain metadata and editor fields.
- Prompt policy and drafting rules.
- Question policy, QA checks, and completion gates.
- Review, approval, permissions, publication, and retention policy.
- Operator, technician, customer, or software-user viewers.
- Product-specific print and export formats.

## Current State

### Built and Working

- Local Flask application with persisted jobs, sections, packages, settings, and media.
- File upload, storage, hashing, deduplication, and rerun support.
- FFmpeg media probing and audio extraction.
- OpenAI and local Whisper transcription paths with timestamp handling.
- Transcript cue parsing and support for multiple aligned video sources.
- Adaptive video sampling, OpenCV quality scoring, and candidate selection.
- Draft questions and assembly-oriented section/step generation.
- Section authoring, QA checks, media correction, package building, and print preview.
- Processing logs and visible job-stage progress.

### Prototype Limitations

- Assembly vocabulary and assumptions are embedded throughout processing and UI code.
- Drawing/PDF OCR is not yet a complete production pipeline.
- Uploaded still-image analysis is not yet a complete production pipeline.
- Silent-video and weak-audio workflows need a deliberate segmentation strategy.
- Background work uses in-process threads and is not durable across process restarts.
- SQLite currently stores a large application-state document rather than normalized records.
- There is no representative regression/evaluation suite for instruction accuracy.
- The core processing implementation is still concentrated in a large module.
- Corporate rules for retention, confidential media, AI-provider use, and customer publication are not yet defined.

## Protection Rules

The shared-engine work must follow these rules:

1. Preserve the current Assembly Assistant workflow, fields, package snapshots, and saved data.
2. Introduce neutral contracts and compatibility adapters before moving behavior.
3. Establish baseline assembly jobs and expected outputs before refactoring processing.
4. Add regression tests around timing, step boundaries, image selection, drafting, and package compatibility.
5. Never replace the assembly prompt with one generic prompt intended to serve every domain.
6. Do not make profile-specific fields mandatory in the shared core.
7. Do not copy and diverge the processing pipeline between products.
8. Do not introduce a network service until the package boundary is stable locally.

## Source and Evidence Model

Supporting content must have an explicit authority level:

1. **Authoritative** - approved manuals, released drawings, specifications, bulletins, and controlled procedures.
2. **Primary evidence** - source video, chosen voiceover, screen recording, and direct operator/technician narration.
3. **Supporting context** - 3D renders, reference photographs, draft documents, and related examples.
4. **Reference only** - unapproved or uncertain material that can inform questions but cannot establish a technical claim.

Every generated step should retain, where available:

- Source ID, filename, revision, and authority level.
- Transcript word/span timestamps and source audio track.
- Video/frame timestamps and media-selection reason.
- Document page, region, table, callout, or drawing reference.
- Which content was directly supported and which wording was AI-normalized.
- Confidence, conflicting evidence, and unresolved questions.

Conflicting authoritative sources must produce a review question or block publication. The system must not silently choose one.

## Voiceover Roadmap

Voiceover support should cover three practical cases:

1. **Replacement narration recorded while watching the source video** - same timeline and easiest alignment.
2. **Separately uploaded voiceover with approximately the same duration** - align using audio features, pauses, and cue anchors.
3. **Independent narration** - require semantic alignment plus user-confirmed anchors when confidence is low.

The UI should provide:

- Original, voiceover, or mixed transcript-source selection.
- Start offset and at least one manual synchronization anchor.
- Transcript preview with confidence and detected drift.
- Preservation of both original and replacement audio as evidence.

## Delivery Phases

### Phase 0 - Core Model Proof and Baseline

Build the evidence needed to prove and safely change the current model.

- Assemble representative assembly test jobs, including clutter, weak audio, long video, multiple angles, and supporting documents.
- Record expected step boundaries, critical instructions, and acceptable media for each job.
- Add processing-time, stage-time, model, prompt version, candidate-count, and cost metrics.
- Define critical omissions and inventions as hard failures.
- Add smoke tests for saved jobs, sections, package snapshots, and print output.

**Acceptance:** Runs can be compared to a baseline, critical omissions/inventions are measured, and existing saved content still loads.

### Phase 1 - Neutral Contracts and Assembly Profile

Extract the product boundary without changing visible Assembly Assistant behavior.

- Define `Source`, `EvidenceItem`, `ProcessingRun`, `ProcedureDraft`, `SectionDraft`, and `StepDraft` contracts.
- Define a domain-profile interface and registry.
- Move assembly cue vocabulary, prompts, question policy, QA rules, and output mapping into an Assembly profile.
- Add adapters between existing saved state and the neutral contracts.
- Persist profile ID, profile version, prompt version, and provider/model metadata with each processing run.

**Acceptance:** Existing assembly jobs produce equivalent output, old packages still open, and the shared core does not require assembly-only fields.

### Phase 2 - Real Supporting-Document Processing

- Render PDF pages and extract OCR text with coordinates.
- Detect drawing title blocks, revision data, BOM tables, notes, and callouts where practical.
- Extract text and structure from common office/text formats.
- Analyze uploaded photographs and 3D render images.
- Add source authority, revision, page, region, and conflict metadata.
- Make document evidence available to questions, drafting, section editing, and citations.

**Acceptance:** Generated claims can link to actual document evidence, and drawing/image stages no longer report simulated completion.

### Phase 3 - Voiceover and Timeline Alignment

- Add replacement narration uploads and transcript-source controls.
- Support offsets, anchors, drift detection, and confidence reporting.
- Align multiple camera angles and voiceover to one reference timeline.
- Preserve all original timing and provenance.

**Acceptance:** A noisy source video can use a clean voiceover while frames remain aligned to the correct action.

### Phase 4 - General Physical Procedure Profile

Create a reusable physical-work profile for early validation outside assembly, while keeping Assembly Assistant unchanged.

- General action, object, tool, hazard, verification, and result vocabulary.
- Configurable domain extensions for testing, machining, fabrication, paint, and maintenance.
- A minimal evaluation harness that compares profile output on the same neutral evidence.

**Acceptance:** The shared engine can produce a neutral physical procedure without assembly-specific output leaking into it.

### Phase 5 - Software Instruction Product

- Build a separate software-instruction job setup, review, editor, and export UX.
- Add screen-event evidence, UI-element descriptions, OCR/redaction, and before/after screenshot selection.
- Add software-specific QA for prerequisites, navigation, values, expected results, and recovery paths.

**Acceptance:** Gina can create and correct a software procedure from a screen recording without seeing manufacturing-only controls.

### Phase 6 - Field Service Product

- Build a technician-focused authoring UX and Field Service profile.
- Add asset applicability, symptom/diagnostic structure, measurements, specifications, hazards, repair, and return-to-service verification.
- Require authoritative evidence for safety-critical values and technical claims.
- Add field-service review and release policy.

**Acceptance:** A technician can turn service media and approved documents into a traceable draft procedure with explicit applicability and verification.

### Phase 7 - Product Knowledge and Troubleshooting Assistant

- Build an approved, versioned product corpus and ingestion workflow.
- Chunk and index sources with product/model/serial/revision metadata.
- Retrieve evidence using lexical, semantic, and structured filters.
- Generate cited answers with conflict, applicability, safety, and escalation checks.
- Add feedback and expert-curation workflow without allowing feedback to silently alter approved truth.

**Acceptance:** Answers identify their sources and applicability, distinguish facts from diagnostic suggestions, and decline unsupported or unsafe guidance.

### Phase 8 - Durable Shared Service

Move from a local package to a shared service only after contracts and profiles are proven.

- Durable queue and resumable processing runs.
- Object storage and normalized metadata.
- Authentication, authorization, audit history, and tenant/site boundaries.
- Versioned API and backward-compatible contracts.
- Central provider credentials, budgets, observability, and retention controls.

**Acceptance:** Multiple products can submit work to the same engine without sharing UI state or duplicating processing code.

## Evaluation Program

Each domain needs its own test set and acceptance thresholds. Track at least:

- Critical-step recall.
- Unsupported or invented instruction rate.
- Step ordering and boundary accuracy.
- Transcript-to-frame timing error.
- Primary-media acceptance rate.
- Human edits per generated step.
- Correctly surfaced ambiguities and conflicts.
- Processing duration, retries, and cost per input minute.
- Authoring time saved versus the existing manual process.

Safety-critical inventions, incorrect torque/specification values, incorrect software values, and unsupported diagnostic actions are hard failures, not averageable quality scores.

## Immediate Next Target

Complete Phase 0 and the compatibility groundwork for Phase 1 before creating another product UI.

1. Freeze a representative assembly baseline set.
2. Capture expected steps, timing, media, and critical fields.
3. Add run/version metrics and repeatable evaluation output.
4. Define neutral contracts and the domain-profile interface.
5. Wrap current behavior as the first Assembly profile without changing the user-facing workflow.

This is the safest route to reuse: prove the current product, isolate what is truly common, then build the next UX on a stable engine.

## Decisions Made

- Assembly Assistant remains a dedicated manufacturing product.
- Different instruction domains receive different UXs and domain profiles.
- Processing infrastructure is shared; prompts, QA, metadata, approvals, and viewers are product-specific.
- Supporting documents become weighted, cited evidence rather than undifferentiated attachments.
- Voiceover is a first-class timed source and never destroys the original audio evidence.
- The Product Knowledge Assistant is a separate retrieval product built on shared ingestion and evidence infrastructure.
- Human review remains required before controlled manufacturing, service, or customer publication.

## Attention Needed

These are material product or governance decisions Riley and the company will need to make before the relevant phases:

- Which corporate data may be sent to external AI providers and which must remain local.
- Which source systems and document classes are authoritative for technical claims.
- Who may approve assembly, technician, and customer-facing instructions.
- Which operations customers may perform versus technician-only work.
- Retention, access, and audit requirements for source video, reports, and generated content.
- Whether Software Instructions or General Physical Procedures should be the first profile built after Assembly.
