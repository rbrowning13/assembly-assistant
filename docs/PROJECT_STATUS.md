# Project Status

Updated: 2026-09-30

## Current Objective

Prove and stabilize Assembly Assistant while creating a clean boundary for a reusable procedure-processing engine. Do not extract the engine or build another product UX until the compatibility and evaluation gate is satisfied.

## Current Development Slice

**Input foundation - implemented and awaiting representative media validation**

- Stable source IDs and source roles.
- One selected instruction-narration source to prevent duplicate multi-angle transcripts from affecting step boundaries.
- One reference timeline per job.
- Automatic fixed-offset alignment from shared timed transcript words.
- Manual start anchor and optional end anchor for clock-drift correction.
- Source labels, authority, revision, and applicability metadata.
- Real PDF rendering, embedded-text extraction, scanned-page OCR, and page-level evidence.
- Supporting document context included in AI drafting with source-authority instructions.
- Honest image-stage status: uploaded images are cataloged, but semantic still-image analysis remains pending.
- Synthetic unit and isolated API tests for source defaults, offsets, drift, transcript selection, and text evidence.

## Next Authorized Work

Work that does not require production recordings:

1. Add audio-waveform alignment behind the existing timeline contract.
2. Add synthetic FFmpeg video/audio fixtures with known offsets, drift, orientation, and sync markers.
3. Add processing-run manifests with engine/profile/prompt/provider versions and stage metrics.
4. Add document conflict and applicability scaffolding without making technical decisions automatically.
5. Begin neutral `Source`, `EvidenceItem`, and `ProcessingRun` contracts alongside the existing saved format.

Work that should wait for representative media:

- Tune automatic alignment confidence thresholds.
- Tune frame sampling and media assignment across camera angles.
- Validate OCR quality on actual company drawings and scans.
- Establish instruction-accuracy and media-acceptance baselines.
- Decide whether the engine extraction gate has been satisfied.

## Engine Extraction Gate

The engine is not ready to move to a separate repository. Extraction becomes appropriate when:

- Representative assembly fixtures and expected outputs exist.
- Neutral source/evidence/run contracts are in use.
- Assembly prompts, QA, and mapping are isolated behind an Assembly profile.
- Existing jobs, sections, packages, and print output pass regression tests.
- Processing code no longer depends on Assembly Assistant UI state.
- A second-domain spike consumes the boundary without assembly concepts leaking into it.

## Attention Needed

- Riley must provide representative media when available and identify critical steps/facts that cannot be omitted.
- Company policy is still needed for external-AI data, retention, source authority, and customer-facing approval.
- The first post-Assembly validation product remains undecided: General Physical Procedures or Software Instructions.
- Native CAD ingestion is not planned for the initial supporting-document pipeline; controlled PDF exports and rendered images are the current input standard.
