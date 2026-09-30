# Project Status

Updated: 2026-09-30

## Current Objective

Prove and stabilize Assembly Assistant while creating a clean boundary for a reusable procedure-processing engine. Do not extract the engine or build another product UX until the compatibility and evaluation gate is satisfied.

## Current Development Slice

**Neutral evidence and source-policy foundation - implemented and awaiting representative media validation**

- Stable source IDs and source roles.
- One selected instruction-narration source to prevent duplicate multi-angle transcripts from affecting step boundaries.
- One reference timeline per job.
- Automatic fixed-offset alignment from shared timed transcript words.
- Automatic fixed-offset alignment from shared audio waveforms, with confidence scoring and transcript fallback.
- Manual start anchor and optional end anchor for clock-drift correction.
- Source labels, authority, revision, and applicability metadata.
- Real PDF rendering, embedded-text extraction, scanned-page OCR, and page-level evidence.
- Supporting document context included in AI drafting with source-authority instructions.
- Semantic visual analysis for uploaded images and rendered document pages, with local clarity/orientation metrics, cached normalized previews, and explicit local-only/unreadable states.
- Synthetic unit and isolated API tests for source defaults, offsets, drift, transcript selection, and text evidence.
- Temporary FFmpeg fixtures for known offsets, portrait/landscape orientation, alternate angles, voiceover, audible/visible markers, and scripted step times.
- Processing-run manifests containing engine/profile/prompt versions, redacted configuration, source setup, stage outcomes, timing, counts, alignment evidence, and failures.
- Automated acceptance coverage confirms generated alternate-angle alignment within 0.5 seconds and preserves one drafting narration source.
- Versioned neutral `Source`, `EvidenceItem`, and `ProcessingRun` contracts coexist with the existing job format.
- Transcript spans, scored video frames, document pages, and uploaded images receive stable evidence IDs.
- Draft steps preserve supporting transcript and media evidence IDs.
- Uncertain document applicability blocks that source from drafting until the user explicitly approves its use.
- Conflicting torque, dimension, part-number, and safety claims become questions; the system does not select a winner automatically.

## Next Authorized Work

Work that does not require production recordings:

1. Define neutral procedure/section/step draft contracts and the first Assembly profile interface.
2. Add evidence visibility to section authoring so reviewers can inspect the source behind generated claims.
3. Add visual sync-marker alignment as a fallback for silent or weak-audio multi-angle recordings.
4. Extend drawing interpretation from page classification into title-block, revision, BOM, and callout extraction after representative documents are available.

Work that should wait for representative media:

- Tune automatic alignment confidence thresholds.
- Validate fixed-offset waveform alignment against real shop-floor noise and decide whether automatic clock-drift fitting is justified.
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
