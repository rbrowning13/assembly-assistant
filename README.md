# Assembly Assistant

Assembly Assistant is a local Flask application that converts manufacturing videos, narration, images, and supporting material into evidence-linked draft assembly instructions. Manufacturing engineers review and correct the generated sections, then combine completed sections into controlled instruction packages.

This repository is the first product implementation of a planned shared procedure-processing platform. Assembly Assistant remains manufacturing-specific; software instructions, field-service instructions, customer instructions, and product troubleshooting will use separate interfaces and domain rules.

## Product Direction

- **Keep this UX focused:** products, work areas, stations, parts, tools, torque, quality, safety, sections, and packages remain Assembly Assistant concepts.
- **Extract reusable processing:** media ingestion, transcription, timeline alignment, video/image analysis, OCR, evidence records, and processing telemetry move behind neutral contracts.
- **Add domain profiles:** each product supplies its own vocabulary, prompts, question policy, QA rules, output mapping, and review gates.
- **Build separate products:** other domains consume the shared engine without inheriting assembly-only screens or fields.
- **Preserve provenance:** generated instructions should remain traceable to transcript spans, frames, document regions, and source revisions.

Read the [roadmap](ROADMAP.md) for delivery phases, [project status](docs/PROJECT_STATUS.md) for the active work and readiness gate, and [platform architecture](docs/PLATFORM_ARCHITECTURE.md) for the shared-engine design.

## Current Capabilities

- Create and process jobs while continuing work elsewhere in the app.
- Upload large videos and supporting source files with hashing and deduplication.
- Extract audio with FFmpeg and transcribe through configured providers.
- Retain transcript timing and use narration cues to guide step boundaries.
- Assign source roles, choose one drafting narration, and align alternate camera timelines with automatic or manual anchors.
- Sample and score video frames with OpenCV and semantic analysis.
- Render PDF pages, extract embedded text, OCR scanned pages when local tools are available, and retain page-level evidence.
- Generate clarification questions and assembly-oriented draft sections.
- Review and edit steps, fields, media, and QA items.
- Save completed sections into a searchable library.
- Build frozen package snapshots and render print-friendly output.
- Export processing logs for diagnosis and model improvement.

## Important Prototype Limits

- The current drafting and QA behavior is intentionally assembly-oriented.
- PDF rendering/text/OCR is implemented, but drawing title-block, revision, BOM, table, callout, and conflict interpretation is not complete.
- Uploaded still images are available as authoring evidence but do not yet receive semantic analysis.
- Multi-angle alignment supports transcript matching and manual drift anchors; audio-waveform and visual-marker alignment are not implemented yet.
- Background processing currently runs inside the Flask process and is not durable across a restart.
- The application-state persistence model is suitable for local proving work, not yet for a multi-user production deployment.
- Generated instructions require qualified human review.

## Run Locally

```bash
python3 app.py
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

## Runtime Notes

- Application state and uploaded/generated media are stored locally under `storage/`.
- The OpenAI API key is entered in Settings and is not committed to Git.
- FFmpeg is required for media probing and audio extraction.
- Poppler (`pdftotext` and `pdftoppm`) is required for PDF evidence extraction and rendering.
- Tesseract is optional and is used to OCR rendered PDF pages that do not contain embedded text.
- OpenCV and a configured transcription provider are required for the full processing path.
- Local source files, generated media, logs, and credentials must remain excluded from version control.

## Development Rule

Do not copy the processing pipeline into another application. First extract neutral contracts and wrap current behavior in an Assembly profile. New products should depend on that versioned engine boundary and add their own UX and profile. This keeps improvements to transcription, OCR, timing, evidence, and media selection available to every product without allowing one domain's rules to weaken another.
