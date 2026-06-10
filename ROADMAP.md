# Assembly Assistant Roadmap

## Product Direction

Assembly Assistant turns messy build content into reusable assembly instruction sections.

Core rule:

> AI processes jobs into candidate sections. Humans edit sections and build packages.

The first product focus is processing quality and section writing. Reviews, permissions, reports, and the operator viewer come later.

## Current State

Built:

- Local Python browser app prototype
- Jobs page with processing status columns
- Job Review page
- Fake background processing stages
- Question cards before section drafting
- Clarification files scoped to questions
- Draft section output
- Sections library
- Simple package builder stub
- Settings stub

Current app URL:

- http://127.0.0.1:8765

Limitations:

- Data is in memory
- Uploaded files are tracked by name only
- Processing is simulated
- Sections are not fully editable yet
- Package builder is only a workflow stub

## Core Objects

### Job

A temporary processing workspace.

Contains:

- job metadata
- original uploaded content
- processing status
- AI questions
- logs
- draft sections

Jobs should be archived when complete, not deleted.

### Question

A structured decision the AI needs before drafting sections.

Contains:

- prompt
- source reference
- reason for asking
- suggested choices
- answer
- optional clarification files

Questions happen before section drafting.

### Section

A reusable instruction building block.

Contains:

- title
- metadata
- tags
- ordered steps
- parts
- tools
- torque
- safety
- quality checks
- media attachments
- source job reference

Sections must be completed before use in packages.

### Package

A human-built instruction set made from completed section versions.

AI should not decide package structure.

## MVP Phase 1: UX Skeleton

Goal:

Tune the workflow before real processing makes the system harder to change.

Build:

- Jobs table refinements
- Job Review layout refinements
- Question card UX
- Section preview UX
- Section library UX
- Section editor first pass
- Package builder first pass

Acceptance:

- Riley can step through a fake job end to end
- The workflow feels simple and visual
- No screen feels overloaded
- Progress always feels alive

## MVP Phase 2: Persistence

Goal:

Stop losing data on restart.

Build:

- SQLite database
- local storage folder
- saved jobs
- saved files
- saved questions
- saved answers
- saved sections
- saved package drafts
- app settings

Recommended storage shape:

```text
storage/
  jobs/
    <job_id>/
      originals/
      audio/
      frames/
      thumbnails/
      transcripts/
      ocr/
      analysis/
  sections/
  packages/
```

Acceptance:

- Restarting the app preserves all work
- Duplicate uploads are detected
- Original files are never overwritten

## MVP Phase 3: Real File Uploads

Goal:

Make jobs use real uploaded files.

Build:

- multipart upload handling
- file hashing
- duplicate detection
- file type detection
- upload status
- add files to an existing job
- add clarification files to a question

Acceptance:

- User can upload video, PDF, image, scan, and text files
- Files are stored safely
- Jobs can be rerun with added files
- Exact duplicates are skipped

## MVP Phase 4: Processing Framework

Goal:

Replace fake stages with real worker tasks without changing the UX.

Build:

- processing task model
- background worker
- stage status updates
- retry failed stage
- cancel job
- rerun job
- advanced logs
- error handling

Stages:

1. Upload check
2. Transcription
3. Drawing/PDF OCR
4. Image review
5. Video frame selection
6. Question pass
7. Draft section generation

Acceptance:

- Processing runs outside web requests
- User can leave the job page while it runs
- Failed stages can be rerun
- Jobs table updates progress clearly

## MVP Phase 5: Transcription

Goal:

Build reliable audio extraction and transcription.

Build:

- ffmpeg audio extraction
- audio normalization
- silence handling
- chunking
- per-chunk retries
- timestamped transcript stitching
- transcription provider adapter

Initial providers:

- OpenAI transcription API
- optional local Whisper/faster-whisper later

Acceptance:

- A 10-minute build video produces a usable transcript
- Failed chunks retry individually
- Transcript timestamps can guide video analysis

## MVP Phase 6: OCR and Drawing Review

Goal:

Extract useful manufacturing data from drawings, PDFs, and scans.

Build:

- PDF page rendering
- OCR text extraction
- title block detection
- revision detection
- BOM/table extraction
- part number candidates
- drawing callout candidates
- AI cleanup and structuring

Acceptance:

- Drawing text is searchable
- BOM-like content becomes candidate parts
- Extracted data links back to source page

## MVP Phase 7: Image and Video Processing

Goal:

Reduce media overload and select only useful visuals.

Build:

- image clarity scoring
- duplicate/near-duplicate image detection
- irrelevant image rejection
- video keyframe extraction
- scene/change detection
- transcript-guided frame selection
- useful frame ranking
- rejected media drawer

Acceptance:

- User never has to sift through thousands of frames
- AI recommends useful frames
- Rejected frames are recoverable
- Each recommended image has a reason

## MVP Phase 8: AI Question Pass

Goal:

Ask only the questions needed to improve section drafting.

Build:

- question generation
- confidence thresholds
- source-linked questions
- answer/bypass behavior
- clarification file handling
- rerun affected analysis after answers

Question categories:

- unclear tool
- part identity
- quality check candidate
- safety callout candidate
- drawing/transcript conflict
- useful image confirmation
- torque only when fastener context suggests torque may matter

Acceptance:

- Questions happen before drafting
- The AI does not ask noisy questions
- User can bypass tool/torque/quality/safety flags

## MVP Phase 9: Section Drafting

Goal:

Turn reviewed evidence and question answers into candidate sections.

Build:

- section boundary detection
- step drafting
- part suggestions from BOM/OCR
- tool suggestions
- torque suggestions only when warranted
- safety callouts
- quality callouts
- media suggestions per step
- source traceability

Acceptance:

- Draft sections are usable starting points
- Every step has linked evidence
- User can approve, discard, split, or merge sections

## MVP Phase 10: Section Editor

Goal:

Make sections production-ready.

Build:

- visual step editor
- step media thumbnails
- expandable image viewer
- tools field
- torque field
- parts/consumables field
- quality checks
- safety notes
- add manual attachments
- duplicate section
- mark complete

Acceptance:

- Sections can be polished without touching the original job
- Completed sections become package-eligible
- Manual attachments are not reprocessed unless explicitly requested

## MVP Phase 11: Package Builder

Goal:

Let the manufacturing engineer build final instruction packages from completed sections.

Build:

- section search/filter
- drag sections into package
- reorder sections
- package metadata
- package preview
- validation
- freeze exact section versions

Filters:

- product
- work area
- station
- tags
- keyword
- tool
- part
- safety/quality presence

Acceptance:

- AI does not build packages
- User can visually assemble a package
- Published package content is stable even if sections change later

## Later Phases

### Review Workflow

- review assignments
- role/dept queues
- approvals
- rejections
- comments
- revision history

### Operator Viewer

- tablet-friendly step viewer
- current step text
- tool callout
- torque callout
- quality callout
- safety callout
- thumbnail media strip
- expandable image/drawing viewer
- floor feedback capture

### Reports

- processing health
- section usage
- package revision history
- common missing data
- review aging

### Network Deployment

- server-hosted app
- Postgres
- shared file storage
- login/roles
- backups
- audit trail

## Immediate Next Steps

Recommended order:

1. Tune current UX prototype.
2. Add a real Section Editor screen.
3. Add persistence with SQLite.
4. Add real file upload and storage.
5. Build the processing task framework.
6. Wire transcription first using the Lego video.
7. Add PDF/image OCR.
8. Add video frame selection.
9. Add real AI question generation.
10. Add real section drafting.

Next build target:

> Section Editor first pass, then SQLite persistence.

Reason:

The section is the core reusable product object. If the Section Editor feels wrong, the rest of the processing pipeline will generate content into the wrong shape.
