# Stage 2 WRITE checkpoint

Status: the complete manuscript draft is delivered in `report/report.tex`, following the supplied nine-section course template. Entry was Stage 2 from existing research; no new training or model scoring was run. Supporting evidence, a bibliography, figures and saved-results aggregation code accompany the draft.

The internal writing evaluation identified four localized fixes: conservative hypothesis-family correction, executable reproduction wording with snapshot prerequisites, paired TeX quotation closure, and accurate figure/uncertainty captions. One writer editing loop resolved all four, and the evaluator issued `accept_with_dissent_note`. The orchestrator additionally narrowed the objective wording to both neural families and named the small individual-seed points in the figure caption. These are internal writing checks, not formal pipeline peer review. The absent separate writer Phase 4a precommitment prevents a claim of full generator-contract certification.

Author metadata: Cao Pham Minh Dang, student ID 21453430; Hugo Centeno Sanz, ID intentionally blank; 50/50 contributions, confirmed by the user in the current session as relayed to the tracker. Dang's stated work covers the initial tokenizer, Transformer and its scoring; Hugo's covers LSTM and n-gram.

Budget: approximately 20–35k tokens estimated for planning, not measured usage. Recorded manuscript round trips: one internal Stage 2 writer editing loop; zero formal review/revision or integrity-fix loops.

## Completed draft checks

Tectonic compilation passed. The PDF has **8 main-text pages**, **1 bibliography page**, and **2 appendix pages** (11 total), meeting the template's 6–8-page main-text limit. The abstract has 200 whitespace-delimited words. The supplied template sections match exactly. All 11 reference keys are used with no citation-key orphans; 18 per-seed table rows and 63 saved-source hashes passed consistency checks. See `draft_checks.json` for the bounded check status and `pipeline_state.json` for SHA-256/byte records of the LaTeX, bibliography, PDF and derived comparison JSON. These are draft build and consistency checks, not formal integrity certification.

## Required next checkpoint

Explicit user confirmation is required before entering Stage 2.5 INTEGRITY. The requested skill says: “After each stage completion, the system must proactively prompt the user and wait for confirmation.” Source: `/home/dang.cpm/.codex/skills/academic-pipeline/SKILL.md`, Adaptive Checkpoint System. Its state machine specifies “Stage 2 WRITE → user confirmation → Stage 2.5.”

Neither formal Stage 2.5 integrity, Stage 3 review, re-review, final integrity nor finalization has been performed. No integrity PASS, external editorial acceptance, or final submission-readiness status is claimed. Later integrity must examine reference/claim verification and the AI Research Failure Mode Checklist, explicitly treating unavailable historical LSTM/KenLM raw scores and checkpoints as evidence limits.

The report's copy comparisons remain descriptive because realized sampling manifests differ. Confirmatory corrections retain untested copy family slots at p=1; `comparison_results.json` is the final numerical source. Earlier nominal p-values remain available only in its audit-only field and the annotated baseline memo.

## Collaboration observation (advisory only)

Status: `insufficient_evidence`. The current stage contains fewer than five user turns, and the tracker does not have a complete exposed raw-turn range. Delegation intensity, vigilance, reallocation and zone are unscored. No inference about the user's character, ability or learning is made. This observation never blocks progress.

Rubric read for this checkpoint: `/home/dang.cpm/.codex/skills/academic-research-suite/ars/shared/collaboration_depth_rubric.md`, version 1.0.1.

## Reader-facing revision

The author requested a rewrite for first-time faculty readers. The revised manuscript uses full model and size names, explains technical terms, removes local paths and experiment identifiers, and includes the requested hypothesis/test sentence. The abstract is 214 words. All 18 individual-run rows and six aggregate rows were rechecked, numerical evidence is byte-identical, and the PDF still has eight main-text pages with no LaTeX warnings. This is writing-stage version 2, not a formal integrity-review or peer-review completion.
