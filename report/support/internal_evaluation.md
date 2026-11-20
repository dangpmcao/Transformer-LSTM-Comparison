# Internal academic-paper evaluation (Phase 6b, initial round)

Reviewed `report/report.tex` as present on 2026-11-20, `references.bib`, supplied proposal/template, writing plan, evidence memos, figure-generation code and PNGs, derived comparison JSON, and relevant saved sources. This is the internal academic-paper quality review, **not** academic-pipeline Stage 2.5 integrity certification or Stage 3 external review. No model inference or new experiment was run. The writer's required Phase 4a precommitment was not supplied before drafting; that workflow limitation cannot be retroactively repaired or represented as compliant contract certification. This review nevertheless applies the evaluator's recorded Phase 6a scoring plan. Main-page count is being checked by root; the abstract has 200 whitespace-delimited words and meets the <=250-word requirement.

## Dimension Scores

### D1: originality
**pass.** Sections 1--2 identify a concrete empirical contribution: separating prediction quality, position-based gains, controlled same-target history sensitivity, copying, and grammatical minimal pairs under closely matched small-model conditions. The report does not claim a new architecture or universal law. Discussion engages the proposal's central confounds rather than listing papers. This is appropriate original empirical evidence for a course report.

### D2: methodological_rigor
**warn.** The method is fundamentally sound but insufficiently detailed in a few consequential places, and one inference convention needs correction. Sections 4--5 accurately explain parameter/token matching, architecture-specific initialization, seed units, checkpoint selection, and mismatched copy samples. However, lines 151/155/300--309 present Holm-adjusted CG results whose computation uses the favorable ordering of copy p-values that the paper explicitly declines to treat as confirmatory. Retain the original comparison-family size without those test results influencing CG correction (R1 below). The current reproduction paragraph also advertises an evaluator command whose frozen proposal path no longer exists (R2). These are repairable reporting/analysis issues, not grounds to discard the experiment.

### D3: evidence_sufficiency
**warn.** Core claims have evidence but some secondary claims lack support in their current exact form. Main table values, seed appendices, parameter counts, BLiMP equivalence conclusions and same-target gains agree with saved summaries/derived data. The causal limitations, baseline raw-file access limits, copied-sample deviation, failed deep control and cross-host metadata uncertainty are explicitly preserved. The precise multiplicity-adjusted p-values should be recomputed under the stated confirmatory scope (R1), and the reproducibility command needs its actual prerequisites (R2). No invented core measurement was found. Bibliographic existence checking remains root's separate source-verification task rather than a certification by this review.

### D4: argument_coherence
**pass.** The abstract, introduction and conclusion consistently support stronger context-use behavior while refusing causal decomposition of the entire perplexity gap. H2 is inconclusive rather than equated with nonsignificance, and H3 is not established despite positive point trends. The KenLM control shows why CG alone is ambiguous; the same-target discussion supplies a useful refinement. Grammar subgroup and length-sensitive findings explain the overall dissociation. No central contradiction was found.

### D5: writing_quality
**warn.** Readable but with several minor word-choice/formatting issues. All nine template sections, contributions and reproducibility sections are present and ordered correctly. Prose interprets quantitative findings rather than merely listing them. Required sentence-quote and figure-caption corrections are small (R3/R4); the current version also compresses reproduction commands into ambiguous fragments. Student ID 21453430, leaving Hugo's ID blank, and a 50/50 contribution split have now been directly confirmed by the author; these are not fabrication concerns. The older writing-plan metadata should be updated to reflect that confirmation.

## Failure Condition Checks

### F1
Not fired: neither mandatory D2 nor mandatory D3 scores block.

### F2
**Fired:** mandatory D2 and D3 score warn. Action: request revision before treating the internal writing-quality gate as cleared.

### F3
Not fired: high-priority D1 and D4 score pass.

### F6
Not fired: normal-priority D5 is warn, not block.

### F4
Not fired: although D5 warns, mandatory D2/D3 do not yet pass, so the prerequisites for accept_with_dissent_note are absent.

### F5
Not fired: this is the initial evaluation; no two-round recurring mandatory block exists.

### F0
Not fired: not all mandatory dimensions pass.

## Review Body

### Strengths grounded in the draft

1. **Faithful hypothesis conclusions.** The report distinguishes sizeable PPL/context effects from uncertain grammar equivalence and unconfirmed scale growth. The abstract's last sentence and Sections 6--9 preserve the original questions rather than redefine successful outcomes.
2. **Informative controls.** KenLM's positive CG but zero same-target penalty is used to explain the metric's confound. The nearly flat Transformer k128 penalty then limits the scaling interpretation. This is critical analysis beyond a result inventory.
3. **Transparent research deviations.** Twelve-trial tuning, changed hidden widths, LSTM initialization, missing cross-host raw arrays, different copy manifests and the failed deep-control seed are visible in the main text. None is concealed by replacing or pooling runs.

### Required fixes

**R1 — Multiplicity correction must agree with the declared inferential scope.**

Locations: H1 paragraph (around line 151), H3 paragraph (around line 155), inference appendix/table, `support/build_report_evidence.py`, and derived `comparison_results.json`.

The report explicitly withholds confirmatory copy-test interpretation because realized samples differ. Its current H1 Holm adjusted p-values nevertheless use the small copy p-values to consume early Holm ranks. The H3 CG adjustment similarly benefits from the earlier caveated copy p-value. To preserve the prespecified six- and two-endpoint families conservatively without pretending the copy tests were usable, assign an unavailable/untested copy hypothesis p=1 when constructing the confirmatory correction. Keep any computed copy contrasts as descriptive diagnostics in a separately labelled field.

Independent recomputation from the saved raw seed-test p-values gives:

| Confirmatory CG contrast | Current Holm p | Holm p with untested copy slots p=1 |
|---|---:|---:|
| H1 S | .00779482 | .01299137 |
| H1 M | .00060748 | .0007289703 |
| H1 L | .00779482 | .01299137 |
| H3 L-minus-S | .19801504 | .39603008 |

All conclusions remain unchanged: H1 CG is significant, H3 CG is inconclusive. State the conservative missing-copy convention transparently. Bonferroni with the original family size is also defensible; do not quietly shrink the family to the available CG tests. If the authors instead retain the original nominal copy p-values, those corrected results must be explicitly labelled exploratory conditional calculations, not confirmatory family-controlled evidence.

**R2 — Make reproduction statements executable and distinguish archived evaluation from report regeneration.**

Location: Reproducibility Statement around line 226.

The listed `dataset.prepare`, `tokenizer.train`, etc. are not full shell commands as written. More materially, `transformer/evaluation/runner.py:prepare` references `PROJECT/'proposal.pdf'`, while the user moved the proposal to `report/proposal/proposal.pdf`. Existing frozen manifests also retain original input paths/hashes. `python -m transformer.evaluation prepare` cannot be advertised as an unconditional working reproduction command in the current tree. Do not modify historical model code or snapshots to hide that fact.

Give the CPU-only saved-results regeneration command explicitly, e.g. `python3 report/support/build_report_evidence.py` in the project environment, and document that it rebuilds report tables/figures without rescoring models. For a fresh training/scoring reproduction, cite full `python -m ...` commands and state that the compatible proposal path, original pinned runtime/source snapshot, and required untracked baseline/checkpoint arrays must first be restored/retrieved. Do not run the model pipeline as part of this report task.

**R3 — Close the qualitative example's quotations.**

Location: grammar-analysis example around line 194. Both sentences start with TeX opening quotes but lack closing `''`. Use complete paired quotes around the question and declarative candidate. Scores and example identity agree with Transformer L/seed13 saved BLiMP diagnostics.

**R4 — Align figure captions with actual panels and uncertainty marks.**

Figure 1 has PPL, CG and BLiMP panels; it has **no copy panel**, so remove its last sentence about copy comparisons or move that caveat to the relevant copy table/text. Its visible small seed points should be identified. Figure 2 left shows mean truncation penalties for six model/size groups without error bars, and right shows shuffle means with sample-SD error bars. Say so; currently the caption does not define those error bars. KenLM's zero penalty is evidence in text, but it is not plotted as a series; phrase that accordingly.

### Advisory fixes

- **A1 — Make item uncertainty explicit once.** Section 5 correctly separates seeds from evaluation items but could briefly name the archived conditional-bootstrap units (article clusters for WikiText and binary pairs for BLiMP, 10,000 replicates) rather than leave readers to the evaluator README. The main figures remain seed SD and should not be relabelled bootstrap intervals.
- **A2 — Clarify all-model objective wording.** “All models optimize teacher-forced next-token NLL” immediately precedes a neural loss. KenLM is fitted by count-based estimation; say both neural families optimize that objective and KenLM estimates the same conditional prediction distribution by modified Kneser--Ney.
- **A3 — Cite subword methodology at the tokenizer description.** `sennrich2016subword` exists in the bibliography but is unused; cite it if its adaptation claim fits the actual byte-level implementation. Do not imply that paper specifies all byte-level runtime details.
- **A4 — Restrict resource assertions to the measured experiments.** “No paid model APIs ... were used” can be read as applying to all AI-assisted writing tools. Prefer “The reported training and scoring used local/university compute, with no model API calls,” if supported by project records; do not invent an external funding declaration.
- **A5 — Update support-plan author metadata.** The author supplied the ID and split after the initial plan. Record the provenance rather than leave an obsolete “confirmation required” note inconsistent with the draft.

### Quantitative checks completed

- Abstract: 200 whitespace-delimited words.
- Main T PPL40.31/30.75/24.70 and BLiMP59.58/61.49/63.97% match saved primary aggregates.
- Baseline means, exact parameter counts, copy-length/random diagnostics, group sensitivities, deep failure/sensitivity and same-target values match support and saved summaries.
- H2 raw TOST p=.705078/.318753/.428015; Holm=.956260 each; quoted 90% intervals and all inconclusive conclusions agree.
- H1/H3 raw contrasts and descriptive 95% intervals agree; required correction is the confirmatory multiplicity convention, not the measured effects.
- Two PNG figures visually inspected; no source re-evaluation required.
- No present raw baseline arrays were assumed available, and no same-ID cross-model qualitative scores were invented.

## Evaluator Decision

evaluator_decision=request_revision

Highest fired condition: F2 (mandatory D2/D3 warn). R1--R4 are localized changes. This decision is an internal manuscript-quality recommendation only; it cannot certify the missing writer Phase 4a precommitment, full bibliographic integrity, the formal multi-reviewer stage, or final submission readiness.

---

## In-pair resolution check after localized revision

This is a continuation of the same Stage 2 internal writing check, not formal pipeline Stage 3' re-review. Source reviewed: `report/report.tex` SHA-256 `5b0389bf2cba024c22cb6b4d120aa285ecd5b9584564007cf0c92a8cb1f111cb`. Root is separately checking the compiled 6--8-page main-text limit.

| Required item | Resolution | Evidence checked |
|---|---|---|
| R1: coherent multiplicity convention | **Resolved** | Experimental Setup explicitly assigns untested copy slots p=1 for adjustment only, retaining six H1 and two H3 slots. Main H1 now prints .0130/.00073/.0130; H3 CG prints .396. Appendix matches. |
| R1: reproducible derived statistics | **Resolved** | `comparison_results.json` primary H1 CG adjusted values are .012991373554492138/.0007289702503076816/.012991373554492138; primary H3 CG=.3960300807641024. An independent rank-based Holm recomputation agrees. Copy primary `p_holm` fields are null, `holm_input_p`=1, and `inference_status` explicitly withholds confirmation. Previous nominal tests live only under `nominal_tests_for_audit_only`; the manuscript does not report nominal copy tests as confirmatory evidence. |
| R2: report-generation command | **Resolved** | Main Reproducibility Statement gives the actual CPU saved-summary command `python3 report/support/build_report_evidence.py` and the LaTeX build command. It does not promise a fresh model evaluation from absent artifacts. |
| R2: moved proposal and frozen prerequisites | **Resolved** | Appendix's evidence paragraph identifies the evaluator's legacy root `proposal.pdf` requirement, current `report/proposal/` location, frozen runtime/code checks, GPU environment and separately retrieved checkpoints/baseline arrays. Historical source/snapshots were not rewritten. |
| R3: example quotes | **Resolved** | Both question and declarative candidate have matched TeX opening/closing quotes; log-probability values and diagnostic selection are unchanged. |
| R4: figure-panel accuracy | **Resolved** | Figure1 caption now concerns PPL/CG/grammar only, with no absent copy panel implied. Figure2 explicitly identifies left seed means and right sample-SD error bars. |

### Remaining checks and bounded acceptance

- All **63** source hashes in the derived evidence file match the present source records. This verifies traceability of report calculations to the inspected saved summaries; it does not independently recreate their original model inference.
- All **11** cited bibliography keys exist and all bibliography entries are cited. This is a key-consistency check, not a complete external bibliographic/citation-context audit.
- Abstract remains **200** whitespace-delimited words. All nine numbered template sections and both required unnumbered statements are present and correctly ordered.
- H1 CG evidence remains significant under the conservative family correction. H2 remains inconclusive at all sizes. H3 remains unsupported. No new measured results were introduced by the revision.
- The deeper-LSTM discussion preserves the diverged primary seed, distinguishes the lower-rate/dropout sensitivity, and does not generalize its outcome to all deeper recurrent models. Context interpretation remains sensitivity evidence rather than a causal decomposition of the entire PPL gap.
- Remaining advisory wording: “All models optimize teacher-forced ... NLL” could be narrowed to “Both neural families ...” because KenLM uses count-based smoothing. Figure1's small individual-seed points could also be named explicitly. These are minor presentation items and do not change the substantive conclusions.

### Updated dimension statuses

D1 originality: **pass**. D2 methodological_rigor: **pass for manuscript reporting scope** after R1/R2. D3 evidence_sufficiency: **pass for the inspected saved-summary claims** after R1. D4 argument_coherence: **pass**. D5 writing_quality: **warn**, limited to the advisory wording/seed-point-caption details and pending compiled main-page check.

F1/F2/F3/F6/F5 do not fire on the revised manuscript. F4 applies because normal-priority D5 warns while mandatory dimensions pass; it takes precedence over F0.

`evaluator_decision=accept_with_dissent_note`

This accepts the revised writing at the internal quality-check level only. The absent pre-drafting writer Phase 4a artifact remains a workflow limitation. Formal academic-pipeline **Stage 2.5 integrity** and **Stage 3 external review remain pending**; this resolution check is not full-workflow certification or a statement that missing baseline raw arrays have been recovered.
