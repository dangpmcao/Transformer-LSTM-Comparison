# Report refinement — 2026-11-21

This revision follows the existing course template and report style. It revises the manuscript and rebuilds its PDF from saved evidence; it does not run training or inference, change the experimental protocol, or advance the historical academic-pipeline tracker. The earlier checkpoint and reader-revision files describe earlier versions. Current consistency and build checks are recorded in `refinement_checks.json`.

## Revision outline and paragraph roles

- Abstract and introduction: state the research question, describe completed work, and distinguish context sensitivity from grammar equivalence and causal attribution.
- Related work: explain how context ablations, repetition studies, and filtered-corpus research motivate different parts of the evaluation.
- Methods and setup: define repetition and sentence scoring, disclose initialization and scheduling details, and distinguish the common run budget from the exposure of validation-selected checkpoints.
- Results: show seed standard deviations for the main prediction and grammar scores; separate supported context-gain tests, descriptive copying results, and unresolved equivalence/scaling predictions.
- Analysis and conclusion: remove an unsupported causal interpretation of the independent-span control and describe the equivalence calculation as a retrospective precision check.
- Appendix: report all context-intervention means and seed standard deviations, plus a separate final-checkpoint sensitivity at exactly 250 million targets.

The methods paragraphs explain the question each measure answers before giving its interpretation. Results paragraphs follow effect, uncertainty, and hypothesis outcome. The conclusion answers the research question while preserving the limits of the design.

## Claim–evidence map

| Claim | Saved evidence | Assessment |
|---|---|---|
| Transformers have lower perplexity at every size | All 18 main `best_validation` views in `comparison_results.json`; the corresponding 18 `final_budget` views were also read | Supported for these runs and training procedures; the final-checkpoint comparison matches exposure exactly |
| Transformer position-based context gain is larger | Recomputed paired-seed differences and Holm correction, retaining all six planned family slots | Supported under the stated seed-level assumptions |
| Transformers are more sensitive to distant context | Same-target truncation and shuffle summaries for the same 69,376 targets | Supported descriptively; not a decomposition of the overall loss gap |
| Transformer repetition gains are larger | Natural and random repetition summaries | Descriptive; different realized samples prevent the intended confirmatory comparison |
| BLiMP accuracies are equivalent within two points | Recomputed TOST probabilities and 90% paired-seed intervals | Not established at any size |
| The context advantage grows with model size | Recomputed large-minus-small contrast and its corrected probability | Inconclusive; lack of significance does not disprove growth |
| Independent preceding text causes a negative control gain | The control compares two different target spans | Not isolated by this design; the causal wording was removed |

## Five-dimension self-review

1. Contribution: the report presents an empirical comparison, without claiming a new architecture or a causal explanation for most of the perplexity gap.
2. Clarity: notation, loss units, BOS/EOS treatment, seed variation, and checkpoint selection are explicit. Completed experiments are no longer presented as future work.
3. Experimental strength: every primary seed remains visible; standard deviations expose large-LSTM variability. Final checkpoints preserve the perplexity ranking. Parameter matching does not imply equal computation or optimization.
4. Evaluation completeness: copying samples differ; raw baseline predictions and matched-validation-quality weights remain unavailable. No missing analysis is presented as completed.
5. Method soundness: seed-level inference, conservative multiplicity correction, validation-only selection, the failed deep-LSTM run, and architecture-specific initialization remain disclosed. The precision check is not described as prospective power analysis.

## Validation and preservation

The numerical check reloads 18 primary and 18 final-checkpoint views, verifies all 63 pre-existing primary source hashes, checks all 18 primary seed rows and the six rows in each aggregate table, and independently recalculates the reported context, equivalence, and scaling tests using SciPy. It records hashes for 126 unique source files read during these checks. All 11 citation keys and all cross-references resolve; section order matches the template.

Tectonic 0.14.1 builds the report using cached resources. The PDF retains eight main-text pages and eleven pages overall. The abstract is 208 whitespace-delimited words. PDF inspection checks the main table, appendix tables, text boundaries, and figures. No model-code tests are warranted because the changes are confined to the report.

The existing comparison JSON, bibliography, figures, proposal, model implementations, datasets, checkpoints, and experimental results are preserved. The pre-revision manuscript and PDF were copied to `/tmp/nlp-report-refinement-20261121/` before editing. These checks establish manuscript consistency with the available summaries; they do not recover historical baseline predictions or constitute formal peer review.
