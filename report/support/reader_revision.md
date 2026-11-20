# Revision for first-time readers

This is an author-requested editorial revision of the writing-stage draft. It does not advance the academic pipeline to formal integrity review or peer review.

## Revision outline

- State the hypothesis, comparison, and planned tests in the abstract before presenting completed results.
- Define the task, model sizes, random seeds, tokenizer, context windows, and scoring units at first use.
- Explain context, repetition, grammar, and scaling comparisons using scientific descriptions rather than experiment identifiers.
- Replace abbreviated model labels in all tables and figures with full family and size names.
- Keep reproducibility information useful to readers, with public repository links; remove local paths, commands, and workspace-management details from the manuscript.
- Preserve measurements, inference, caveats, author information, and template sections.

## Paragraph roles

Abstract: motivation, hypothesis and test design, completed comparison, principal findings, bounded conclusion.

Introduction: what perplexity cannot distinguish; controlled design; three substantive hypotheses; contribution and limits.

Methods and setup: define the data and measures, explain why each control is needed, and state selection and statistical procedures before reporting results.

Results and analysis: connect each comparison to its hypothesis, then explain boundary effects, repetition behavior, heterogeneous grammar results, uncertainty, and depth-control limits.

Conclusion: stronger context use is supported; grammar equivalence and increasing advantage with scale remain unresolved.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Transformers predict text better at all three sizes | Recorded test perplexities for all 18 main runs | Supported under the stated training procedures |
| Transformers benefit more from available context | Context-gain differences and fixed-target truncation/shuffling results | Supported; not a decomposition of the total perplexity gap |
| Repetition behavior differs substantially | Recorded natural/random repetition gains | Descriptive because the architectures used different sampled sequences |
| Grammar equivalence has not been established | Three paired equivalence tests and their intervals | Inconclusive, not evidence that no grammar difference exists |
| Increasing context advantage with scale is unresolved | Large-minus-small contrast and near-constant 128-token penalty | Not statistically supported |

## Five-dimension writing check

1. Contribution: the report separates prediction quality, context use, and grammar without claiming a new architecture.
2. Clarity: all reader-facing experiment identifiers, local paths, shell commands, model-family letters, and unexplained size initials were removed. Established technical terms are defined or explained.
3. Experimental strength: the rewrite retains all sizes and repetitions, actual parameter matching, and the different-GPU caveat.
4. Evaluation completeness: the copy-sample difference, failed depth-control run, unavailable individual baseline predictions, and missing matched-quality comparison remain visible.
5. Method soundness: the rewrite preserves validation-only selection, seed-level inference, conservative multiplicity adjustment, and the distinction between equivalence and nonsignificance.

The requested abstract sentence is framed as a hypothesis, followed by completed results that qualify it. It does not assert that grammar equivalence or a causal majority explanation was demonstrated.

The previous manuscript and PDF are preserved as editorial history. Numerical result records and research implementations were not renamed or modified.
