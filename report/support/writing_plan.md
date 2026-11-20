# Report writing plan and argument blueprint

Stage: academic-pipeline 2 (WRITE), using the supplied course template. No Stage 2.5 integrity certification or Stage 3 external review has been performed.

## Configuration

- Title: Context, Not Grammar? Where Transformers Beat LSTMs in Small-Scale Language Modeling.
- Authors: Cao Pham Minh Dang and Hugo Centeno Sanz. Student IDs and contribution percentages are author-confirmation items; do not fabricate.
- Language: English. Output: report/report.tex, numeric references with plain bibliography style.
- Structure and presentation: exactly the supplied report_template.tex section order and 10pt, 0.72-inch margin formatting; 6–8 main-text pages, references/appendix excluded. Target approximately 4,200–4,800 words, to be adjusted against the actual rendered page count.
- Source hierarchy: proposal and TA feedback define research questions; frozen configuration, source code, and saved JSON/CSV establish methods and measurements; READMEs are navigation aids only.
- Primary comparison: Transformer mb16 best-validation S/M/L × seeds 13/37/73 vs matched production two-layer LSTM runs. Transformer mb64 and seed-101 trials are sensitivity/history rather than extra seeds. Deep LSTM is a depth control, with failures retained. KenLM is a finite-history control.

## Section allocation and purpose

1. Abstract (200–240 words): task, controlled comparison, principal results, limits and bounded conclusion.
2. Introduction (400): why perplexity cannot answer grammar/context question; predeclared H1/H2/H3 and main finding.
3. Related Work (300): architecture/context, corpus and grammar evaluation, length bias, finite-history baseline.
4. Data and Task Formulation (400): train-only tokenizer and official splits, boundaries/alignment/masks, grammar labels and source terms.
5. Approach (600): actual matched architecture parameter table, teacher-forced objective, CG/copy/same-target/BLiMP definitions.
6. Experimental Setup (550): 250M token budget, three seeds, architecture-specific S tuning, validation-only selection, compute, uncertainty, comparability boundaries.
7. Results (600): main seed-mean table plus paired effects/intervals only where defensible; answer H1/H2/H3 explicitly.
8. Analysis (800): same-target intervention vs position confound; copy length/control and 5-gram invariance; grammar groups/length bias/disagreements; deep control and matched-quality feasibility; interpretation beyond listing.
9. Limitations and Responsible Use (400): three-seed precision, fixed budgets not compute equality, task validity, corpus bias and source terms, methodological deviations.
10. Conclusion (200): state which predictions survive, which are contradicted/inconclusive, and few evidence-based next steps.
11. Team Contributions/AI and Reproducibility (250): truthful work split and material AI-writing disclosure, source/code/artifact paths and reproduction entry points.
12. Appendix: every primary seed, supplementary controls and source traceability sufficient to audit main text without dumping all experiments.

## Claim–evidence–reasoning chains

- H1 context use: early/late CG alone confounds target difficulty. Establish positive architecture differences with saved seed metrics, then test the same targets under full/truncated/distant-shuffled context. Interpret convergence across measures as evidence of greater context sensitivity, without attributing the entire perplexity gap causally to long context.
- H1 repetition: repeated-span gains should exceed finite-history boundary gains and independent-span controls. Report natural Copy(100) as primary; random-token sequences and other lengths diagnose the mechanism and are secondary. Verify sample-manifest compatibility before paired inference.
- H2 grammar: compare full BLiMP summed-score accuracy and equal-BPE-length/length-normalized sensitivity. A smaller grammar gap than the perplexity/context gap is not equivalence; distinguish paired seed TOST support, contradictory sign, and inconclusive confidence intervals.
- H3 scaling: show architecture-gap trajectory and prespecified L-minus-S change, distinguishing better absolute large-model scores from a demonstrably increasing architecture gap. Keep depth failures and unmatched validation-quality controls visible.
- Fairness: actual non-embedding counts plus shared token windows address parameter/data confounds; separate S hyperparameters, depth stability, fixed tokens and GPU time prevent claiming architecture-only or equal-compute causality beyond the design.

## Draft quality gates

Every central number must map to a saved source/key. No new model training/scoring. No invented citations, IDs, contributions, qualitative examples or formal review certifications. Explicitly record proposal deviations and missing cross-model comparability. Repeated-count tables should be accompanied by explanatory prose. Avoid changing the template to manufacture page compliance.

## Completed Stage 2 draft and internal corrections

The report follows all template sections, includes a 200-word abstract, actual parameter counts, all 18 primary seed rows, two evidence-derived figures, paired CG/grammar contrasts, and explicit H1/H2/H3 interpretations. Confirmed author metadata: Dang ID 21453430, Hugo ID omitted by request, contributions 50/50.

Internal advisory evaluation identified a real multiplicity correction: because copy comparisons are withheld from confirmatory inference after discovering different realized samples, their original family slots are assigned 1 for adjustment only. This preserves six H1 slots and two H3 slots without using favorable copy ranks. CG significance survives; H3 remains inconclusive. The manuscript does not call these placeholder adjustment values measured copy p-values.

The original root proposal was moved by the user into report/proposal/. Model evaluator preparation still expects root proposal.pdf, so the main reproducibility statement gives the working CPU summary-to-report command; the appendix records model-rerun prerequisites. The training/evaluation implementation and frozen outputs were not modified.

Writer Phase 4a physical paper-blind precommitment was not emitted before drafting and cannot be honestly reconstructed afterward. The internal evaluation is advisory; no formal pipeline integrity or external-review certification is asserted. Stage 2.5 and later pipeline checkpoints remain the parent orchestrator's responsibility.
