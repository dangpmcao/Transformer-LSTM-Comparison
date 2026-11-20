# Academic-paper internal evaluator: Phase 6a precommitment

Contract: `academic_paper/evaluator_full/v1`, baseline v3.6.6. This is the internal writing-quality gate, not formal academic-pipeline Stage 3 peer review and not an integrity certification. Metadata: *Context, Not Grammar? Where Transformers Beat LSTMs in Small-Scale Language Modeling*; NLP; empirical course report; approximately 4,500 words; 6--8 main-text pages; supplied report template. The draft `report/report.tex` has not been read in this turn. The writer's Phase 4a precommitment was not supplied with this assignment and is not yet visible among support files; its consistency check remains pending for Phase 6b. No draft is scored here.

## Contract Paraphrase

### D1: originality
The paper should make a concrete, appropriately scoped empirical contribution to the context-versus-grammar question, distinguish that contribution from the nearest relevant prior work, and use the discussion to explain what the new comparison adds. A course replication can contribute controlled measurements and critical evidence without claiming a new architecture or unsupported novelty.

### D2: methodological_rigor
The research design must answer the stated question and be described well enough to reproduce. Data boundaries, architectural matching, checkpoint selection, random seeds, training budgets, metric definitions, uncertainty units, and acknowledged protocol deviations should support valid interpretation. Design limitations must be analyzed rather than merely named; causal or equivalence language needs the corresponding design or tests.

### D3: evidence_sufficiency
Claims must be supported by appropriate scholarly sources or traceable project evidence, including negative or inconclusive findings. Quantitative reports must preserve the measured population, units, checkpoint views, campaign roles, and uncertainty source. Unavailable raw baseline evidence cannot be upgraded to independent verification, and secondary analyses cannot silently replace the proposal's primary hypotheses. Search-bounded novelty or absence claims are acceptable only with documented search provenance and nearest-prior-work context.

### D4: argument_coherence
The abstract, introduction, experiments, results, analysis, and conclusion should form one consistent argument that answers the proposed hypotheses. Tables should lead to interpretation rather than replace it. Alternative explanations, contradictory evidence, and qualification of the title should be integrated into the conclusion rather than hidden in an appendix.

### D5: writing_quality
The report must read as clear, precise, professional NLP research prose and follow the user's exact course template, length constraint, and citation style. Figures and tables should be intelligible and referenced in the narrative; contributions and reproducibility information should be concrete. Stylistic flaws should be distinguished from scientific failures, with missing metadata flagged honestly rather than invented.

## Scoring Plan

### D1: originality
- dimension_id: D1
- what_to_look_for: A specific contribution and substantive engagement with relevant context-use, repetition, and grammar-evaluation work show what this controlled empirical report adds.
- what_triggers_block: Contribution statement is vague or absent, or the paper merely restates known conclusions with no added empirical evidence or defensible perspective.
- what_triggers_warn: Contribution exists but is weakly differentiated from cited prior work, or discussion engagement is uneven across relevant sources.

### D2: methodological_rigor
- dimension_id: D2
- what_to_look_for: Replicable data/model/training/scoring definitions, explicit primary-versus-sensitivity roles, valid seed-level tests and equivalence logic, and engagement with copy-sample and baseline-access limitations support the stated question.
- what_triggers_block: Method does not match the research question, significant design flaws are concealed, limitations are not discussed, or analysis cannot be replicated from the description, including misrepresented exact pairing or causal/equivalence claims unsupported by design.
- what_triggers_warn: Method is fundamentally sound but insufficiently detailed, some choices lack justification, or limitations are mentioned but not engaged with, including incomplete descriptions of sampled-copy comparability or missing raw baseline evidence.

### D3: evidence_sufficiency
- dimension_id: D3
- what_to_look_for: Core quantitative conclusions trace to saved evidence or documented calculations, related-work claims cite appropriate sources, and both supportive and contrary outcomes remain within their stated verification limits.
- what_triggers_block: Multiple core claims lack evidence, Discussion inferences exceed what the data supports, or cherry-picking of favorable evidence occurs, including fabricated results, treating nonsignificance as equivalence, or declaring absent raw baselines independently verified.
- what_triggers_warn: Core claims have evidence but some secondary claims lack support, citation density is uneven across sections, or secondary numbers cannot be traced to saved sources or declared calculations.

### D4: argument_coherence
- dimension_id: D4
- what_to_look_for: Abstract-to-conclusion consistency, explicit answers to H1/H2/H3, and result-linked interpretation sustain one research argument while recognizing scope limits and counterevidence.
- what_triggers_block: Multiple logical gaps between sections, a conclusion disconnected from the introduction's research question, or self-contradictory passages leave the central claim unsupported.
- what_triggers_warn: Overall logic holds but some transitions feel forced, or minor inconsistencies between section claims obscure which findings are supported, contradicted, or inconclusive.

### D5: writing_quality
- dimension_id: D5
- what_to_look_for: Professional connected prose, the supplied ordered section structure, a <=250-word abstract, 6--8 main-text pages, consistent references, and clearly interpreted tables meet the course-report format.
- what_triggers_block: Difficult to understand, numerous grammar errors, colloquial register, citation format inconsistent across the manuscript, or missing required template sections prevents a usable report.
- what_triggers_warn: Readable but with several grammar or word-choice issues, some paragraphs overly long, minor citation format inconsistencies, or unresolved author metadata or length checks needs a limited editorial correction.

## Decision and boundary commitments

Scores will be `block`, `warn`, or `pass`, with manuscript locations and evidence rather than impressionistic assertions. Mandatory D2 or D3 block fires F1; mandatory warn fires F2. High-priority D1/D4 block fires F3. Normal-priority D5 block fires F6. A high/normal warn with mandatory passes fires F4 and permits `accept_with_dissent_note`. F5 is reserved for a recurring mandatory block after the second in-pair revision attempt, not the initial review. Otherwise all mandatory dimensions passing with no blocks permits F0/`accept`, unless the higher-severity F4 applies. The numerical five-dimension rubric may supplement this decision but cannot override the contract priorities. No final publication readiness or formal pipeline Stage 2.5/3 completion will be implied by this internal gate.

[PRE-COMMITMENT-ACKNOWLEDGED]
