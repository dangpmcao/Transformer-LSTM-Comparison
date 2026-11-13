# Context, Not Grammar?
## Where Transformers Beat LSTMs in Small-Scale Language Modeling

**COMP 4221: Natural Language Processing** • Fall 2026 • Prof. Yi R. (May) Fung — Project Proposal

**Authors:** Cao Pham Minh Dang and Hugo Centeno Sanz

---

## 1. Problem and motivation

**Question.** When classic LSTMs and causal Transformers are trained at comparable small scales, does the Transformer advantage in language modeling arise mainly from *using long context*, rather than *recognizing sentence-level grammar*? Overall perplexity cannot distinguish these capabilities. The project makes a controlled architectural comparison, not a claim about all recurrent models.

**Hypotheses.**

- **H1:** Transformers achieve larger late-context loss reductions and repeated-text gains than size-matched LSTMs.
- **H2:** Their overall grammatical minimal-pair accuracies are equivalent within a predeclared ±2 percentage-point margin.
- **H3:** The context advantage grows across model sizes while the grammar gap remains small.

Any outcome can falsify these predictions.

## 2. Related work

Kaplan et al. (2020) report that the Transformer–LSTM loss gap grows at later context positions and larger model sizes. Khandelwal et al. (2018) measure how recurrent models use nearby and distant tokens; Olsson et al. (2022) connect repeated-pattern completion to attention. Patil et al. (2024) motivate separating linguistic generalization from perplexity in controlled training. BLiMP (Warstadt et al., 2020) supplies targeted grammar evaluations, while Ueda et al. (2024) identify token-length bias in minimal-pair scoring. Unlike a perplexity-only benchmark, our design tests *both skills on the same matched model checkpoints*, across sizes and seeds. Results apply to classic LSTMs, not modern variants such as xLSTM.

## 3. Data or tasks

**Training data:** WikiText-103 `wikitext-103-raw-v1`, using official article-level train/validation/test splits. Train a *single* 16,384-type byte-level BPE tokenizer on training data and reuse it for every model. Validation selects checkpoints and hyperparameters; the test split is held out for final reporting.

**Evaluation tasks:**

1. Position-wise next-token loss on 512-token test windows.
2. Exact repetitions of held-out text or sampled tokens, with span lengths $L \in \{25, 50, 100, 200\}$.
3. BLiMP's 67 grammatical minimal-pair paradigms (12 phenomenon groups), used *only for evaluation*.

We will report natural-text and random-token repeats separately.

## 4. Method and baseline

**Controlled change:** sequence architecture — decoder-only, causal Transformer *versus* classic two-layer LSTM. Both learn the same next-token cross-entropy objective from scratch. We hold fixed the WikiText split, tokenizer, 512-token context, embedding dimension at each size, training-token budget, validation-based selection protocol, and number of seeds. The LSTM hidden state is reset every window, matching the Transformer's available history; input/output embeddings are tied in both families.

| Size | Total params. | Transformer               | LSTM                          |
|------|---------------|---------------------------|-------------------------------|
| S    | ~7.3M         | 4 layers, d = 256, 4 heads  | 2 layers, h = 460, d = 256  |
| M    | ~16.9M        | 6 layers, d = 384, 6 heads  | 2 layers, h = 864, d = 384  |
| L    | ~49.8M        | 8 layers, d = 640, 10 heads | 2 layers, h = 1680, d = 640 |

Non-embedding parameters are targeted to match within ±5% (verified in code). The experiment has **18 main runs**: 2 architectures × 3 sizes × 3 seeds. The **baselines/controls** are a 5-gram KenLM model (only four tokens of history) and a size-M four-layer LSTM (h = 580; three seeds) to assess the depth confound. Each neural run receives the same number of training tokens (target ~250M); at size S, each architecture receives an equal eight-configuration learning-rate/dropout search. We record model size, training FLOPs, time, and validation loss: *equal tokens do not imply equal compute*.

## 5. Evaluation

**Primary measurements** (Transformer minus LSTM at each size):

- **Long-context gain (H1):** plot mean loss $\bar{\ell}(t)$ by token position and calculate $\mathrm{CG} = \mathrm{mean}_{t=1:32}\,\bar{\ell}(t) - \mathrm{mean}_{t=385:512}\,\bar{\ell}(t)$. Larger CG indicates greater late-context improvement.
- **Copy gain (H1):** $\mathrm{Copy}(L) = \ell(\text{first span}) - \ell(\text{repeated span})$; $L = 100$ is the primary endpoint. Compare performance as repeated spans lengthen.
- **Grammar (H2):** BLiMP accuracy, testing whether $\log P(\text{grammatical}) > \log P(\text{ungrammatical})$ using summed subword log-probabilities; report overall, phenomenon-group, and length-bias sensitivity results.
- **Scaling (H3):** plot architecture gaps for S/M/L and compare scores at matched validation perplexity across saved checkpoints. Report test perplexity as a complementary measure.

**Evidence and falsification.** Present every seed and 95% uncertainty intervals, with Holm-adjusted comparisons across sizes. H1 fails if the Transformer shows no reliable context/copy advantage; H3 fails if the predicted increase with size is absent. Test H2 with two one-sided equivalence tests using the fixed ±2-point margin; a non-significant difference is *not* proof of equivalence. Report cases where models disagree on repeated names or BLiMP minimal pairs, and verify the 5-gram has no long-history gain.

## 6. Resources and risks

**Resources:** PyTorch, Hugging Face Datasets/Tokenizers, KenLM, statistical Python packages, and university GPU compute; no paid model API or purchased dataset. The original training estimate is *under three GPU-days*, subject to a small-model timing pilot; log storage, checkpoints and GPU costs must be checked. WikiText-103 and BLiMP are public (source-listed licenses: CC BY-SA 3.0 and CC BY 4.0); cite the datasets and verify terms before publishing artifacts.

**Risks and mitigations:**

- Sequential LSTM training may exceed the compute budget → prioritize the S milestone, then M.
- Tuning might favor one architecture → equal search budgets and documented grid checks.
- Three seeds may yield unstable estimates → report seed-level spread, add S seeds if needed.
- Minimal-pair comparisons can be affected by BPE length → analyze equal-token-length pairs and alternate scoring.

These limitations will be reported rather than hidden.

## 7. Work plan

- **Weeks 1–2:** build and test shared data/BPE/model pipeline (**Dang**); build 5-gram baseline and BLiMP scorer (**Hugo**).
- **Weeks 3–4:** equal-budget S tuning and three-seed S experiments (**Dang: training/context**; **Hugo: grammar/statistics**; **both: fairness review**).
- **Milestone — minimum viable pipeline:** matched S Transformer and LSTM, three seeds each, plus 5-gram; position-wise loss, CG and BLiMP accuracy with seed-level results and uncertainty intervals.
- **Weeks 5–7:** M/L models, deep-LSTM control and copy/perturbation tests (**Dang**; **Hugo** maintains BLiMP evaluation).
- **Weeks 8–10:** statistics and figures (**Hugo**), reproducibility checks (**Dang**), and final report/presentation (**both**).

Release versioned code, configurations, data identifiers and seeds.

## Key references

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020).
- Khandelwal et al., *Sharp Nearby, Fuzzy Far Away* (ACL 2018).
- Olsson et al., *In-Context Learning and Induction Heads* (2022).
- Patil et al., *Filtered Corpus Training* (TACL 2024).
- Warstadt et al., *BLiMP* (TACL 2020).
- Ueda et al., *Token-Length Bias in Minimal-Pair Paradigm Datasets* (LREC-COLING 2024).

---

## 8. TA feedback on the proposal

> This project compares small Transformers and LSTMs trained under matched conditions to ask whether the Transformer advantage comes mainly from using longer context rather than sentence-level grammar. This is a strong research question, and separating context use from grammar makes the study more informative than a single perplexity comparison. Shared tokenization, multiple seeds, and the deeper-LSTM control show real attention to fairness. Your distinction between equivalence and a non-significant difference is especially careful. The following changes would strengthen the study:

1. **Add a same-target context intervention.** Position-wise loss can change because tokens at different positions differ in difficulty. Score the same target tokens with full versus truncated or shuffled distant context, so the long-context interpretation does not rely only on early-versus-late loss.
2. **Validate the parameter table in executable model definitions**, including any projection needed to tie embeddings when LSTM hidden and embedding dimensions differ. Report actual non-embedding counts and the tuning budget for each size; small-model tuning may not transfer equally across architectures.
3. **Profile one full small-model run before launching the 18-run matrix.** Prioritize replicated S and M comparisons over an underpowered large-model extension. Define uncertainty separately over training seeds and evaluation items, and assess whether the planned data support the ±2-point grammar equivalence margin.
4. **In the final presentation and evaluation, connect the matched-model results to each hypothesis.** Present actual parameter and compute counts, same-target context ablations, BLiMP results, and seed-level variation, and clearly distinguish supported conclusions from inconclusive equivalence tests.

### Action items derived from the feedback

- [ ] Implement a same-target context ablation: evaluate identical target tokens with full, truncated, and shuffled distant context (complements CG for H1).
- [ ] Add an output projection to the LSTM (h → d) when `h ≠ d` so embeddings can be tied, and count it in the parameter budget.
- [ ] Write a script/test that instantiates every model config and prints actual total and non-embedding parameter counts; check the ±5% match.
- [ ] Document the tuning budget per size and per architecture.
- [ ] Run and time one full S run (both architectures) before launching the full matrix; update the compute estimate.
- [ ] Prioritize S and M with replicated seeds; treat L as optional/extension.
- [ ] Separate uncertainty sources: across training seeds vs. across evaluation items (e.g., bootstrap over BLiMP pairs).
- [ ] Power check: estimate whether BLiMP size and seed count can support the ±2-point TOST equivalence margin.
- [ ] Final report: map each result explicitly to H1/H2/H3 and separate supported conclusions from inconclusive equivalence tests.
