# NLP-Project

**Context, Not Grammar?** compares Transformers, LSTMs, and a 5-gram language
model: how much does context help them predict text and recognize English grammar?

The datasets, trained tokenizer, and shared model inputs are ready. Transformer
S/M/L training is complete for the exploratory runs: 250M prediction tokens per
model, seed 101, completed on 2026-11-11. See the
[training results](transformer/README.md#completed-training) for metrics and artifacts.
The production run matrix and final evaluation remain pending.

## Start here

| What you need | Read |
|---|---|
| Download data, understand saved files, or load batches | [Dataset guide](dataset/README.md) |
| Train, load, or inspect the text tokenizer | [Tokenizer guide](tokenizer/README.md) |
| Understand experiment settings and the timing-trial requirement | [Experiment guide](comparison/README.md) |
| Build and train the Transformer on a GPU | [Transformer guide](transformer/README.md) |
| Read the research question and planned experiments | [Project proposal](proposal.pdf) |

All commands below run from the repository root, the folder containing this file.

## Set up Python

Use Linux x86_64, **Python 3.12 with `venv` and pip**, Bash, internet access, and
several GB of free disk space. The current preparation steps run on CPU.

```bash
bash dataset/setup_environment.sh
source .venv/bin/activate
python -m pip check
```

The script installs [requirements.txt](requirements.txt), including Tokenizers
0.22.1 and CPU-only PyTorch. It expects `python3.12` on your PATH; otherwise use
`STAGE1_PYTHON=/path/to/python3.12 bash dataset/setup_environment.sh`.
Activate `.venv/` in every new terminal. GPU training uses the separately
validated [CUDA environment](transformer/README.md#run-an-exploratory-timingtraining-trial-on-one-gpu).

## Understand the data

| Dataset | Download source | Purpose |
|---|---|---|
| WikiText-103, `wikitext-103-raw-v1` | [Salesforce on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext) | Training text, validation text, and final test text |
| BLiMP, 67,000 sentence pairs | [alexwarstadt/blimp on GitHub](https://github.com/alexwarstadt/blimp) | Final grammar evaluation only |

The source versions and file checksums are fixed in
[dataset/sources.lock.json](dataset/sources.lock.json).

```text
parent/
├── NLP-Project/                  # This repository
│   ├── dataset/                  # Python scripts, not the downloaded corpus
│   ├── tokenizer/artifacts/      # Trained text-to-token converter
│   └── comparison/               # Shared experiment settings
└── dataset/                      # Corpus and generated model inputs
    ├── raw/                      # Original downloads
    └── processed/                # Articles, token IDs, and window indexes
```

Each checkout uses its sibling `dataset/` folder automatically. To use another
location, follow the [custom data-path instructions](dataset/README.md#use-another-data-location).

## Prepare a new checkout

Download the original data, reconstruct articles, and verify them:

```bash
python -m dataset.download
python -m dataset.prepare
python -m dataset.validate
```

The tokenizer converts text to token IDs. The trained byte-level BPE tokenizer
has **16,384 IDs**, including padding/start/end/unknown tokens, and was fitted
only on **28,472 training articles**. Reuse `tokenizer/artifacts/` when supplied.
If that directory is absent, train it:

```bash
python -m tokenizer.train
```

Then convert articles to IDs, split them into model examples, check one article
end-to-end, and save the experiment settings:

```bash
python -m dataset.encode_articles
python -m dataset.build_windows
python -m dataset.check_model_inputs
python -m comparison.experiment_settings
```

A **prediction window** is one model example with up to 512 next-token targets.
It stays within one article, starts with a beginning token, and ignores padding
in the loss. Both neural models will use the same examples and training order.
See the dataset guide for a concrete input/target example and custom paths.

## Current preparation results

| Split | Articles | Prediction targets, including article-end tokens | Windows |
|---|---:|---:|---:|
| Train | 28,472 | 122,022,510 | 252,574 |
| Validation | 60 | 255,099 | 529 |
| Test | 60 | 291,710 | 602 |

The saved tokenizer check reports exact text recovery for all 28,592 articles.
The model-input check passes for batch sizes 1 and 2. These are data-correctness
results; completed Transformer validation results are reported in the
[Transformer guide](transformer/README.md#completed-training).

The settings snapshot currently has `production_ready: false`. A measured timing
trial covering both Transformer and LSTM must establish exact training-token
budgets and batch accumulation before production training. The completed
exploratory runs used 250M tokens each. The
[experiment guide](comparison/README.md) explains the production requirements.

## Recheck existing work and share it

Creation commands refuse to replace existing tokenizer, token-array, window,
and settings-snapshot outputs. Verify existing outputs instead:

```bash
python -m dataset.encode_articles --check
python -m dataset.build_windows --check
python -m comparison.experiment_settings --check
python -m pytest dataset/tests tokenizer/tests comparison/tests -q
```

Share the source, requirements, source lock, and all eight files in
`tokenizer/artifacts/` to reuse the trained tokenizer. Create a fresh `.venv/` on
the other machine. The external corpus and generated inputs can be recreated;
archive the experiment settings snapshot with any future training runs.
