# Dataset preparation and model inputs

This folder contains Python code. Downloaded text and generated arrays belong
in the **external data folder**, which defaults to `../dataset/` relative to the
repository root. Run the commands here from the repository root, after following
[the Python setup instructions](../README.md#set-up-python).

The full preparation flow is:

```text
Original downloads → complete articles → token IDs → prediction windows → batches
```

BLiMP follows a separate path: download and verify its sentence pairs, then
reserve them for final grammar evaluation. It is never used to fit the tokenizer
or train a language model.

## Datasets and download sources

| Dataset | Source/version | What is downloaded |
|---|---|---|
| WikiText-103 | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-103-raw-v1` | Four Parquet files containing official train/validation/test text, plus the source README |
| BLiMP | [alexwarstadt/blimp](https://github.com/alexwarstadt/blimp) | 67 JSONL files with 1,000 grammatical/ungrammatical sentence pairs each, plus the source README |

[sources.lock.json](sources.lock.json) records the exact source revisions,
download URLs, expected file sizes, and checksums. WikiText is fixed to revision
`b08601e04326c79dfdd32d625aee71d232d685c3`; BLiMP is fixed to
`3e56b06fcabca9b30822fc66435fca6b1aa40bb1`.
Use the **raw** WikiText variant: the vocabulary-filtered `.tokens` variant has
already replaced some text with unknown-token markers.

## Download, prepare, and verify text

```bash
python -m dataset.download
python -m dataset.prepare
python -m dataset.validate
```

| Step | What it does | Main output |
|---|---|---|
| Download | Fetch fixed source files, resume partial downloads, verify sizes/checksums | `raw/`, `downloads.json` |
| Prepare | Join WikiText source rows into complete articles; record duplicates and BLiMP grammar groups | `processed/articles/`, `manifest.json` |
| Validate | Independently check source files, exact text preservation, article boundaries/IDs, and BLiMP records | `validation.json` with `status: PASS` |

Case, headings, accents, whitespace, and Unicode remain unchanged. Subsection
headings and table/equation fragments stay inside their article. The official
train, validation, and test splits are preserved.

| WikiText split | Original text rows | Reconstructed articles |
|---|---:|---:|
| Train | 1,801,350 | 28,472 |
| Validation | 3,760 | 60 |
| Test | 4,358 | 60 |

BLiMP has 67,000 pairs in 67 test categories, called *paradigms*. The separate
`paradigm_groups.json` file maps them to 12 broader grammar groups; original
source labels remain unchanged.

## Convert articles to model inputs

First reuse or train the shared tokenizer as explained in the
[tokenizer guide](../tokenizer/README.md). Then run:

```bash
python -m dataset.encode_articles
python -m dataset.build_windows
python -m dataset.check_model_inputs
```

Encoding reads all three WikiText splits using the already trained tokenizer.
It does not learn new tokenization rules. It stores each article's content IDs
followed by **one end-of-sequence token (EOS)**. It then creates indexes for
**prediction windows**: independent model examples with up to 512 targets.

Example with three content tokens:

```text
Stored article IDs:  [token_A, token_B, token_C, EOS]

Model input:         [BOS,     token_A, token_B, token_C, PAD, ...]
Prediction targets:  [token_A, token_B, token_C, EOS,     PAD, ...]
Loss mask:           [true,    true,    true,    true,    false, ...]
```

BOS marks the beginning of each window; EOS marks the end of each article;
PAD fills unused positions. Every window has 512 tensor positions. Padding
contributes neither loss nor training-token count. A long article is split into
nonoverlapping windows, each starting from BOS with reset model state. If an
article has exactly 512 content tokens, its EOS gets a separate one-target window.
Windows never cross article boundaries.

| Split | Content tokens | Prediction targets including EOS | Windows |
|---|---:|---:|---:|
| Train | 121,994,038 | 122,022,510 | 252,574 |
| Validation | 255,039 | 255,099 | 529 |
| Test | 291,650 | 291,710 | 602 |

The difference between the two token counts is exactly the number of articles:
each article contributes one EOS. BOS and padding are excluded from both counts.

The input check verifies the first real training article through tokenization,
window creation, batching, and loss/gradient calculations. It checks batch sizes
1 and 2, tensor types, shifted targets, position resets, and fresh LSTM hidden
and cell states. This is a small correctness check, not model training.

## Where files are saved

Here `<split>` means `train`, `validation`, or `test`.

```text
../dataset/
├── raw/
│   ├── wikitext103_raw/                  # Original Parquet files and README
│   └── blimp/data/                       # Original sentence-pair JSONL files
├── processed/
│   ├── articles/
│   │   ├── <split>.jsonl                 # Complete article text and metadata
│   │   └── <split>.manifest.json         # Article metadata without the full text
│   ├── blimp/paradigm_groups.json        # Mapping to 12 grammar groups
│   ├── duplicates.json                  # Duplicate-article report
│   └── tokens/
│       ├── <split>.tokens.npy            # Content IDs + one EOS per article
│       ├── <split>.offsets.npy           # Where each article starts/ends in the IDs
│       ├── manifest.json                # Token counts, file hashes, source identity
│       ├── checks.json                  # One-article model-input check result
│       └── windows/
│           ├── <split>.windows.npy       # Which article/targets form each window
│           └── manifest.json            # Window counts, order, and file hashes
├── downloads.json                       # Verified downloaded-file list
├── manifest.json                        # Prepared-data inventory and source records
├── validation.json                      # Full text/data verification result
└── ATTRIBUTION.json                     # Credits and recorded license information
```

Existing `environment.json`, `kenlm_build.json`, and `checks/` contain records
from earlier environment/KenLM checks. They are not tokenized training data;
current preparation does not require the historical KenLM build to be present.

### Article records and IDs

Each line in `<split>.jsonl` is one JSON object:

| Field | Meaning |
|---|---|
| `article_id` | Stable ID, such as `train:000000`; numbering starts at zero within each split |
| `split` | `train`, `validation`, or `test` |
| `title`, `text` | Original article title and complete preserved text |
| `row_start`, `row_end` | Source-row range: start included, end excluded |
| `heading_row` | Source-row number containing the article heading |
| `text_sha256` | Checksum of the exact UTF-8 article text |

### Token arrays and article offsets

`.npy` is NumPy's array format. Token IDs are stored as unsigned 16-bit integers;
article offsets use unsigned 64-bit integers. The model loader converts selected
windows to PyTorch `int64` tensors. Memory mapping reads portions of the arrays
without loading the whole corpus into RAM.

If offsets are `[0, 4, 9]`, article 0 occupies token-array positions `[0:4]` and
article 1 occupies `[4:9]`. The final offset is the total array length. There
are always `article_count + 1` offsets.

### Window indexes and IDs

Each row of `<split>.windows.npy` contains three integers:

| Column | Meaning |
|---|---|
| `article_index` | Zero-based article number within the split |
| `target_start` | Zero-based position of the first target within that article's stored IDs |
| `valid_targets` | Number of real prediction targets in this window, from 1 to 512 |

For example, `[0, 512, 100]` selects 100 targets beginning at position 512 of
article 0. Its training-window ID is `train:000000:000000512`. Article numbers
are formatted with six digits; target starts use nine digits. Validation/test
windows remain in article order and then target-start order. Training uses a
repeatable shuffle based on the run seed and epoch.

## Load a training batch

```python
from dataset.common import DEFAULT_DATA_ROOT
from dataset.model_inputs import NextTokenWindowDataset, create_batch_loader

training_windows = NextTokenWindowDataset(DEFAULT_DATA_ROOT / "processed/tokens", "train")
training_batches = create_batch_loader(
    training_windows, batch_size=2, training=True, seed=13, epoch=0,
)
batch = next(iter(training_batches))

assert batch["input_ids"].shape == (2, 512)
assert batch["target_ids"].shape == (2, 512)
```

| Batch field | Meaning |
|---|---|
| `input_ids` | Tokens given to the model |
| `target_ids` | Correct next-token answers |
| `loss_mask` | Positions included in loss and token counts |
| `attention_mask` | Real input positions versus padding; causal attention must also be enforced by the model |
| `position_ids` | Positions 0–511, restarted for each window |
| `window_id` | Stable example IDs for tracing and resuming data order |

For validation/test, choose that split and leave `training=False`. Their order
cannot be shuffled. See the [experiment guide](../comparison/README.md) for loss
normalization, LSTM state initialization, and exact token-budget handling.

## Use another data location

Pass the same data folder to the download, preparation, validation, and encoding
steps. Point window construction and checking to the resulting token folder:

```bash
python -m dataset.download --data-root /path/to/data
python -m dataset.prepare --data-root /path/to/data
python -m dataset.validate --data-root /path/to/data
python -m dataset.encode_articles --data-root /path/to/data
python -m dataset.build_windows --token-dir /path/to/data/processed/tokens
python -m dataset.check_model_inputs --data-root /path/to/data \
  --token-dir /path/to/data/processed/tokens
```

If the tokenizer must also be trained, use
`python -m tokenizer.train --data-root /path/to/data` before encoding. Encoding
and input checking accept `--tokenizer-dir /path/to/tokenizer` for a copied tokenizer
bundle. Window construction accepts `--output-dir`; input checking accepts
`--window-dir` and `--report-file` for a custom index folder or check report.

## Recheck existing files

```bash
python -m dataset.encode_articles --check
python -m dataset.build_windows --check
python -m pytest dataset/tests tokenizer/tests comparison/tests -q
```

These array/index checks verify saved files without rebuilding them. New token
and window outputs refuse to overwrite existing directories. Downloading
verifies existing files and resumes partial ones. Text preparation regenerates
its articles and inventory; validation and input checking refresh their reports.
If you regenerate upstream files, recheck downstream files and create a new
experiment-settings snapshot rather than silently reusing an older snapshot.

The official data contain one exact train/test article overlap and seven BLiMP
pairs whose candidates are identical. They are recorded, retained, and must be
reported when interpreting final evaluation. BLiMP group mapping does not alter
the original pairs. Source credits and the conflicting WikiText license statements
are preserved in `ATTRIBUTION.json`.

### Compare with the reference hashes

[expected_outputs.lock.json](expected_outputs.lock.json) lists the SHA-256 and
byte size of every generated token, offset, and window array (paths relative to
`<data root>/processed`). If your hashes match, your token IDs, article
boundaries, and 512-target windows are byte-identical to the reference. Its
`status` says whether a second machine has confirmed them yet. Compare the
array hashes, not `manifest.json` hashes: manifests contain creation times.

```bash
cd ../dataset/processed   # or your custom data root
sha256sum tokens/*.npy tokens/windows/*.npy
```

A mismatch in `tokens.npy` points to tokenization (check the tokenizer hash and
`tokenizers`/`numpy` versions); a match there but not in `windows.npy` points to
window construction.

See the [naming guide](../NAMING.md) for the older equivalent commands and the
meaning of common abbreviations. Both interfaces use the same implementations;
existing saved data and settings snapshots remain compatible.

## Code in this folder

| File or folder | Responsibility |
|---|---|
| `setup_environment.sh` | Create the local Python environment and install pinned packages |
| `sources.lock.json` | Fixed download URLs, revisions, sizes, and checksums |
| `download.py` | Download and verify original dataset files |
| `prepare.py` | Reconstruct articles and write duplicate/grouping reports |
| `validate.py` | Independently verify the original and prepared text |
| `encode_articles.py` | Public command for creating/checking token arrays |
| `build_windows.py` | Public command for creating/checking prediction-window indexes |
| `check_model_inputs.py` | Public command for the one-article batch/loss check |
| `model_inputs.py` | Public dataset, batch-loader, loss, and LSTM-state names |
| `encode.py`, `windows.py`, `check_inputs.py` | Existing implementations used by the public interfaces |
| `common.py` | Shared paths, JSON input/output, and checksum helpers |
| `environment.py` | Optional report of installed packages and machine capabilities |
| `tests/` | Synthetic-data and command-line integration tests |

The optional `python -m dataset.environment` command refreshes
`../dataset/environment.json`. It reports the observed environment; it does not
install packages or establish readiness for GPU model training.
