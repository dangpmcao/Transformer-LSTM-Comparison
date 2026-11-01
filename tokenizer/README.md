# Text tokenizer

The tokenizer converts text into integer token IDs and can convert those IDs
back into the original text. All project models will use this same converter.
The trained tokenizer is saved in [`artifacts/`](artifacts/).

Run commands from the repository root after completing the
[Python setup](../README.md#set-up-python). The
[naming guide](../NAMING.md) explains abbreviations used in saved files.

## What was trained

This is **byte-level BPE (byte-pair encoding)**, trained from scratch with
Hugging Face Tokenizers 0.22.1. It starts with all 256 byte symbols and learns
which adjacent pieces to join. A token may be punctuation, part of a word, a
whole word, or a byte; it is not necessarily a word.

```text
4 reserved tokens + 256 byte symbols + 16,124 learned merges = 16,384 token IDs
```

Only the **28,472 WikiText training articles** in
`../dataset/processed/articles/train.jsonl` are used to learn the vocabulary.
They contain 539,295,549 UTF-8 text bytes. Articles are read in saved order,
without sampling or shuffling. Validation/test articles are checked only after
the vocabulary is fixed. BLiMP is excluded from both tokenizer training and checks.

The [configuration](config.json) fixes these choices:

| Setting | Meaning |
|---|---|
| `vocab_size: 16384` | Total token IDs, including the four reserved IDs |
| `min_frequency: 2` | Minimum adjacent-pair frequency for a learned merge |
| `normalization: null` | Keep case, accents, whitespace, and Unicode unchanged |
| `add_prefix_space: false` | Do not add a space at the start of text |
| `use_regex: true` | Use ByteLevel's initial text-splitting rules |
| `bpe_dropout: null` | Do not randomly skip merge operations |
| `automatic_special_tokens: false` | Add beginning/end tokens only when explicitly requested |
| `batch_size: 64` | Read 64 articles at a time; this is unrelated to model-training batch size |
| `parallelism: false` | Train on one CPU thread; no GPU is needed |

| Reserved token | ID | Meaning |
|---|---:|---|
| `<pad>` | 0 | Fill unused positions in a model batch |
| `<bos>` | 1 | Beginning of a prediction sequence |
| `<eos>` | 2 | End of an article or scored sequence |
| `<unk>` | 3 | Unknown-token fallback; not needed by the checked corpus |

The text `"literal <bos> stays text"` is treated as ordinary content. It does
not insert the beginning-token ID. The tokenizer does not automatically pad,
truncate, or choose model-example boundaries; the
[dataset code](../dataset/README.md) does that.

## Train or recheck

Reuse the supplied `tokenizer/artifacts/` directory. If it is absent, prepare
and validate the dataset, then train:

```bash
python -m tokenizer.train
```

The command verifies the data, learns merges from every training article,
saves/reloads the tokenizer, and checks exact text recovery for every WikiText
article. It also tests Unicode, whitespace, literal reserved-token strings,
batch versus individual encoding, and explicit BOS/EOS insertion. The complete
bundle is published only after all checks pass.

If the bundle already exists, recheck it instead:

```bash
python -m tokenizer.check
```

Training refuses to overwrite an existing output directory. It has no resumable
intermediate checkpoint: BPE learns vocabulary/merge rules rather than neural
weights. To compare another configuration, copy `config.json`, change its
`vocab_size`, `min_frequency`, or `batch_size`, and select a separate output:

```bash
python -m tokenizer.train --config /path/to/experiment-config.json \
  --output /path/to/new-tokenizer
```

Both commands accept `--data-root /path/to/data`. Training accepts `--output`
and `--config`; checking accepts `--artifacts /path/to/tokenizer`. Those existing
option names remain unchanged because they are part of the verified tokenizer
implementation. The naming guide maps *artifacts* to the trained-tokenizer files.

## Saved training result

The saved check from **2026-10-28** reports **PASS**: all **28,592 articles**
recover exactly after encoding/decoding, all **13 test strings** pass, and no
reserved/unknown IDs appear in article content.

| Split | Content tokens | Tokens per word | Median article length in tokens | Longest article in tokens |
|---|---:|---:|---:|---:|
| Train | 121,994,038 | 1.203 | 3,202 | 31,858 |
| Validation | 255,039 | 1.192 | 3,248.5 | 21,496 |
| Test | 291,650 | 1.209 | 3,176.5 | 16,962 |

Here a word is a whitespace-separated string. For example, 1.203 tokens/word
means about 120 tokens per 100 such words. Counts exclude BOS, EOS, and padding;
the dataset's later prediction-target counts include one EOS per article.
These checks establish correct conversion, not language-model quality.

## What each saved file contains

Keep all eight files in [`artifacts/`](artifacts/) together:

| File | Purpose |
|---|---|
| [`tokenizer.json`](artifacts/tokenizer.json) | Main trained converter: text splitting, vocabulary, merge rules, and decoding |
| [`vocab.json`](artifacts/vocab.json) | Token string → token ID dictionary; also included in `tokenizer.json` |
| [`merges.txt`](artifacts/merges.txt) | Learned pair-joining rules in priority order; also included in `tokenizer.json` |
| [`config.json`](artifacts/config.json) | Exact configuration used for this training run |
| [`special_tokens.json`](artifacts/special_tokens.json) | The four reserved token names and IDs |
| [`manifest.json`](artifacts/manifest.json) | Training date, source data, configuration, code/file checksums, and pre-save test encodings |
| [`checks.json`](artifacts/checks.json) | Check status, date, text-recovery results, and checksums identifying the checked bundle |
| [`statistics.csv`](artifacts/statistics.csv) | Per-split token counts, tokens/word, bytes/token, and article-length summaries |

For example, the rule `h e` joins `h` and `e` into `he`. The character `Ġ` in
some token strings represents a space byte in ByteLevel's readable mapping;
decoding restores a normal space.

In `statistics.csv`, `article_tokens_p50` is the median article length;
`p0`/`p100` are the minimum/maximum. `p25`, `p75`, `p95`, and `p99` are other
percentiles. A percentile can be fractional because it is interpolated between
article lengths. `bytes_per_token` measures UTF-8 text bytes per token, not
compressed file size.

## Load and use the tokenizer

```python
from tokenizer.core import encode_text, load_tokenizer

tokenizer = load_tokenizer()
text = "Hello, world! Literal <bos> stays text."

content_token_ids = encode_text(tokenizer, text)
assert tokenizer.decode(content_token_ids, skip_special_tokens=False) == text

sequence_token_ids = encode_text(tokenizer, text, add_bos=True, add_eos=True)
assert sequence_token_ids == [1, *content_token_ids, 2]
```

For several texts at once:

```python
texts = ["First article.", "Second article."]
encoded_texts = tokenizer.encode_batch(texts, add_special_tokens=False)
batch_token_ids = [encoded.ids for encoded in encoded_texts]
```

Always use `load_tokenizer()`: it checks the saved files and matching `PASS`
report, and restores the setting that keeps literal `<bos>` text as content.
Tokenizers 0.22.1 does not preserve that setting when loading directly with
`Tokenizer.from_file(...)`.

Loading needs the project code and Python environment, but not the corpus.
For a copied bundle, pass its folder as a `Path`:

```python
from pathlib import Path
tokenizer = load_tokenizer(Path("/path/to/copied/artifacts"))
```

Rechecking needs the same prepared corpus. It accepts identical data copied or
prepared on another machine even when its absolute path or preparation date
differs. Source versions, split file contents, and article/byte counts must match.
`tokenizer.check` refreshes the check/statistics reports without changing learned
merges, vocabulary, or the training record.
