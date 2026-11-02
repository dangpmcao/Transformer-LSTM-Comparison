"""BLiMP scoring rules using synthetic pairs and fake models with known probabilities."""

import json
import math

import pytest
import torch
import yaml

from comparison.blimp_scorer import (
    evaluate_blimp, load_pairs, make_window_batch, neural_window_scorer, score_pairs,
    split_target_windows, sum_log_probabilities, summarize,
)
from comparison.protocol import DEFAULT_PROTOCOL
from dataset.common import file_record, read_json, sha256_file, write_json
from dataset.prepare import EXPECTED_GROUPS, validate_blimp
from tokenizer.core import encode_text, load_tokenizer
from tokenizer.tests.test_tokenizer import fixture_data, bundle  # Reuse the established tiny corpus fixtures.

VOCAB = 300
BOS, EOS, PAD = 1, 2, 0


class BigramModel(torch.nn.Module):
    """Causal fake: each token's next-token distribution comes from a fixed known table."""

    def __init__(self):
        super().__init__()
        self.table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(7))
        self.inputs = []

    def forward(self, batch):
        self.inputs.append(batch["input_ids"].clone())
        return self.table[batch["input_ids"]]

    def expected(self, window):
        log_probs = torch.log_softmax(self.table.double(), dim=-1)
        previous = [BOS, *window[:-1]]
        return sum(log_probs[a, b].item() for a, b in zip(previous, window, strict=True))


def pair(good, bad, *, paradigm="p", group="g", pair_id="0"):
    return {"paradigm": paradigm, "pair_id": pair_id, "group": group, "sentence_good": good, "sentence_bad": bad}


def result(good, bad, *, targets=(1, 1), paradigm="p", group="g"):
    return {"paradigm": paradigm, "group": group, "good_log_probability": good, "bad_log_probability": bad,
            "good_targets": targets[0], "bad_targets": targets[1], "identical_candidates": False,
            "correct": good > bad, "length_normalized_correct": good / targets[0] > bad / targets[1]}


@pytest.fixture
def tokenizer(bundle):
    return load_tokenizer(bundle[2])


@pytest.fixture
def blimp_root(tmp_path):
    """Synthetic layout with one extra raw label that the mapping folds into argument_structure."""
    root = tmp_path / "blimp_data"
    terms = sorted(EXPECTED_GROUPS) + ["s-selection"]
    (root / "raw/blimp/data").mkdir(parents=True)
    files = []
    for number, term in enumerate(terms):
        path = root / f"raw/blimp/data/paradigm_{number:02d}.jsonl"
        rows = [{"sentence_good": f"Good sentence {i} of {term}.", "sentence_bad": f"Bad sentence {i} of {term}.",
                 "field": "syntax", "linguistics_term": term, "UID": path.stem, "pairID": str(i)} for i in range(2)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        files.append({"path": f"raw/blimp/data/{path.name}", "size": path.stat().st_size,
                      "sha256": sha256_file(path), "source": "blimp"})
    lock = tmp_path / "sources.lock.json"
    expected = {"expected_paradigms": len(terms), "pairs_per_paradigm": 2, "expected_groups": 12}
    write_json(lock, {"blimp": expected, "files": files})
    validate_blimp(root, expected)
    return root, lock


@pytest.fixture
def settings_file(bundle, tmp_path):
    config = yaml.safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    config["inputs"]["tokenizer_sha256"] = sha256_file(bundle[2] / "tokenizer.json")
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("length,counts", [(1, [1]), (511, [511]), (512, [512]), (513, [512, 1]),
                                           (1024, [512, 512]), (1025, [512, 512, 1])])
def test_windows_keep_every_target_in_order(length, counts):
    targets = list(range(length))
    windows = split_target_windows(targets)
    assert [len(window) for window in windows] == counts
    assert sum(windows, []) == targets


def test_batch_shifts_inputs_and_masks_padding_like_shared_windows():
    batch = make_window_batch([[5, 6, EOS], [7]])
    assert batch["input_ids"].tolist() == [[BOS, 5, 6], [BOS, PAD, PAD]]
    assert batch["target_ids"].tolist() == [[5, 6, EOS], [7, PAD, PAD]]
    assert batch["loss_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["attention_mask"].tolist() == batch["loss_mask"].tolist()
    assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]


@pytest.mark.parametrize("windows", [[], [[]], [[5], []], [[5] * 513]])
def test_invalid_window_batches_are_rejected(windows):
    with pytest.raises(ValueError, match="targets"):
        make_window_batch(windows)


def test_scores_equal_known_log_probabilities_including_eos():
    model = BigramModel().eval()
    windows = [[10, 20, EOS], [30, EOS], [40, 50, 60, 70, EOS]]
    scores = neural_window_scorer(model)(windows)
    assert scores == pytest.approx([model.expected(window) for window in windows], abs=1e-4)
    assert all(score < 0 for score in scores)


def test_sum_matches_hand_computed_uniform_distribution():
    logits = torch.zeros(1, 3, 4)
    batch = make_window_batch([[1, 2, 3]])
    assert sum_log_probabilities(logits, batch["target_ids"], batch["loss_mask"]).item() == pytest.approx(3 * math.log(1 / 4))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
def test_scores_do_not_depend_on_batching_or_input_order(batch_size):
    model = BigramModel().eval()
    windows = [[10, EOS], [40, 50, 60, 70, EOS], [30, 31, EOS], [20, EOS]]
    expected = neural_window_scorer(model, batch_size=1)(windows)
    assert neural_window_scorer(model, batch_size=batch_size)(windows) == pytest.approx(expected, abs=1e-4)
    backward = neural_window_scorer(model, batch_size=batch_size)(windows[::-1])
    assert backward[::-1] == pytest.approx(expected, abs=1e-4)


def test_padding_logits_cannot_change_scores_but_scored_nans_are_rejected():
    batch = make_window_batch([[5, EOS], [6, 7, 8, EOS]])
    logits = torch.zeros(2, 4, VOCAB)
    logits[0, 2:] = float("nan")
    totals = sum_log_probabilities(logits, batch["target_ids"], batch["loss_mask"])
    assert totals.tolist() == pytest.approx([2 * -math.log(VOCAB), 4 * -math.log(VOCAB)])
    logits[0, 1] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        sum_log_probabilities(logits, batch["target_ids"], batch["loss_mask"])


def test_module_in_training_mode_is_rejected():
    model = BigramModel()
    with pytest.raises(ValueError, match="eval"):
        neural_window_scorer(model)([[5, EOS]])
    assert neural_window_scorer(model.eval())([[5, EOS]])[0] < 0


def test_targets_are_sentence_ids_plus_eos_with_implicit_bos(tokenizer):
    seen = []

    def record(windows):
        seen.extend(windows)
        return [-1.0] * len(windows)

    row = score_pairs([pair("The fox.", "The dog.")], tokenizer, record)[0]
    assert seen == [encode_text(tokenizer, text, add_bos=True, add_eos=True)[1:] for text in ("The fox.", "The dog.")]
    assert row["good_targets"] == len(seen[0]) and row["bad_targets"] == len(seen[1])
    assert BOS not in seen[0] and seen[0][-1] == EOS


def test_only_a_strictly_higher_good_score_is_correct(tokenizer):
    good, bad = encode_text(tokenizer, "The fox.", add_eos=True), encode_text(tokenizer, "The dog.", add_eos=True)
    for values, correct in [((-1.0, -2.0), True), ((-2.0, -1.0), False), ((-1.5, -1.5), False)]:
        table = {tuple(good): values[0], tuple(bad): values[1]}
        row = score_pairs([pair("The fox.", "The dog.")], tokenizer, lambda ws: [table[tuple(w)] for w in ws])[0]
        assert row["correct"] is correct
        assert row["good_log_probability"] == values[0] and row["bad_log_probability"] == values[1]


def test_identical_candidates_tie_even_if_the_scorer_is_noisy(tokenizer):
    calls = iter(range(100))
    rows = score_pairs([pair("Same text.", "Same text."), pair("Other one.", "Same text.", pair_id="1")],
                       tokenizer, lambda windows: [-1.0 - next(calls) for _ in windows])
    assert rows[0]["identical_candidates"] is True and rows[0]["correct"] is False
    assert rows[0]["good_log_probability"] == rows[0]["bad_log_probability"]
    assert rows[1]["bad_log_probability"] == rows[0]["bad_log_probability"]
    assert summarize(rows)["counts"]["identical_candidate_pairs"] == 1


def test_overlength_sentences_score_every_target_in_independent_windows(tokenizer):
    long_text = "a " * 700
    targets = encode_text(tokenizer, long_text, add_eos=True)
    assert len(targets) > 512
    model = BigramModel().eval()
    rows = score_pairs([pair(long_text, "Short.")], tokenizer, neural_window_scorer(model))
    expected = sum(model.expected(window) for window in split_target_windows(targets))
    assert rows[0]["good_targets"] == len(targets)
    assert rows[0]["good_log_probability"] == pytest.approx(expected, abs=1e-3)
    assert all((inputs[:, 0] == BOS).all() for inputs in model.inputs)
    assert summarize(rows)["counts"]["overlength_sentences"] == 1


@pytest.mark.parametrize("bad_scores", [[], [-1.0], [-1.0, float("nan")], [-1.0, float("inf")]])
def test_scorers_returning_bad_values_are_rejected(tokenizer, bad_scores):
    with pytest.raises(ValueError, match="window score"):
        score_pairs([pair("The fox.", "The dog.")], tokenizer, lambda windows: bad_scores)


def test_summary_reports_overall_groups_and_paradigms():
    rows = [
        result(-1.0, -2.0, paradigm="a", group="x"), result(-2.0, -1.0, paradigm="a", group="x"),
        result(-1.0, -2.0, paradigm="b", group="x"), result(-3.0, -3.0, paradigm="c", group="y"),
    ]
    summary = summarize(rows)
    primary = summary["primary"]
    assert primary["overall"] == {"pairs": 4, "correct": 2, "accuracy": 0.5}
    assert primary["groups"]["x"] == {"pairs": 3, "correct": 2, "accuracy": 2 / 3}
    assert primary["groups"]["y"]["accuracy"] == 0
    assert primary["paradigms"]["a"]["accuracy"] == 0.5 and primary["paradigms"]["b"]["accuracy"] == 1
    assert summary["counts"]["tied_pairs"] == 1


def test_sensitivities_handle_length_differences():
    rows = [
        # Good loses on total score, but wins per target once its extra tokens are normalized away.
        result(-6.0, -4.0, targets=(6, 2), paradigm="a"),
        result(-1.0, -2.0, targets=(3, 3), paradigm="b"),
    ]
    sensitivities = summarize(rows)["sensitivities"]
    equal = sensitivities["equal_bpe_length_pairs"]
    assert equal["overall"] == {"pairs": 1, "correct": 1, "accuracy": 1.0}
    assert equal["paradigms"]["a"] == {"pairs": 0, "correct": 0, "accuracy": None}
    assert sensitivities["length_normalized_log_probability"]["overall"]["correct"] == 2
    assert summarize(rows)["primary"]["overall"]["correct"] == 1


def test_load_pairs_keeps_source_order_and_maps_twelve_groups(blimp_root):
    root, lock = blimp_root
    pairs = load_pairs(root, lock)
    assert len(pairs) == 26 and len({row["group"] for row in pairs}) == 12
    assert pairs[0]["pair_id"] == "0" and pairs[1]["pair_id"] == "1"
    assert "s-selection" not in {row["group"] for row in pairs}
    assert sum(row["group"] == "argument_structure" for row in pairs) == 4


def test_changed_download_and_stale_mapping_are_rejected(blimp_root):
    root, lock = blimp_root
    mapping = root / "processed/blimp/paradigm_groups.json"
    rows = read_json(mapping)
    rows[0]["source_linguistics_term"] = "something_else"
    write_json(mapping, rows)
    with pytest.raises(ValueError, match="Stale"):
        load_pairs(root, lock)
    with (root / "raw/blimp/data/paradigm_00.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="mismatch"):
        load_pairs(root, lock)


def test_evaluation_publishes_traceable_results_and_refuses_overwrite(bundle, blimp_root, settings_file, tmp_path):
    root, lock = blimp_root
    output = tmp_path / "results/transformer_s_seed13"
    model = BigramModel().eval()
    arguments = dict(model_label="fake bigram", data_root=root, tokenizer_dir=bundle[2],
                     protocol=settings_file, source_lock=lock)
    summary = evaluate_blimp(neural_window_scorer(model), output, **arguments)
    rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(path.name for path in output.iterdir()) == ["pairs.jsonl", "summary.json"]
    assert read_json(output / "summary.json") == summary
    assert summary["pairs_file"] == file_record(output / "pairs.jsonl", output)
    assert summary["protocol_sha256"] == sha256_file(settings_file)
    assert summary["tokenizer_sha256"] == sha256_file(bundle[2] / "tokenizer.json")
    assert summary["model_label"] == "fake bigram" and summary["status"] == "PASS"
    assert len(rows) == 26 and summary["primary"]["overall"]["pairs"] == 26
    assert summary["primary"]["overall"]["correct"] == sum(row["correct"] for row in rows)
    assert len(summary["primary"]["groups"]) == 12 and len(summary["primary"]["paradigms"]) == 13
    assert all(row["correct"] == (row["good_log_probability"] > row["bad_log_probability"]) for row in rows)
    before = sha256_file(output / "summary.json")
    with pytest.raises(FileExistsError, match="Refusing"):
        evaluate_blimp(neural_window_scorer(model), output, **arguments)
    assert sha256_file(output / "summary.json") == before


def test_failed_scoring_leaves_no_published_output(bundle, blimp_root, settings_file, tmp_path):
    root, lock = blimp_root
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="window score"):
        evaluate_blimp(lambda windows: [float("nan")] * len(windows), output, model_label="broken", data_root=root,
                       tokenizer_dir=bundle[2], protocol=settings_file, source_lock=lock)
    assert not output.exists()


def test_changed_protocol_rules_or_tokenizer_are_rejected(bundle, blimp_root, settings_file, tmp_path):
    root, lock = blimp_root
    arguments = dict(model_label="fake", data_root=root, tokenizer_dir=bundle[2], source_lock=lock)
    config = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
    config["evaluation"]["blimp"]["include_eos"] = False
    changed_rules = tmp_path / "changed_rules.yaml"
    changed_rules.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="BLiMP rules"):
        evaluate_blimp(lambda windows: [-1.0] * len(windows), tmp_path / "a", protocol=changed_rules, **arguments)
    with pytest.raises(ValueError, match="shared tokenizer"):
        evaluate_blimp(lambda windows: [-1.0] * len(windows), tmp_path / "b", protocol=DEFAULT_PROTOCOL, **arguments)
    with pytest.raises(ValueError, match="label"):
        evaluate_blimp(lambda windows: [-1.0] * len(windows), tmp_path / "c", protocol=settings_file,
                       **{**arguments, "model_label": ""})
