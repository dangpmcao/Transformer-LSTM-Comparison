"""Shared-input invariants using synthetic corpora; never write production artifacts."""

import json
import math

import numpy as np
import pytest
import torch

from dataset.common import file_record, read_json, sha256_file, write_json
from dataset.encode import encode, load_cache, split_arrays
from dataset.windows import (
    ArticleWindows, WindowOrder, budget_mask, build_windows, initial_lstm_state,
    load_windows, make_index, masked_cross_entropy, window_loader,
)
from tokenizer.core import encode_text, load_tokenizer
from tokenizer.tests.test_tokenizer import fixture_data, bundle  # Reuse the established tiny corpus fixtures.


@pytest.fixture
def shared(bundle, tmp_path):
    data, _, artifacts = bundle
    tokens = tmp_path / "tokens"
    encode(data, artifacts, tokens)
    build_windows(tokens)
    return data, artifacts, tokens


def test_cache_matches_text_ids_and_one_eos_per_article(shared):
    data, artifacts, root = shared
    tokenizer = load_tokenizer(artifacts)
    manifest = load_cache(root)
    for split in ("train", "validation", "test"):
        ids, offsets = split_arrays(root, split)
        articles = [json.loads(line) for line in (data / f"processed/articles/{split}.jsonl").read_text().splitlines()]
        assert ids.dtype == np.dtype("<u2") and offsets.dtype == np.dtype("<u8")
        for i, article in enumerate(articles):
            assert ids[int(offsets[i]):int(offsets[i + 1])].tolist() == encode_text(tokenizer, article["text"], add_eos=True)
        assert np.count_nonzero(ids == 2) == len(articles)
        assert not np.isin(ids, [0, 1, 3]).any()
        assert manifest["splits"][split]["prediction_tokens"] == len(ids)


def test_encode_refuses_overwrite_and_wrong_data(shared):
    data, artifacts, root = shared
    with pytest.raises(FileExistsError):
        encode(data, artifacts, root)
    manifest = read_json(data / "manifest.json")
    manifest["wikitext_splits"]["train"]["text_bytes"] += 1
    write_json(data / "manifest.json", manifest)
    write_json(data / "validation.json", {"status": "PASS", "manifest_sha256": sha256_file(data / "manifest.json")})
    with pytest.raises(ValueError, match="identity"):
        encode(data, artifacts, root.parent / "other")


def test_corrupt_cache_and_windows_are_rejected(shared):
    _, _, root = shared
    path = root / "train.tokens.npy"
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="Changed token cache"):
        load_cache(root)


@pytest.mark.parametrize("length,counts", [(0, [1]), (1, [2]), (511, [512]), (512, [512, 1]),
                                         (513, [512, 2]), (1023, [512, 512]), (1024, [512, 512, 1])])
def test_window_lengths_cover_content_and_eos_without_dropping_targets(length, counts):
    index = make_index(np.array([0, length + 1], dtype="<u8"))
    assert index[:, 2].tolist() == counts
    assert index[:, 1].tolist() == list(range(0, length + 1, 512))
    assert int(index[:, 2].sum()) == length + 1


def test_windows_do_not_cross_articles_and_have_stable_ids(shared):
    _, _, root = shared
    dataset = ArticleWindows(root, "train")
    reconstructed = [[] for _ in range(len(dataset.offsets) - 1)]
    for i in range(len(dataset)):
        item = dataset[i]
        article, local, valid = map(int, dataset.index[i])
        mask = item["loss_mask"]
        assert item["window_id"] == f"train:{article:06d}:{local:09d}"
        assert item["input_ids"][0] == 1
        assert torch.equal(item["input_ids"][1:valid], item["target_ids"][:valid - 1])
        assert item["input_ids"][valid:].eq(0).all()
        assert item["target_ids"][~mask].eq(0).all()
        assert torch.equal(item["position_ids"], torch.arange(512))
        reconstructed[article].extend(item["target_ids"][mask].tolist())
    for i, ids in enumerate(reconstructed):
        assert ids == dataset.tokens[int(dataset.offsets[i]):int(dataset.offsets[i + 1])].tolist()
        assert ids[-1] == 2 and ids.count(2) == 1


def test_eos_only_tail_starts_from_bos_and_next_article_is_independent(shared):
    _, _, root = shared
    ids = np.array([4] * 512 + [2, 5, 2], dtype="<u2")
    offsets = np.array([0, 513, 515], dtype="<u8")
    np.save(root / "train.tokens.npy", ids, allow_pickle=False)
    np.save(root / "train.offsets.npy", offsets, allow_pickle=False)
    manifest = read_json(root / "manifest.json")
    manifest["splits"]["train"].update({
        "articles": 2, "content_tokens": 513, "prediction_tokens": 515,
        **{key: file_record(root / f"train.{key}.npy", root) for key in ("tokens", "offsets")},
    })
    write_json(root / "manifest.json", manifest)
    output = root / "exact-windows"
    build_windows(root, output)
    dataset = ArticleWindows(root, "train", output)
    assert len(dataset) == 3
    assert dataset[1]["target_ids"][0] == 2
    assert dataset[1]["input_ids"][0] == 1
    assert dataset[1]["loss_mask"].sum() == 1
    assert dataset[1]["input_ids"][1:].eq(0).all()
    assert dataset[2]["input_ids"][:2].tolist() == [1, 5]
    assert dataset[2]["target_ids"][:2].tolist() == [5, 2]


def test_paired_order_is_independent_of_torch_rng_and_eval_is_fixed(shared):
    _, _, root = shared
    dataset = ArticleWindows(root, "train")
    first = list(WindowOrder(dataset, training=True, seed=13, epoch=2))
    torch.rand(123)
    assert first == list(WindowOrder(dataset, training=True, seed=13, epoch=2))
    assert sorted(first) == list(range(len(dataset)))
    assert list(WindowOrder(dataset, training=False)) == list(range(len(dataset)))
    validation = ArticleWindows(root, "validation")
    with pytest.raises(ValueError, match="never be shuffled"):
        WindowOrder(validation, training=True)
    state = torch.get_rng_state().clone()
    batches = list(window_loader(dataset, 2, training=True, seed=13, epoch=2))
    assert torch.equal(state, torch.get_rng_state())
    assert [identifier for batch in batches for identifier in batch["window_id"]] == [dataset[i]["window_id"] for i in first]
    assert sum(len(b["window_id"]) for b in batches) == len(dataset)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_masked_loss_matches_hand_computation_and_ignores_padding(batch_size):
    logits = torch.tensor([[[0., math.log(3.)], [float("nan"), float("nan")]]] * batch_size, requires_grad=True)
    labels = torch.tensor([[1, 0]] * batch_size)
    mask = torch.tensor([[True, False]] * batch_size)
    summed, count = masked_cross_entropy(logits, labels, mask)
    assert count == batch_size
    assert summed.item() == pytest.approx(-math.log(0.75) * batch_size)
    (summed / count).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[:, 1].eq(0).all()


def test_loss_uses_target_count_not_mean_of_batch_means():
    logits = torch.tensor([[[0., 0.], [0., math.log(3.)]]])
    labels = torch.tensor([[1, 1]])
    summed, count = masked_cross_entropy(logits, labels, torch.ones_like(labels, dtype=torch.bool))
    assert (summed / count).item() == pytest.approx((math.log(2) - math.log(.75)) / 2)
    with pytest.raises(ValueError, match="zero prediction"):
        masked_cross_entropy(logits, labels, torch.zeros_like(labels, dtype=torch.bool))
    with pytest.raises(ValueError, match="Expected colocated"):
        masked_cross_entropy(logits, labels.float(), torch.ones_like(labels, dtype=torch.bool))


@pytest.mark.parametrize("remaining", [0, 1, 3, 4, 100])
def test_budget_mask_counts_exact_nonpadding_targets(remaining):
    original = torch.tensor([[True, False, True], [True, True, False]])
    result = budget_mask(original, remaining)
    assert int(result.sum()) == min(remaining, 4)
    assert not (result & ~original).any()
    assert original.sum() == 4
    if remaining == 3:
        assert result.tolist() == [[True, False, True], [True, False, False]]


def test_both_lstm_states_are_fresh_and_on_requested_dtype_device():
    ids = torch.ones(2, 512, dtype=torch.long)
    hidden, cell = initial_lstm_state(ids, 2, 7, dtype=torch.float64)
    assert hidden.shape == cell.shape == (2, 2, 7)
    assert hidden.dtype == cell.dtype == torch.float64
    assert hidden.device == cell.device == ids.device
    hidden.fill_(1)
    cell.fill_(2)
    next_hidden, next_cell = initial_lstm_state(ids, 2, 7, dtype=torch.float64)
    assert not next_hidden.any() and not next_cell.any()


def test_index_integrity_and_no_overwrite(shared):
    _, _, root = shared
    with pytest.raises(FileExistsError):
        build_windows(root)
    path = root / "windows/test.windows.npy"
    index = np.load(path)
    index[0, 1] += 1
    np.save(path, index, allow_pickle=False)
    # Even if a bad writer updates the file hash, semantic coverage must fail.
    manifest_path = root / "windows/manifest.json"
    manifest = read_json(manifest_path)
    manifest["splits"]["test"]["index"] = file_record(path, path.parent)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="cover articles"):
        load_windows(path.parent, root)
