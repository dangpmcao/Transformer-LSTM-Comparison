"""Protocol checks with invented fixtures; no benchmark examples used for fitting."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset.common import contained_path
from dataset.download import download_one, verify_file
from dataset.prepare import article_title, iter_articles, prepare_split, validate_blimp


def test_top_level_headings_only():
    assert article_title(" = A title = \n") == "A title"
    assert article_title(" = = A subsection = = \n") is None
    assert article_title(" = = = Detail = = = \n") is None
    assert article_title(" =  = A subsection =  = \n") is None
    assert article_title("ordinary prose = x") is None


def test_paper_group_mapping_preserves_selection_as_argument_structure():
    from dataset.prepare import GROUP_ALIASES

    assert GROUP_ALIASES["s-selection"] == "argument_structure"


def test_article_partition_preserves_exact_rows_and_unicode():
    rows = [" \n", " = Café = \n", "\n", " Pokémon 😀\n", " = = Section = = \n",
            "\n", " = Second = \n", "\n", " End without newline"]
    articles = list(iter_articles(rows, "train"))
    assert len(articles) == 2
    assert "".join(a["text"] for a in articles) == "".join(rows)
    assert (articles[0]["row_start"], articles[0]["row_end"]) == (0, 6)
    assert articles[0]["heading_row"] == 1
    assert articles[1]["article_id"] == "train:000001"


def test_unrecognized_preamble_fails_instead_of_dropping_text():
    with pytest.raises(ValueError, match="before first article"):
        list(iter_articles(["unexpected body\n", " = Title = \n"], "test"))


def test_table_and_equation_fragments_do_not_start_articles():
    rows = ["\n", " = Article = \n", "\n", " Note : Pos\n", " = Position ; GP = \n",
            " Games played\n", "\n", " = Volume 6 = \n", " July 1979\n"]
    articles = list(iter_articles(rows, "test"))
    assert len(articles) == 1
    assert articles[0]["text"] == "".join(rows)


def test_empty_and_headingless_sources():
    assert list(iter_articles([], "test")) == []
    with pytest.raises(ValueError, match="No article heading"):
        list(iter_articles([" \n"], "test"))


def test_parquet_to_articles_end_to_end(tmp_path):
    raw = tmp_path / "raw/wikitext103_raw"
    raw.mkdir(parents=True)
    # Article extends across shard boundary; subsection must not split it.
    pq.write_table(pa.table({"text": ["\n", " = First = \n", "\n", " body\n"]}), raw/"train-00000.parquet")
    pq.write_table(pa.table({"text": [" = = Subsection = = \n", "\n", " = Last = \n", "\n", " tail\n"]}), raw/"train-00001.parquet")
    summary = prepare_split(tmp_path, "train", 9)
    assert summary["rows"] == 9 and summary["articles"] == 2
    articles = [json.loads(line) for line in (tmp_path/summary["file"]["path"]).read_text().splitlines()]
    assert articles[0]["text"] == "\n = First = \n\n body\n = = Subsection = = \n\n"
    assert articles[1]["row_end"] == 9
    with pytest.raises(ValueError, match="Row count mismatch"):
        prepare_split(tmp_path, "train", 10)


def test_hash_verification_detects_corruption(tmp_path):
    path = tmp_path/"file"
    path.write_bytes(b"original")
    verify_file(path, {"size": 8, "sha256": hashlib.sha256(b"original").hexdigest()})
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(path, {"size": 8, "sha256": hashlib.sha256(b"original").hexdigest()})


def test_git_blob_hash_and_path_containment(tmp_path):
    path = tmp_path/"file"
    path.write_bytes(b"abc")
    digest = hashlib.sha1(b"blob 3\0abc", usedforsecurity=False).hexdigest()
    verify_file(path, {"size": 3, "git_blob_sha1": digest})
    with pytest.raises(ValueError, match="escapes root"):
        contained_path(tmp_path, "../outside")


@pytest.mark.parametrize("status,partial,body", [(200, b"", b"abcdef"),
                                               (200, b"abc", b"abcdef"),
                                               (206, b"abc", b"def")])
def test_download_resume_and_ignored_range(tmp_path, monkeypatch, status, partial, body):
    import dataset.download as download

    class Response:
        status_code = status
        headers = {"Content-Range": "bytes 3-5/6"} if status == 206 else {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield body

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url, *, headers, **kwargs):
            assert headers == ({"Range": "bytes=3-"} if partial else {})
            return Response()

    monkeypatch.setattr(download, "make_session", Session)
    if partial:
        (tmp_path/"item.part").write_bytes(partial)
    specification = {"path": "item", "url": "https://example.invalid/item", "size": 6,
                     "source": "fixture", "sha256": hashlib.sha256(b"abcdef").hexdigest()}
    result = download_one(tmp_path, specification)
    assert (tmp_path/"item").read_bytes() == b"abcdef"
    assert not (tmp_path/"item.part").exists()
    assert result["bytes"] == 6


def test_blimp_requires_complete_benchmark(tmp_path):
    with pytest.raises(ValueError, match="67 BLiMP paradigms"):
        validate_blimp(tmp_path, {"expected_paradigms": 67})


def test_blimp_preserves_and_flags_identical_candidates(tmp_path, monkeypatch):
    import dataset.prepare as prepare

    monkeypatch.setattr(prepare, "EXPECTED_GROUPS", {"anaphor_agreement"})
    data = tmp_path / "raw/blimp/data"
    data.mkdir(parents=True)
    rows = [{"UID": "toy", "pairID": str(i), "field": "morphology",
             "linguistics_term": "anaphor_agreement", "sentence_good": "Invented text.",
             "sentence_bad": "Invented text." if i == 0 else "Other text."} for i in range(2)]
    path = data / "toy.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(original)
    result = validate_blimp(tmp_path, {"expected_paradigms": 1, "pairs_per_paradigm": 2, "expected_groups": 1})
    assert result["pairs"] == 2
    assert result["identical_candidate_pairs"] == [{"UID": "toy", "pairID": "0"}]
    assert path.read_text() == original
    rows[1]["pairID"] = "0"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Duplicate pair ID"):
        validate_blimp(tmp_path, {"expected_paradigms": 1, "pairs_per_paradigm": 2, "expected_groups": 1})
