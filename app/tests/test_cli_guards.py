"""Fast guard tests: no model download, no fit."""

import pytest

from jlens_inspector import fit_cli, inspection


@pytest.mark.parametrize("flag", ["--load-in-4bit", "--load-in-8bit"])
def test_fit_refuses_quantized_flags(flag, capsys, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main(
            ["--model", "x", "--prompts", str(tmp_path / "p.txt"), flag]
        )
    assert excinfo.value.code == 2
    assert "quantized" in capsys.readouterr().err.lower()


def test_fit_requires_prompts_unless_merge(capsys, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main(["--model", "x", "--out", str(tmp_path / "lens.pt")])
    assert excinfo.value.code == 2
    assert "--prompts" in capsys.readouterr().err


def test_load_prompts_txt_and_jsonl(tmp_path):
    txt = tmp_path / "p.txt"
    txt.write_text("one two three\n\nfour five six\n")
    assert fit_cli.load_prompts(txt, max_prompts=10) == [
        "one two three",
        "four five six",
    ]
    assert fit_cli.load_prompts(txt, max_prompts=1) == ["one two three"]

    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text('{"text": "from object"}\n"bare string"\n')
    assert fit_cli.load_prompts(jsonl, max_prompts=10) == [
        "from object",
        "bare string",
    ]


def _fake_wikitext(n):
    """Stand-in for the streaming loader: no network, identifiable strings."""
    return [f"wiki {i}" for i in range(n)]


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("".join(f"file {i}\n" for i in range(10)))
    return path


def test_build_corpus_file_ignores_wikitext(corpus_file):
    corpus = fit_cli.build_corpus(
        "file", corpus_file, 4, wikitext_loader=_fake_wikitext
    )
    assert corpus == ["file 0", "file 1", "file 2", "file 3"]


def test_build_corpus_wikitext_ignores_file(corpus_file):
    corpus = fit_cli.build_corpus(
        "wikitext", None, 3, wikitext_loader=_fake_wikitext
    )
    assert corpus == ["wiki 0", "wiki 1", "wiki 2"]


def test_build_corpus_mixed_honors_fraction_and_total(corpus_file):
    corpus = fit_cli.build_corpus(
        "mixed", corpus_file, 8, wikitext_fraction=0.25,
        wikitext_loader=_fake_wikitext,
    )
    assert len(corpus) == 8
    assert sum(c.startswith("wiki") for c in corpus) == 2
    assert sum(c.startswith("file") for c in corpus) == 6


def test_build_corpus_mixed_tops_up_from_wikitext_when_file_is_short(corpus_file):
    """The file holds 10 lines but its half-share of 100 is 50; WikiText covers
    the shortfall so the corpus still reaches --max-prompts."""
    corpus = fit_cli.build_corpus(
        "mixed", corpus_file, 100, wikitext_fraction=0.5,
        wikitext_loader=_fake_wikitext,
    )
    assert len(corpus) == 100
    assert sum(c.startswith("file") for c in corpus) == 10
    assert sum(c.startswith("wiki") for c in corpus) == 90


def test_build_corpus_mixed_is_seeded_and_interleaved(corpus_file):
    """Same seed reproduces the order (resume replays by index), and the two
    sources are actually interleaved rather than concatenated."""
    kwargs = dict(wikitext_fraction=0.5, wikitext_loader=_fake_wikitext)
    first = fit_cli.build_corpus("mixed", corpus_file, 10, seed=0, **kwargs)
    again = fit_cli.build_corpus("mixed", corpus_file, 10, seed=0, **kwargs)
    other = fit_cli.build_corpus("mixed", corpus_file, 10, seed=1, **kwargs)
    assert first == again
    assert first != other
    # A concatenated list would have all 5 file prompts in the first half.
    assert sum(c.startswith("file") for c in first[:5]) != 5


def test_fit_rejects_prompts_with_wikitext_source(capsys, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main(
            ["--prompt-source", "wikitext", "--prompts", str(tmp_path / "p.txt")]
        )
    assert excinfo.value.code == 2
    assert "unused with --prompt-source wikitext" in capsys.readouterr().err


def test_fit_mixed_requires_prompts(capsys):
    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main(["--prompt-source", "mixed"])
    assert excinfo.value.code == 2
    assert "--prompts is required" in capsys.readouterr().err


def test_fit_rejects_out_of_range_wikitext_fraction(capsys, corpus_file):
    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main(
            ["--prompt-source", "mixed", "--prompts", str(corpus_file),
             "--wikitext-fraction", "1.5"]
        )
    assert excinfo.value.code == 2
    assert "--wikitext-fraction must be between 0 and 1" in capsys.readouterr().err


def test_inspect_never_fits_implicitly(tmp_path):
    """A missing local lens file is a hard error pointing at fit_lens.py."""
    missing = tmp_path / "nope.pt"
    with pytest.raises(SystemExit) as excinfo:
        inspection.main(
            ["--model", "x", "--lens", str(missing), "--prompt", "hi"]
        )
    assert "fit_lens.py" in str(excinfo.value)


def test_adapter_rejects_gguf():
    from jlens_inspector import adapter

    with pytest.raises(ValueError, match="GGUF"):
        adapter.load_model("model.gguf")
