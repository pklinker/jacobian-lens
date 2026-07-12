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
