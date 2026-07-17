"""3D viewer tests: CLI guards and payload assembly. No model, no network."""

import base64
import gzip
import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from jlens_inspector import viewer3d, viewer3d_cli


@dataclass
class FakeSlice:
    """SliceData-shaped test double (build_viewer_page only reads attributes)."""

    seq_len: int
    layers: list
    context_token_ids: list
    context_token_strs: list
    top_ids: np.ndarray
    top_ranks: np.ndarray
    tracked_token_ids: list
    rank_tensor: np.ndarray
    vocab_fragment: dict
    vocab_size: int = 0
    pinned_token_ids: list = field(default_factory=list)
    ctx_offset: int = 0


def make_slice(**overrides):
    T, L, topn, tracked = 3, 4, 2, [7, 9]
    rng = np.random.default_rng(0)
    data = dict(
        seq_len=T,
        layers=[0, 1, 2, 5],
        context_token_ids=[7, 9, 11],
        context_token_strs=["a", " b", " c"],
        top_ids=rng.integers(0, 12, (T, L, topn)).astype(np.int32),
        top_ranks=np.tile(np.arange(topn, dtype=np.int32), (T, L, 1)),
        tracked_token_ids=tracked,
        rank_tensor=rng.integers(0, 999, (T, L, len(tracked))).astype(np.int32),
        vocab_fragment={i: f"tok{i}" for i in range(12)},
        vocab_size=50_000,
        pinned_token_ids=[7],
    )
    data.update(overrides)
    return FakeSlice(**data)


def test_viewer_never_fits_implicitly(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(SystemExit) as excinfo:
        viewer3d_cli.main(
            ["--model", "x", "--lens", str(missing), "--prompt", "hi"]
        )
    assert "fit_lens.py" in str(excinfo.value)


class FakeTokenizer:
    """Single-token vocab is {' euro': 5, 'x': 6}; everything else splits."""

    def encode(self, text, add_special_tokens=False):
        return {" euro": [5], "x": [6]}.get(text, [1, 2])

    def decode(self, ids):
        return "piece"


def test_pin_resolves_with_leading_space():
    assert viewer3d_cli.pin_token_ids(FakeTokenizer(), ["euro", "x"]) == {5, 6}


def test_pin_rejects_multi_token():
    with pytest.raises(SystemExit) as excinfo:
        viewer3d_cli.pin_token_ids(FakeTokenizer(), ["not a token"])
    assert "does not tokenize to a single token" in str(excinfo.value)


def test_bootstrap_roundtrip():
    fake = make_slice()
    boot, payload_bytes = viewer3d.build_bootstrap(
        fake, "a b c", title="t", description="d"
    )
    assert boot["mode"] == "embed"
    meta = boot["meta"]
    assert meta["T"] == 3
    assert meta["layers"] == [0, 1, 2, 5]
    assert meta["top_n"] == 2
    assert meta["ctx_strs"] == ["a", " b", " c"]
    assert meta["tracked"] == [7, 9]
    assert meta["pinned"] == [7]
    assert meta["vocab_size"] == 50_000
    assert meta["vocab"]["7"] == "tok7"

    # slice.bin gunzips back to the top-K grids, ids block then ranks block.
    raw = gzip.decompress(base64.b64decode(boot["files"]["slice.bin"]))
    n = fake.top_ids.size
    ids = np.frombuffer(raw[: n * 4], dtype="<i4").reshape(fake.top_ids.shape)
    ranks = np.frombuffer(raw[n * 4 :], dtype="<i4").reshape(fake.top_ids.shape)
    np.testing.assert_array_equal(ids, fake.top_ids)
    np.testing.assert_array_equal(ranks, fake.top_ranks)

    # one rank file per tracked token, gunzipping to that token's [T, L] slab
    for i, tid in enumerate(fake.tracked_token_ids):
        raw = gzip.decompress(base64.b64decode(boot["files"][f"ranks/{tid}.bin"]))
        np.testing.assert_array_equal(
            np.frombuffer(raw, dtype="<i4").reshape(3, 4),
            fake.rank_tensor[:, :, i],
        )
    assert payload_bytes == sum(
        len(base64.b64decode(b)) for b in boot["files"].values()
    )


def make_record():
    """Minimal inspect_prompt event record: 2 positions, 2 lens layers + final."""
    def readout(pos, toks):
        return {
            "position": pos,
            "top_k": [
                {"token": s, "id": tid, "rank": r + 1, "logit": 0.0}
                for r, (s, tid) in enumerate(toks)
            ],
        }

    return {
        "timestamp": "t",
        "model_id": "m",
        "lens_id": "l",
        "prompt": "the country shaped like a boot",
        "positions": [-2, -1],
        "top_k": 2,
        "n_layers": 6,
        "layers": [
            {"layer": 0, "is_model_output": False,
             "readouts": [readout(-2, [(" boots", 11), (" shoes", 12)]),
                          readout(-1, [(" then", 13), (" very", 14)])]},
            {"layer": 2, "is_model_output": False,
             "readouts": [readout(-2, [(" shoes", 12), (" boots", 11)]),
                          readout(-1, [(" euro", 15), (" then", 13)])]},
            {"layer": 5, "is_model_output": True,
             "readouts": [readout(-2, [(" is", 16), (" was", 17)]),
                          readout(-1, [(" the", 18), (" euro", 15)])]},
        ],
    }


def test_slice_from_record():
    s = viewer3d.slice_from_record(make_record())
    assert s.seq_len == 2
    assert s.layers == [0, 2, 5]
    assert s.top_ids.shape == (2, 3, 2)
    assert s.pos_labels == ["-2", "-1"]
    # grid: [t, layer_idx, k]; ranks are 0-based
    assert s.vocab_fragment[11] == " boots"
    assert s.top_ids[0, 0].tolist() == [11, 12]
    assert s.top_ids[1, 2].tolist() == [18, 15]
    assert s.top_ranks[0, 0].tolist() == [0, 1]
    # rank tensor: top-k slot where present, -1 (unknown) elsewhere
    i_boots = s.tracked_token_ids.index(11)
    assert s.rank_tensor[0, :, i_boots].tolist() == [0, 1, -1]
    assert s.rank_tensor[1, :, i_boots].tolist() == [-1, -1, -1]
    # context: prompt prefix + one marker per position, windowed by ctx_offset
    assert s.ctx_offset == 1
    assert s.context_token_strs[0].startswith("the country")
    assert s.context_token_strs[1:] == [" [-2]", " [-1]"]


def test_slice_from_record_rejects_non_record():
    with pytest.raises(ValueError, match="not an inspect.py record"):
        viewer3d.slice_from_record({"layers": []})


def test_record_cli_builds_page_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer3d, "_THREE_BUNDLE_CACHE", "<script>/*three*/</script>")
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(make_record()), encoding="utf-8")
    out = tmp_path / "page.html"
    viewer3d_cli.main(
        ["--record", str(record_path), "--out", str(out), "--pin", "euro"]
    )
    page = out.read_text(encoding="utf-8")
    blob = page.split('<script id="bootstrap" type="application/json">')[1]
    meta = json.loads(blob.split("</script>")[0])["meta"]
    assert meta["pos_labels"] == ["-2", "-1"]
    assert meta["pinned"] == [15]  # " euro" resolved via the record vocabulary
    assert "imported record" in meta["what"]


def test_record_cli_guards(tmp_path):
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(make_record()), encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot be combined"):
        viewer3d_cli.main(["--record", str(record_path), "--model", "x"])
    with pytest.raises(SystemExit, match="inspect.py --out"):
        viewer3d_cli.main(["--record", str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit, match="required unless --record"):
        viewer3d_cli.main(["--model", "x", "--lens", "y"])
    with pytest.raises(SystemExit, match="not in the record's top-k vocabulary"):
        viewer3d_cli.main(
            ["--record", str(record_path), "--out", str(tmp_path / "p.html"),
             "--pin", "zebra"]
        )


def test_page_embeds_escaped_bootstrap(monkeypatch):
    # No network: stub the memoised three.js bundle.
    monkeypatch.setattr(viewer3d, "_THREE_BUNDLE_CACHE", "<script>/*three*/</script>")
    fake = make_slice(vocab_fragment={0: "</script><b>", 1: "ok"})
    page, _ = viewer3d.build_viewer_page(
        fake, "a b c", title="ti<tle", description="de&sc"
    )
    # placeholders are gone, header text is HTML-escaped
    assert "__BOOTSTRAP__" not in page and "__THREE__" not in page
    assert "ti&lt;tle" in page and "de&amp;sc" in page
    # a hostile vocab string cannot close the bootstrap <script>: the JSON
    # blob contains no literal "</", so it parses back losslessly
    blob = page.split('<script id="bootstrap" type="application/json">')[1]
    blob = blob.split("</script>")[0]
    assert "</" not in blob
    assert json.loads(blob)["meta"]["vocab"]["0"] == "</script><b>"
