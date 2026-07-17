"""Build the self-contained 3D layer-viewer page from a computed slice.

The 3D page reuses the exact payload format of jlens's 2D slice page
(``meta`` + gzip'd ``slice.bin`` + lazy ``ranks/{tid}.bin``, base64-embedded
as a virtual filesystem), but the assembly lives here rather than importing
jlens's private helpers — ``adapter.py`` stays the only jlens seam.

The page's only external dependencies are the three.js r147 UMD builds
(core + OrbitControls), fetched once with SRI verification and inlined, the
same way jlens.vis inlines d3. r147 is the last release shipping UMD builds;
everything the scene uses is stable in it.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # SliceData lives in jlens.vis, which imports torch.
    from jlens_inspector.adapter import SliceData

_TEMPLATE_NAME = "viewer3d.html"

#: SRI-pinned three.js r147 UMD scripts (core + OrbitControls), the page's
#: only external dependency. A fetch or integrity failure raises — the
#: builder never silently emits a CDN-dependent page.
_THREE_SCRIPTS = [
    (
        "https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js",
        "sha384-vV17nr/rMaJqmeZkFUzXLpHdQ+ME5QHKdydaqqN+3Ga39RJlNrTatJxHwGV4ml2C",
    ),
    (
        "https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js",
        "sha384-I0DMsfimAPIqWT8lF+oA997gRgdUi3jhidoTq3fN0zmn35smXZukD01FLnsPRU9i",
    ),
]

_THREE_BUNDLE_CACHE: str | None = None


def _three_bundle() -> str:
    """Fetch and inline the pinned three.js scripts, verified against their
    SRI hashes. Memoised per process."""
    global _THREE_BUNDLE_CACHE
    if _THREE_BUNDLE_CACHE is not None:
        return _THREE_BUNDLE_CACHE
    parts = []
    for url, expected_sri in _THREE_SCRIPTS:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
        except OSError as exc:
            raise RuntimeError(
                f"could not fetch {url}; the 3D viewer page inlines three.js, "
                "so building it needs network access once"
            ) from exc
        sri = "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode()
        if sri != expected_sri:
            raise RuntimeError(f"integrity check failed for {url} (got {sri})")
        parts.append(f"<script>\n{body.decode()}\n</script>")
    _THREE_BUNDLE_CACHE = "\n".join(parts)
    return _THREE_BUNDLE_CACHE


def _load_template() -> str:
    from importlib.resources import files

    return (files("jlens_inspector") / "data" / _TEMPLATE_NAME).read_text(
        encoding="utf-8"
    )


@dataclass
class RecordSlice:
    """SliceData-shaped view of an inspect.py JSON event record.

    The record only carries top-k readouts, so ``rank_tensor`` holds the
    0-based top-k slot where a token appears and ``-1`` (unknown) elsewhere;
    the page breaks trajectory lines at unknowns. ``context_token_strs`` is
    the prompt as a single prefix string followed by one ``[pos]`` marker per
    readout position (``ctx_offset=1`` windows the markers), and
    ``pos_labels`` carries the record's position indices for display.
    """

    seq_len: int
    layers: list[int]
    context_token_ids: list[int]
    context_token_strs: list[str]
    top_ids: np.ndarray
    top_ranks: np.ndarray
    tracked_token_ids: list[int]
    rank_tensor: np.ndarray
    vocab_fragment: dict[int, str]
    vocab_size: int = 0
    pinned_token_ids: list[int] = field(default_factory=list)
    ctx_offset: int = 1
    pos_labels: list[str] = field(default_factory=list)


def slice_from_record(record: dict) -> RecordSlice:
    """Convert an ``inspect_prompt`` event record into a renderable slice."""
    try:
        positions = record["positions"]
        entries = record["layers"]
        top_n = record["top_k"]
        prompt = record["prompt"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "not an inspect.py record: expected keys 'positions', 'layers', "
            "'top_k', 'prompt' (produce one with inspect.py --out FILE)"
        ) from exc

    seq_len, n_layers = len(positions), len(entries)
    top_ids = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    top_ranks = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    vocab: dict[int, str] = {}
    for layer_idx, entry in enumerate(entries):
        by_pos = {r["position"]: r for r in entry["readouts"]}
        for t, pos in enumerate(positions):
            for k, tok in enumerate(by_pos[pos]["top_k"][:top_n]):
                top_ids[t, layer_idx, k] = tok["id"]
                top_ranks[t, layer_idx, k] = tok["rank"] - 1
                vocab[int(tok["id"])] = tok["token"]

    tracked = sorted({int(t) for t in np.unique(top_ids)})
    rank_tensor = np.full((seq_len, n_layers, len(tracked)), -1, dtype=np.int32)
    for i, tid in enumerate(tracked):
        t_idx, l_idx, k_idx = np.nonzero(top_ids == tid)
        rank_tensor[t_idx, l_idx, i] = top_ranks[t_idx, l_idx, k_idx]

    return RecordSlice(
        seq_len=seq_len,
        layers=[e["layer"] for e in entries],
        context_token_ids=[0] * (seq_len + 1),
        context_token_strs=[prompt + "  "] + [f" [{p}]" for p in positions],
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab,
        pos_labels=[str(p) for p in positions],
    )


def _meta(
    slice_data: SliceData, prompt: str, title: str, description: str
) -> dict:
    """Everything the page needs besides the binary grids; same shape as the
    2D slice page's meta so the loader JS is shared."""
    pinned = set(slice_data.pinned_token_ids) & set(slice_data.tracked_token_ids)
    meta = {
        "title": title,
        "what": description,
        "prompt": prompt,
        "T": slice_data.seq_len,
        "layers": slice_data.layers,
        "top_n": int(slice_data.top_ids.shape[2]),
        "ctx_strs": slice_data.context_token_strs,
        "tracked": slice_data.tracked_token_ids,
        "vocab": {str(k): v for k, v in slice_data.vocab_fragment.items()},
        "pinned": sorted(pinned),
    }
    if slice_data.vocab_size:
        meta["vocab_size"] = slice_data.vocab_size
    if slice_data.ctx_offset:
        meta["ctx_offset"] = slice_data.ctx_offset
    pos_labels = getattr(slice_data, "pos_labels", None)
    if pos_labels:
        meta["pos_labels"] = pos_labels
    return meta


def build_bootstrap(
    slice_data: SliceData, prompt: str, *, title: str, description: str
) -> tuple[dict, int]:
    """Assemble the embed-mode bootstrap: meta + base64 virtual filesystem
    (``slice.bin``, one ``ranks/{tid}.bin`` per tracked token).

    Returns ``(bootstrap, payload_bytes)`` where ``payload_bytes`` is the
    compressed size of the embedded files.
    """
    slice_bin = gzip.compress(
        slice_data.top_ids.astype("<i4").tobytes()
        + slice_data.top_ranks.astype("<i4").tobytes(),
        compresslevel=6,
    )
    ranks = slice_data.rank_tensor.astype("<i4")
    files = {"slice.bin": slice_bin} | {
        f"ranks/{tid}.bin": gzip.compress(ranks[:, :, i].tobytes(), compresslevel=6)
        for i, tid in enumerate(slice_data.tracked_token_ids)
    }
    bootstrap = {
        "mode": "embed",
        "meta": _meta(slice_data, prompt, title, description),
        "files": {name: base64.b64encode(body).decode() for name, body in files.items()},
    }
    return bootstrap, sum(len(body) for body in files.values())


def build_viewer_page(
    slice_data: SliceData, prompt: str, *, title: str, description: str
) -> tuple[str, int]:
    """Render ``slice_data`` into a fully self-contained 3D viewer page.

    Returns ``(html, payload_bytes)``.
    """
    bootstrap, payload_bytes = build_bootstrap(
        slice_data, prompt, title=title, description=description
    )
    # ``</`` -> ``<\/`` so a vocab string can't close the <script> tag.
    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False).replace("</", "<\\/")
    page = (
        _load_template()
        .replace("__THREE__", _three_bundle())
        .replace("__TITLE__", html.escape(title))
        .replace("__WHAT__", html.escape(description))
        .replace("__BOOTSTRAP__", bootstrap_json)
    )
    return page, payload_bytes
