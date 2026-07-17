"""CLI for the 3D layer viewer: compute a slice, render a self-contained page.

Two data sources:

- ``--model/--lens/--prompt``: load the model, apply the lens, compute a full
  slice (top-k grid + full-vocab rank tensors), mirroring ``slice_cli.py``.
- ``--record FILE``: render an ``inspect.py --out`` JSON event record without
  touching the model — trajectories are then limited to top-k depth, since
  the record carries no full-vocab ranks.

The scene itself lives in ``data/viewer3d.html`` and the payload assembly in
``viewer3d.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from jlens_inspector.inspection import MISSING_LENS_MSG

MULTI_TOKEN_PIN_MSG = (
    "--pin {text!r} does not tokenize to a single token (got {pieces}); pin "
    "one of those pieces instead"
)

RECORD_CONFLICT_MSG = (
    "--record renders a saved inspect.py record without loading the model; "
    "it cannot be combined with --model/--lens/--prompt or slice options"
)

MODEL_ARGS_REQUIRED_MSG = (
    "--model, --lens and --prompt are required unless --record is given"
)

MISSING_RECORD_MSG = (
    "record file {path!r} does not exist. Produce one with "
    "inspect.py --out FILE."
)

PIN_NOT_IN_RECORD_MSG = (
    "--pin {text!r} is not in the record's top-k vocabulary, so there is "
    "nothing to pin"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a self-contained 3D Jacobian-lens layer viewer for a "
            "prompt: layers stacked as planes, same-token links across depth."
        )
    )
    parser.add_argument("--model", help="HF Hub id or local path")
    parser.add_argument(
        "--lens", help="lens .pt from fit_lens.py, or a Hub repo id"
    )
    parser.add_argument("--prompt")
    parser.add_argument(
        "--record",
        metavar="FILE",
        help="render an inspect.py --out JSON record instead of running the model",
    )
    parser.add_argument("--out", default="out/viewer3d.html")
    parser.add_argument("--title", default="Jacobian lens 3D viewer")
    parser.add_argument(
        "--top-n", type=int, default=10, help="top tokens kept per (position, layer)"
    )
    parser.add_argument(
        "--layer-stride", type=int, default=1, help="render every Nth fitted layer"
    )
    parser.add_argument(
        "--last-n-tokens",
        type=int,
        default=None,
        help="compute the grid only for the last N positions",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--mask-display",
        action="store_true",
        help="show only word-like tokens in cells (ranks stay full-vocab)",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="TOKEN",
        help="token string to pin at load (repeatable; must be a single token)",
    )
    return parser


def pin_token_ids(tokenizer, pins: Sequence[str]) -> set[int]:
    """Resolve ``--pin`` strings to single token ids.

    Tries the string as-is, then with a leading space (the usual in-context
    form for BPE vocabularies). A string that is multiple tokens either way
    is an error naming the pieces.
    """
    ids: set[int] = set()
    for text in pins:
        for candidate in (text, " " + text):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
                break
        else:
            pieces = [
                tokenizer.decode([t])
                for t in tokenizer.encode(text, add_special_tokens=False)
            ]
            raise SystemExit(MULTI_TOKEN_PIN_MSG.format(text=text, pieces=pieces))
    return ids


def pin_ids_from_vocab(vocab: dict[int, str], pins: Sequence[str]) -> list[int]:
    """Resolve ``--pin`` strings against a record's token-string vocabulary."""
    by_str = {s: tid for tid, s in vocab.items()}
    ids = []
    for text in pins:
        tid = by_str.get(text, by_str.get(" " + text))
        if tid is None:
            raise SystemExit(PIN_NOT_IN_RECORD_MSG.format(text=text))
        ids.append(tid)
    return ids


def _slice_from_record_file(args) -> tuple:
    """Load ``--record`` and return ``(slice_data, prompt, description)``."""
    from jlens_inspector import viewer3d

    non_default = (
        args.model or args.lens or args.prompt
        or args.top_n != 10 or args.layer_stride != 1
        or args.last_n_tokens is not None or args.max_seq_len != 512
        or args.mask_display
    )
    if non_default:
        raise SystemExit(RECORD_CONFLICT_MSG)
    if not os.path.exists(args.record):
        raise SystemExit(MISSING_RECORD_MSG.format(path=args.record))
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    try:
        slice_data = viewer3d.slice_from_record(record)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    slice_data.pinned_token_ids = pin_ids_from_vocab(
        slice_data.vocab_fragment, args.pin
    )
    description = (
        f"model: {record.get('model_id')} - lens: {record.get('lens_id')} - "
        "imported record (trajectories limited to top-k depth)"
    )
    return slice_data, record["prompt"], description


def _slice_from_model(args) -> tuple:
    """Load the model + lens and return ``(slice_data, prompt, description)``."""
    if not (args.model and args.lens and args.prompt):
        raise SystemExit(MODEL_ARGS_REQUIRED_MSG)
    looks_local = args.lens.endswith(".pt") or os.sep in args.lens
    if looks_local and not os.path.exists(args.lens):
        raise SystemExit(MISSING_LENS_MSG.format(path=args.lens))

    import torch

    from jlens_inspector import adapter

    device_map = "auto" if torch.cuda.is_available() else None
    hf_model, tokenizer = adapter.load_model(args.model, device_map=device_map)
    model = adapter.wrap(hf_model, tokenizer)
    lens = adapter.load_lens(args.lens)

    slice_data = adapter.slice_for_prompt(
        model,
        lens,
        args.prompt,
        top_n=args.top_n,
        layer_stride=args.layer_stride,
        last_n_tokens=args.last_n_tokens,
        mask_display=args.mask_display,
        max_seq_len=args.max_seq_len,
        pinned_token_ids=pin_token_ids(tokenizer, args.pin),
    )
    return slice_data, args.prompt, f"model: {args.model} - lens: {args.lens}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.record:
        slice_data, prompt, description = _slice_from_record_file(args)
    else:
        slice_data, prompt, description = _slice_from_model(args)

    from jlens_inspector import viewer3d

    n_cells = slice_data.seq_len * len(slice_data.layers) * slice_data.top_ids.shape[2]
    if n_cells > 200_000:
        print(
            f"warning: {n_cells} cells will strain the page; consider "
            "--layer-stride, --last-n-tokens, or a smaller --top-n"
        )
    page, payload_bytes = viewer3d.build_viewer_page(
        slice_data, prompt, title=args.title, description=description
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, "
        f"{payload_bytes / 1024:.0f} KiB payload); open it in a browser"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
