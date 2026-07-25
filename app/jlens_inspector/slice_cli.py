"""Render the interactive layer x position slice page for one prompt.

Mirrors the final step of the repo's walkthrough.ipynb: a single
self-contained HTML file (d3 inlined, so the first render needs network
access to fetch d3 once).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from jlens_inspector.inspection import LENS_APPLY_FAILED_MSG, MISSING_LENS_MSG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a self-contained Jacobian-lens slice page for a prompt."
    )
    parser.add_argument("--model", required=True, help="HF Hub id or local path")
    parser.add_argument(
        "--models-dir",
        default="models",
        metavar="DIR",
        help="local model store: downloads are saved here and found here on "
        "later runs; point at another drive if needed (default: models)",
    )
    parser.add_argument(
        "--lens", required=True, help="lens .pt from fit_lens.py, or a Hub repo id"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="out/slice.html")
    parser.add_argument("--title", default="Jacobian lens slice")
    parser.add_argument(
        "--layer-stride", type=int, default=1, help="render every Nth fitted layer"
    )
    parser.add_argument(
        "--mask-display",
        action="store_true",
        help="show only word-like tokens in cells (ranks stay full-vocab)",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    looks_local = args.lens.endswith(".pt") or os.sep in args.lens
    if looks_local and not os.path.exists(args.lens):
        raise SystemExit(MISSING_LENS_MSG.format(path=args.lens))

    import torch

    from jlens_inspector import adapter

    device_map = "auto" if torch.cuda.is_available() else None
    hf_model, tokenizer = adapter.load_model(
        args.model, device_map=device_map, models_dir=args.models_dir
    )
    model = adapter.wrap(hf_model, tokenizer)
    lens = adapter.load_lens(args.lens)

    try:
        slice_data = adapter.slice_for_prompt(
            model,
            lens,
            args.prompt,
            layer_stride=args.layer_stride,
            mask_display=args.mask_display,
            max_seq_len=args.max_seq_len,
        )
    except ValueError as exc:
        raise SystemExit(
            LENS_APPLY_FAILED_MSG.format(
                lens=args.lens, model=args.model, error=exc
            )
        ) from exc
    page = adapter.render_slice_page(
        slice_data,
        args.prompt,
        title=args.title,
        description=f"model: {args.model} - lens: {args.lens}",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB); open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
