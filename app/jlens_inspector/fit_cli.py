"""Fit a Jacobian lens (offline build step), or merge partial lenses.

The heavy imports (torch/transformers via the adapter) happen only after
argument validation, so bad flags fail fast.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
LARGE_MODEL_PARAMS = 9e9

QUANTIZED_FLAGS_MSG = (
    "quantized loading (--load-in-4bit/--load-in-8bit) is refused: "
    "Jacobian fitting backprops through the model and quantized weights "
    "break the backward passes; the model is always loaded in bf16"
)


def load_prompts(path: Path, max_prompts: int) -> list[str]:
    """One sequence per line; ``.jsonl`` lines may be a string or an object
    with a ``text`` field."""
    prompts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                record = json.loads(line)
                line = record["text"] if isinstance(record, dict) else str(record)
            prompts.append(line)
            if len(prompts) >= max_prompts:
                break
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def print_vram_report(hf_model) -> None:
    n_params = sum(p.numel() for p in hf_model.parameters())
    weights_gb = n_params * 2 / 1024**3  # bf16
    print(
        f"model: {n_params / 1e9:.2f}B params, ~{weights_gb:.1f} GiB of bf16 "
        f"weights; fitting retains per-block graphs, so budget roughly "
        f"{3 * weights_gb:.0f}-{4 * weights_gb:.0f} GiB of gradient-capable VRAM"
    )
    if n_params > LARGE_MODEL_PARAMS:
        print(
            "WARNING: >9B params — a single-machine fit is likely infeasible "
            "on home-lab hardware. Fit disjoint prompt slices on separate "
            "runs (--prompts split files) and combine them with --merge."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a Jacobian lens for an HF decoder model (offline build step)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF Hub id or local path")
    parser.add_argument("--prompts", help="text/JSONL corpus, one sequence per line")
    parser.add_argument("--out", default="out/jacobian_lens.pt", help="lens output path")
    parser.add_argument(
        "--checkpoint",
        help="resumable fit checkpoint path (default: <out>.ckpt)",
    )
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=8,
        help="output dims per backward pass; higher = more VRAM, fewer passes",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="LENS_PT",
        help="merge partial lens files into --out instead of fitting",
    )
    # Recognized only to be refused with a clear message (hard requirement).
    parser.add_argument("--load-in-4bit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--load-in-8bit", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.load_in_4bit or args.load_in_8bit:
        parser.error(QUANTIZED_FLAGS_MSG)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from jlens_inspector import adapter

    if args.merge:
        lens = adapter.merge_lens_files(args.merge)
        lens.save(str(out))
        print(f"merged {len(args.merge)} lenses ({lens.n_prompts} prompts) -> {out}")
        return 0

    if not args.prompts:
        parser.error("--prompts is required (unless --merge)")
    prompts = load_prompts(Path(args.prompts), args.max_prompts)

    import torch

    adapter.configure_logging()
    device_map = "auto" if torch.cuda.is_available() else None
    hf_model, tokenizer = adapter.load_model(args.model, device_map=device_map)
    print_vram_report(hf_model)
    model = adapter.wrap(hf_model, tokenizer)

    checkpoint = args.checkpoint or f"{out}.ckpt"
    lens = adapter.fit_lens(
        model,
        prompts,
        checkpoint_path=checkpoint,
        max_seq_len=args.seq_len,
        dim_batch=args.dim_batch,
    )
    lens.save(str(out))
    print(f"saved {lens!r} -> {out}")
    print(f"fit checkpoint kept at {checkpoint} (delete once you're happy with the lens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
