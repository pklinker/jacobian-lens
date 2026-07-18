"""Fit a Jacobian lens (offline build step), or merge partial lenses.

The heavy imports (torch/transformers via the adapter) happen only after
argument validation, so bad flags fail fast.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
LARGE_MODEL_PARAMS = 9e9

PROMPT_SOURCES = ("file", "wikitext", "mixed")

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


def build_corpus(
    source: str,
    prompts_path: Path | None,
    max_prompts: int,
    *,
    wikitext_fraction: float = 0.5,
    seed: int = 0,
    wikitext_loader: Callable[[int], list[str]] | None = None,
) -> list[str]:
    """Assemble the fit corpus from ``--prompts``, WikiText-103, or both.

    ``max_prompts`` is the size of the corpus in every mode. Under ``mixed``,
    WikiText supplies ``wikitext_fraction`` of it and the file supplies the
    rest; if the file holds fewer lines than its share, the shortfall is drawn
    from WikiText so the total still lands on ``max_prompts``.

    The mixed list is shuffled with ``seed`` rather than concatenated. Order is
    irrelevant to a completed fit — :func:`jlens.fitting.fit` averages over the
    whole list — but it decides what an *interrupted* fit averages over, since
    resume replays the list by index. Concatenated, a fit stopped halfway would
    have seen one source and none of the other. The shuffle is seeded because
    that same index replay means a reshuffled list on resume would double-count
    some prompts and skip others.

    Args:
        source: One of :data:`PROMPT_SOURCES`.
        prompts_path: Corpus file; unused when ``source`` is ``"wikitext"``.
        max_prompts: Total prompts to return.
        wikitext_fraction: ``mixed`` only; share drawn from WikiText.
        seed: Seed for the ``mixed`` shuffle.
        wikitext_loader: Injectable for tests; defaults to the adapter's
            streaming loader.
    """
    if source not in PROMPT_SOURCES:
        raise ValueError(f"unknown prompt source {source!r}")
    if wikitext_loader is None:
        from jlens_inspector import adapter

        wikitext_loader = adapter.load_wikitext_prompts

    if source == "file":
        assert prompts_path is not None
        return load_prompts(prompts_path, max_prompts)
    if source == "wikitext":
        return wikitext_loader(max_prompts)

    assert prompts_path is not None
    n_from_file = max_prompts - round(max_prompts * wikitext_fraction)
    file_prompts = load_prompts(prompts_path, n_from_file) if n_from_file else []
    corpus = file_prompts + wikitext_loader(max_prompts - len(file_prompts))
    random.Random(seed).shuffle(corpus)
    return corpus


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
    parser.add_argument(
        "--models-dir",
        default="models",
        metavar="DIR",
        help="local model store: downloads are saved here and found here on "
        "later runs; point at another drive if needed (default: models)",
    )
    parser.add_argument("--prompts", help="text/JSONL corpus, one sequence per line")
    parser.add_argument(
        "--prompt-source",
        choices=PROMPT_SOURCES,
        default="file",
        help="where the corpus comes from: the --prompts file, streamed "
        "WikiText-103, or both (default: file)",
    )
    parser.add_argument(
        "--wikitext-fraction",
        type=float,
        default=0.5,
        help="--prompt-source mixed only: share of the corpus drawn from "
        "WikiText-103 (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the mixed-corpus shuffle; keep it fixed across resumes "
        "of the same fit (default: 0)",
    )
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

    # Validate the corpus flags before the heavy adapter import below, so a bad
    # combination fails immediately instead of after loading torch.
    if not args.merge:
        needs_file = args.prompt_source in ("file", "mixed")
        if needs_file and not args.prompts:
            parser.error(
                f"--prompts is required for --prompt-source {args.prompt_source} "
                "(unless --merge)"
            )
        if args.prompt_source == "wikitext" and args.prompts:
            parser.error(
                "--prompts is unused with --prompt-source wikitext; drop it, or "
                "use --prompt-source mixed to fit on both"
            )
        if not 0.0 <= args.wikitext_fraction <= 1.0:
            parser.error(
                f"--wikitext-fraction must be between 0 and 1, got "
                f"{args.wikitext_fraction}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from jlens_inspector import adapter

    if args.merge:
        lens = adapter.merge_lens_files(args.merge)
        lens.save(str(out))
        print(f"merged {len(args.merge)} lenses ({lens.n_prompts} prompts) -> {out}")
        return 0

    prompts = build_corpus(
        args.prompt_source,
        Path(args.prompts) if args.prompts else None,
        args.max_prompts,
        wikitext_fraction=args.wikitext_fraction,
        seed=args.seed,
        wikitext_loader=adapter.load_wikitext_prompts,
    )
    print(f"corpus: {len(prompts)} prompts (--prompt-source {args.prompt_source})")

    import torch

    adapter.configure_logging()
    device_map = "auto" if torch.cuda.is_available() else None
    hf_model, tokenizer = adapter.load_model(
        args.model, device_map=device_map, models_dir=args.models_dir
    )
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
