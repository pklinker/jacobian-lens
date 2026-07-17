"""Apply a fitted lens to a prompt: CLI table + importable, JSON-friendly API.

``inspect_prompt`` is the embedding surface for agent-observability
pipelines: it returns the raw per-layer logits plus an append-friendly,
JSON-serializable event record.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

MISSING_LENS_MSG = (
    "lens file {path!r} does not exist. The inspector never fits a lens "
    "implicitly — run fit_lens.py first, or point --lens at a Hub repo id."
)


def inspect_prompt(
    model: Any,
    tok: Any,
    lens: Any,
    prompt: str,
    positions: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    top_k: int = 5,
    model_id: str | None = None,
    lens_id: str | None = None,
) -> dict:
    """Read out ``prompt`` at ``positions`` across ``layers``.

    Args:
        model: A wrapped lens model (``adapter.wrap(hf, tok)``).
        tok: The HF tokenizer (used to decode top-k token ids).
        lens: A fitted lens (``adapter.load_lens(...)``).
        layers: Lens layers to read out; ``None`` = every fitted layer. The
            model's final layer is always appended and flagged
            ``is_model_output`` — that row is the model's actual next-token
            prediction, not a lens readout.

    Returns:
        ``{"record": <JSON-serializable event>, "lens_logits": {layer:
        Tensor[n_positions, vocab]}, "model_logits": Tensor[n_positions,
        vocab]}``.
    """
    from jlens_inspector import adapter

    positions = list(positions)
    lens_logits, model_logits, _ = adapter.apply_lens(
        lens, model, prompt, layers=layers, positions=positions
    )
    final_layer = model.n_layers - 1

    def top_tokens(logits_row) -> list[dict]:
        values, indices = logits_row.topk(top_k)
        return [
            {
                "token": tok.decode([int(token_id)]),
                "id": int(token_id),
                "rank": rank + 1,
                "logit": float(value),
            }
            for rank, (value, token_id) in enumerate(zip(values, indices))
        ]

    layer_entries = [
        {
            "layer": layer,
            "is_model_output": False,
            "readouts": [
                {"position": pos, "top_k": top_tokens(lens_logits[layer][row])}
                for row, pos in enumerate(positions)
            ],
        }
        for layer in sorted(lens_logits)
    ]
    layer_entries.append(
        {
            "layer": final_layer,
            "is_model_output": True,
            "readouts": [
                {"position": pos, "top_k": top_tokens(model_logits[row])}
                for row, pos in enumerate(positions)
            ],
        }
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "lens_id": lens_id,
        "prompt": prompt,
        "positions": positions,
        "top_k": top_k,
        "n_layers": model.n_layers,
        "layers": layer_entries,
    }
    return {
        "record": record,
        "lens_logits": lens_logits,
        "model_logits": model_logits,
    }


def print_table(record: dict) -> None:
    print(f"prompt: {record['prompt']!r}")
    print(f"{'layer':>8}  {'pos':>5}  top-{record['top_k']} (token, rank)")
    for entry in record["layers"]:
        label = f"L{entry['layer']}" + ("*" if entry["is_model_output"] else "")
        for readout in entry["readouts"]:
            cells = "  ".join(
                f"{t['token']!r}({t['rank']})" for t in readout["top_k"]
            )
            print(f"{label:>8}  {readout['position']:>5}  {cells}")
    print("* final layer: the model's actual next-token prediction")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a fitted Jacobian lens to a prompt (read-only)."
    )
    parser.add_argument("--model", required=True, help="HF Hub id or local path")
    parser.add_argument(
        "--lens", required=True, help="lens .pt from fit_lens.py, or a Hub repo id"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--positions", nargs="+", type=int, default=[-2, -1])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None, help="default: all fitted layers"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the JSON event record, not a table"
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help=(
            "also write the JSON event record (per-layer top-k readouts) to "
            "FILE; serve_3d.py --record renders it without reloading the model"
        ),
    )
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
    hf_model, tokenizer = adapter.load_model(args.model, device_map=device_map)
    model = adapter.wrap(hf_model, tokenizer)
    lens = adapter.load_lens(args.lens)

    result = inspect_prompt(
        model,
        tokenizer,
        lens,
        args.prompt,
        args.positions,
        layers=args.layers,
        top_k=args.top_k,
        model_id=args.model,
        lens_id=args.lens,
    )
    if args.json:
        print(json.dumps(result["record"], ensure_ascii=False))
    else:
        print_table(result["record"])
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result["record"], ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote record to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
