# Jacobian Lens Inspector

Fit and apply a [Jacobian lens](https://github.com/anthropics/jacobian-lens)
to a small local HuggingFace decoder model: a one-time **fit** step produces a
lens artifact (`out/jacobian_lens.pt`), and a separate **inspect** step loads
it and prints what the model is "disposed to say" at every intermediate layer.

The lens is read-only — model weights are never modified. All use of the
vendored `jlens` library (the repo root, one directory up) goes through
[`jlens_inspector/adapter.py`](jlens_inspector/adapter.py); to swap in a fork,
edit that one file (or repoint the `[tool.uv.sources]` entry in
`pyproject.toml`).

## Install

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/). From this
`app/` directory:

```bash
uv sync --extra dev
```

This installs pinned `torch`/`transformers` (exact versions in `uv.lock`) and
`jlens` as an editable path dependency on the repo root.

### Hard constraint: bf16, never quantized

Models are always loaded via `transformers` in **bf16**. Quantized weights
(bitsandbytes 4-bit/8-bit) break the backward passes the Jacobian estimator
needs, so `fit_lens.py` refuses `--load-in-4bit`/`--load-in-8bit`, and GGUF
checkpoints are rejected — use an HF-format checkpoint (Hub ID or a local
directory with `config.json` + safetensors).

### Gated models (e.g. `google/gemma-2-2b-it`)

If you pick a gated model you'll get an error telling you to run
`huggingface-cli login` and accept the license on the model's Hub page. Do
both, then retry.

## Fit (offline build step)

```bash
uv run python fit_lens.py --model Qwen/Qwen2.5-0.5B --prompts data/prompts.txt
```

Produces `out/jacobian_lens.pt`. Useful flags: `--out`, `--max-prompts`
(default 200), `--seq-len` (default 128), `--dim-batch` (VRAM knob: output
dims per backward pass), `--checkpoint` (default `<out>.ckpt`). Fitting is
checkpointed after every prompt — re-running the same command resumes where
an interrupted fit stopped.

The default model is `Qwen/Qwen2.5-3B` (a 2–4B class decoder). Fitting needs
gradient-capable VRAM, roughly 3–4x the bf16 weight size; the CLI prints an
estimate and warns above 9B params.

### Parallel fit + merge

For big models or big corpora, fit disjoint prompt slices in separate runs
and combine the partial lenses:

```bash
uv run python fit_lens.py --model M --prompts slice_a.txt --out out/part_a.pt
uv run python fit_lens.py --model M --prompts slice_b.txt --out out/part_b.pt
uv run python fit_lens.py --merge out/part_a.pt out/part_b.pt --out out/jacobian_lens.pt
```

## Inspect (runtime)

The inspector never fits implicitly — it errors if the lens file is missing.

```bash
uv run python inspect.py --model Qwen/Qwen2.5-0.5B \
  --lens out/jacobian_lens.pt \
  --prompt "Fact: The currency used in the country shaped like a boot is"
```

Prints a layer x position x top-k table. The last row (marked `*`) is the
model's actual next-token prediction; every other row is the lens readout at
that layer. Flags: `--positions` (default `-2 -1`), `--top-k` (default 5),
`--layers` (default all fitted layers), `--json` (emit the event record
instead of a table). `--lens` also accepts a HuggingFace Hub repo id hosting
a pre-fitted lens.

### Embedding in a pipeline

```python
from jlens_inspector import adapter, inspect_prompt

hf, tok = adapter.load_model("Qwen/Qwen2.5-0.5B")
model = adapter.wrap(hf, tok)
lens = adapter.load_lens("out/jacobian_lens.pt")

result = inspect_prompt(model, tok, lens, "some prompt", positions=[-1],
                        model_id="Qwen/Qwen2.5-0.5B", lens_id="out/jacobian_lens.pt")
result["record"]        # JSON-serializable event: timestamp, model/lens ids,
                        #   per-layer top-k tokens with ranks
result["lens_logits"]   # {layer: Tensor[n_positions, vocab]} raw logits
result["model_logits"]  # the model's own final-layer logits
```

## Slice page (optional)

Render the repo's interactive layer x position visualisation as a single
self-contained HTML file (first run fetches d3 once, so it needs network):

```bash
uv run python serve_slice.py --model Qwen/Qwen2.5-0.5B \
  --lens out/jacobian_lens.pt --prompt "..." --out out/slice.html
```

## Tests

```bash
uv run pytest
```

The smoke tests download `sshleifer/tiny-gpt2` (a few MB) on first run and
run on CPU; skip them with `-m "not network"`.
