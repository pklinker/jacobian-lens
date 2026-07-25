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

### Local model store

Every entry point keeps downloaded models in a local store (`models/` by
default, gitignored): the first run of a Hub id downloads it once and saves
it there, and later runs load it straight from disk — no network and no Hub
login, which also sidesteps re-auth for gated models. `--models-dir DIR`
points the store somewhere else (e.g. a bigger drive):

```bash
uv run python inspect.py --model Qwen/Qwen2.5-3B --models-dir /Volumes/big/models ...
```

A `--model` that is an explicit local path bypasses the store entirely. To
re-download a model, delete its `models/<org>--<name>/` directory.

## Fit (offline build step)

```bash
uv run python fit_lens.py --model Qwen/Qwen2.5-0.5B --prompts data/prompts.txt
```

Produces `out/jacobian_lens.pt`. Useful flags: `--out`, `--max-prompts`
(default 200), `--seq-len` (default 128), `--dim-batch` (VRAM knob: output
dims per backward pass), `--checkpoint` (default `<out>.ckpt`). Fitting is
checkpointed after every prompt — re-running the same command resumes where
an interrupted fit stopped.

`--max-prompts` caps the corpus in every mode, and it defaults to **200**.
`data/prompts.txt` ships with 324 lines, so the command above fits on the
first 200 and ignores the rest; pass `--max-prompts 324` to use all of them.

### Corpus source

`J_l` is an expectation over a generic web-text corpus, so what you fit on
shapes the lens. `--prompt-source` selects where the corpus comes from:

| Source | Corpus |
| --- | --- |
| `file` (default) | `--prompts` only — full control, narrower distribution |
| `wikitext` | WikiText-103 streamed from the Hub — closest to pretraining text |
| `mixed` | both, in a `--wikitext-fraction` split (default 0.5) |

```bash
# half hand-written prompts, half WikiText-103
uv run python fit_lens.py --model Qwen/Qwen2.5-0.5B \
  --prompts data/prompts.txt --prompt-source mixed \
  --wikitext-fraction 0.5 --max-prompts 1000
```

`--max-prompts` is the total either way. Under `mixed`, if the file holds
fewer lines than its share (324 lines against a 500-line share of 1000), the
shortfall is drawn from WikiText so the corpus still reaches the total.

The mixed corpus is shuffled rather than concatenated, so that an interrupted
fit has still seen both sources. The shuffle is seeded (`--seed`, default 0)
because resume replays the prompt list by index: **the checkpoint records the
fit geometry but not the corpus**, so resuming after changing `--seed`,
`--prompt-source`, `--wikitext-fraction`, or the contents of `--prompts` will
silently average over a different set of prompts than the run it resumes.
Delete the `.ckpt` when you change any of them.

`wikitext` and `mixed` stream from the Hub, so they need network and the
`datasets` package, which is not part of the default install:

```bash
uv sync --extra wikitext
```

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
A lens is tied to the model it was fitted on (its `J_l` are `d_model x
d_model` and its layer indices address that model's blocks), so `--model`
must name that same model; a mismatch is reported before the readout runs.

```bash
uv run python inspect.py --model Qwen/Qwen2.5-0.5B \
  --lens out/jacobian_lens.pt \
  --prompt "Fact: The currency used in the country shaped like a boot is"
```

Prints a layer x position x top-k table. The last row (marked `*`) is the
model's actual next-token prediction; every other row is the lens readout at
that layer. Flags: `--positions` (default `-2 -1`), `--top-k` (default 5),
`--layers` (default all fitted layers), `--json` (emit the event record
instead of a table), `--out FILE` (also write the record to a file — the 3D
viewer below can render it without reloading the model). `--lens` also
accepts a HuggingFace Hub repo id hosting a pre-fitted lens.

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

## 3D layer viewer (optional)

Render an interactive 3D view of the same slice: each layer is a plane
(rows = prompt positions, columns = top-k slots) stacked by depth, with lines
linking the same token across adjacent layers — the graph of candidates
persisting, migrating, and dying out as the model converges on its answer.
Output is a single self-contained HTML file (three.js inlined; the first
build fetches it once, so it needs network):

```bash
uv run python serve_3d.py --model Qwen/Qwen2.5-0.5B \
  --lens out/jacobian_lens.pt --prompt "..." --out out/viewer3d.html
```

In the page: drag to orbit, hover a cell for details, click a token (or use
the search box) to trace its full-vocabulary rank trajectory through every
layer; pin several tokens to compare them, and "sweep" animates a pass
through the layer stack. Flags: `--top-n` (default 10), `--layer-stride`,
`--last-n-tokens`, `--max-seq-len`, `--mask-display` (as in
`serve_slice.py`), plus `--pin TOKEN` (repeatable) to pre-pin tokens at
load. Very long prompts make heavy pages — the CLI warns above ~200k cells;
window with `--last-n-tokens` or thin layers with `--layer-stride`.

### Rendering a saved inspect record

The viewer can also render an `inspect.py --out` record without touching the
model — useful when the inspection ran elsewhere (a GPU box, a pipeline):

```bash
uv run python inspect.py --model M --lens L --prompt "..." --out out/record.json
uv run python serve_3d.py --record out/record.json --out out/viewer3d.html
```

Any open viewer page can likewise load a record at runtime via its
**import record JSON…** button (reset returns to the page's embedded data).
Either way the record only carries top-k readouts, so rank trajectories are
limited to top-k depth — cells and links are exact, but a token's curve
breaks where it leaves the top-k (a full `serve_3d.py --model ...` build has
true full-vocabulary curves).

## Tests

```bash
uv run pytest
```

The smoke tests download `sshleifer/tiny-gpt2` (a few MB) on first run and
run on CPU; skip them with `-m "not network"`.
