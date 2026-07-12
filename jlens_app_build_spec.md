# Build Spec: Local Model Jacobian Lens Inspector

## Objective

Build a Python application that uses Anthropic's `jlens` library
(https://github.com/anthropics/jacobian-lens) to inspect what a small local
HuggingFace decoder model is "disposed to say" at intermediate layers. The app
has two phases: a one-time **fit** step that produces a lens artifact, and a
runtime **apply/inspect** step that loads the artifact and reads out per-layer
token predictions for arbitrary prompts.

## Background (for the implementer)

The Jacobian lens linearly transports a residual-stream activation at any
(layer, position) into the final-layer basis, then decodes it with the model's
own unembedding into a ranked list of vocabulary tokens. The lens is fit as
the average input–output Jacobian over a text corpus. It is read-only: model
weights are never modified. The lens itself is saved as a `.pt` file, separate
from the model.

Reference implementation notes from the repo:

- Fitting is dominated by the model's backward pass (needs gradient-capable
  VRAM, not just inference headroom).
- ~100 prompts is usable; the paper uses 1000 sequences of 128 tokens from a
  pretraining-like corpus. Quality saturates quickly.
- Fitting can be parallelized across disjoint prompt slices and combined with
  `JacobianLens.merge()`.
- Examples in the repo use Qwen; other HF decoders adapt cleanly.

## Hard requirements

1. **Model loading** must go through `transformers`
   (`AutoModelForCausalLM.from_pretrained` + `AutoTokenizer.from_pretrained`),
   in **bf16**, NOT quantized (no bitsandbytes 4-bit/8-bit) — quantized weights
   break the Jacobian estimation's backward passes. GGUF checkpoints are not
   usable; require an HF-format checkpoint (Hub ID or local directory).
2. **Fit and apply must be separate entry points.** Fit is an offline build
   step producing `out/jacobian_lens.pt`; the inspector must never trigger a
   fit implicitly.
3. **Checkpointing**: pass `checkpoint_path` to `jlens.fit()` so an
   interrupted fit can resume.
4. Default target model: a 2–4B class decoder (e.g. `Qwen/Qwen2.5-3B` or
   `google/gemma-2-2b-it`), configurable via CLI flag / config file. If a
   gated model (Gemma) is selected, surface a clear error telling the user to
   run `huggingface-cli login` and accept the license on the Hub page.
5. Use `uv` or `pip` with a `pyproject.toml`; pin `torch`, `transformers`, and
   install `jlens` from the GitHub repo (`pip install -e .` after clone, or a
   git dependency). Treat jlens as vendored/unmaintained: do not assume
   upstream fixes; wrap its API behind a thin adapter module so a fork can be
   swapped in.

## Repository layout

This repo is a clone of the upstream `jacobian-lens` library itself: the root
`pyproject.toml`, `uv.lock`, `jlens/`, and `tests/` are vendored upstream code
and must not be modified (see hard requirement 5). The app lives in a new
top-level `app/` directory, set up as its own self-contained uv project:

```
app/
├── pyproject.toml        # pins torch/transformers; depends on jlens via
│                         #   [tool.uv.sources] path dependency (../, editable)
├── fit_lens.py           # thin CLI entry script (spec acceptance criteria)
├── inspect.py            # thin CLI entry script (spec acceptance criteria)
├── serve_slice.py        # thin CLI entry script (optional component)
├── jlens_inspector/      # the package: adapter.py, fitting CLI impl,
│                         #   inspection impl, slice-page rendering
├── tests/                # app smoke tests, separate from upstream tests/
├── data/prompts.txt      # small default fitting corpus
└── out/                  # gitignored lens artifacts and checkpoints
```

Rationale and constraints:

- Never add app code or dependency pins to the repo root — that creates merge
  conflicts when pulling upstream and mixes app code into the vendored tree.
- The path dependency on `..` (editable) satisfies the "install jlens from the
  cloned repo" requirement with no network fetch, and keeps the app pinned to
  the exact vendored jlens version in one git history.
- `app/inspect.py` shadows the stdlib `inspect` module for *any* Python
  process with `app/` on `sys.path` (script dir for `python <script>.py`, cwd
  for `python -c`/REPL) — torch imports stdlib `inspect` and crashes. Every
  top-level entry script therefore strips `app/` from `sys.path` before heavy
  imports (the installed `jlens_inspector` package remains importable), and
  `inspect.py` additionally hands over to the real stdlib module if it was
  imported as `inspect` by accident. The package itself must not contain an
  import-root module named `inspect.py`.

## Components to build

### 1. `fit_lens.py` (CLI)

- Args: `--model` (HF id or path), `--prompts` (path to a text/JSONL corpus,
  one sequence per line), `--out` (lens path, default `out/jacobian_lens.pt`),
  `--checkpoint`, `--max-prompts` (default 200), `--seq-len` (default 128).
- Loads model in bf16 with `device_map="auto"` (or `.cuda()` for single GPU),
  wraps with `jlens.from_hf(hf, tok)`, calls `jlens.fit(...)`, saves the lens.
- Print VRAM estimate and warn if the model is >9B params (fitting a 26B on
  home-lab hardware is likely infeasible; suggest slice-parallel fit + merge).
- Include a `--merge` mode that combines multiple partial lens files via
  `JacobianLens.merge()`.

### 2. `inspect.py` (CLI + importable API)

- Args: `--model`, `--lens`, `--prompt`, `--positions` (default `[-2, -1]`),
  `--top-k` (default 5), `--layers` (default: all).
- Loads model + lens (`JacobianLens.from_pretrained` for Hub-hosted lenses, or
  local load for `fit_lens.py` output), calls
  `lens.apply(model, prompt, positions=...)`, and prints a table: layer ×
  position × top-k decoded tokens with ranks. Bottom row (final layer) is the
  model's actual next-token prediction — label it as such.
- Also expose a Python function `inspect_prompt(model, tok, lens, prompt,
  positions) -> dict` returning raw per-layer logits and decoded top-k, so the
  app can be embedded in a larger agent-observability pipeline. Output should
  be JSON-serializable (an append-friendly event record: prompt, positions,
  per-layer top-k tokens + ranks, timestamp, model id, lens id).

### 3. `serve_slice.py` (optional, nice-to-have)

- Render the repo's interactive layer × position "slice page" (d3-based HTML,
  self-contained file) for a given prompt, mirroring `walkthrough.ipynb`'s
  final step. Output a single HTML file to `out/`.

### 4. Tests

- Smoke test with a tiny model (e.g. `sshleifer/tiny-gpt2` or
  `Qwen/Qwen2.5-0.5B`) that: fits a lens on 5 short prompts, saves it,
  reloads it, applies it to one prompt, and asserts the returned structure
  (dict of layer -> logits tensor of vocab size) and that the final-layer lens
  output matches the model's own logits within tolerance.
- Test that the fit entry point refuses quantized load flags.

## Reference usage from the jlens README (adapt, do not invent APIs)

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])

# Fitting:
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

Before coding, read the repo's `README.md`, `walkthrough.ipynb`, and the
`jlens/fitting.py` module docstring (it documents the precise estimator) to
confirm exact signatures — do not guess parameters beyond what's shown above.

## Acceptance criteria

All commands below are run from the `app/` directory.

- `python fit_lens.py --model Qwen/Qwen2.5-0.5B --prompts data/prompts.txt`
  produces `out/jacobian_lens.pt` without modifying model weights.
- `python inspect.py --model Qwen/Qwen2.5-0.5B --lens out/jacobian_lens.pt
  --prompt "Fact: The currency used in the country shaped like a boot is"`
  prints per-layer top-k tokens, with mid-layer readouts differing from the
  final layer and the final layer matching the model's own prediction.
- All smoke tests pass on CPU or a single consumer GPU.
- README documents: install, HF auth for gated models, fit, inspect, merge,
  and the bf16/no-quantization constraint.

## Non-goals

- No weight modification, fine-tuning, or activation steering.
- No support for GGUF/llama.cpp/Ollama runtimes.
- No production hot-path optimization; this is an analysis/observability tool.
