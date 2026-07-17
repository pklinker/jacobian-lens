# Implementation Plan: 3D Layer Viewer (`serve_3d.py`)

## Goal

An interactive 3D visualization of a Jacobian-lens slice: each transformer
layer is a horizontal plane (rows = prompt positions, columns = top-k slots),
planes stacked vertically by layer depth. Edges link occurrences of the same
token across adjacent layers, so the viewer shows the *graph* of how candidate
tokens appear, persist, migrate between rank slots, and die out as the
residual stream converges on the model's final prediction.

Delivered as a self-contained static HTML file (three.js inlined), mirroring
the existing `serve_slice.py` pattern. All new code lives in `app/` — the
vendored repo root (`jlens/`, root `pyproject.toml`, root `tests/`) is not
touched.

## Scene design

```
            ▲ layer (Y)
  L23* ────────────────────────  ← final layer plane (model output, visually
  L22  ────────────────────────     distinguished: brighter frame + "*" label)
   ...        │ ╲   edges: same token in adjacent layers'
  L1   ───────┼──╲─────────────    top-k, drawn rank-slot → rank-slot
  L0   ────────────────────────
        position (X) →   top-k slot (Z) ↗
```

- **Cell**: one quad per `(position, layer, k)` holding the decoded token
  string, colored by rank (k=0 bright → k=top_n-1 dim). Rendered as a single
  `InstancedMesh` with a canvas texture atlas of the unique token strings
  (unique tokens ≪ cells; the atlas is built once at load).
- **Edges**: for each pair of adjacent rendered layers and each position, a
  line segment wherever the same token id appears in both layers' top-k at
  that position. One `LineSegments` buffer for all edges; per-vertex color
  encodes the rank at each end (so a token climbing to rank 1 visibly
  "brightens" upward). Cross-position edges are *not* drawn by default —
  the graph links columns (same position across layers); rows are the plane
  grid itself.
- **Trajectory highlight** (click / search / pin): for a selected token,
  re-color its cells and edges, dim everything else, and additionally draw
  its *full* rank trajectory per position from the tracked-token rank tensor
  — a polyline through all layers where Z position within the plane maps
  `log(rank)` (so the token remains visible even where it falls out of
  top-k, exactly what the 2D page's rank charts show, but in-scene).
- **Scale**: typical case ~30 positions x ~25 layers x 10 slots = 7,500
  instances plus a few thousand edge segments — trivial for WebGL. Guard
  rail: warn in the page header above ~200k cells and suggest
  `--layer-stride` / `--last-n-tokens`.

## Data flow — reuse, don't recompute

`jlens.vis.compute_slice` (via `adapter.slice_for_prompt`) already produces
everything the scene needs, and `build_page`'s embed bootstrap format
(`meta` + base64 `slice.bin` + `ranks/{tid}.bin`, gzip'd int32 arrays) is
already designed as a virtual filesystem with a lazy rank loader. The 3D page
**reuses the same payload format and loader JS** as the d3 page:

- `slice.bin` → `top_ids`/`top_ranks` `[seq_len, n_layers, top_n]` — cells + edges.
- `ranks/{tid}.bin` → `[seq_len, n_layers]` full-vocab ranks — trajectory
  polylines, decoded lazily on pin/click (same as the 2D page).
- `meta` → layers, context token strings, vocab fragment, tracked/pinned ids,
  `ctx_offset`, `vocab_size`.

No changes to `jlens/vis.py`: the payload builder in `app` calls the *public*
pieces (`compute_slice`, `SliceData`) through `adapter.py` and re-implements
the ~30 lines of embed-bootstrap assembly (gzip + base64 + `</`-escaping) in
app code rather than importing private helpers (`_slice_bin`, `_slice_meta`)
— keeping the adapter the only jlens seam and avoiding reliance on
underscore-private API in a vendored library.

## Files

```
app/
├── serve_3d.py                        # thin entry script (sys.path guard, like serve_slice.py)
└── jlens_inspector/
    ├── viewer3d_cli.py                # argparse + model/lens load + orchestration
    ├── viewer3d.py                    # payload assembly + template fill (no torch imports)
    └── data/
        └── viewer3d.html              # three.js page template (package data)
```

- `serve_3d.py`: copy of the `serve_slice.py` preamble (strip `app/` from
  `sys.path` before heavy imports — required because `app/inspect.py` shadows
  stdlib `inspect`), then delegates to `jlens_inspector.viewer3d_cli:main`.
- `viewer3d_cli.py`: mirrors `slice_cli.py`. Flags:
  `--model`, `--lens`, `--prompt`, `--out` (default `out/viewer3d.html`),
  `--title`, `--top-n` (default 10), `--layer-stride`, `--last-n-tokens`,
  `--max-seq-len` (default 512), `--mask-display`, `--pin` (token strings to
  pin at load, tokenized single-token-or-error). Same missing-lens guard
  reusing `MISSING_LENS_MSG`.
- `viewer3d.py`:
  - `build_viewer_page(slice_data, prompt, *, title, description) -> str` —
    assembles the embed bootstrap dict, fills `__TITLE__ / __WHAT__ /
    __BOOTSTRAP__ / __THREE__` placeholders in the template.
  - three.js embedding: pin **three r147** — the last release shipping UMD
    builds of both `three.min.js` and `examples/js/controls/OrbitControls.js`
    — fetch both once with SRI SHA-384 verification, and inline them as
    plain `<script>` text, byte-for-byte the same mechanism `_template
    ("embed")` uses for d3. No import maps, no ESM, no novel machinery;
    everything the scene needs (InstancedMesh, LineSegments, instanced
    raycasting, OrbitControls) is stable in r147, and pinning a frozen
    version matches the repo's vendoring philosophy. Memoise per process
    like `_TEMPLATE_FOR_MODE`. A fetch or integrity failure raises — never
    silently emit a CDN-dependent page.
- `adapter.py` (small extension, existing seam file): pass `top_n`,
  `last_n_tokens`, and `pinned_token_ids` through `slice_for_prompt` — they
  are existing `compute_slice` kwargs the adapter currently doesn't expose.
- `pyproject.toml`: add `jlens_inspector/data/*.html` as package data.

## Template (`viewer3d.html`) structure

1. **Loader** — lifted from `slice_vis.html`: bootstrap `<script
   type="application/json">`, `b64bytes`, gzip `DecompressionStream`,
   virtual-filesystem `read(name)`, int32 array views. Embed mode only
   (no fetch mode in v1 — cut scope; the payload format doesn't preclude
   adding it later).
2. **Scene build** (`<script type="module">`):
   - token atlas: draw each unique top-k token string once into a canvas
     atlas; instanced quads sample their glyph via per-instance UV offsets.
   - planes: wireframe grid + layer label (`L{n}`, final layer `L{n}*` styled
     as model output, matching the CLI table's convention).
   - edge pass: single iteration over `[seq_len, n_layers-1, top_n]`
     building matched-token segment list into one `BufferGeometry`.
3. **Interaction layer**:
   - OrbitControls (rotate/zoom/pan), raycaster hover → HTML tooltip
     (token, id, rank, layer, position, and the position's context string).
   - click cell → select token → highlight trajectory (see scene design),
     `Esc`/background click clears.
   - **HTML side panel** (plain DOM, no framework, styled like the d3 page):
     - token search box over `meta.vocab` fragments; selecting = same as click.
     - pin list (load-time pins from `--pin` pre-selected); pins persist as
       colored trajectories simultaneously, each with a legend swatch.
     - layer range double-slider + top-k display slider (filter without
       recompute — just instance visibility masks).
     - position window brush (highlight + optionally cull positions).
     - **layer sweep**: play button animating a highlight plane from L0 to
       the final layer, dimming layers above the sweep line — shows the
       "convergence movie" through depth.

## Testing

Follow the existing app test conventions (`tests/test_cli_guards.py`,
`tests/test_smoke.py`, `network` marker):

1. **Guard tests** (no network): `serve_3d.py --lens missing.pt` exits with
   `MISSING_LENS_MSG`; `--pin` with a multi-token string errors clearly.
2. **Payload unit tests** (no network, no torch model): build a synthetic
   `SliceData` (tiny numpy arrays), assert the bootstrap JSON round-trips —
   layers/ctx strings present, `slice.bin` gunzips to the input arrays,
   `</` escaped in vocab strings, one `ranks/{tid}.bin` per tracked token.
3. **Smoke test** (`network` marker, tiny-gpt2 like existing smoke tests):
   fit-free path using a lens fixture from the existing smoke fixtures,
   render the page, assert it's a single file containing the three.js
   payload marker, the bootstrap, and a known context token string.
4. **Browser check** (manual, not CI): open the generated page, verify scene
   renders and console is clean. (Optional follow-up: a playwright smoke
   test; out of scope for v1.)

## Milestones

1. **Skeleton + payload** — entry script, CLI, adapter passthroughs, payload
   assembly, template with loader and an empty scene; guard + payload tests.
2. **Static scene** — planes, instanced token cells, atlas text, rank
   coloring, orbit controls.
3. **Graph edges** — adjacent-layer same-token segments with rank-gradient
   coloring.
4. **Core interaction** — hover tooltip, click-to-trace with full-rank
   trajectory polylines from `ranks/{tid}.bin`.
5. **Explorer panel** — search, pinning, layer/top-k/position filters, layer
   sweep animation.
6. **Polish + docs** — smoke test, README section (install unchanged; new
   "3D viewer" section beside "Slice page"), payload-size guard rail.

Each milestone leaves the tool runnable end-to-end
(`uv run python serve_3d.py --model ... --lens ... --prompt ...`).

## Risks / notes

- **three.js version pin**: r147 (Dec 2022) is frozen — no upstream fixes
  will land, same trade the repo already makes for `jlens` itself. If a
  future feature genuinely needs modern three (e.g. WebGPU), the upgrade
  path is a data-URL import map carrying the ESM build; documented here so
  it isn't rediscovered, deliberately not v1. Milestone 1 still proves the
  inlined r147 renders a cube in the generated file before any scene work.
- **Token text legibility in 3D** is the main UX risk. Mitigations: billboard
  the cell quads toward the camera (sprite mode toggle), scale font atlas by
  device pixel ratio, and lean on hover/tooltip rather than trying to make
  every distant label readable.
- **Payload size**: same envelope as the 2D embed page (rank files dominate,
  `~tracked * seq_len * n_layers * 4` bytes pre-gzip). The existing
  `--max-seq-len` / `--layer-stride` / `last-n-tokens` knobs are the control;
  print the same "wrote N KiB" line as `serve_slice.py`.
- **Vendored-root constraint**: everything above is `app/`-only; if a private
  `jlens.vis` helper turns out to be needed after all, the rule from the
  build spec applies — extend via `adapter.py`, never edit `jlens/`.
