"""Single seam between this app and the vendored jlens library.

jlens is a reference implementation and is not maintained upstream. Every
jlens API call in this app goes through this module, so swapping in a fork
means editing exactly one file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers

import jlens
from jlens import examples as jlens_examples
from jlens.lens import JacobianLens
from jlens.vis import SliceData, build_page, compute_slice

__all__ = [
    "JacobianLens",
    "SliceData",
    "apply_lens",
    "configure_logging",
    "fit_lens",
    "load_lens",
    "load_model",
    "local_model_path",
    "load_wikitext_prompts",
    "merge_lens_files",
    "render_slice_page",
    "slice_for_prompt",
    "wrap",
]

NO_QUANTIZATION_MSG = (
    "quantized weights are not usable: Jacobian estimation backprops through "
    "the model, and 4-bit/8-bit kernels break those backward passes. Load the "
    "checkpoint in bf16 instead."
)

_GATED_MSG = (
    "Model {model_id!r} is gated on the HuggingFace Hub. Run "
    "`huggingface-cli login`, accept the license on "
    "https://huggingface.co/{model_id}, then retry."
)

_NO_DATASETS_MSG = (
    "streaming WikiText-103 needs the `datasets` package, which is not "
    "installed. Run `uv sync --extra wikitext` from the app/ directory, or "
    "fit on a local corpus with `--prompt-source file`."
)


def configure_logging() -> None:
    jlens.configure_logging()


def local_model_path(model_id: str, models_dir: str | Path) -> Path:
    """Directory in the local model store for a Hub model id."""
    return Path(models_dir) / str(model_id).replace("/", "--")


def load_model(
    model_id: str,
    *,
    device_map: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    models_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load an HF-format causal LM + tokenizer in bf16, never quantized.

    ``device_map="auto"`` shards across available GPUs (requires accelerate);
    ``None`` leaves the model where transformers puts it (CPU).

    ``models_dir`` names a local model store: a Hub id already present there
    is loaded from disk (no network, no Hub auth), and one that isn't is
    downloaded once and then saved into the store for later runs. Explicit
    local paths bypass the store, and ``None`` disables it.
    """
    if str(model_id).endswith(".gguf"):
        raise ValueError(
            f"{model_id!r} is a GGUF checkpoint; jlens needs an HF-format "
            "checkpoint (Hub ID or a directory with config.json + safetensors)"
        )
    source = str(model_id)
    store_to: Path | None = None
    if models_dir is not None and not os.path.isdir(source):
        local = local_model_path(source, models_dir)
        if (local / "config.json").exists():
            print(f"loading {model_id} from local model store: {local}")
            source = str(local)
        else:
            store_to = local
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, device_map=device_map
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    except OSError as exc:
        message = str(exc).lower()
        if "gated" in message or "403" in message:
            raise RuntimeError(_GATED_MSG.format(model_id=model_id)) from exc
        raise
    if getattr(hf_model.config, "quantization_config", None) is not None:
        raise ValueError(NO_QUANTIZATION_MSG)
    if store_to is not None:
        # A failed save must not kill the run — the model is already loaded.
        try:
            store_to.mkdir(parents=True, exist_ok=True)
            hf_model.save_pretrained(store_to)
            tokenizer.save_pretrained(store_to)
            print(f"stored {model_id} in local model store: {store_to}")
        except OSError as exc:
            print(f"warning: could not store {model_id} at {store_to}: {exc}")
    return hf_model, tokenizer


def wrap(hf_model: Any, tokenizer: Any) -> Any:
    """Wrap a loaded HF model as a jlens ``LensModel``."""
    return jlens.from_hf(hf_model, tokenizer)


def fit_lens(
    model: Any,
    prompts: Sequence[str],
    *,
    checkpoint_path: str,
    max_seq_len: int = 128,
    dim_batch: int = 8,
) -> JacobianLens:
    """Fit ``J_l`` over ``prompts``; resumable via ``checkpoint_path``."""
    return jlens.fit(
        model,
        prompts,
        checkpoint_path=checkpoint_path,
        max_seq_len=max_seq_len,
        dim_batch=dim_batch,
    )


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """Stream ``n_prompts`` WikiText-103 records of at least ``min_chars``
    characters from the HuggingFace Hub. Needs the ``wikitext`` extra and
    network access."""
    try:
        return jlens_examples.load_wikitext_prompts(n_prompts, min_chars=min_chars)
    except ImportError as exc:
        raise RuntimeError(_NO_DATASETS_MSG) from exc


def load_lens(name_or_path: str, *, filename: str = "lens.pt") -> JacobianLens:
    """Load a lens from a local ``.pt`` file, a directory, or a Hub repo id."""
    return JacobianLens.from_pretrained(name_or_path, filename=filename)


def merge_lens_files(paths: Sequence[str]) -> JacobianLens:
    """Combine partial lenses fitted on disjoint prompt slices."""
    return JacobianLens.merge([JacobianLens.load(str(p)) for p in paths])


def apply_lens(
    lens: JacobianLens,
    model: Any,
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Read out ``prompt`` at ``(layers, positions)``.

    Returns ``(lens_logits, model_logits, input_ids)``; ``lens_logits`` maps
    layer -> ``[n_positions, vocab]`` and ``model_logits`` is the model's own
    final-layer logits at the same positions.
    """
    return lens.apply(model, prompt, layers=layers, positions=positions)


def slice_for_prompt(
    model: Any,
    lens: JacobianLens,
    prompt: str,
    *,
    top_n: int = 10,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    mask_display: bool = False,
    max_seq_len: int = 512,
    pinned_token_ids: set[int] | None = None,
) -> SliceData:
    return compute_slice(
        model,
        lens,
        prompt,
        top_n=top_n,
        layer_stride=layer_stride,
        last_n_tokens=last_n_tokens,
        mask_display=mask_display,
        max_seq_len=max_seq_len,
        pinned_token_ids=pinned_token_ids,
    )


def render_slice_page(
    slice_data: SliceData, prompt: str, *, title: str, description: str
) -> str:
    """Render a fully self-contained HTML slice page (d3 inlined; needs
    network access once to fetch d3)."""
    page, _, _ = build_page(
        slice_data, prompt, title=title, description=description, mode="embed"
    )
    return page
