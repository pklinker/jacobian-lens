"""End-to-end smoke test on a tiny model: fit -> save -> reload -> apply.

Downloads sshleifer/tiny-gpt2 (a few MB) from the Hub on first run.
"""

import json

import pytest
import torch

from jlens_inspector import adapter
from jlens_inspector.inspection import inspect_prompt

TINY_MODEL = "sshleifer/tiny-gpt2"

# Each prompt must exceed jlens's skip_first (16) + 1 tokens to be usable.
FIT_PROMPTS = [
    "the quick brown fox jumps over the lazy dog " * 4,
    "a stitch in time saves nine and practice makes perfect " * 3,
    "every good boy deserves fudge while cows eat grass slowly " * 3,
    "pack my box with five dozen liquor jugs right now please " * 3,
    "how vexingly quick daft zebras jump over fences at dawn " * 3,
]
PROMPT = "Fact: The currency used in the country shaped like a boot is"
POSITIONS = [-2, -1]


@pytest.fixture(scope="module")
def tiny():
    hf_model, tokenizer = adapter.load_model(TINY_MODEL)
    return hf_model, tokenizer, adapter.wrap(hf_model, tokenizer)


@pytest.mark.network
def test_fit_save_reload_apply(tmp_path, tiny):
    hf_model, tokenizer, model = tiny
    weights_before = {
        name: param.detach().clone() for name, param in hf_model.named_parameters()
    }

    lens = adapter.fit_lens(
        model,
        FIT_PROMPTS,
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        max_seq_len=64,
        dim_batch=8,
    )
    assert lens.n_prompts == len(FIT_PROMPTS)
    # Fitted layers are everything below the final layer.
    assert lens.source_layers == list(range(model.n_layers - 1))
    for J in lens.jacobians.values():
        assert J.shape == (model.d_model, model.d_model)

    # Fitting is read-only: weights are untouched.
    for name, param in hf_model.named_parameters():
        assert torch.equal(param, weights_before[name]), name

    lens_path = tmp_path / "jacobian_lens.pt"
    lens.save(str(lens_path))
    reloaded = adapter.load_lens(str(lens_path))
    assert reloaded.source_layers == lens.source_layers

    result = inspect_prompt(
        model,
        tokenizer,
        reloaded,
        PROMPT,
        POSITIONS,
        model_id=TINY_MODEL,
        lens_id=str(lens_path),
    )

    # Structure: dict of layer -> [n_positions, vocab] logits.
    vocab_size = hf_model.config.vocab_size
    assert set(result["lens_logits"]) == set(reloaded.source_layers)
    for logits in result["lens_logits"].values():
        assert logits.shape == (len(POSITIONS), vocab_size)
    assert result["model_logits"].shape == (len(POSITIONS), vocab_size)

    # The final-layer row matches the model's own logits within tolerance.
    input_ids = model.encode(PROMPT)
    with torch.no_grad():
        reference = hf_model(input_ids).logits[0, POSITIONS, :].float().cpu()
    torch.testing.assert_close(
        result["model_logits"], reference, rtol=5e-2, atol=5e-2
    )
    # Mid-layer readouts are not just the final-layer logits.
    first_fitted = reloaded.source_layers[0]
    assert not torch.equal(result["lens_logits"][first_fitted], result["model_logits"])

    # The event record is JSON-serializable and labels the model-output row.
    encoded = json.dumps(result["record"])
    assert TINY_MODEL in encoded
    layers = result["record"]["layers"]
    assert [entry["is_model_output"] for entry in layers] == [False] * (
        len(layers) - 1
    ) + [True]
    assert layers[-1]["layer"] == model.n_layers - 1
    for entry in layers:
        for readout in entry["readouts"]:
            assert [t["rank"] for t in readout["top_k"]] == [1, 2, 3, 4, 5]


@pytest.mark.network
def test_merge_partial_lenses(tmp_path, tiny):
    _, _, model = tiny
    paths = []
    for i, prompt in enumerate(FIT_PROMPTS[:2]):
        lens = adapter.fit_lens(
            model,
            [prompt],
            checkpoint_path=str(tmp_path / f"ckpt{i}.pt"),
            max_seq_len=64,
            dim_batch=8,
        )
        path = tmp_path / f"part{i}.pt"
        lens.save(str(path))
        paths.append(str(path))

    merged = adapter.merge_lens_files(paths)
    assert merged.n_prompts == 2
    assert merged.source_layers == list(range(model.n_layers - 1))
