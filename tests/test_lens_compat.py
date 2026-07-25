# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""A lens is only valid for the model it was fitted on."""

from __future__ import annotations

import pytest

from jlens.fitting import fit
from jlens.vis import compute_slice

from .tiny import TinyDecoder

PROMPT = "the quick brown fox"
FIT_PROMPTS = ["abcdefghij " * 5, "klmnopqrst " * 5]


def _fit(model: TinyDecoder):
    return fit(model, FIT_PROMPTS, dim_batch=4, max_seq_len=64)


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def deeper_lens():
    """Fitted on a 6-layer model: source layers 0..4 overrun a 4-layer one."""
    return _fit(TinyDecoder(n_layers=6, d_model=8))


@pytest.fixture(scope="module")
def wider_lens():
    return _fit(TinyDecoder(n_layers=4, d_model=16))


def test_too_many_layers_is_a_clear_error_not_an_index_error(model, deeper_lens):
    # Without the check this surfaced as IndexError from the forward hook.
    with pytest.raises(ValueError, match="out of range for a 4-layer model"):
        compute_slice(model, deeper_lens, PROMPT)
    with pytest.raises(ValueError, match="out of range for a 4-layer model"):
        deeper_lens.apply(model, PROMPT)


def test_d_model_mismatch_is_a_clear_error(model, wider_lens):
    with pytest.raises(ValueError, match="lens d_model=16 but model d_model=8"):
        compute_slice(model, wider_lens, PROMPT)
    with pytest.raises(ValueError, match="lens d_model=16 but model d_model=8"):
        wider_lens.apply(model, PROMPT)


def test_matching_lens_still_applies(model):
    lens = _fit(model)
    lens.check_compatible(model)
    assert compute_slice(model, lens, PROMPT).layers == [0, 1, 2, 3]
