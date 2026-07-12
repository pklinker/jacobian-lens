"""Jacobian lens inspector: fit/apply tooling around the vendored jlens.

All jlens API usage is confined to :mod:`jlens_inspector.adapter`.
"""

from jlens_inspector.inspection import inspect_prompt

__all__ = ["inspect_prompt"]
