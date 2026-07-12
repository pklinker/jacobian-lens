#!/usr/bin/env python
"""Render an interactive slice page. See jlens_inspector/slice_cli.py.

This directory contains ``inspect.py``, which shares its name with a stdlib
module, and Python puts the script's directory first on ``sys.path`` — so a
bare ``import inspect`` inside torch/transformers would import that file
instead of the stdlib. Strip this directory from ``sys.path`` before any
heavy import; the jlens_inspector package is installed into the environment,
so it stays importable.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

from jlens_inspector.slice_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
