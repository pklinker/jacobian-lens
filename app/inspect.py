#!/usr/bin/env python
"""Apply a fitted Jacobian lens to a prompt. See jlens_inspector/inspection.py.

This script shares its name with the stdlib ``inspect`` module. Whenever this
directory lands on ``sys.path`` (script dir when running ``python
inspect.py``, cwd when running ``python -c`` / a REPL from here), a bare
``import inspect`` inside torch/transformers would import this file instead
of the stdlib. Two defenses:

1. Strip this directory from ``sys.path`` before any heavy import (the
   jlens_inspector package is installed into the environment, so it stays
   importable).
2. If this file was already imported *as* ``inspect``, load the real stdlib
   module, put it back in ``sys.modules``, and mirror its namespace here.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

if __name__ == "inspect":
    del sys.modules["inspect"]
    import inspect as _stdlib_inspect  # resolves to the stdlib now that _HERE is gone

    sys.modules["inspect"] = _stdlib_inspect
    globals().update(_stdlib_inspect.__dict__)
else:
    from jlens_inspector.inspection import main

    if __name__ == "__main__":
        raise SystemExit(main())
