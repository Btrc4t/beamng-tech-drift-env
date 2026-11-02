"""The array backend selection: cupy when installed, numpy otherwise, and
the USE_CUPY environment flag that forces the CPU backend.

array.py reads the flag at import time, so each case runs in a subprocess
with the controlled environment.
"""
import os
import subprocess
import sys

import pytest

_CHECK = (
    "import sys\n"
    "import numpy as np\n"
    "from beamng_tech_env.core import array\n"
    "x = array.to_gpu(np.arange(3, dtype='float32'))\n"
    "y = array.to_cpu(x)\n"
    "print(array.USE_CUPY)\n"
    "print(array.np is np)\n"
    "print('cupy' in sys.modules)\n"
    "print(type(x) is np.ndarray)\n"
    "print(y is x)\n"
    "print(array.make_stream() is None)\n"
)


def _probe(env):
    """Import the array module in a subprocess; the backend state as booleans
    (USE_CUPY, np is numpy, cupy imported, to_gpu->numpy, to_cpu identity,
    no stream)."""
    out = subprocess.run(
        [sys.executable, "-c", _CHECK],
        env=env, capture_output=True, text=True, check=True)
    return [line == "True" for line in out.stdout.splitlines()]


def _env(**extra):
    env = dict(os.environ)
    env.pop("USE_CUPY", None)
    env.update(extra)
    return env


def test_use_cupy_flag_disables_cupy():
    # every falsy value forces the CPU backend: cupy is not even imported,
    # and to_gpu/to_cpu/make_stream become no-ops
    for value in ("0", "false", "no", "off", "OFF"):
        assert _probe(_env(USE_CUPY=value)) == \
            [False, True, False, True, True, True], value


def test_use_cupy_flag_keeps_cupy():
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")
    for value in (None, "1", "true"):
        env = _env() if value is None else _env(USE_CUPY=value)
        assert _probe(env) == [True, False, True, False, False, False], value
