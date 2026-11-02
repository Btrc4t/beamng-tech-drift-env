"""A drop-in array module: cupy (GPU) when available, else numpy (CPU).

The hot path (the observation / point-cloud processing) imports :data:`np`
from here instead of ``numpy`` directly, so it runs on the GPU when cupy is
installed and falls back to the CPU otherwise. The math is written in the
shared numpy/cupy API, so the same code runs on either backend.

Data that comes from the game (the LiDAR shared memory, the State sensor) is
a CPU numpy array. :func:`to_gpu` moves it to the GPU (a no-op when cupy is
not used), and :func:`to_cpu` moves the finished observation back to the CPU
(a no-op when cupy is not used) so the rest of the stack (gymnasium, the
collector, the callbacks) keeps seeing plain numpy arrays.

By default cupy runs on the **last** CUDA device (``CUPY_DEVICE`` selects a
specific one): the game's renderer usually occupies the first GPU, so using a
different one avoids contending with it for GPU time.

Set the ``USE_CUPY`` environment variable to ``0`` (or ``false``/``no``/
``off``, case-insensitive) to force the CPU backend even when cupy is
installed: cupy is not imported at all and ``to_gpu``/``to_cpu`` become
no-ops, so the observation / point-cloud processing stays on the CPU.
"""
import os

import numpy as _numpy

_env = os.environ.get("USE_CUPY", "1").strip().lower()
if _env in ("0", "false", "no", "off"):
    _cupy = None
else:
    try:
        import cupy as _cupy
    except ImportError:
        _cupy = None

if _cupy is not None:
    np = _cupy
    USE_CUPY = True
    _n_dev = _cupy.cuda.runtime.getDeviceCount()
    _env = os.environ.get("CUPY_DEVICE")
    _device = int(_env) if _env is not None else _n_dev - 1
    CUPY_DEVICE = _device if 0 <= _device < _n_dev else 0
    _cupy.cuda.Device(CUPY_DEVICE).use()
else:
    np = _numpy
    USE_CUPY = False
    CUPY_DEVICE = None


def use_device():
    """Make the current thread use the selected CUDA device (a no-op when cupy
    is not used). A new thread defaults to device 0, so a worker thread that
    runs cupy ops must call this first."""
    if USE_CUPY:
        _cupy.cuda.Device(CUPY_DEVICE).use()


def make_stream():
    """A new non-blocking CUDA stream for concurrent work (None when cupy is
    not used). Kernels submitted in different streams can run at the same
    time, so an independent sub-computation (e.g. the road BEV) can overlap
    the point-cloud path."""
    return _cupy.cuda.Stream(non_blocking=True) if USE_CUPY else None


def to_gpu(x):
    """Move ``x`` to the GPU (a no-op when cupy is not used).

    A numpy array is copied to the GPU; a cupy array is returned as-is (no
    copy). The result is always an array of the active backend.
    """
    return _cupy.asarray(x) if USE_CUPY else _numpy.asarray(x)


def to_cpu(x):
    """Move ``x`` back to the CPU (a no-op when cupy is not used).

    A cupy array is copied to the CPU (a numpy array); anything else is
    returned as-is.
    """
    if USE_CUPY and isinstance(x, _cupy.ndarray):
        return x.get()
    return x
