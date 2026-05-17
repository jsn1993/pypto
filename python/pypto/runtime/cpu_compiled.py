"""
CPU-compiled function wrapper — loads a .so via ctypes and marshals torch tensors.
"""

from ctypes import CDLL, c_int32, c_int64
from typing import Any

import numpy as np


class CPUCompiledFunction:
    """Callable wrapper around a CPU-compiled shared library.

    Marshals ``torch.Tensor`` to raw pointers, calls the C function via ctypes,
    and wraps results back as ``torch.Tensor``.
    """

    def __init__(
        self,
        lib: CDLL,
        func_name: str,
        param_metas: list[dict],
        return_metas: list[dict],
    ) -> None:
        self._lib = lib
        self._func_name = func_name
        self._param_metas = param_metas
        self._return_metas = return_metas
        self._func = getattr(lib, func_name)

    def __call__(self, *args: Any) -> Any:
        import ctypes

        import torch

        c_args: list = []
        arg_types: list = []
        # Track output tensors: index → numpy array (for reading back results)
        output_indices: list[int] = []

        for i, (meta, arg) in enumerate(zip(self._param_metas, args)):
            if meta["kind"] == "tensor":
                arr = _to_numpy(arg)
                arr = np.ascontiguousarray(arr)
                ptr_type = _np_to_ctypes_ptr(_np_dtype_to_c_np(meta.get("dtype", "fp32")))
                c_args.append(arr.ctypes.data_as(ptr_type))
                arg_types.append(ptr_type)
                for d in arr.shape:
                    c_args.append(c_int64(d))
                    arg_types.append(c_int64)
                if meta.get("direction") == "out":
                    output_indices.append(i)
            else:
                val, vtype = _scalar_to_ctypes(arg, meta.get("dtype", "index"))
                c_args.append(val)
                arg_types.append(vtype)

        # Set up the ctypes function signature
        self._func.argtypes = arg_types
        self._func.restype = None

        # Call
        self._func(*c_args)

        # Return results from output tensors (read back after in-place modification)
        results = []
        for i in output_indices:
            arr = _to_numpy(args[i])
            results.append(torch.from_numpy(arr.copy()))

        if len(results) == 0:
            return None
        if len(results) == 1:
            return results[0]
        return tuple(results)


# ── conversion helpers ───────────────────────────────────────────────────────


def _to_numpy(arg) -> np.ndarray:
    """Convert a tensor-like argument to a NumPy array."""
    import torch

    if isinstance(arg, torch.Tensor):
        return arg.detach().cpu().numpy()
    if isinstance(arg, np.ndarray):
        return arg
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(arg).__name__}")


def _np_dtype_to_c_np(dtype_name: str):
    """Map PyPTO dtype name to numpy dtype."""
    mapping = {
        "fp32": np.float32,
        "fp64": np.float64,
        "fp16": np.float16,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool_": np.bool_,
        "index": np.int64,
    }
    return mapping.get(str(dtype_name).lower(), np.float32)


def _np_to_ctypes_ptr(dtype) -> type:
    """Map numpy dtype to corresponding ctypes pointer type."""
    import ctypes

    mapping = {
        np.dtype(np.float32): ctypes.POINTER(ctypes.c_float),
        np.dtype(np.float64): ctypes.POINTER(ctypes.c_double),
        np.dtype(np.int32): ctypes.POINTER(ctypes.c_int32),
        np.dtype(np.int64): ctypes.POINTER(ctypes.c_int64),
    }
    for key, val in mapping.items():
        if np.dtype(dtype) == key:
            return val
    return ctypes.POINTER(ctypes.c_float)


def _scalar_to_ctypes(value, dtype_name: str) -> tuple:
    import ctypes

    name = str(dtype_name).lower()
    if name in ("int32",):
        return ctypes.c_int32(value), ctypes.c_int32
    if name in ("int64", "index"):
        return ctypes.c_int64(value), ctypes.c_int64
    if name in ("fp64", "float64", "double"):
        return ctypes.c_double(float(value)), ctypes.c_double
    return ctypes.c_float(float(value)), ctypes.c_float
