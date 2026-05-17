"""
CPU compilation driver — generates C from transformed IR, invokes g++, loads .so.
"""

import os
import shutil
import subprocess
import tempfile

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes
from pypto.pypto_core.ir import ParamDirection, is_incore_type

from .cpu_codegen import CCodegen, _sanitize_name

_CPU_VEC_TYPES_H = os.path.join(os.path.dirname(__file__), "cpu_vec_types.h")
_CPU_TENSOR_TYPES_H = os.path.join(os.path.dirname(__file__), "cpu_tensor_types.h")


def cpu_compile(
    program: _ir_core.Program,
    *,
    vec_width: int = 1,
    opt_level: int = 2,
    work_dir: str | None = None,
    verbose: bool = False,
):
    """Compile an already-transformed ir.Program to CPU-native machine code.

    The caller is responsible for running the pass pipeline (via PassManager
    with CPUDefault strategy) before calling this function.

    Args:
        program: Transformed IR Program (passes already applied).
        vec_width: Vector width (1=scalar, 4=SSE, 8=AVX, 16=AVX-512).
        opt_level: GCC optimization level (default: 2).
        work_dir: Working directory for .c and .so artifacts.  If None, creates a
            temp directory (not cleaned up automatically).
        verbose: Print GCC output.

    Returns:
        A CPUCompiledFunction callable wrapper.
    """
    from pypto.runtime.cpu_compiled import CPUCompiledFunction  # noqa: PLC0415

    # 1. Find the entry-point function — matches PTOCodegen's InCore iteration.
    #    For single-kernel programs (no outlining), falls back to first non-Inline.
    incore_funcs = [f for f in program.functions.values() if is_incore_type(f.func_type)]
    if incore_funcs:
        entry_func = incore_funcs[0]
    else:
        for func in program.functions.values():
            if str(func.func_type) != "Inline":
                entry_func = func
                break
        else:
            raise ValueError("Program has no InCore or non-Inline functions to compile")

    # 2. Generate C code
    vec_type = f"FP32Vec{vec_width}"
    tile_type = "ScalarTile"
    codegen = CCodegen(vec_type=vec_type, tile_type=tile_type)
    c_source = codegen.generate_function(entry_func)

    # 3. Write to work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="pypto_cpu_")
    os.makedirs(work_dir, exist_ok=True)

    func_name_safe = _sanitize_name(entry_func.name)
    c_path = os.path.join(work_dir, f"{func_name_safe}.cc")
    so_path = os.path.join(work_dir, f"{func_name_safe}.so")

    with open(c_path, "w") as f:
        f.write(c_source)

    # Ship the runtime headers alongside the generated .cc
    for header in (_CPU_VEC_TYPES_H, _CPU_TENSOR_TYPES_H):
        if os.path.exists(header):
            shutil.copy2(header, os.path.join(work_dir, os.path.basename(header)))

    # 4. Compile with g++
    gpp_cmd = [
        "g++",
        "-fopenmp",
        f"-O{opt_level}",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-I", work_dir,
        "-o", so_path,
        c_path,
        "-lm",
    ]
    if verbose:
        print(f"[cpu_compile] {' '.join(gpp_cmd)}")
    try:
        result = subprocess.run(gpp_cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "g++ not found. Install g++ to use the CPU backend."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed:\n{result.stderr}\n\nGenerated C source:\n{c_source}"
        )

    # 5. Load via ctypes
    import ctypes

    lib = ctypes.CDLL(so_path)

    # 6. Build metadata for the wrapper
    params = entry_func.params
    param_dirs = entry_func.param_directions if hasattr(entry_func, "param_directions") else []
    param_metas: list[dict] = []
    for i, p in enumerate(params):
        meta = {"name": p.name_hint}
        direction = param_dirs[i] if i < len(param_dirs) else ParamDirection.In
        if hasattr(p.type, "shape") and hasattr(p.type, "dtype"):
            meta["kind"] = "tensor"
            meta["dtype"] = str(p.type.dtype)
            meta["direction"] = "out" if direction == ParamDirection.Out else "in"
        else:
            meta["kind"] = "scalar"
            meta["dtype"] = str(p.type.dtype) if hasattr(p.type, "dtype") else "index"
        param_metas.append(meta)

    return_types = entry_func.return_types if hasattr(entry_func, "return_types") else []
    return_metas: list[dict] = []
    for rt in return_types:
        meta = {"dtype": str(rt.dtype) if hasattr(rt, "dtype") else "fp32"}
        if hasattr(rt, "shape"):
            meta["shape"] = [int(str(d)) for d in rt.shape]
        return_metas.append(meta)

    return CPUCompiledFunction(
        lib=lib,
        func_name=func_name_safe,
        param_metas=param_metas,
        return_metas=return_metas,
    )
