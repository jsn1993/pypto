"""Tests for the CPU backend — compilation + execution for scalar tile ops."""

import pytest
import torch

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.compile import compile as _ir_compile
from pypto.pypto_core.ir import Program


_compile_n = 0


def _cpu_compile(func):
    global _compile_n
    _compile_n += 1
    prog = Program([func], f"cpu_{_compile_n}", func.span)
    return _ir_compile(prog, backend_type=BackendType.CPU)


# ── tile.load / tile.store ─────────────────────────────────────────────────


def test_load_store_identity():
    """tile.load + tile.store preserves tensor data."""

    @pl.function
    def copy_kernel(
        a: pl.Tensor[[4, 4], pl.FP32], b: pl.Out[pl.Tensor[[4, 4], pl.FP32]]
    ):
        t = pl.load(a, [0, 0], [4, 4])
        return pl.store(t, [0, 0], b)

    compiled = _cpu_compile(copy_kernel)
    src = torch.randn(4, 4)
    dst = torch.zeros(4, 4)
    result = compiled(src, dst)
    assert torch.allclose(result, src), f"copy failed: {result} vs {src}"


# ── BackendType and handler ────────────────────────────────────────────────


def test_cpu_backend_type_exists():
    """BackendType.CPU is accessible from Python."""
    from pypto.backend import BackendCPU
    assert BackendType.CPU is not None
    inst = BackendCPU.instance()
    assert inst.get_type_name() == "CPU"


def test_cpu_handler_values():
    """CPUBackendHandler returns CPU-appropriate defaults."""
    from pypto.pypto_core.backend import BackendCPU
    handler = BackendCPU.instance().get_handler()
    assert not handler.requires_gm_pipe_buffer()
    assert not handler.requires_vto_c_fractal_adapt()
    assert handler.get_gm_access_granularity_bytes() == 64
    assert handler.get_l2_cache_line_bytes() == 64


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
