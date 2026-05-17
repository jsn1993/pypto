"""Tests for advanced CPU ops — reduce, transpose, cast, reshape."""

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
    prog = Program([func], f"cpu_adv_{_compile_n}", func.span)
    return _ir_compile(prog, backend_type=BackendType.CPU)


# ═══════ reduce ops ════════════════════════════════════════════════════

def test_sum_axis_1():
    """Row-wise sum: MxN -> Mx1."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 1], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4])
        return pl.store(pl.sum(t, 1), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    o = torch.zeros(4, 1)
    expected = a.sum(dim=1, keepdim=True)
    assert torch.allclose(run(a, o), expected)


def test_max_axis_1():
    """Row-wise max: MxN -> Mx1."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 1], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4])
        return pl.store(pl.max(t, 1), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.randn(4, 4)
    o = torch.zeros(4, 1)
    expected = a.max(dim=1, keepdim=True).values
    assert torch.allclose(run(a, o), expected)


# ═══════ transform ops ═════════════════════════════════════════════════

def test_transpose():
    """Transpose MxN -> NxM."""

    @pl.function
    def kernel_(a: pl.Tensor[[3, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 3], pl.FP32]]):
        t = pl.load(a, [0, 0], [3, 4])
        return pl.store(pl.transpose(t, 0, 1), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])
    o = torch.zeros(4, 3)
    expected = a.T
    assert torch.allclose(run(a, o), expected)


def test_reshape():
    """Reshape MxN -> KxL (same element count)."""

    @pl.function
    def kernel_(a: pl.Tensor[[2, 6], pl.FP32], o: pl.Out[pl.Tensor[[3, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [2, 6])
        return pl.store(pl.reshape(t, [3, 4]), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    o = torch.zeros(3, 4)
    expected = a.reshape(3, 4)
    assert torch.allclose(run(a, o), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
