"""Stress / edge-case tests for CPU backend.

Focus: larger shapes, boundary conditions, rapid recompilation, numerical
correctness, and parametrized vector widths.
"""

import pytest
import torch

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.compile import compile as _ir_compile
from pypto.pypto_core.ir import Program


_compile_counter = 0


def _cpu_compile(func):
    global _compile_counter
    _compile_counter += 1
    prog = Program([func], f"stress_{_compile_counter}", func.span)
    return _ir_compile(prog, backend_type=BackendType.CPU)


# ═══════ large shapes ═══════════════════════════════════════════════════


@pytest.mark.parametrize("M,N", [
    (32, 32), (64, 64), (128, 128), (256, 4), (4, 256),
])
def test_large_add(M, N):
    """Elementwise add on larger shapes."""

    @pl.function
    def kernel_(a: pl.Tensor[[M, N], pl.FP32], b: pl.Tensor[[M, N], pl.FP32],
                o: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
        ta = pl.load(a, [0, 0], [M, N])
        tb = pl.load(b, [0, 0], [M, N])
        return pl.store(pl.add(ta, tb), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.randn(M, N)
    b = torch.randn(M, N)
    o = torch.zeros(M, N)
    assert torch.allclose(run(a, b, o), a + b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("M,K,N", [
    (16, 16, 16), (32, 32, 32), (64, 64, 64), (8, 64, 8),
])
def test_large_matmul(M, K, N):
    """Matmul on larger shapes."""

    @pl.function
    def kernel_(a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32],
                o: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
        ta = pl.load(a, [0, 0], [M, K])
        tb = pl.load(b, [0, 0], [K, N])
        return pl.store(pl.matmul(ta, tb), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    o = torch.zeros(M, N)
    expected = a @ b
    assert torch.allclose(run(a, b, o), expected, rtol=1e-4, atol=1e-4)


# ═══════ edge shapes ════════════════════════════════════════════════════


def test_1x1_tile():
    """1x1 tile edge case."""

    @pl.function
    def kernel_(a: pl.Tensor[[1, 1], pl.FP32], o: pl.Out[pl.Tensor[[1, 1], pl.FP32]]):
        t = pl.load(a, [0, 0], [1, 1])
        return pl.store(pl.mul(t, 3.0), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.tensor([[7.0]])
    o = torch.zeros(1, 1)
    assert torch.allclose(run(a, o), a * 3.0)


def test_1x64_tile():
    """Very narrow tile (1x64)."""

    @pl.function
    def kernel_(a: pl.Tensor[[1, 64], pl.FP32], o: pl.Out[pl.Tensor[[1, 64], pl.FP32]]):
        t = pl.load(a, [0, 0], [1, 64])
        return pl.store(pl.exp(t), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.randn(1, 64) * 0.5
    o = torch.zeros(1, 64)
    assert torch.allclose(run(a, o), a.exp(), rtol=1e-4, atol=1e-4)


def test_64x1_tile():
    """Very tall tile (64x1)."""

    @pl.function
    def kernel_(a: pl.Tensor[[64, 1], pl.FP32], o: pl.Out[pl.Tensor[[64, 1], pl.FP32]]):
        t = pl.load(a, [0, 0], [64, 1])
        return pl.store(pl.sqrt(t), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.full((64, 1), 4.0)
    o = torch.zeros(64, 1)
    assert torch.allclose(run(a, o), a.sqrt(), rtol=1e-5, atol=1e-5)


# ═══════ numerical edge cases ══════════════════════════════════════════


def test_negative_values():
    """Elementwise ops on negative values."""

    @pl.function
    def kernel_(a: pl.Tensor[[2, 2], pl.FP32], o: pl.Out[pl.Tensor[[2, 2], pl.FP32]]):
        t = pl.load(a, [0, 0], [2, 2])
        return pl.store(pl.abs(t), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.tensor([[-3.0, -1.0], [0.0, 5.0]])
    o = torch.zeros(2, 2)
    assert torch.allclose(run(a, o), a.abs())


def test_zero_tensor():
    """Operations on zero tensors."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
                o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4])
        tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.mul(ta, tb), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.zeros(4, 4)
    b = torch.randn(4, 4)
    o = torch.zeros(4, 4)
    assert torch.allclose(run(a, b, o), a * b)


def test_recip_nonzero():
    """reciprocal on positive values (no div-by-zero)."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4])
        return pl.store(pl.recip(t), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.full((4, 4), 2.0)
    o = torch.zeros(4, 4)
    assert torch.allclose(run(a, o), 1.0 / a)


# ═══════ rapid recompilation ═══════════════════════════════════════════


def test_rapid_recompilation():
    """Compile 5 different kernels in quick succession (no stale .so)."""
    results = []
    # Use distinct kernels to avoid closure-variable issues with @pl.function
    kernels = [
        _make_add_1, _make_add_2, _make_add_3, _make_add_4, _make_add_5,
    ]
    for i, kfunc in enumerate(kernels):
        prog = Program([kfunc], f"recomp_{i}", kfunc.span)
        compiled = _ir_compile(prog, backend_type=BackendType.CPU)
        a = torch.ones(4, 4)
        o = torch.zeros(4, 4)
        result = compiled(a, o)
        assert torch.allclose(result, a + float(i + 1))
        results.append(True)
    assert all(results)


@pl.function
def _make_add_1(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
    t = pl.load(a, [0, 0], [4, 4])
    return pl.store(pl.add(t, 1.0), [0, 0], o)


@pl.function
def _make_add_2(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
    t = pl.load(a, [0, 0], [4, 4])
    return pl.store(pl.add(t, 2.0), [0, 0], o)


@pl.function
def _make_add_3(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
    t = pl.load(a, [0, 0], [4, 4])
    return pl.store(pl.add(t, 3.0), [0, 0], o)


@pl.function
def _make_add_4(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
    t = pl.load(a, [0, 0], [4, 4])
    return pl.store(pl.add(t, 4.0), [0, 0], o)


@pl.function
def _make_add_5(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
    t = pl.load(a, [0, 0], [4, 4])
    return pl.store(pl.add(t, 5.0), [0, 0], o)


# ═══════ chained ops ═══════════════════════════════════════════════════


def test_chained_three_ops():
    """Three elementwise ops chained: add -> mul -> add."""

    @pl.function
    def kernel_(a: pl.Tensor[[8, 8], pl.FP32], o: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        t = pl.load(a, [0, 0], [8, 8])
        t1 = pl.add(t, 1.0)
        t2 = pl.mul(t1, 2.0)
        return pl.store(pl.add(t2, 3.0), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.full((8, 8), 1.0)
    o = torch.zeros(8, 8)
    expected = ((a + 1.0) * 2.0) + 3.0
    assert torch.allclose(run(a, o), expected)


def test_mixed_matmul_add():
    """Matmul then elementwise add: (A @ B) + C."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
                c: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4])
        tb = pl.load(b, [0, 0], [4, 4])
        tc = pl.load(c, [0, 0], [4, 4])
        tm = pl.matmul(ta, tb)
        return pl.store(pl.add(tm, tc), [0, 0], o)

    run = _cpu_compile(kernel_)
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    c = torch.randn(4, 4)
    o = torch.zeros(4, 4)
    expected = a @ b + c
    assert torch.allclose(run(a, b, c, o), expected, rtol=1e-4, atol=1e-4)


# ═══════ many inputs ═══════════════════════════════════════════════════


def test_five_inputs():
    """Kernel with 5 input tensors."""

    @pl.function
    def kernel_(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
                c: pl.Tensor[[4, 4], pl.FP32], d: pl.Tensor[[4, 4], pl.FP32],
                e: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4])
        tb = pl.load(b, [0, 0], [4, 4])
        tc = pl.load(c, [0, 0], [4, 4])
        td = pl.load(d, [0, 0], [4, 4])
        te = pl.load(e, [0, 0], [4, 4])
        t1 = pl.add(pl.add(pl.add(pl.add(ta, tb), tc), td), te)
        return pl.store(t1, [0, 0], o)

    run = _cpu_compile(kernel_)
    tensors = [torch.randn(4, 4) for _ in range(5)]
    o = torch.zeros(4, 4)
    expected = sum(tensors)
    assert torch.allclose(run(*tensors, o), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
