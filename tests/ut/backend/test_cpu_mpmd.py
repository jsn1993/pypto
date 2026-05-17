"""MPMD tests for CPU backend — multiple incore blocks compiled to OpenMP sections."""

import pytest
import torch

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.compile import compile as _ir_compile
from pypto.pypto_core.ir import Program

_compile_n = 0


def _compile(func):
    global _compile_n
    _compile_n += 1
    prog = Program([func], f"cpu_mpmd_{_compile_n}", func.span)
    return _ir_compile(prog, backend_type=BackendType.CPU)


# ── two independent incore blocks ────────────────────────────────────────────


def test_two_incore_parallel():
    """Two pl.incore() blocks operating on disjoint output regions."""

    @pl.function
    def two_kernel(
        a: pl.Tensor[[4, 8], pl.FP32], o: pl.Out[pl.Tensor[[4, 8], pl.FP32]]
    ):
        with pl.incore():
            t = pl.load(a, [0, 0], [4, 4])
            pl.store(pl.mul(t, 2.0), [0, 0], o)
        with pl.incore():
            t = pl.load(a, [0, 4], [4, 4])
            pl.store(pl.add(t, 1.0), [0, 4], o)

    run = _compile(two_kernel)
    a = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ]
    )
    o = torch.zeros(4, 8)
    result = run(a, o)
    expected = torch.cat([a[:, :4] * 2.0, a[:, 4:] + 1.0], dim=1)
    assert torch.allclose(result, expected)


def test_three_incore_parallel():
    """Three pl.incore() blocks — regions of different widths."""

    @pl.function
    def three_kernel(
        a: pl.Tensor[[4, 9], pl.FP32], o: pl.Out[pl.Tensor[[4, 9], pl.FP32]]
    ):
        with pl.incore():
            t = pl.load(a, [0, 0], [4, 3])
            pl.store(pl.mul(t, 2.0), [0, 0], o)
        with pl.incore():
            t = pl.load(a, [0, 3], [4, 3])
            pl.store(pl.add(t, 1.0), [0, 3], o)
        with pl.incore():
            t = pl.load(a, [0, 6], [4, 3])
            pl.store(pl.sub(t, 1.0), [0, 6], o)

    run = _compile(three_kernel)
    a = torch.randn(4, 9)
    o = torch.zeros(4, 9)
    result = run(a, o)
    expected = torch.cat(
        [a[:, :3] * 2.0, a[:, 3:6] + 1.0, a[:, 6:9] - 1.0], dim=1
    )
    assert torch.allclose(result, expected)


def test_single_incore():
    """Single pl.incore() block — omp parallel, still correct."""

    @pl.function
    def single_kernel(
        a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]
    ):
        with pl.incore():
            t = pl.load(a, [0, 0], [4, 4])
            pl.store(pl.mul(t, 3.0), [0, 0], o)

    run = _compile(single_kernel)
    a = torch.randn(4, 4)
    o = torch.zeros(4, 4)
    result = run(a, o)
    assert torch.allclose(result, a * 3.0)


def test_incore_with_tile_create():
    """InCore blocks that each create their own intermediate tiles."""

    @pl.function
    def create_tile_kernel(
        a: pl.Tensor[[4, 8], pl.FP32], o: pl.Out[pl.Tensor[[4, 8], pl.FP32]]
    ):
        with pl.incore():
            t = pl.load(a, [0, 0], [4, 4])
            tmp = pl.create_tile([4, 4], dtype=pl.FP32)
            pl.store(pl.mul(t, 2.0), [0, 0], o)
        with pl.incore():
            t = pl.load(a, [0, 4], [4, 4])
            pl.store(pl.exp(t), [0, 4], o)

    run = _compile(create_tile_kernel)
    a = torch.abs(torch.randn(4, 8))
    o = torch.zeros(4, 8)
    result = run(a, o)
    expected = torch.cat([a[:, :4] * 2.0, a[:, 4:].exp()], dim=1)
    assert torch.allclose(result, expected, rtol=1e-4)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
