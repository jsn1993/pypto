"""Comprehensive CPU operator tests — ported from Ascend example kernels."""

import pytest
import torch

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.compile import compile as _ir_compile
from pypto.pypto_core.ir import Program


# ═══════ helpers ══════════════════════════════════════════════════════

_compile_n = 0


def _compile(func):
    global _compile_n
    _compile_n += 1
    prog = Program([func], f"cpu_{_compile_n}", func.span)
    return _ir_compile(prog, backend_type=BackendType.CPU)


# ═══════ binary elementwise ═══════════════════════════════════════════

def test_add():
    @pl.function
    def add_kernel(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4]); tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.add(ta, tb), [0, 0], o)
    run = _compile(add_kernel)
    a, b, c = torch.full((4, 4), 2.0), torch.full((4, 4), 3.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, b, c), a + b)

def test_sub():
    @pl.function
    def sub_kernel(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4]); tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.sub(ta, tb), [0, 0], o)
    run = _compile(sub_kernel)
    a, b, c = torch.full((4, 4), 5.0), torch.full((4, 4), 2.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, b, c), a - b)

def test_mul_tt():
    @pl.function
    def mul_tt_kernel(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4]); tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.mul(ta, tb), [0, 0], o)
    run = _compile(mul_tt_kernel)
    a, b, c = torch.full((4, 4), 3.0), torch.full((4, 4), 4.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, b, c), a * b)

def test_div():
    @pl.function
    def div_kernel(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4]); tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.div(ta, tb), [0, 0], o)
    run = _compile(div_kernel)
    a, b, c = torch.full((4, 4), 6.0), torch.full((4, 4), 2.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, b, c), a / b)

# ═══════ scalar-tile binary ═══════════════════════════════════════════

def test_adds():
    @pl.function
    def adds_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.add(t, 1.0), [0, 0], o)
    run = _compile(adds_kernel)
    a, c = torch.full((4, 4), 3.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a + 1.0)

def test_subs():
    @pl.function
    def subs_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.sub(t, 2.0), [0, 0], o)
    run = _compile(subs_kernel)
    a, c = torch.full((4, 4), 5.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a - 2.0)

def test_muls():
    @pl.function
    def muls_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.mul(t, 3.0), [0, 0], o)
    run = _compile(muls_kernel)
    a, c = torch.full((4, 4), 2.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a * 3.0)

def test_divs():
    @pl.function
    def divs_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.div(t, 2.0), [0, 0], o)
    run = _compile(divs_kernel)
    a, c = torch.full((4, 4), 6.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a / 2.0)

# ═══════ unary ════════════════════════════════════════════════════════

def test_neg():
    @pl.function
    def neg_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.neg(t), [0, 0], o)
    run = _compile(neg_kernel)
    a, c = torch.full((4, 4), 3.0), torch.zeros(4, 4)
    assert torch.allclose(run(a, c), -a)

def test_abs():
    @pl.function
    def abs_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.abs(t), [0, 0], o)
    run = _compile(abs_kernel)
    a = torch.tensor([[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0],
                      [-9.0, 10.0, -11.0, 12.0], [13.0, -14.0, 15.0, -16.0]])
    c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a.abs())

def test_exp():
    @pl.function
    def exp_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.exp(t), [0, 0], o)
    run = _compile(exp_kernel)
    a = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 4.0
    c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a.exp(), rtol=1e-4, atol=1e-4)

def test_rsqrt():
    @pl.function
    def rsqrt_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.rsqrt(t), [0, 0], o)
    run = _compile(rsqrt_kernel)
    a = torch.full((4, 4), 4.0); c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), torch.rsqrt(a), rtol=1e-5, atol=1e-5)

def test_recip():
    @pl.function
    def recip_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.recip(t), [0, 0], o)
    run = _compile(recip_kernel)
    a = torch.full((4, 4), 2.0); c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), 1.0 / a)

def test_log():
    @pl.function
    def log_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.log(t), [0, 0], o)
    run = _compile(log_kernel)
    a = torch.full((4, 4), 2.0); c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a.log(), rtol=1e-5, atol=1e-5)

def test_relu():
    @pl.function
    def relu_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.relu(t), [0, 0], o)
    run = _compile(relu_kernel)
    a = torch.tensor([[-1.0, 2.0, -3.0, 4.0],
                      [5.0, -6.0, 7.0, -8.0],
                      [-9.0, 10.0, -11.0, 12.0],
                      [13.0, -14.0, 15.0, -16.0]])
    c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a.relu())

def test_sqrt():
    @pl.function
    def sqrt_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4]); return pl.store(pl.sqrt(t), [0, 0], o)
    run = _compile(sqrt_kernel)
    a = torch.full((4, 4), 4.0); c = torch.zeros(4, 4)
    assert torch.allclose(run(a, c), a.sqrt())

# ═══════ matmul ═══════════════════════════════════════════════════════

def test_matmul():
    @pl.function
    def matmul_kernel(a: pl.Tensor[[4, 4], pl.FP32], b: pl.Tensor[[4, 4], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        ta = pl.load(a, [0, 0], [4, 4]); tb = pl.load(b, [0, 0], [4, 4])
        return pl.store(pl.matmul(ta, tb), [0, 0], o)
    run = _compile(matmul_kernel)
    a = torch.randn(4, 4); b = torch.randn(4, 4); c = torch.zeros(4, 4)
    assert torch.allclose(run(a, b, c), a @ b, rtol=1e-4, atol=1e-4)

def test_matmul_rect():
    @pl.function
    def matmul_rect_kernel(a: pl.Tensor[[2, 6], pl.FP32], b: pl.Tensor[[6, 3], pl.FP32],
          o: pl.Out[pl.Tensor[[2, 3], pl.FP32]]):
        ta = pl.load(a, [0, 0], [2, 6]); tb = pl.load(b, [0, 0], [6, 3])
        return pl.store(pl.matmul(ta, tb), [0, 0], o)
    run = _compile(matmul_rect_kernel)
    a = torch.randn(2, 6); b = torch.randn(6, 3); c = torch.zeros(2, 3)
    assert torch.allclose(run(a, b, c), a @ b, rtol=1e-4, atol=1e-4)

def test_gemv():
    @pl.function
    def gemv_kernel(a: pl.Tensor[[1, 4], pl.FP32], b: pl.Tensor[[4, 3], pl.FP32],
          o: pl.Out[pl.Tensor[[1, 3], pl.FP32]]):
        ta = pl.load(a, [0, 0], [1, 4]); tb = pl.load(b, [0, 0], [4, 3])
        return pl.store(pl.gemv(ta, tb), [0, 0], o)
    run = _compile(gemv_kernel)
    a = torch.randn(1, 4); b = torch.randn(4, 3); c = torch.zeros(1, 3)
    assert torch.allclose(run(a, b, c), a @ b, rtol=1e-4, atol=1e-4)

# ═══════ fusion patterns ══════════════════════════════════════════════

def test_silu():
    @pl.function
    def silu_kernel(x: pl.Tensor[[4, 8], pl.FP32], o: pl.Out[pl.Tensor[[4, 8], pl.FP32]]):
        t = pl.load(x, [0, 0], [4, 8])
        neg_x = pl.mul(t, -1.0); denom = pl.add(pl.exp(neg_x), 1.0)
        return pl.store(pl.mul(t, pl.recip(denom)), [0, 0], o)
    run = _compile(silu_kernel)
    x = torch.randn(4, 8); c = torch.zeros(4, 8)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(run(x, c), expected, rtol=1e-3, atol=1e-3)

def test_gelu():
    @pl.function
    def gelu_kernel(x: pl.Tensor[[4, 8], pl.FP32], o: pl.Out[pl.Tensor[[4, 8], pl.FP32]]):
        t = pl.load(x, [0, 0], [4, 8])
        scaled = pl.mul(t, 1.702); denom = pl.add(pl.exp(pl.mul(scaled, -1.0)), 1.0)
        return pl.store(pl.mul(t, pl.recip(denom)), [0, 0], o)
    run = _compile(gelu_kernel)
    x = torch.randn(4, 8); c = torch.zeros(4, 8)
    expected = x * torch.sigmoid(1.702 * x)
    assert torch.allclose(run(x, c), expected, rtol=1e-3, atol=1e-3)

def test_swiglu():
    @pl.function
    def swiglu_kernel(gate: pl.Tensor[[4, 8], pl.FP32], up: pl.Tensor[[4, 8], pl.FP32],
          o: pl.Out[pl.Tensor[[4, 8], pl.FP32]]):
        tg = pl.load(gate, [0, 0], [4, 8]); tu = pl.load(up, [0, 0], [4, 8])
        denom = pl.add(pl.exp(pl.mul(tg, -1.0)), 1.0)
        swish = pl.mul(tg, pl.recip(denom))
        return pl.store(pl.mul(swish, tu), [0, 0], o)
    run = _compile(swiglu_kernel)
    gate = torch.randn(4, 8); up = torch.randn(4, 8); c = torch.zeros(4, 8)
    expected = gate * torch.sigmoid(gate) * up
    assert torch.allclose(run(gate, up, c), expected, rtol=1e-3, atol=1e-3)

def test_fused_add_scale():
    @pl.function
    def fused_add_scale_kernel(a: pl.Tensor[[8, 8], pl.FP32], b: pl.Tensor[[8, 8], pl.FP32],
          o: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        ta = pl.load(a, [0, 0], [8, 8]); tb = pl.load(b, [0, 0], [8, 8])
        return pl.store(pl.mul(pl.add(ta, tb), 2.0), [0, 0], o)
    run = _compile(fused_add_scale_kernel)
    a, b, c = torch.full((8, 8), 2.0), torch.full((8, 8), 3.0), torch.zeros(8, 8)
    assert torch.allclose(run(a, b, c), (a + b) * 2.0)

# ═══════ output-direction tracking ═══════════════════════════════════

def test_out_param_only_returned():
    """Only Out-annotated parameters appear in compiled result."""

    @pl.function
    def out_param_only_returned_kernel(a: pl.Tensor[[4, 4], pl.FP32], o: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(a, [0, 0], [4, 4])
        return pl.store(pl.mul(t, 2.0), [0, 0], o)

    run = _compile(out_param_only_returned_kernel)
    a = torch.full((4, 4), 3.0)
    o = torch.zeros(4, 4)
    result = run(a, o)
    assert torch.allclose(result, a * 2.0)


# ═══════ main ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
