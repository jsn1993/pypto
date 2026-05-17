/*
 * PyPTO CPU vector types — polymorphic scalar/SIMD abstraction.
 *
 * Usage in generated code::
 *
 *   #include "cpu_vec_types.h"
 *
 *   for (int64_t j = 0; j < N; j += VEC_T::VEC_ELEM_NUM) {
 *       VEC_T va(&a[i * ld + j]);
 *       VEC_T vb(&b[i * ld + j]);
 *       VEC_T vc = va + vb;
 *       vc.save(&c[i * ld + j]);
 *   }
 *
 * The *VEC_T* macro (FP32Vec1 / FP32Vec4 / FP32Vec8 / FP32Vec16) is supplied
 * by the codegen as a ``-DVEC_T=FP32Vec1`` compiler flag.
 */

#ifndef PYPTO_CPU_VEC_TYPES_H_
#define PYPTO_CPU_VEC_TYPES_H_

#include <cmath>
#include <cstdint>
#include <cstring>

#define PYPTO_FORCE_INLINE __attribute__((always_inline)) inline

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time loop unroller (C++17 fold expression)
// ═══════════════════════════════════════════════════════════════════════════

template <typename T, T... indexes, typename F>
constexpr void _unroll_item(std::integer_sequence<T, indexes...>, F&& f) {
    (f(std::integral_constant<T, indexes>{}), ...);
}

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
    _unroll_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

// ═══════════════════════════════════════════════════════════════════════════
// CRTP base
// ═══════════════════════════════════════════════════════════════════════════

template <typename T>
struct Vec {
    static constexpr int get_elem_num() { return T::VEC_ELEM_NUM; }
};

// ═══════════════════════════════════════════════════════════════════════════
// Storage<N> and FP32VecImpl<N>
// ═══════════════════════════════════════════════════════════════════════════

template <int N>
struct Storage { float val[N]; };

template <int N>
struct FP32VecImpl : public Vec<FP32VecImpl<N>> {
    static constexpr int VEC_ELEM_NUM = N;
    Storage<N> reg;

    explicit FP32VecImpl(float v) {
        if constexpr (N == 1) { reg.val[0] = v; }
        else { unroll_loop<int, N>([&v, this](int i) { reg.val[i] = v; }); }
    }
    explicit FP32VecImpl() {
        if constexpr (N == 1) { reg.val[0] = 0.0f; }
        else { unroll_loop<int, N>([this](int i) { reg.val[i] = 0.0f; }); }
    }
    explicit FP32VecImpl(const float* ptr)
        : reg(*reinterpret_cast<const Storage<N>*>(ptr)) {}

    // -- arithmetic ---------------------------------------------------------
    PYPTO_FORCE_INLINE FP32VecImpl operator+(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(reg.val[0] + b.reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = reg.val[i] + b.reg.val[i]; }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl operator-(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(reg.val[0] - b.reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = reg.val[i] - b.reg.val[i]; }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl operator*(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(reg.val[0] * b.reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = reg.val[i] * b.reg.val[i]; }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl operator/(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(reg.val[0] / b.reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = reg.val[i] / b.reg.val[i]; }); return FP32VecImpl(r); }
    }

    // -- comparisons --------------------------------------------------------
    PYPTO_FORCE_INLINE FP32VecImpl max(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(fmaxf(reg.val[0], b.reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = fmaxf(reg.val[i], b.reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl min(const FP32VecImpl& b) const {
        if constexpr (N == 1) return FP32VecImpl(fminf(reg.val[0], b.reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = fminf(reg.val[i], b.reg.val[i]); }); return FP32VecImpl(r); }
    }

    // -- unary --------------------------------------------------------------
    PYPTO_FORCE_INLINE FP32VecImpl exp() const {
        if constexpr (N == 1) return FP32VecImpl(expf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = expf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl log() const {
        if constexpr (N == 1) return FP32VecImpl(logf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = logf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl sqrt() const {
        if constexpr (N == 1) return FP32VecImpl(sqrtf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = sqrtf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl rsqrt() const {
        if constexpr (N == 1) return FP32VecImpl(1.0f / sqrtf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = 1.0f / sqrtf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl neg() const {
        if constexpr (N == 1) return FP32VecImpl(-reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = -reg.val[i]; }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl abs() const {
        if constexpr (N == 1) return FP32VecImpl(fabsf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = fabsf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl sin() const {
        if constexpr (N == 1) return FP32VecImpl(sinf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = sinf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl cos() const {
        if constexpr (N == 1) return FP32VecImpl(cosf(reg.val[0]));
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = cosf(reg.val[i]); }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl recip() const {
        if constexpr (N == 1) return FP32VecImpl(1.0f / reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = 1.0f / reg.val[i]; }); return FP32VecImpl(r); }
    }
    PYPTO_FORCE_INLINE FP32VecImpl relu() const {
        if constexpr (N == 1) return FP32VecImpl(reg.val[0] < 0.0f ? 0.0f : reg.val[0]);
        else { Storage<N> r; unroll_loop<int, N>([&](int i) { r.val[i] = reg.val[i] < 0.0f ? 0.0f : reg.val[i]; }); return FP32VecImpl(r); }
    }

    // -- store --------------------------------------------------------------
    void save(float* ptr) const {
        *reinterpret_cast<Storage<N>*>(ptr) = reg;
    }

 private:
    explicit FP32VecImpl(Storage<N> data) : reg(data) {}
};

// ═══════════════════════════════════════════════════════════════════════════
// Public type aliases (used by generated code)
// ═══════════════════════════════════════════════════════════════════════════

using FP32Vec1  = FP32VecImpl<1>;
using FP32Vec4  = FP32VecImpl<4>;
using FP32Vec8  = FP32VecImpl<8>;
using FP32Vec16 = FP32VecImpl<16>;

// ═══════════════════════════════════════════════════════════════════════════
// FMA helper — fused multiply-add: acc += a * b
// ═══════════════════════════════════════════════════════════════════════════

template <typename T>
PYPTO_FORCE_INLINE void v_fma(T& acc, const T& a, const T& b) { acc = acc + a * b; }

// ═══════════════════════════════════════════════════════════════════════════
// Cache-line prefetch (non-temporal hint)
// ═══════════════════════════════════════════════════════════════════════════

PYPTO_FORCE_INLINE void v_prefetch(const void* addr) {
    __builtin_prefetch(addr, 0, 3);
}

#endif  // PYPTO_CPU_VEC_TYPES_H_
