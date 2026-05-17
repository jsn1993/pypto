/*
 * PyPTO CPU vector types — polymorphic scalar/SIMD abstraction (PR 1: minimal).
 *
 * Full vector API (operators, unary, FMA) is added in PR 2.
 */

#ifndef PYPTO_CPU_VEC_TYPES_H_
#define PYPTO_CPU_VEC_TYPES_H_

#define PYPTO_FORCE_INLINE __attribute__((always_inline)) inline

template <typename T>
struct Vec {
    static constexpr int get_elem_num() { return T::VEC_ELEM_NUM; }
};

template <int N>
struct Storage { float val[N]; };

template <int N>
struct FP32VecImpl : public Vec<FP32VecImpl<N>> {
    static constexpr int VEC_ELEM_NUM = N;
    Storage<N> reg;

    explicit FP32VecImpl(float v) { reg.val[0] = v; }
    explicit FP32VecImpl()       { reg.val[0] = 0.0f; }
    explicit FP32VecImpl(const float* ptr)
        : reg(*reinterpret_cast<const Storage<N>*>(ptr)) {}

    void save(float* ptr) const {
        *reinterpret_cast<Storage<N>*>(ptr) = reg;
    }
};

using FP32Vec1  = FP32VecImpl<1>;
using FP32Vec4  = FP32VecImpl<4>;
using FP32Vec8  = FP32VecImpl<8>;
using FP32Vec16 = FP32VecImpl<16>;

#endif  // PYPTO_CPU_VEC_TYPES_H_
