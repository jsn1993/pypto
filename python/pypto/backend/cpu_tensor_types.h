/*
 * PyPTO CPU tensor tile types — polymorphic 2D tile abstraction.
 *
 * Usage in generated code::
 *
 *   #include "cpu_tensor_types.h"
 *
 *   ScalarTile<128, 128> ta;
 *   ta.load(tensor_a + base_off, stride);
 *   ScalarTile<128, 128> tb;
 *   tb.load(tensor_b + base_off, stride);
 *   ScalarTile<128, 128> tc;
 *   tc.matmul(ta, tb);
 *   tc.store(tensor_c + base_off, stride);
 *
 * The *TILE_T* macro is supplied by the codegen as ``-DTILE_T=ScalarTile``
 * (or ``AMXTile`` / ``SMETile`` when those backends are added).
 */

#ifndef PYPTO_CPU_TENSOR_TYPES_H_
#define PYPTO_CPU_TENSOR_TYPES_H_

#include <cmath>
#include <cstdint>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════
// ScalarTile<M, N> — pure scalar 2D tile
//
// Operations are plain C loops; the compiler auto-vectorizes the inner
// dimension when vec_width > 1.  AMX / SME specialisations will provide
// the same API surface backed by tile-register intrinsics.
// ═══════════════════════════════════════════════════════════════════════════

template <int M_, int N_>
struct ScalarTile {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int elem_count = M * N;

    float data[M][N];

    // -- load / store -------------------------------------------------------
    void load(const float* src, int64_t ld) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                data[i][j] = src[i * ld + j];
    }

    void store(float* dst, int64_t ld) const {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                dst[i * ld + j] = data[i][j];
    }

    // -- zero / fill --------------------------------------------------------
    void zero() { std::memset(data, 0, sizeof(data)); }

    void fill(float v) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                data[i][j] = v;
    }

    // -- matmul: this = a @ b -----------------------------------------------
    template <int K>
    void matmul(const ScalarTile<M, K>& a, const ScalarTile<K, N>& b) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += a.data[i][k] * b.data[k][j];
                data[i][j] = sum;
            }
        }
    }

    // -- matmul_acc: this += a @ b ------------------------------------------
    template <int K>
    void matmul_acc(const ScalarTile<M, K>& a, const ScalarTile<K, N>& b) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += a.data[i][k] * b.data[k][j];
                data[i][j] += sum;
            }
        }
    }

};

// ═══════════════════════════════════════════════════════════════════════════
// Convenience alias — set by codegen via -DTILE_T=ScalarTile
// ═══════════════════════════════════════════════════════════════════════════

#ifndef TILE_T
#define TILE_T ScalarTile
#endif

#endif  // PYPTO_CPU_TENSOR_TYPES_H_
