/*
 * PyPTO CPU tensor tile types — polymorphic 2D tile abstraction (PR 1: minimal).
 *
 * Matmul, zero/fill, and elementwise ops added in PR 2.
 */

#ifndef PYPTO_CPU_TENSOR_TYPES_H_
#define PYPTO_CPU_TENSOR_TYPES_H_

#include <cstdint>

template <int M_, int N_>
struct ScalarTile {
    static constexpr int M = M_;
    static constexpr int N = N_;

    float data[M][N];

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
};

#endif  // PYPTO_CPU_TENSOR_TYPES_H_
